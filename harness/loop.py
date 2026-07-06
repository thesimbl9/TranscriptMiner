"""
harness/loop.py — AgentLoop 状态机

解决的问题（失控模式 #1）：
  长任务崩溃后无法恢复。run_agent.py 是顺序脚本，跑到第 25 轮 API 挂了，
  前 24 轮的状态全丢。没有断点续跑，没有异常隔离。

Harness 解法：
  显式状态机 + 每步 checkpoint + 启动时检测 checkpoint 自动恢复。
  每次状态转换都记录到 Tracer。

面试要点：
  - "我的 Agent Loop 有显式状态机，不是顺序脚本"
  - "长任务中断后可从 checkpoint 恢复，不是从头跑"
  - "接近 token 预算上限时自动注入收敛指令"

使用：
  loop = AgentLoop(tracer=tracer, checkpoint_dir=...)
  loop.register_tool("hypothesis", hypothesis_fn, ...)
  loop.run(max_iter=31, token_budget=500_000)
"""

from __future__ import annotations

import json
import signal
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable

from harness.tracer import Tracer


# ═══════════════════════════════════════════════════════════════════════════════════
# 状态定义
# ═══════════════════════════════════════════════════════════════════════════════════

class AgentState(Enum):
    """Agent Loop 的显式状态

    每个状态有明确的进入条件和退出动作。
    状态转换通过 Tracer 记录，支持回放。
    """
    INIT        = auto()   # 初始化：加载 checkpoint / 准备资源
    PLANNING    = auto()   # 规划：HypothesisAgent 生成 feature_spec
    ACTING      = auto()   # 执行：ExtractionAgent 检索 + 打分
    OBSERVING   = auto()   # 观察：ValidationAgent 计算统计量
    EVALUATING  = auto()   # 评估：GovernanceAgent 硬规则检查
    DIAGNOSING  = auto()   # 诊断：DiagnosisAgent 失败根因分析（仅 FAIL 时）
    FINALIZING  = auto()   # 收尾：写入结果 / 更新 Memory
    RETRYING    = auto()   # 重试：单步失败后的恢复状态
    DONE        = auto()   # 完成：所有迭代结束
    FAILED      = auto()   # 失败：不可恢复的错误


# 合法的状态转换
_TRANSITIONS: dict[AgentState, set[AgentState]] = {
    AgentState.INIT:        {AgentState.PLANNING, AgentState.DONE, AgentState.FAILED},
    AgentState.PLANNING:    {AgentState.ACTING, AgentState.RETRYING, AgentState.FAILED},
    AgentState.ACTING:      {AgentState.OBSERVING, AgentState.RETRYING, AgentState.FAILED},
    AgentState.OBSERVING:   {AgentState.EVALUATING, AgentState.RETRYING, AgentState.FAILED},
    AgentState.EVALUATING:  {AgentState.DIAGNOSING, AgentState.FINALIZING, AgentState.RETRYING},
    AgentState.DIAGNOSING:  {AgentState.FINALIZING, AgentState.RETRYING, AgentState.FAILED},
    AgentState.FINALIZING:  {AgentState.PLANNING, AgentState.DONE, AgentState.FAILED},
    AgentState.RETRYING:    {AgentState.PLANNING, AgentState.ACTING, AgentState.OBSERVING,
                             AgentState.EVALUATING, AgentState.DIAGNOSING, AgentState.FAILED},
    AgentState.DONE:        set(),
    AgentState.FAILED:      set(),
}


# ═══════════════════════════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════════════════════════

@dataclass
class StepResult:
    """单步执行的结果"""
    ok: bool
    data: Any = None
    error: str | None = None
    should_retry: bool = False
    retry_after_s: float = 5.0


@dataclass
class Checkpoint:
    """断点续跑的快照

    包含恢复执行所需的所有状态。
    """
    iteration: int = 0
    state: str = "INIT"
    completed_features: list[str] = field(default_factory=list)
    current_feature_name: str | None = None
    token_used: int = 0
    step_id: int = 0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "state": self.state,
            "completed_features": self.completed_features,
            "current_feature_name": self.current_feature_name,
            "token_used": self.token_used,
            "step_id": self.step_id,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Checkpoint":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ═══════════════════════════════════════════════════════════════════════════════════
# 超时保护
# ═══════════════════════════════════════════════════════════════════════════════════

class StepTimeout(Exception):
    """单步执行超时"""
    pass


@contextmanager
def step_timeout(seconds: float):
    """单步超时保护——超时不杀整个 loop，抛 StepTimeout 由 loop 决定重试或跳过

    Windows 注意：signal.SIGALRM 在 Windows 上不可用，使用 threading.Timer 回退。
    """
    if seconds <= 0:
        yield
        return

    # Windows 不支持 SIGALRM，使用 threading.Timer
    if not hasattr(signal, 'SIGALRM'):
        import threading
        timed_out = [False]

        def _timeout_cb():
            timed_out[0] = True

        timer = threading.Timer(seconds, _timeout_cb)
        timer.daemon = True
        timer.start()
        try:
            yield
            if timed_out[0]:
                raise StepTimeout(f"Step timed out after {seconds}s (threading timer)")
        finally:
            timer.cancel()
        return

    def _handler(signum, frame):
        raise StepTimeout(f"Step timed out after {seconds}s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# ═══════════════════════════════════════════════════════════════════════════════════
# AgentLoop
# ═══════════════════════════════════════════════════════════════════════════════════

class AgentLoop:
    """Agent 运行时状态机

    职责：
      - 管理 Agent 执行的状态转换
      - 每步执行后自动 checkpoint
      - Token 预算管理
      - 单步超时保护
      - 启动时自动检测并恢复 checkpoint

    不负责：
      - 具体的 Agent 业务逻辑（由注册的工具函数实现）
      - 上下文装配（由 ContextConstructor 负责）
      - 质量门控（由 GuardrailPipeline 负责）
    """

    def __init__(
        self,
        tracer: Tracer,
        checkpoint_dir: Path | str | None = None,
        max_step_timeout_s: float = 600.0,    # 单步超时（秒）
        token_budget: int = 1_000_000,         # 总 token 预算
        token_warning_ratio: float = 0.85,     # 触发收敛指令的预算比例
        auto_resume: bool = True,              # 启动时自动恢复 checkpoint
    ):
        self.tracer = tracer
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("agent_output")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_path = self.checkpoint_dir / "loop_checkpoint.json"

        self.max_step_timeout_s = max_step_timeout_s
        self.token_budget = token_budget
        self.token_warning_ratio = token_warning_ratio
        self.auto_resume = auto_resume

        # 运行时状态
        self.state: AgentState = AgentState.INIT
        self.iteration: int = 0
        self.max_iter: int = 31
        self.token_used: int = 0
        self.step_id: int = 0
        self.completed_features: list[str] = []
        self.current_feature_name: str | None = None

        # Pipeline shared context: carriers data between state handlers
        # (feature_spec → feature_df → val_result → gov_result → diagnosis)
        self.context: dict[str, Any] = {}

        # 工具注册表（简化版——完整 ToolRegistry 在 harness/tools.py）
        self._tools: dict[str, Callable] = {}

        # 状态处理函数
        self._handlers: dict[AgentState, Callable] = {}

        # 生命周期钩子
        self.on_checkpoint: Callable | None = None
        self.on_state_change: Callable | None = None

    # ── 工具注册 ───────────────────────────────────────────────────

    def register_tool(self, name: str, fn: Callable) -> None:
        """注册一个工具函数到 loop

        Args:
            name: 工具名称（对应 AgentState.PLANNING → "hypothesis" 等）
            fn: 工具函数，签名为 fn(loop, **kwargs) -> StepResult
        """
        self._tools[name] = fn

    def register_handler(self, state: AgentState, handler: Callable) -> None:
        """注册状态处理函数

        Args:
            state: 要处理的 AgentState
            handler: 处理函数，签名为 handler(loop) -> StepResult
        """
        self._handlers[state] = handler

    # ── 状态转换 ───────────────────────────────────────────────────

    def transition_to(self, target: AgentState) -> bool:
        """执行状态转换

        Returns:
            True 如果转换合法且已执行
        Raises:
            ValueError 如果转换不合法
        """
        if target not in _TRANSITIONS.get(self.state, set()):
            raise ValueError(
                f"Illegal state transition: {self.state.name} → {target.name}. "
                f"Allowed: {[s.name for s in _TRANSITIONS.get(self.state, set())]}"
            )

        old_state = self.state
        self.state = target

        # 记录到 tracer
        self.tracer.record_step(
            step_id=self.step_id,
            state=target.name,
            iteration=self.iteration,
            observation=f"State: {old_state.name} → {target.name}",
            decision="CONTINUE",
        )

        # 触发钩子
        if self.on_state_change:
            self.on_state_change(old_state, target)

        return True

    # ── 预算管理 ───────────────────────────────────────────────────

    def budget_remaining(self) -> int:
        """剩余 token 预算"""
        return max(0, self.token_budget - self.token_used)

    def budget_warning(self) -> bool:
        """是否接近预算上限（需要注入收敛指令）"""
        if self.token_budget <= 0:
            return False
        return self.token_used / self.token_budget >= self.token_warning_ratio

    def consume_tokens(self, n: int) -> None:
        self.token_used += n

    # ── Checkpoint ─────────────────────────────────────────────────

    def _build_checkpoint(self) -> Checkpoint:
        # Serialize context: only keep serializable values
        ctx_serializable = {}
        for k, v in self.context.items():
            try:
                json.dumps(v, default=str)
                ctx_serializable[k] = v
            except (TypeError, ValueError):
                ctx_serializable[k] = str(v)[:500]
            else:
                # Re-test without default=str — if it fails, store str version
                try:
                    json.dumps(v)
                except (TypeError, ValueError):
                    ctx_serializable[k] = str(v)[:500]
        return Checkpoint(
            iteration=self.iteration,
            state=self.state.name,
            completed_features=list(self.completed_features),
            current_feature_name=self.current_feature_name,
            token_used=self.token_used,
            step_id=self.step_id,
            extra={"context": ctx_serializable},
        )

    def checkpoint(self) -> None:
        """保存当前状态到磁盘（原子写入，防止写崩时损坏）"""
        try:
            ckpt = self._build_checkpoint()
            tmp_path = self._checkpoint_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(ckpt.to_dict(), f, ensure_ascii=False, indent=2)
            tmp_path.replace(self._checkpoint_path)  # 原子替换

            if self.on_checkpoint:
                self.on_checkpoint(ckpt)
        except Exception as _ckpt_err:
            print(f"[AgentLoop] checkpoint write FAILED: {_ckpt_err}", flush=True)

    def _load_checkpoint(self) -> Checkpoint | None:
        """从磁盘加载 checkpoint，不存在则返回 None"""
        if not self._checkpoint_path.exists():
            return None
        try:
            with open(self._checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return Checkpoint.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return None

    def _resume_from_checkpoint(self, ckpt: Checkpoint) -> None:
        """从 checkpoint 恢复运行时状态"""
        self.iteration = ckpt.iteration
        self.state = AgentState[ckpt.state]
        self.completed_features = ckpt.completed_features
        self.current_feature_name = ckpt.current_feature_name
        self.token_used = ckpt.token_used
        self.step_id = ckpt.step_id
        self.context = ckpt.extra.get("context", {})
        print(f"[AgentLoop] 从 checkpoint 恢复: iteration={self.iteration}, "
              f"state={self.state.name}, completed={len(self.completed_features)}")

    # ── 主循环 ─────────────────────────────────────────────────────

    def run(self, max_iter: int = 31) -> dict[str, Any]:
        """启动 Agent Loop

        Args:
            max_iter: 最大迭代轮数

        Returns:
            {"status": "done"|"failed", "iterations": int, "completed_features": [...], ...}
        """
        self.max_iter = max_iter

        # ── 检查并恢复 checkpoint ──
        if self.auto_resume:
            ckpt = self._load_checkpoint()
            if ckpt and ckpt.iteration > 0:
                self._resume_from_checkpoint(ckpt)

        # ── 主循环 ──
        try:
            # Only transition to INIT if we can legally get there from current state.
            # On checkpoint resume, we may already be in a valid workflow state (e.g. ACTING).
            if self.state != AgentState.INIT and AgentState.INIT in _TRANSITIONS.get(self.state, set()):
                self.transition_to(AgentState.INIT)

            while self.state not in (AgentState.DONE, AgentState.FAILED):
                self.step_id += 1
                self._step()
                self.checkpoint()

            status = "done" if self.state == AgentState.DONE else "failed"

        except Exception as e:
            self.tracer.record_step(
                step_id=self.step_id,
                state="FAILED",
                iteration=self.iteration,
                tool_error=str(e),
                observation=traceback.format_exc()[-500:],
                decision="FAIL",
            )
            self.state = AgentState.FAILED
            self.checkpoint()
            status = "failed"

        return {
            "status": status,
            "run_id": self.tracer.run_id,
            "iterations": self.iteration,
            "completed_features": self.completed_features,
            "token_used": self.token_used,
            "stats": self.tracer.stats_summary(),
        }

    def _step(self) -> None:
        """执行单步——当前状态 → 对应 handler → 根据结果决定下一状态"""
        handler = self._handlers.get(self.state)

        if handler is None:
            # 没有注册 handler 的状态走默认逻辑
            self._default_step()
            return

        try:
            with step_timeout(self.max_step_timeout_s):
                result: StepResult = handler(self)
        except StepTimeout:
            result = StepResult(
                ok=False,
                error=f"Step timeout after {self.max_step_timeout_s}s at state {self.state.name}",
                should_retry=False,
            )

        # 根据结果决定下一步
        if result.ok:
            self._advance_state()
        elif result.should_retry:
            self.transition_to(AgentState.RETRYING)
            time.sleep(result.retry_after_s)
            # 回到重试前的状态
            # RETRYING handler 负责跳回原状态
        else:
            self.transition_to(AgentState.FAILED)

    def _default_step(self) -> None:
        """没有注册 handler 时的默认状态推进逻辑"""
        # ── EVALUATING 条件路由 ──
        if self.state == AgentState.EVALUATING:
            gov_result = self.context.get("gov_result", {})
            if gov_result and not gov_result.get("passed", True):
                self.transition_to(AgentState.DIAGNOSING)
            else:
                self.transition_to(AgentState.FINALIZING)
            return

        flow = {
            AgentState.INIT:        AgentState.PLANNING,
            AgentState.PLANNING:    AgentState.ACTING,
            AgentState.ACTING:      AgentState.OBSERVING,
            AgentState.OBSERVING:   AgentState.EVALUATING,
            AgentState.DIAGNOSING:  AgentState.FINALIZING,
            AgentState.FINALIZING:  AgentState.PLANNING,
            AgentState.RETRYING:    AgentState.PLANNING,
        }

        next_state = flow.get(self.state)
        if next_state == AgentState.PLANNING:
            if self.iteration >= self.max_iter:
                self.transition_to(AgentState.DONE)
                return
            self.iteration += 1

        if next_state:
            self.transition_to(next_state)

    def _advance_state(self) -> None:
        """根据当前状态推进到下一状态（正常流程）

        EVALUATING 特殊处理：根据 gov_result.passed 决定进入 DIAGNOSING 还是 FINALIZING。
        """
        # ── EVALUATING 条件路由：FAIL → DIAGNOSING，PASS → FINALIZING ──
        if self.state == AgentState.EVALUATING:
            gov_result = self.context.get("gov_result", {})
            if gov_result and not gov_result.get("passed", True):
                # Guardrail failed → diagnose before finalizing
                self.transition_to(AgentState.DIAGNOSING)
                return
            # Guardrail passed → skip diagnosis, go directly to finalize
            self.transition_to(AgentState.FINALIZING)
            return

        flow = {
            AgentState.INIT:        AgentState.PLANNING,
            AgentState.PLANNING:    AgentState.ACTING,
            AgentState.ACTING:      AgentState.OBSERVING,
            AgentState.OBSERVING:   AgentState.EVALUATING,
            AgentState.DIAGNOSING:  AgentState.FINALIZING,
            AgentState.FINALIZING:  AgentState.PLANNING,
            AgentState.RETRYING:    AgentState.PLANNING,
        }

        next_state = flow.get(self.state)

        if next_state == AgentState.PLANNING:
            if self.iteration >= self.max_iter:
                self.transition_to(AgentState.DONE)
                return
            self.iteration += 1

        if next_state:
            self.transition_to(next_state)

    # ── 调试信息 ───────────────────────────────────────────────────

    def status_line(self) -> str:
        """单行状态摘要"""
        budget_info = f"tokens={self.token_used}/{self.token_budget}" if self.token_budget > 0 else ""
        warn = " ⚠BUDGET" if self.budget_warning() else ""
        return (
            f"[{self.state.name}] iter={self.iteration}/{self.max_iter} "
            f"step={self.step_id} {budget_info}{warn} "
            f"completed={len(self.completed_features)}"
        )
