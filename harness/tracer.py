"""
harness/tracer.py — 结构化 Trace 系统

解决的问题（失控模式 #6）：
  只知道最终 FAIL，不知道链路中哪一步开始偏离。
  排查靠 debug_{feature}.txt 人工肉眼对比，一个特征半小时。

Harness 解法：
  每步结构化记录 → 回放定位根因 → 聚合统计 → 30 秒定位问题。

使用：
  tracer = Tracer(output_dir)
  tracer.record(StepTrace(step_id=1, state="PLANNING", ...))
  tracer.replay(run_id)   # 回放完整轨迹
  tracer.stats(run_id)    # 聚合统计
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class StepTrace:
    """单步执行的结构化记录"""
    step_id: int
    state: str                          # AgentState 名称
    iteration: int                      # 当前迭代轮次
    tool_called: str | None = None      # 被调用的工具名称
    tool_args: dict | None = None       # 工具参数（脱敏后）
    tool_latency_ms: float | None = None
    tool_ok: bool = True                # 工具调用是否成功
    tool_error: str | None = None       # 工具失败原因
    context_tokens: int = 0             # 注入上下文的 token 数
    observation: str = ""               # 工具返回的关键观察
    decision: str = ""                  # CONTINUE | RETRY | DONE | FAIL
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def summary(self) -> str:
        """单行摘要，用于 WorkingMemory 压缩"""
        parts = [f"[Step{self.step_id}|{self.state}]"]
        if self.tool_called:
            status = "✓" if self.tool_ok else "✗"
            parts.append(f"{self.tool_called}{status}")
        if self.tool_latency_ms:
            parts.append(f"{self.tool_latency_ms:.0f}ms")
        if self.decision:
            parts.append(f"→{self.decision}")
        if self.tool_error:
            parts.append(f"err={self.tool_error[:60]}")
        return " ".join(parts)


@dataclass
class RunStats:
    """单次 run 的聚合统计"""
    run_id: str
    total_steps: int = 0
    total_iterations: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    tool_success_rate: float = 0.0
    avg_steps_per_iteration: float = 0.0
    states_visited: list[str] = field(default_factory=list)
    tool_calls: dict[str, int] = field(default_factory=dict)  # tool_name -> count
    gate_intercepts: dict[str, int] = field(default_factory=dict)  # gate_name -> count
    decisions: dict[str, int] = field(default_factory=dict)   # decision -> count


class Tracer:
    """结构化 Trace 系统

    职责：
      - 记录每一步的完整上下文到 JSONL
      - 支持按 run_id 回放任意轨迹
      - 聚合统计（收敛步数、工具成功率、token 消耗、Gate 拦截率）
    """

    def __init__(self, output_dir: Path | str, run_id: str | None = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or time.strftime("run_%Y%m%d_%H%M%S")
        self._traces: list[StepTrace] = []
        self._trace_path = self.output_dir / f"trace_{self.run_id}.jsonl"

    # ── 记录 ───────────────────────────────────────────────────────

    def record(self, trace: StepTrace) -> None:
        """记录一步 trace 并实时写盘"""
        self._traces.append(trace)
        with open(self._trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(trace), ensure_ascii=False) + "\n")

    def record_step(
        self,
        step_id: int,
        state: str,
        iteration: int,
        tool_called: str | None = None,
        tool_args: dict | None = None,
        tool_latency_ms: float | None = None,
        tool_ok: bool = True,
        tool_error: str | None = None,
        context_tokens: int = 0,
        observation: str = "",
        decision: str = "",
    ) -> StepTrace:
        """便捷方法：一步完成 trace 构建 + 记录"""
        trace = StepTrace(
            step_id=step_id,
            state=state,
            iteration=iteration,
            tool_called=tool_called,
            tool_args=tool_args,
            tool_latency_ms=tool_latency_ms,
            tool_ok=tool_ok,
            tool_error=tool_error,
            context_tokens=context_tokens,
            observation=observation,
            decision=decision,
        )
        self.record(trace)
        return trace

    # ── 回放 ───────────────────────────────────────────────────────

    def replay(self, run_id: str | None = None) -> list[StepTrace]:
        """回放指定 run 的完整轨迹

        Args:
            run_id: 要回放的 run ID，默认当前 run

        Returns:
            按 step_id 排序的轨迹列表
        """
        path = self._trace_path if run_id is None else self.output_dir / f"trace_{run_id}.jsonl"
        if not path.exists():
            return []

        traces = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    traces.append(StepTrace(**json.loads(line)))
        return sorted(traces, key=lambda t: t.step_id)

    def find_divergence_point(self, run_id: str | None = None) -> StepTrace | None:
        """定位轨迹中第一次出现问题的步骤（tool_ok=False 或 decision=FAIL）"""
        traces = self.replay(run_id)
        for t in traces:
            if not t.tool_ok or t.decision in ("FAIL", "RETRY"):
                return t
        return None

    # ── 统计 ───────────────────────────────────────────────────────

    def stats(self, run_id: str | None = None) -> RunStats:
        """聚合统计：收敛步数、工具成功率、token 消耗、Gate 拦截率"""
        traces = self.replay(run_id)
        if not traces:
            return RunStats(run_id=run_id or self.run_id)

        rs = RunStats(run_id=run_id or self.run_id)
        rs.total_steps = len(traces)
        rs.total_iterations = max(t.step_id for t in traces)  # 近似
        rs.total_tokens = sum(t.context_tokens for t in traces)
        rs.total_latency_ms = sum(t.tool_latency_ms or 0 for t in traces)

        # 工具成功率
        tool_total = sum(1 for t in traces if t.tool_called)
        tool_ok_count = sum(1 for t in traces if t.tool_called and t.tool_ok)
        rs.tool_success_rate = tool_ok_count / tool_total if tool_total > 0 else 1.0

        # 平均每迭代步数
        iterations = set(t.iteration for t in traces if t.iteration > 0)
        rs.avg_steps_per_iteration = rs.total_steps / len(iterations) if iterations else 0

        # 状态分布
        rs.states_visited = list(dict.fromkeys(t.state for t in traces))

        # 工具调用分布
        for t in traces:
            if t.tool_called:
                rs.tool_calls[t.tool_called] = rs.tool_calls.get(t.tool_called, 0) + 1

        # Gate 拦截统计（从 observation 中提取）
        for t in traces:
            if "Gate" in t.observation or "G1_" in t.observation or "G2_" in t.observation:
                for gate in ["G1_coverage", "G2_zero_ratio", "G3_t_stat", "G4_direction"]:
                    if gate in t.observation:
                        rs.gate_intercepts[gate] = rs.gate_intercepts.get(gate, 0) + 1

        # 决策分布
        for t in traces:
            if t.decision:
                rs.decisions[t.decision] = rs.decisions.get(t.decision, 0) + 1

        return rs

    def stats_summary(self, run_id: str | None = None) -> str:
        """可读的统计摘要"""
        s = self.stats(run_id)
        return (
            f"RunStats({s.run_id}): "
            f"steps={s.total_steps} iters={s.total_iterations} "
            f"tokens={s.total_tokens} latency={s.total_latency_ms/1000:.1f}s "
            f"tool_ok={s.tool_success_rate:.0%} "
            f"avg_steps/iter={s.avg_steps_per_iteration:.1f} "
            f"gates={s.gate_intercepts}"
        )

    # ── 最近轨迹查询 ───────────────────────────────────────────────

    @property
    def last_trace(self) -> StepTrace | None:
        return self._traces[-1] if self._traces else None

    @property
    def step_count(self) -> int:
        return len(self._traces)
