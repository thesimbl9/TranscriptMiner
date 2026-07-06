"""
harness/guardrail.py — 可插拔质量门控管道

解决的问题（失控模式 #5）：
  GovernanceAgent 4 个硬编码门控，加新门控 = 改核心代码。
  不同特征类型无法用不同门控组合（prepared 特征可能不需要 DirectionConsistency）。
  加完新门控还要手动验证前 4 个没被改坏。

Harness 解法：
  可插拔检查器链：pre_exec / post_exec / on_failure 三种钩子位置。
  支持 add / remove / insert_after / clone。
  每个检查器独立测试，Pipeline 只负责编排。

使用：
  pipeline = GuardrailPipeline()
  pipeline.add_post_hook(CoverageGate(0.05))
  pipeline.add_post_hook(ZeroRatioGate(uniform=0.70, concentrated=0.45))
  pipeline.add_post_hook(TStatGate(1.5))
  pipeline.on_failure(DiagnosisHook())
  result = pipeline.run(ctx)

面试得分点：
  "Guardrail 是可插拔的 Pipeline——随时加新门控不碰业务代码"
  "不同特征类型用不同的门控组合：qa_pipeline = pipeline.clone().remove('DirectionConsistencyGate')"
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable


# ═══════════════════════════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════════════════════════

@dataclass
class GuardrailResult:
    """门控检查结果"""
    passed: bool = True
    blocked_by: str | None = None           # 被哪个 hook 拦截（pre_exec）
    failed_by: str | None = None            # 被哪个 hook 判定失败（post_exec）
    failures: list[str] = field(default_factory=list)  # 所有失败项的描述
    recovery_actions: list[str] = field(default_factory=list)  # 触发过的恢复动作
    hook_results: dict[str, bool] = field(default_factory=dict)  # hook_name → passed


class GuardrailHook(ABC):
    """门控检查器基类

    子类只需实现：
      - name: 唯一标识名
      - check(ctx) -> bool: 检查逻辑
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """唯一标识名，如 'G2_zero_ratio'"""
        ...

    @abstractmethod
    def check(self, ctx: dict) -> bool:
        """执行检查

        Args:
            ctx: 执行上下文，至少包含 validation_result / feature_spec 等

        Returns:
            True = 通过，False = 拦截
        """
        ...

    @property
    def description(self) -> str:
        return f"Hook: {self.name}"


class GuardrailPipeline:
    """可插拔的门控检查器管道

    三种钩子位置：
      pre_exec   — 执行前检查（如 feature_spec schema 校验）
      post_exec  — 执行后检查（如 G1-G4 门控，这是主要使用位置）
      on_failure — 失败时触发（如 DiagnosisAgent 诊断，可以有多个）

    操作：
      - add_pre_hook / add_post_hook / on_failure
      - remove(name)
      - insert_after(name, hook)
      - clone() → 派生新的 Pipeline
    """

    def __init__(self, name: str = "default"):
        self.name = name
        self._pre_hooks: list[GuardrailHook] = []
        self._post_hooks: list[GuardrailHook] = []
        self._failure_handlers: list[Callable[[dict, GuardrailResult], None]] = []

    # ── 添加钩子 ───────────────────────────────────────────────────

    def add_pre_hook(self, hook: GuardrailHook) -> "GuardrailPipeline":
        """添加执行前检查器"""
        self._pre_hooks.append(hook)
        return self

    def add_post_hook(self, hook: GuardrailHook) -> "GuardrailPipeline":
        """添加执行后检查器（主要使用位置）"""
        self._post_hooks.append(hook)
        return self

    def on_failure(self, handler: Callable[[dict, GuardrailResult], None]) -> "GuardrailPipeline":
        """添加失败处理器（DiagnosisAgent 等）"""
        self._failure_handlers.append(handler)
        return self

    # ── 移除/插入 ──────────────────────────────────────────────────

    def remove(self, name: str) -> "GuardrailPipeline":
        """按名称移除钩子"""
        self._pre_hooks = [h for h in self._pre_hooks if h.name != name]
        self._post_hooks = [h for h in self._post_hooks if h.name != name]
        return self

    def insert_after(self, after_name: str, hook: GuardrailHook,
                     position: str = "post") -> "GuardrailPipeline":
        """在指定钩子后面插入新钩子"""
        target = self._post_hooks if position == "post" else self._pre_hooks
        for i, h in enumerate(target):
            if h.name == after_name:
                target.insert(i + 1, hook)
                return self
        # 没找到 → 追加到末尾
        target.append(hook)
        return self

    def replace(self, name: str, new_hook: GuardrailHook,
                position: str = "post") -> "GuardrailPipeline":
        """替换指定名称的钩子"""
        target = self._post_hooks if position == "post" else self._pre_hooks
        for i, h in enumerate(target):
            if h.name == name:
                target[i] = new_hook
                return self
        return self

    # ── 克隆 ───────────────────────────────────────────────────────

    def clone(self, name: str | None = None) -> "GuardrailPipeline":
        """深拷贝 pipeline（用于不同特征类型的定制化门控组合）"""
        new_pipeline = GuardrailPipeline(name=name or f"{self.name}_clone")
        new_pipeline._pre_hooks = list(self._pre_hooks)
        new_pipeline._post_hooks = list(self._post_hooks)
        new_pipeline._failure_handlers = list(self._failure_handlers)
        return new_pipeline

    # ── 执行 ───────────────────────────────────────────────────────

    def run(self, ctx: dict) -> GuardrailResult:
        """运行完整门控管道

        Args:
            ctx: 执行上下文，包含：
                - validation_result: 验证结果 dict
                - feature_spec: 特征定义 dict
                - (pre_exec hooks 可能还需要其他字段)

        Returns:
            GuardrailResult
        """
        result = GuardrailResult()

        # ── Pre-exec hooks：执行前拦截 ──
        for hook in self._pre_hooks:
            ok = hook.check(ctx)
            result.hook_results[hook.name] = ok
            if not ok:
                result.passed = False
                result.blocked_by = hook.name
                result.failures.append(f"PRE_BLOCKED by {hook.name}")
                return result  # pre-exec 拦截 → 立即返回，不执行后续

        # ── Post-exec hooks：执行后检查 ──
        for hook in self._post_hooks:
            ok = hook.check(ctx)
            result.hook_results[hook.name] = ok
            if not ok:
                result.passed = False
                if result.failed_by is None:
                    result.failed_by = hook.name
                result.failures.append(f"FAILED {hook.name}")

        # ── On-failure handlers：失败时触发恢复动作 ──
        if not result.passed:
            for handler in self._failure_handlers:
                try:
                    handler(ctx, result)
                    result.recovery_actions.append(handler.__name__)
                except Exception as e:
                    result.failures.append(f"Recovery handler {handler.__name__} failed: {e}")

        return result

    def run_with_executor(
        self, ctx: dict, executor: Callable[[dict], Any]
    ) -> tuple[Any, GuardrailResult]:
        """执行 + 门控的组合模式

        如果 pre_exec 全部通过 → 执行 executor(ctx) → post_exec 检查 → 返回结果和门控结果

        Args:
            ctx: 执行上下文
            executor: 实际执行函数，签名为 executor(ctx) -> Any

        Returns:
            (executor_result, guardrail_result)
        """
        # Pre-exec
        pre_result = self.run_pre(ctx)
        if not pre_result.passed:
            return None, pre_result

        # Execute
        exec_result = executor(ctx)
        ctx["_exec_result"] = exec_result

        # Post-exec
        full_result = self.run_post(ctx, pre_result)
        return exec_result, full_result

    def run_pre(self, ctx: dict) -> GuardrailResult:
        """仅运行 pre-exec hooks"""
        result = GuardrailResult()
        for hook in self._pre_hooks:
            ok = hook.check(ctx)
            result.hook_results[hook.name] = ok
            if not ok:
                result.passed = False
                result.blocked_by = hook.name
                result.failures.append(f"PRE_BLOCKED by {hook.name}")
                return result
        return result

    def run_post(self, ctx: dict, pre_result: GuardrailResult | None = None) -> GuardrailResult:
        """仅运行 post-exec hooks（在 pre_result 基础上追加）"""
        result = pre_result or GuardrailResult()

        for hook in self._post_hooks:
            ok = hook.check(ctx)
            result.hook_results[hook.name] = ok
            if not ok:
                result.passed = False
                if result.failed_by is None:
                    result.failed_by = hook.name
                result.failures.append(f"FAILED {hook.name}")

        if not result.passed:
            for handler in self._failure_handlers:
                try:
                    handler(ctx, result)
                    result.recovery_actions.append(handler.__name__)
                except Exception:
                    pass

        return result

    # ── 查询 ───────────────────────────────────────────────────────

    @property
    def hooks(self) -> dict[str, list[str]]:
        """返回所有注册的 hook 名称"""
        return {
            "pre": [h.name for h in self._pre_hooks],
            "post": [h.name for h in self._post_hooks],
            "failure_handlers": [
                h.__name__ if hasattr(h, '__name__') else str(h)
                for h in self._failure_handlers
            ],
        }

    def describe(self) -> str:
        """可读的 Pipeline 描述"""
        lines = [f"GuardrailPipeline '{self.name}':"]
        if self._pre_hooks:
            lines.append("  Pre-exec: " + " → ".join(h.name for h in self._pre_hooks))
        if self._post_hooks:
            lines.append("  Post-exec: " + " → ".join(h.name for h in self._post_hooks))
        if self._failure_handlers:
            names = [getattr(h, '__name__', str(h)) for h in self._failure_handlers]
            lines.append("  On-failure: " + ", ".join(names))
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════════
# 内置检查器（从现有 governance_agent.py 迁移）
# ═══════════════════════════════════════════════════════════════════════════════════

class CoverageGate(GuardrailHook):
    """G1: 测试期覆盖率检查"""
    name = "G1_coverage"

    def __init__(self, min_coverage: float = 0.05):
        self.min_coverage = min_coverage

    def check(self, ctx: dict) -> bool:
        val = ctx.get("validation_result", {})
        if val.get("coverage_failure"):
            return False
        ratio = val.get("test_coverage_ratio", 1.0)
        return ratio >= self.min_coverage


class ZeroRatioGate(GuardrailHook):
    """G2: 零值率检查（区分均匀型和集中型）"""
    name = "G2_zero_ratio"

    def __init__(self, uniform_threshold: float = 0.70, concentrated_threshold: float = 0.45):
        self.uniform_threshold = uniform_threshold
        self.concentrated_threshold = concentrated_threshold

    def check(self, ctx: dict) -> bool:
        import statistics
        val = ctx.get("validation_result", {})
        zr = val.get("zero_ratio", 1.0)
        zbs = val.get("zero_by_sector", {})

        if len(zbs) < 3:
            return zr < self.concentrated_threshold  # unknown → strict

        vals = list(zbs.values())
        var = statistics.variance(vals)
        max_z = max(vals)
        min_z = min(vals)

        if var < 0.02:
            return zr < self.uniform_threshold  # uniform → lenient
        if max_z > 0.80 and min_z < 0.40:
            return zr < self.concentrated_threshold  # concentrated → strict
        return zr < self.concentrated_threshold  # unknown → strict


class TStatGate(GuardrailHook):
    """G3: |t-stat| 显著性检查"""
    name = "G3_t_stat"

    def __init__(self, min_abs_t: float = 1.5):
        self.min_abs_t = min_abs_t

    def check(self, ctx: dict) -> bool:
        val = ctx.get("validation_result", {})
        t = abs(val.get("t_stat", 0.0))
        return t >= self.min_abs_t


class DirectionConsistencyGate(GuardrailHook):
    """G4: 方向一致性检查"""
    name = "G4_direction"

    def __init__(self, min_consistency: float = 0.60):
        self.min_consistency = min_consistency

    def check(self, ctx: dict) -> bool:
        val = ctx.get("validation_result", {})
        dc = val.get("direction_consistency", 0.0)
        return dc >= self.min_consistency


class MaxDrawdownGate(GuardrailHook):
    """G5（可选）: 最大回撤检查"""
    name = "G5_max_drawdown"

    def __init__(self, max_mdd: float = 0.03):
        self.max_mdd = max_mdd

    def check(self, ctx: dict) -> bool:
        val = ctx.get("validation_result", {})
        mdd = abs(val.get("max_drawdown", 0.0))
        return mdd <= self.max_mdd


# ═══════════════════════════════════════════════════════════════════════════════════
# 工厂函数：创建标准默认 Pipeline
# ═══════════════════════════════════════════════════════════════════════════════════

def create_default_guardrail_pipeline(
    diagnosis_handler: Callable | None = None,
) -> GuardrailPipeline:
    """创建与现有 governance_agent.py 兼容的默认门控管道"""
    pipeline = GuardrailPipeline("default")
    pipeline.add_post_hook(CoverageGate(0.05))
    pipeline.add_post_hook(ZeroRatioGate(uniform_threshold=0.70, concentrated_threshold=0.45))
    pipeline.add_post_hook(TStatGate(1.5))
    pipeline.add_post_hook(DirectionConsistencyGate(0.60))
    if diagnosis_handler:
        pipeline.on_failure(diagnosis_handler)
    return pipeline
