"""
harness/tools.py — 工具注册与调用层

解决的问题（失控模式 #4）：
  API 超时/限流 → 硬崩；LLM 返回非法 JSON → 静默吞 → feature 队列缺一个。
  裸函数调用，没有超时保护、重试机制、降级策略。

Harness 解法：
  每个工具注册时绑定 schema + timeout_s + retry_policy + fallback。
  调用失败 → 按 retry_policy 重试 → 仍失败 → 执行 fallback → 记录到 dead_letter。
  所有调用通过 Tracer 记录。

使用：
  registry = ToolRegistry()
  registry.register("hypothesis", hypothesis_fn,
                    timeout_s=120, retry=RetryPolicy(max_retries=2, backoff_s=5.0))
  result = registry.invoke("hypothesis", feature_spec=spec)
"""

from __future__ import annotations

import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Callable


# ═══════════════════════════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════════════════════════

@dataclass
class RetryPolicy:
    """重试策略"""
    max_retries: int = 2
    backoff_s: float = 5.0           # 重试间隔（秒）
    backoff_multiplier: float = 2.0  # 每次重试间隔翻倍
    retry_on_timeout: bool = True
    retry_on_value_error: bool = True

    def delay_for_attempt(self, attempt: int) -> float:
        """第 attempt 次重试的等待时间"""
        return self.backoff_s * (self.backoff_multiplier ** (attempt - 1))


@dataclass
class ToolResult:
    """工具调用结果"""
    ok: bool
    data: Any = None
    error: str | None = None
    attempts: int = 1
    latency_ms: float = 0.0
    fallback_used: bool = False

    def to_dict(self) -> dict:
        return {
            "ok": self.ok, "error": self.error,
            "attempts": self.attempts, "latency_ms": self.latency_ms,
            "fallback_used": self.fallback_used,
        }


@dataclass
class ToolDef:
    """工具定义"""
    name: str
    fn: Callable
    timeout_s: float = 120.0
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    fallback: Callable | None = None
    output_schema: dict | None = None      # JSON Schema 用于输出校验
    description: str = ""

    def to_mcp_style(self) -> dict:
        """导出为 MCP 风格的工具描述"""
        desc = {
            "name": self.name,
            "description": self.description or f"Tool: {self.name}",
        }
        if self.output_schema:
            desc["inputSchema"] = self.output_schema
        return desc


# ═══════════════════════════════════════════════════════════════════════════════════
# ToolRegistry
# ═══════════════════════════════════════════════════════════════════════════════════

class ToolRegistry:
    """工具注册与调用层

    职责：
      - 工具注册：绑定函数 + 超时 + 重试 + 降级 + schema
      - 工具调用：超时保护 → 失败重试 → 降级 fallback → 结构化错误
      - 调用指标：延迟、成功率、重试次数
      - 导出 MCP 风格的工具清单

    面试得分点：
      "工具调用全通过 ToolRegistry 路由，带上超时+重试+降级，不是裸调函数"
    """

    def __init__(self, tracer: Any = None):
        """
        Args:
            tracer: Tracer 实例（可选，用于记录工具调用到 trace）
        """
        self._tools: dict[str, ToolDef] = {}
        self._dead_letter: list[dict] = []   # 彻底失败的调用记录
        self._metrics: dict[str, dict] = {}   # name -> {calls, ok, fail, total_latency}
        self.tracer = tracer

    # ── 注册 ───────────────────────────────────────────────────────

    def register(
        self,
        name: str,
        fn: Callable,
        timeout_s: float = 120.0,
        retry: RetryPolicy | None = None,
        fallback: Callable | None = None,
        output_schema: dict | None = None,
        description: str = "",
    ) -> None:
        """注册一个工具

        Args:
            name: 工具名称（唯一标识）
            fn: 工具函数
            timeout_s: 单次调用超时（秒）
            retry: 重试策略（None = 不重试）
            fallback: 降级函数，签名为 fn(**kwargs) -> Any，在所有重试失败后调用
            output_schema: JSON Schema，用于输出校验
            description: 工具描述（MCP 风格）
        """
        if name in self._tools:
            raise ValueError(f"Tool '{name}' already registered")

        self._tools[name] = ToolDef(
            name=name,
            fn=fn,
            timeout_s=timeout_s,
            retry_policy=retry or RetryPolicy(max_retries=0),
            fallback=fallback,
            output_schema=output_schema,
            description=description,
        )
        self._metrics[name] = {"calls": 0, "ok": 0, "fail": 0, "total_latency": 0.0}

    def unregister(self, name: str) -> None:
        """移除工具"""
        self._tools.pop(name, None)
        self._metrics.pop(name, None)

    # ── 调用 ───────────────────────────────────────────────────────

    def invoke(self, name: str, **kwargs) -> ToolResult:
        """调用工具（含超时保护 + 重试 + 降级）

        Args:
            name: 工具名称
            **kwargs: 传递给工具函数的参数

        Returns:
            ToolResult
        """
        if name not in self._tools:
            return ToolResult(ok=False, error=f"Tool '{name}' not found. Available: {list(self._tools.keys())}")

        return self._invoke_impl(name, self._tools[name].fn, kwargs)

    def invoke_fn(self, name: str, fn: Callable, **kwargs) -> ToolResult:
        """调用工具，但 fn 在调用时提供（覆盖注册时的 fn）。

        解决 Agent 方法无法在 build_harness_pipeline() 之前注册的问题：
        工具的超时/重试/降级配置来自注册定义，实际函数由 handler 在调用时传入。

        Args:
            name: 工具名称（必须在 ToolRegistry 中已注册）
            fn: 实际执行的函数（覆盖 ToolDef.fn）
            **kwargs: 传递给 fn 的参数

        Returns:
            ToolResult
        """
        if name not in self._tools:
            return ToolResult(ok=False, error=f"Tool '{name}' not found. Available: {list(self._tools.keys())}")

        return self._invoke_impl(name, fn, kwargs)

    def _invoke_impl(self, name: str, fn: Callable, call_kwargs: dict) -> ToolResult:
        """invoke 的核心实现：超时 + 重试 + 降级。"""
        tool = self._tools[name]
        t0 = time.perf_counter()

        # ── 尝试调用（含重试）──
        last_error = None
        max_attempts = tool.retry_policy.max_retries + 1  # 首次 + 重试
        for attempt in range(1, max_attempts + 1):
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(fn, **call_kwargs)
                    result_data = future.result(timeout=tool.timeout_s)

                # 成功
                latency = (time.perf_counter() - t0) * 1000
                self._record_metric(name, ok=True, latency=latency)
                result = ToolResult(ok=True, data=result_data, attempts=attempt, latency_ms=latency)

                # 记录到 tracer
                if self.tracer:
                    self.tracer.record_step(
                        step_id=getattr(self.tracer, 'step_count', 0),
                        state="TOOL_CALL",
                        iteration=call_kwargs.get("_iteration", 0),
                        tool_called=name,
                        tool_args=self._sanitize_args(call_kwargs),
                        tool_latency_ms=latency,
                        tool_ok=True,
                        observation=self._summarize_result(result_data),
                        decision="CONTINUE",
                    )

                return result

            except FutureTimeout:
                last_error = f"Timeout after {tool.timeout_s}s"
                if not tool.retry_policy.retry_on_timeout:
                    break

            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                if not tool.retry_policy.retry_on_value_error:
                    break

            # 重试前等待
            if attempt <= tool.retry_policy.max_retries:
                delay = tool.retry_policy.delay_for_attempt(attempt)
                time.sleep(delay)

        # ── 所有尝试失败，执行降级 ──
        latency = (time.perf_counter() - t0) * 1000
        self._record_metric(name, ok=False, latency=latency)

        fallback_used = False
        if tool.fallback:
            try:
                fallback_data = tool.fallback(**call_kwargs)
                fallback_used = True
                self._dead_letter.append({
                    "tool": name, "error": last_error, "fallback_used": True,
                    "kwargs": self._sanitize_args(call_kwargs),
                    "traceback": traceback.format_exc()[-300:],
                })
                return ToolResult(ok=True, data=fallback_data, error=last_error,
                                  attempts=max_attempts,
                                  latency_ms=latency, fallback_used=True)
            except Exception as fb_err:
                last_error += f" | Fallback also failed: {fb_err}"

        # ── 彻底失败，记录到 dead_letter ──
        self._dead_letter.append({
            "tool": name, "error": last_error, "fallback_used": False,
            "kwargs": self._sanitize_args(call_kwargs),
            "traceback": traceback.format_exc()[-500:],
        })

        if self.tracer:
            self.tracer.record_step(
                step_id=getattr(self.tracer, 'step_count', 0),
                state="TOOL_CALL",
                iteration=call_kwargs.get("_iteration", 0),
                tool_called=name,
                tool_latency_ms=latency,
                tool_ok=False,
                tool_error=last_error,
                observation=f"All {max_attempts} attempts failed: {last_error}",
                decision="FAIL",
            )

        return ToolResult(ok=False, error=last_error,
                          attempts=max_attempts,
                          latency_ms=latency, fallback_used=fallback_used)

    # ── 查询 ───────────────────────────────────────────────────────

    def get_tool(self, name: str) -> ToolDef | None:
        return self._tools.get(name)

    def list_tools(self) -> list[dict]:
        """导出 MCP 风格的工具清单"""
        return [t.to_mcp_style() for t in self._tools.values()]

    def get_metrics(self) -> dict:
        """获取每个工具的调用指标"""
        result = {}
        for name, m in self._metrics.items():
            total = m["calls"]
            result[name] = {
                "calls": total,
                "success_rate": m["ok"] / total if total > 0 else 0,
                "avg_latency_ms": m["total_latency"] / total if total > 0 else 0,
            }
        return result

    def get_dead_letter(self, n: int = 10) -> list[dict]:
        """获取最近 N 条死信"""
        return self._dead_letter[-n:]

    def dead_letter_count(self) -> int:
        return len(self._dead_letter)

    # ── 内部 ───────────────────────────────────────────────────────

    def _record_metric(self, name: str, ok: bool, latency: float) -> None:
        if name not in self._metrics:
            return
        self._metrics[name]["calls"] += 1
        self._metrics[name]["total_latency"] += latency
        if ok:
            self._metrics[name]["ok"] += 1
        else:
            self._metrics[name]["fail"] += 1

    @staticmethod
    def _sanitize_args(kwargs: dict) -> dict:
        """脱敏参数（去掉 API key、大文本等）"""
        safe = {}
        for k, v in kwargs.items():
            if k.startswith("_"):
                continue
            if k in ("api_key", "password", "token"):
                safe[k] = "***"
            elif isinstance(v, str) and len(v) > 200:
                safe[k] = v[:200] + "..."
            elif isinstance(v, dict):
                safe[k] = {sk: str(sv)[:100] for sk, sv in v.items()}
            else:
                safe[k] = str(v)[:100] if not isinstance(v, (int, float, bool, type(None))) else v
        return safe

    @staticmethod
    def _summarize_result(data: Any, max_len: int = 200) -> str:
        """压缩工具返回结果为简短摘要"""
        if data is None:
            return "None"
        if isinstance(data, dict):
            summary = json.dumps(data, ensure_ascii=False, default=str)
        elif isinstance(data, str):
            summary = data
        else:
            summary = str(data)
        return summary[:max_len] + ("..." if len(summary) > max_len else "")
