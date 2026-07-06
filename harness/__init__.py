"""
harness — 通用 Agent 运行时框架

从 EarningsSignal Agent 的 31 轮迭代中抽象出的可复用 Harness 层。
每个组件背后对应一种真实失控模式的工程防御。

模块：
  loop      — AgentLoop 状态机（checkpoint + 断点续跑）
  context   — ContextConstructor（动态上下文装配）
  memory    — 三层 Memory（Working / Episodic / Semantic）
  tools     — ToolRegistry（超时 + 重试 + 降级 + 结构化错误）
  guardrail — GuardrailPipeline（可插拔检查器链）
  tracer    — Tracer（结构化 trace + 回放 + 聚合统计）
"""

from harness.loop import AgentLoop, AgentState, StepResult, Checkpoint
from harness.context import ContextConstructor, ContextStrategy
from harness.memory import WorkingMemory, EpisodicMemory
from harness.tools import ToolRegistry, RetryPolicy, ToolResult
from harness.guardrail import (
    GuardrailPipeline, GuardrailHook, GuardrailResult,
    CoverageGate, ZeroRatioGate, TStatGate,
    DirectionConsistencyGate, MaxDrawdownGate,
    create_default_guardrail_pipeline,
)
from harness.tracer import Tracer, StepTrace, RunStats
from harness.adapters import register_all_handlers

__all__ = [
    # loop
    "AgentLoop", "AgentState", "StepResult", "Checkpoint",
    # context
    "ContextConstructor", "ContextStrategy",
    # memory
    "WorkingMemory", "EpisodicMemory",
    # tools
    "ToolRegistry", "RetryPolicy", "ToolResult",
    # guardrail
    "GuardrailPipeline", "GuardrailHook", "GuardrailResult",
    "CoverageGate", "ZeroRatioGate", "TStatGate",
    "DirectionConsistencyGate", "MaxDrawdownGate",
    "create_default_guardrail_pipeline",
    # tracer
    "Tracer", "StepTrace", "RunStats",
    # adapters
    "register_all_handlers",
]
