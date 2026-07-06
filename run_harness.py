"""
run_harness.py — EarningsSignal Agent with Harness Framework

This is the P2-migrated version of run_agent.py. All 5 agents are wired through
the Harness framework:

  原版 run_agent.py                →  Harness 版 run_harness.py
  ─────────────────────────────────────────────────────────────────
  for i in range(31):              →  AgentLoop 状态机 + checkpoint
  agent.next_feature()             →  PLANNING handler (adapters)
  extract_feature_global()         →  ACTING handler (adapters)
  validate()                       →  OBSERVING handler (adapters)
  governance_agent.check()         →  GuardrailPipeline (pluggable gates)
  diag_agent.diagnose()            →  DIAGNOSING handler (adapters)
  agent.record_result()            →  FINALIZING handler + EpisodicMemory.store()
  裸函数调用，无超时/重试          →  ToolRegistry.invoke() 四层防护
  feature_history.jsonl 只写不读   →  EpisodicMemory 检索复用
  硬编码 prompt 拼接                →  ContextConstructor 动态装配
  debug_*.txt 人工排查              →  Tracer 结构化 trace + 回放

用法:
  python run_harness.py [--max-iter 10] [--dry-run] [--symbols AAPL MSFT]

  --max-iter:  最大迭代轮数（默认 10）
  --dry-run:   只跑 HypothesisAgent，不调 LLM API
  --symbols:   只处理指定 symbol（调试用）
  --years:     只处理指定年份
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agent_core.config import OUTPUT_DIR, HISTORY_PATH, API_KEY
OUTPUT_DIR.mkdir(exist_ok=True)

from agent_core.hypothesis_agent import HypothesisAgent
from agent_core.extraction_agent import extract_feature_global
from agent_core.validation_agent import validate
from agent_core.diagnosis_agent import DiagnosisAgent

# ── Harness components ──
from harness.loop import AgentLoop, AgentState
from harness.tracer import Tracer
from harness.context import ContextConstructor, ContextStrategy
from harness.tools import ToolRegistry, RetryPolicy
from harness.guardrail import (
    GuardrailPipeline,
    CoverageGate, ZeroRatioGate, TStatGate,
    DirectionConsistencyGate, MaxDrawdownGate,
    create_default_guardrail_pipeline,
)
from harness.memory.working import WorkingMemory
from harness.memory.episodic import EpisodicMemory
from harness.adapters import register_all_handlers


def build_harness_pipeline(
    api_key: str,
    max_iter: int = 10,
    symbols: list[str] | None = None,
    years: list[int] | None = None,
    checkpoint_dir: Path | str | None = None,
) -> dict:
    """Assemble the complete Harness pipeline for EarningsSignal Agent.

    Returns a dict with all constructed components, ready for loop.run().
    """

    # ── 0. Tracer: 全链路可观测 ──────────────────────────────────────
    tracer = Tracer(
        run_id=f"earnings_signal_{Path(checkpoint_dir or 'agent_output').name}",
        output_dir=checkpoint_dir or OUTPUT_DIR,
    )

    # ── 1. AgentLoop: 状态机 + checkpoint ────────────────────────────
    loop = AgentLoop(
        tracer=tracer,
        checkpoint_dir=checkpoint_dir or OUTPUT_DIR,
        max_step_timeout_s=600.0,  # ExtractionAgent 单步最长 ~5min
        token_budget=1_000_000,
        auto_resume=True,
    )

    # ── 2. ToolRegistry: 超时 + 重试 + 降级 ──────────────────────────
    tool_registry = ToolRegistry(tracer=tracer)

    # 注册 5 个 Agent 工具（配置超时/重试策略，fn 由 handler 在调用时通过 invoke_fn 提供）
    # 这种设计允许在 agent 实例创建后，handler 通过闭包将具体函数传入 ToolRegistry，
    # 同时复用注册时的超时/重试/降级配置。
    tool_registry.register(
        "hypothesis",
        fn=None,  # 由 invoke_fn 覆盖
        timeout_s=120.0,
        retry=RetryPolicy(max_retries=2, backoff_s=10.0),
        description="Generate next feature hypothesis via Theory-First RAG",
    )

    tool_registry.register(
        "extraction",
        fn=None,
        timeout_s=600.0,
        retry=RetryPolicy(max_retries=1, backoff_s=30.0),
        description="Extract feature values from transcript chunks via GPU retrieval + LLM scoring",
    )

    tool_registry.register(
        "validation",
        fn=None,
        timeout_s=120.0,
        retry=RetryPolicy(max_retries=0),
        description="Validate feature predictive power via LightGBM walk-forward IC",
    )

    tool_registry.register(
        "diagnosis",
        fn=None,
        timeout_s=60.0,
        retry=RetryPolicy(max_retries=1, backoff_s=5.0),
        description="Diagnose failure root cause via theory RAG + LLM",
    )

    # ── 3. GuardrailPipeline: 可插拔门控（替代 governance_agent.check()） ──
    guardrail = create_default_guardrail_pipeline()
    # Add an optional G5 gate (can be removed per feature type)
    # guardrail.add_post_hook(MaxDrawdownGate(0.03))

    # ── 4. Memory ────────────────────────────────────────────────────
    # WorkingMemory: 会话内最近 K 轮摘要
    working_memory = WorkingMemory(max_entries=20)

    # EpisodicMemory: 跨会话轨迹检索复用
    episodic_memory = EpisodicMemory(
        store_dir=OUTPUT_DIR / "episodic",
        encoder=None,  # BGE-M3 encoder injectable later
        max_records=1000,
    )

    # ── 5. ContextConstructor: 动态 prompt 装配 ───────────────────────
    context_constructor = ContextConstructor(
        max_tokens=8000,
        strategy=ContextStrategy.RETRIEVAL_AUGMENTED,  # 优先展示 episodic hints
    )

    # ── 6. Agents (existing business logic) ──────────────────────────
    hypothesis_agent = HypothesisAgent(api_key=api_key)
    diag_agent = DiagnosisAgent(
        history_path=hypothesis_agent.history_path,
        explored_names=hypothesis_agent._explored_names,
    )

    # ── 7. Wire everything together ──────────────────────────────────
    register_all_handlers(
        loop=loop,
        hypothesis_agent=hypothesis_agent,
        diagnosis_agent=diag_agent,
        api_key=api_key,
        guardrail_pipeline=guardrail,
        episodic_memory=episodic_memory,
        working_memory=working_memory,
        context_constructor=context_constructor,
        tool_registry=tool_registry,
        output_dir=OUTPUT_DIR,
        symbols=symbols,
        years=years,
    )

    return {
        "loop": loop,
        "tracer": tracer,
        "tool_registry": tool_registry,
        "guardrail": guardrail,
        "working_memory": working_memory,
        "episodic_memory": episodic_memory,
        "context_constructor": context_constructor,
        "hypothesis_agent": hypothesis_agent,
        "diag_agent": diag_agent,
    }


def run_harness(
    api_key: str,
    max_iter: int = 10,
    dry_run: bool = False,
    symbols: list[str] | None = None,
    years: list[int] | None = None,
    checkpoint_dir: Path | str | None = None,
):
    """Run the full EarningsSignal Agent through the Harness framework.

    This is the drop-in replacement for run_agent.run_loop().
    """
    print("=" * 60)
    print("EarningsSignal Agent — Harness Edition")
    print(f"  max_iter={max_iter}  dry_run={dry_run}")
    if symbols:
        print(f"  symbols={symbols}")
    if years:
        print(f"  years={years}")
    print("=" * 60)

    if dry_run:
        # Dry-run: only HypothesisAgent, no API calls
        agent = HypothesisAgent(api_key=api_key)
        for i in range(1, max_iter + 1):
            spec = agent.next_feature()
            print(f"\n[DRY-RUN {i}/{max_iter}] {spec['feature_name']}")
            print(f"  definition: {spec['definition']}")
            print(f"  query: {spec['retrieval_query']}")
            print(f"  scope: {spec['condition_scope']}")
        return {"status": "dry_run_complete", "iterations": max_iter}

    # Build pipeline
    components = build_harness_pipeline(
        api_key=api_key,
        max_iter=max_iter,
        symbols=symbols,
        years=years,
        checkpoint_dir=checkpoint_dir,
    )

    loop = components["loop"]
    hypothesis_agent = components["hypothesis_agent"]

    # Run
    print(f"\n[Harness] Starting AgentLoop with {max_iter} max iterations...")
    result = loop.run(max_iter=max_iter)

    # Summary
    summary = hypothesis_agent.summary()
    print(f"\n{'='*60}")
    print(f"Agent 运行完成 (Harness Edition)")
    print(f"  Status: {result['status']}")
    print(f"  总探索: {summary['total_explored']} 个特征")
    print(f"  PASS:  {summary['passed']} 个")
    print(f"  FAIL:  {summary['failed']} 个")
    if summary["passed_features"]:
        print(f"  已通过特征: {summary['passed_features']}")

    # Harness-specific stats
    print(f"\n[Harness Stats]")
    print(f"  Trace entries: {len(loop.tracer._traces)}")
    print(f"  Episodic records: {components['episodic_memory'].record_count}")
    print(f"  Dead letters: {components['tool_registry'].dead_letter_count()}")
    stats = loop.tracer.stats_summary()
    if stats:
        print(f"  {stats}")

    print("=" * 60)

    # Save final trace
    trace_path = OUTPUT_DIR / f"trace_{loop.tracer.run_id}.json"
    try:
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(
                {"run_id": loop.tracer.run_id, "steps": [s.__dict__ for s in loop.tracer._traces]},
                f, ensure_ascii=False, indent=2, default=str,
            )
        print(f"[Harness] Full trace saved: {trace_path}")
    except Exception:
        pass

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EarningsSignal Agent — Harness Edition")
    parser.add_argument("--api-key", required=False, default=None,
                        help="API key (default: from .env)")
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--years", nargs="+", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    args = parser.parse_args()

    api_key = args.api_key or API_KEY
    if not api_key:
        print("ERROR: API key required. Set SILICONFLOW_API_KEY in .env or pass --api-key.")
        sys.exit(1)

    run_harness(
        api_key=api_key,
        max_iter=args.max_iter,
        dry_run=args.dry_run,
        symbols=args.symbols,
        years=args.years,
        checkpoint_dir=args.checkpoint_dir,
    )
