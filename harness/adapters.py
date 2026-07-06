"""
harness/adapters.py — 将现有5个Agent适配为Harness状态处理器

每个handler签名为 handler(loop) -> StepResult，职责：
  1. 从 loop.context 读取上一步的输出
  2. 调用对应Agent执行业务逻辑
  3. 将结果写入 loop.context 供下一步使用
  4. 更新 loop.current_feature_name / loop.completed_features
  5. 通过 loop.tracer 记录关键指标

P2.5 修复（2026-07-05）：
  - ToolRegistry 通过 invoke_fn() 真正接入超时/重试/降级
  - EpisodicMemory hints 注入到 HypothesisAgent.next_feature()
  - ContextConstructor 动态装配 session context 并控制 token 预算
  - WorkingMemory 在 FINALIZING 时更新
  - EVALUATING FAIL → DIAGNOSING 流转（在 loop._advance_state 中实现）
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Ensure agent_core is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.loop import AgentLoop, AgentState, StepResult


# ═══════════════════════════════════════════════════════════════════════════════════
# Helper: extract serializable summary from agent results
# ═══════════════════════════════════════════════════════════════════════════════════

def _safe_summary(obj: Any, max_len: int = 300) -> str:
    """Safe summary for tracer recording."""
    if obj is None:
        return "None"
    if isinstance(obj, dict):
        keys = list(obj.keys())
        return f"dict with keys: {keys[:10]}"
    if isinstance(obj, pd.DataFrame):
        return f"DataFrame({len(obj)} rows, {list(obj.columns)[:5]} cols)"
    s = str(obj)
    return s[:max_len] + ("..." if len(s) > max_len else "")


# ═══════════════════════════════════════════════════════════════════════════════════
# State Handler: PLANNING → HypothesisAgent.next_feature()
# ═══════════════════════════════════════════════════════════════════════════════════

def create_planning_handler(
    hypothesis_agent,
    episodic_memory=None,
    working_memory=None,
    context_constructor=None,
    registry=None,
):
    """Factory: create PLANNING state handler.

    P2.5: EpisodicMemory hints + ContextConstructor → agent.next_feature(episodic_hints=...)
    ToolRegistry: uses registry.invoke_fn() with closure wrapper for timeout/retry protection.
    """

    def _planning_fn(**kwargs) -> dict:
        """Closure wrapper for ToolRegistry.invoke_fn."""
        hints = kwargs.pop("_episodic_hints", "")
        return hypothesis_agent.next_feature(episodic_hints=hints)

    def handle_planning(loop: AgentLoop) -> StepResult:
        t0 = time.perf_counter()

        try:
            # ── Step 1: 检索 EpisodicMemory（跨会话历史教训）──
            episodic_hints = ""
            if episodic_memory is not None:
                try:
                    # Use the last feature's definition if available, else broad query
                    prev_spec = loop.context.get("feature_spec")
                    query_text = prev_spec.get("definition", "") if prev_spec else ""
                    if query_text:
                        episodic_hints = episodic_memory.generate_hint(query_text, k=3)
                    else:
                        # First iteration: retrieve recent failure patterns
                        recent_fails = episodic_memory.retrieve_by_pattern("G2_zero_ratio", k=2)
                        if recent_fails:
                            episodic_hints = episodic_memory.generate_hint(
                                recent_fails[0].definition, k=3
                            )
                except Exception:
                    pass  # Memory retrieval is non-critical

            # ── Step 2: 获取 WorkingMemory（会话内最近K轮摘要）──
            wm_ctx = ""
            if working_memory is not None:
                try:
                    wm_ctx = working_memory.get_context(n=5)
                except Exception:
                    pass

            # ── Step 3: ContextConstructor 动态装配 session context ──
            session_ctx = ""
            if context_constructor is not None and (episodic_hints or wm_ctx):
                try:
                    session_ctx = context_constructor.assemble(
                        system="",
                        task="",
                        working_memory=wm_ctx,
                        episodic_hints=episodic_hints,
                    )
                except Exception:
                    session_ctx = episodic_hints  # Fallback: raw hints
            else:
                session_ctx = episodic_hints or wm_ctx

            # ── Step 4: 调用 HypothesisAgent（通过 ToolRegistry 保护）──
            if registry is not None:
                result = registry.invoke_fn(
                    "hypothesis",
                    _planning_fn,
                    _episodic_hints=session_ctx,
                    _iteration=loop.iteration,
                    _loop_state=loop.state.name,
                )
                if not result.ok:
                    return StepResult(
                        ok=False, error=result.error,
                        should_retry=True, retry_after_s=10.0,
                    )
                # _planning_fn returns feature_spec dict directly (not wrapped)
                feature_spec = result.data
            else:
                feature_spec = hypothesis_agent.next_feature(episodic_hints=session_ctx)

            feature_name = feature_spec["feature_name"]
            loop.current_feature_name = feature_name

            # ── Store in context for downstream handlers ──
            loop.context["feature_spec"] = feature_spec
            loop.context["episodic_hints"] = session_ctx
            loop.context["csv_path"] = None

            latency_ms = (time.perf_counter() - t0) * 1000

            # ── Trace ──
            loop.tracer.record_step(
                step_id=loop.step_id,
                state="PLANNING",
                iteration=loop.iteration,
                tool_called="hypothesis",
                tool_latency_ms=latency_ms,
                tool_ok=True,
                observation=(
                    f"Feature: {feature_name} | "
                    f"cluster: {feature_spec.get('_theory_cluster', 'seed')} | "
                    f"episodic_hints: {len(session_ctx)} chars"
                ),
                decision="CONTINUE",
            )

            return StepResult(ok=True, data=feature_spec)

        except Exception as e:
            loop.tracer.record_step(
                step_id=loop.step_id,
                state="PLANNING",
                iteration=loop.iteration,
                tool_called="hypothesis",
                tool_ok=False,
                tool_error=str(e),
                observation=traceback.format_exc()[-300:],
                decision="FAIL" if not _is_retryable(e) else "RETRY",
            )
            return StepResult(
                ok=False, error=f"{type(e).__name__}: {e}",
                should_retry=_is_retryable(e),
            )

    return handle_planning


# ═══════════════════════════════════════════════════════════════════════════════════
# State Handler: ACTING → ExtractionAgent.extract_feature_global()
# ═══════════════════════════════════════════════════════════════════════════════════

def create_acting_handler(
    api_key: str,
    output_dir: Path | None = None,
    symbols: list[str] | None = None,
    years: list[int] | None = None,
    registry=None,
    sample_n: int = 400,
):
    """Factory: create ACTING state handler.

    P2.5: uses registry.invoke_fn() with closure wrapping extract_feature_global.
    P3: sample_n=400 — discovery-phase stratified sampling for fast formula validation.
    """

    def handle_acting(loop: AgentLoop) -> StepResult:
        t0 = time.perf_counter()
        feature_spec = loop.context.get("feature_spec")
        if feature_spec is None:
            return StepResult(ok=False, error="No feature_spec in context. PLANNING must run first.")

        feature_name = feature_spec["feature_name"]
        output_dir_path = output_dir or Path("agent_output")
        csv_path = output_dir_path / f"{feature_name}.csv"
        loop.context["csv_path"] = str(csv_path)

        # ── Duplicate guard: if this feature already extracted (CSV exists with data), skip ──
        csv_path_obj = Path(csv_path)
        if csv_path_obj.exists() and csv_path_obj.stat().st_size > 100:
            try:
                feature_df = pd.read_csv(csv_path)
                feature_df["earnings_date"] = pd.to_datetime(feature_df["earnings_date"])
                loop.context["feature_df"] = feature_df
                loop.context["n_episodes"] = len(feature_df)
                print(f"[ACTING] Cache HIT: {feature_name} → {len(feature_df)} rows from {csv_path}", flush=True)
                latency_ms = (time.perf_counter() - t0) * 1000
                loop.tracer.record_step(
                    step_id=loop.step_id, state="ACTING", iteration=loop.iteration,
                    tool_called="extraction", tool_latency_ms=latency_ms, tool_ok=True,
                    observation=f"Cached: {len(feature_df)} rows from {csv_path}",
                    decision="CONTINUE",
                )
                return StepResult(ok=True, data={"n_episodes": len(feature_df), "cached": True})
            except Exception as _cache_err:
                print(f"[ACTING] Cache load FAILED for {csv_path}: {_cache_err}", flush=True)
                import traceback as _tb
                _tb.print_exc()
                # Cache corrupt, re-extract below

        # ── Call ExtractionAgent ──
        print(f"[ACTING] Starting extraction: {feature_name} → {csv_path} (step={loop.step_id}, iter={loop.iteration})", flush=True)
        try:
            from agent_core.extraction_agent import extract_feature_global

            # Closure wrapper for ToolRegistry
            def _extraction_fn(**kwargs) -> dict:
                spec = kwargs.pop("feature_spec")
                ak = kwargs.pop("api_key", api_key)
                op = kwargs.pop("output_path", str(csv_path))
                sym = kwargs.pop("symbols", symbols)
                yr = kwargs.pop("years", years)
                sn = kwargs.pop("sample_n", sample_n)
                df = extract_feature_global(spec, ak, op, sym, yr,
                    max_workers=4, batch_size=50, sample_n=sn)
                return {"feature_df": df, "n_episodes": len(df) if df is not None else 0}

            if registry is not None:
                result = registry.invoke_fn(
                    "extraction",
                    _extraction_fn,
                    feature_spec=feature_spec,
                    api_key=api_key,
                    output_path=str(csv_path),
                    symbols=symbols,
                    years=years,
                    sample_n=sample_n,
                    _iteration=loop.iteration,
                    _loop_state=loop.state.name,
                )
                if not result.ok:
                    return StepResult(
                        ok=False, error=result.error,
                        should_retry=True, retry_after_s=15.0,
                    )
                feature_df = result.data.get("feature_df") if isinstance(result.data, dict) else result.data
            else:
                feature_df = extract_feature_global(
                    feature_spec=feature_spec,
                    api_key=api_key,
                    output_path=str(csv_path),
                    symbols=symbols,
                    years=years,
                    max_workers=4,
                    batch_size=50,
                    sample_n=sample_n,
                )

            loop.context["feature_df"] = feature_df
            n_episodes = len(feature_df) if feature_df is not None else 0
            loop.context["n_episodes"] = n_episodes

            latency_ms = (time.perf_counter() - t0) * 1000
            loop.tracer.record_step(
                step_id=loop.step_id, state="ACTING", iteration=loop.iteration,
                tool_called="extraction", tool_latency_ms=latency_ms, tool_ok=True,
                observation=f"Extracted: {n_episodes} episodes",
                decision="CONTINUE",
            )
            return StepResult(ok=True, data={"n_episodes": n_episodes, "cached": False})

        except Exception as e:
            loop.tracer.record_step(
                step_id=loop.step_id, state="ACTING", iteration=loop.iteration,
                tool_called="extraction", tool_ok=False, tool_error=str(e),
                observation=traceback.format_exc()[-300:],
                decision="FAIL" if not _is_retryable(e) else "RETRY",
            )
            return StepResult(
                ok=False, error=f"{type(e).__name__}: {e}",
                should_retry=_is_retryable(e),
            )

    return handle_acting


# ═══════════════════════════════════════════════════════════════════════════════════
# State Handler: OBSERVING → ValidationAgent.validate()
# ═══════════════════════════════════════════════════════════════════════════════════

def create_observing_handler(registry=None):
    """Factory: create OBSERVING state handler.

    P2.5: uses registry.invoke_fn() with closure wrapping validate().
    """

    def handle_observing(loop: AgentLoop) -> StepResult:
        t0 = time.perf_counter()
        feature_spec = loop.context.get("feature_spec")
        feature_df = loop.context.get("feature_df")
        feature_name = feature_spec["feature_name"] if feature_spec else loop.current_feature_name

        if feature_df is None or len(feature_df) == 0:
            # Empty extraction result → skip to FINALIZING with empty result
            loop.context["val_result"] = {
                "feature_name": feature_name,
                "ic": 0.0, "t_stat": 0.0, "n_quarters": 0,
                "zero_ratio": 1.0, "direction_consistency": 0.0,
                "per_sector_ic": {}, "season_ic": {},
                "score_dist": {}, "zero_by_sector": {}, "zero_by_year": {},
                "coverage_failure": True, "test_coverage_ratio": 0.0,
            }
            latency_ms = (time.perf_counter() - t0) * 1000
            loop.tracer.record_step(
                step_id=loop.step_id, state="OBSERVING", iteration=loop.iteration,
                tool_called="validation", tool_latency_ms=latency_ms, tool_ok=True,
                observation=f"SKIP: empty extraction for {feature_name}",
                decision="CONTINUE",
            )
            return StepResult(ok=True)

        try:
            from agent_core.validation_agent import validate

            # Closure wrapper for ToolRegistry
            def _validation_fn(**kwargs) -> dict:
                df = kwargs.pop("feature_df")
                fn = kwargs.pop("feature_name")
                return validate(df, feature_name=fn, use_lgbm=True)

            if registry is not None:
                result = registry.invoke_fn(
                    "validation",
                    _validation_fn,
                    feature_df=feature_df,
                    feature_name=feature_name,
                    _iteration=loop.iteration,
                    _loop_state=loop.state.name,
                )
                if not result.ok:
                    return StepResult(ok=False, error=result.error, should_retry=False)
                val_result = result.data
            else:
                val_result = validate(feature_df, feature_name=feature_name, use_lgbm=True)

            loop.context["val_result"] = val_result

            latency_ms = (time.perf_counter() - t0) * 1000
            loop.tracer.record_step(
                step_id=loop.step_id, state="OBSERVING", iteration=loop.iteration,
                tool_called="validation", tool_latency_ms=latency_ms, tool_ok=True,
                observation=(
                    f"IC={val_result.get('ic', 0):+.4f} "
                    f"t={val_result.get('t_stat', 0):+.3f} "
                    f"zero_ratio={val_result.get('zero_ratio', 0):.1%}"
                ),
                decision="CONTINUE",
            )
            return StepResult(ok=True, data=val_result)

        except Exception as e:
            loop.tracer.record_step(
                step_id=loop.step_id, state="OBSERVING", iteration=loop.iteration,
                tool_called="validation", tool_ok=False, tool_error=str(e),
                observation=traceback.format_exc()[-300:],
                decision="FAIL",
            )
            return StepResult(ok=False, error=f"{type(e).__name__}: {e}", should_retry=False)

    return handle_observing


# ═══════════════════════════════════════════════════════════════════════════════════
# State Handler: EVALUATING → GuardrailPipeline.run() (replaces governance_agent.check)
# ═══════════════════════════════════════════════════════════════════════════════════

def create_evaluating_handler(guardrail_pipeline):
    """Factory: create EVALUATING state handler.

    This REPLACES governance_agent.check() with the pluggable GuardrailPipeline.

    P2.5: Always returns ok=True (step succeeded). gov_result.passed controls the
    next state transition: PASS → FINALIZING, FAIL → DIAGNOSING (handled in loop._advance_state).
    """

    def handle_evaluating(loop: AgentLoop) -> StepResult:
        t0 = time.perf_counter()
        feature_spec = loop.context.get("feature_spec")
        val_result = loop.context.get("val_result")

        if val_result is None:
            return StepResult(ok=False, error="No val_result in context. OBSERVING must run first.")

        feature_name = val_result.get("feature_name", loop.current_feature_name or "unknown")

        # ── Build context for GuardrailPipeline ──
        guardrail_ctx = {
            "validation_result": val_result,
            "feature_spec": feature_spec or {},
            "feature_name": feature_name,
        }

        try:
            # Run guardrail pipeline (post-exec hooks: G1/G2/G3/G4)
            gr_result = guardrail_pipeline.run(guardrail_ctx)

            # Build gov_result in same format as original governance_agent.check()
            gov_result = {
                "feature_name": feature_name,
                "passed": gr_result.passed,
                "failures": gr_result.failures,
                "feedback": (
                    f"Feature '{feature_name}' passed all guardrails."
                    if gr_result.passed
                    else f"Feature '{feature_name}' failed {len(gr_result.failures)} guardrail(s): "
                         + "; ".join(gr_result.failures)
                ),
                "ic": val_result.get("ic", 0.0),
                "t_stat": val_result.get("t_stat", 0.0),
                "zero_ratio": val_result.get("zero_ratio", 1.0),
                "direction_consistency": val_result.get("direction_consistency", 0.0),
                "score_dist": val_result.get("score_dist", {}),
                "zero_by_sector": val_result.get("zero_by_sector", {}),
                "zero_by_year": val_result.get("zero_by_year", {}),
            }
            loop.context["gov_result"] = gov_result

            latency_ms = (time.perf_counter() - t0) * 1000
            loop.tracer.record_step(
                step_id=loop.step_id, state="EVALUATING", iteration=loop.iteration,
                tool_called="guardrail_pipeline", tool_latency_ms=latency_ms,
                tool_ok=gr_result.passed,
                observation=(
                    f"{'PASS' if gr_result.passed else 'FAIL'}: "
                    + (gr_result.failed_by or "all passed")
                    + (" → DIAGNOSING" if not gr_result.passed else " → FINALIZING")
                ),
                decision="CONTINUE",
            )

            # Always return ok=True — the state machine routes to DIAGNOSING
            # or FINALIZING based on gov_result.passed (in _advance_state).
            return StepResult(ok=True, data=gov_result)

        except Exception as e:
            loop.tracer.record_step(
                step_id=loop.step_id, state="EVALUATING", iteration=loop.iteration,
                tool_called="guardrail_pipeline", tool_ok=False, tool_error=str(e),
                observation=traceback.format_exc()[-300:],
                decision="FAIL",
            )
            return StepResult(ok=False, error=f"{type(e).__name__}: {e}", should_retry=False)

    return handle_evaluating


# ═══════════════════════════════════════════════════════════════════════════════════
# State Handler: DIAGNOSING → DiagnosisAgent.diagnose()
# ═══════════════════════════════════════════════════════════════════════════════════

def create_diagnosing_handler(diagnosis_agent, registry=None):
    """Factory: create DIAGNOSING state handler.

    P2.5: uses registry.invoke_fn() with closure wrapping diagnose().
    Non-fatal on failure — diagnosis failure doesn't crash the pipeline.
    """

    def handle_diagnosing(loop: AgentLoop) -> StepResult:
        t0 = time.perf_counter()
        feature_spec = loop.context.get("feature_spec")
        val_result = loop.context.get("val_result")
        gov_result = loop.context.get("gov_result")

        if gov_result is None or val_result is None:
            return StepResult(ok=False, error="No gov_result/val_result in context.")

        # Skip diagnosis if passed
        if gov_result.get("passed", False):
            loop.context["diagnosis"] = None
            return StepResult(ok=True, data={"skipped": True, "reason": "PASS"})

        try:
            # Closure wrapper for ToolRegistry
            def _diagnosis_fn(**kwargs) -> dict:
                spec = kwargs.pop("feature_spec")
                val = kwargs.pop("val_result")
                gov = kwargs.pop("gov_result")
                return diagnosis_agent.diagnose(spec, val, gov)

            if registry is not None:
                result = registry.invoke_fn(
                    "diagnosis",
                    _diagnosis_fn,
                    feature_spec=feature_spec,
                    val_result=val_result,
                    gov_result=gov_result,
                    _iteration=loop.iteration,
                    _loop_state=loop.state.name,
                )
                if not result.ok:
                    # Diagnosis failure is non-fatal — continue without diagnosis
                    loop.tracer.record_step(
                        step_id=loop.step_id, state="DIAGNOSING", iteration=loop.iteration,
                        tool_called="diagnosis", tool_ok=False, tool_error=result.error,
                        observation=f"Diagnosis failed (non-fatal): {result.error}",
                        decision="CONTINUE",
                    )
                    loop.context["diagnosis"] = {
                        "root_cause": f"Diagnosis failed: {result.error}",
                        "fix": "", "avoid": "",
                    }
                    return StepResult(ok=True, data={"fallback": True})
                diagnosis = result.data
            else:
                diagnosis = diagnosis_agent.diagnose(feature_spec, val_result, gov_result)

            loop.context["diagnosis"] = diagnosis
            gov_result["diagnosis"] = diagnosis  # merge back for FINALIZING

            latency_ms = (time.perf_counter() - t0) * 1000
            root_cause = diagnosis.get("root_cause", "")[:120] if diagnosis else "None"
            loop.tracer.record_step(
                step_id=loop.step_id, state="DIAGNOSING", iteration=loop.iteration,
                tool_called="diagnosis", tool_latency_ms=latency_ms, tool_ok=True,
                observation=f"Root cause: {root_cause}",
                decision="CONTINUE",
            )
            return StepResult(ok=True, data=diagnosis)

        except Exception as e:
            loop.tracer.record_step(
                step_id=loop.step_id, state="DIAGNOSING", iteration=loop.iteration,
                tool_called="diagnosis", tool_ok=False, tool_error=str(e),
                observation=traceback.format_exc()[-300:],
                decision="CONTINUE",  # Non-fatal
            )
            loop.context["diagnosis"] = {"root_cause": f"Error: {e}", "fix": "", "avoid": ""}
            return StepResult(ok=True, data={"fallback": True})

    return handle_diagnosing


# ═══════════════════════════════════════════════════════════════════════════════════
# State Handler: FINALIZING → record_result + EpisodicMemory.store() + WorkingMemory
# ═══════════════════════════════════════════════════════════════════════════════════

def create_finalizing_handler(
    hypothesis_agent,
    episodic_memory=None,
    working_memory=None,
):
    """Factory: create FINALIZING state handler.

    P2.5: Also updates WorkingMemory so subsequent PLANNING steps have session context.
    """

    def handle_finalizing(loop: AgentLoop) -> StepResult:
        t0 = time.perf_counter()
        feature_spec = loop.context.get("feature_spec")
        gov_result = loop.context.get("gov_result")

        if feature_spec is None:
            return StepResult(ok=True, data={"skipped": True, "reason": "No feature_spec"})

        feature_name = feature_spec.get("feature_name", loop.current_feature_name or "unknown")
        outcome = "PASS" if (gov_result or {}).get("passed") else "FAIL"
        ic = (gov_result or {}).get("ic", 0.0)
        t = (gov_result or {}).get("t_stat", 0.0)
        zr = (gov_result or {}).get("zero_ratio", 0.0)

        # ── Record result to HypothesisAgent (writes feature_history.jsonl) ──
        if hypothesis_agent is not None and gov_result is not None:
            hypothesis_agent.record_result(feature_spec, gov_result)

        # ── Store to EpisodicMemory (跨会话检索复用) ──
        if episodic_memory is not None and gov_result is not None:
            try:
                val_result = loop.context.get("val_result", {})
                diagnosis = loop.context.get("diagnosis", {})
                episodic_memory.store_from_iteration(
                    feature_name=feature_name,
                    definition=feature_spec.get("definition", ""),
                    retrieval_query=feature_spec.get("retrieval_query", ""),
                    condition_scope=feature_spec.get("condition_scope", {}),
                    outcome=outcome,
                    ic=float(gov_result.get("ic", 0.0)),
                    t_stat=float(gov_result.get("t_stat", 0.0)),
                    zero_ratio=float(gov_result.get("zero_ratio", 1.0)),
                    direction_consistency=float(gov_result.get("direction_consistency", 0.0)),
                    failures=gov_result.get("failures", []),
                    diagnosis_root_cause=diagnosis.get("root_cause", "") if diagnosis else "",
                    diagnosis_fix=diagnosis.get("fix", "") if diagnosis else "",
                    iteration=loop.iteration,
                )
            except Exception:
                pass  # Episodic storage is non-critical

        # ── Update WorkingMemory (会话内最近K轮摘要) ──
        if working_memory is not None:
            try:
                key_metrics = f"IC={ic:+.4f} t={t:+.3f} zr={zr:.1%}"
                working_memory.add(
                    iteration=loop.iteration,
                    feature_name=feature_name,
                    outcome=outcome,
                    key_metrics=key_metrics,
                )
            except Exception:
                pass

        # ── Mark completed ──
        if feature_name not in loop.completed_features:
            loop.completed_features.append(feature_name)

        # ── Write report JSON (same format as original run_agent.py) ──
        try:
            from agent_core.config import OUTPUT_DIR
            val_result = loop.context.get("val_result", {})
            diag = loop.context.get("diagnosis")
            if gov_result and diag:
                gov_result["diagnosis"] = diag
            report = {
                "iteration": loop.iteration,
                "feature_spec": feature_spec,
                "val_result": {k: v for k, v in val_result.items() if k != "season_ic"}
                if isinstance(val_result, dict) else {},
                "gov_result": gov_result,
                "level": "L3_full",
            }
            report_path = OUTPUT_DIR / f"report_{feature_name}.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        except Exception:
            pass

        latency_ms = (time.perf_counter() - t0) * 1000

        loop.tracer.record_step(
            step_id=loop.step_id, state="FINALIZING", iteration=loop.iteration,
            tool_called="record_result", tool_latency_ms=latency_ms, tool_ok=True,
            observation=(
                f"Result: {outcome} | IC={ic:+.4f} t={t:+.3f} zero_ratio={zr:.1%} | "
                f"Total completed: {len(loop.completed_features)}"
            ),
            decision="CONTINUE",
        )

        # Clean context for next iteration (keep persistent references)
        for key in ("feature_df", "val_result", "gov_result", "diagnosis", "episodic_hints"):
            loop.context.pop(key, None)

        return StepResult(ok=True)

    return handle_finalizing


# ═══════════════════════════════════════════════════════════════════════════════════
# Utility: determine if error is retryable
# ═══════════════════════════════════════════════════════════════════════════════════

def _is_retryable(exc: Exception) -> bool:
    """Check if an exception suggests a transient failure worth retrying."""
    name = type(exc).__name__
    msg = str(exc).lower()
    retryable_names = {
        "Timeout", "ConnectionError", "RemoteDisconnected", "ReadTimeout",
        "ConnectTimeout", "RateLimitError", "ServiceUnavailableError",
        "APITimeoutError", "APIConnectionError",
    }
    retryable_keywords = [
        "timeout", "rate limit", "too many requests", "service unavailable",
        "connection reset", "connection refused", "try again",
    ]
    if name in retryable_names:
        return True
    return any(kw in msg for kw in retryable_keywords)


# ═══════════════════════════════════════════════════════════════════════════════════
# Convenience: register all handlers on a loop instance at once
# ═══════════════════════════════════════════════════════════════════════════════════

def register_all_handlers(
    loop: AgentLoop,
    hypothesis_agent,
    diagnosis_agent,
    api_key: str,
    guardrail_pipeline,
    episodic_memory=None,
    working_memory=None,
    context_constructor=None,
    tool_registry=None,
    output_dir: Path | None = None,
    symbols: list[str] | None = None,
    years: list[int] | None = None,
) -> AgentLoop:
    """Register all 7 state handlers on the given AgentLoop instance.

    This is the one-call setup for the full EarningsSignal pipeline.
    All Harness components (ToolRegistry, GuardrailPipeline, EpisodicMemory,
    WorkingMemory, ContextConstructor) are wired through in a single call.

    P2.5: Added working_memory and context_constructor parameters.
    """
    loop.register_handler(
        AgentState.PLANNING,
        create_planning_handler(
            hypothesis_agent,
            episodic_memory=episodic_memory,
            working_memory=working_memory,
            context_constructor=context_constructor,
            registry=tool_registry,
        ),
    )
    loop.register_handler(
        AgentState.ACTING,
        create_acting_handler(
            api_key=api_key,
            output_dir=output_dir,
            symbols=symbols,
            years=years,
            registry=tool_registry,
        ),
    )
    loop.register_handler(
        AgentState.OBSERVING,
        create_observing_handler(registry=tool_registry),
    )
    loop.register_handler(
        AgentState.EVALUATING,
        create_evaluating_handler(guardrail_pipeline),
    )
    loop.register_handler(
        AgentState.DIAGNOSING,
        create_diagnosing_handler(diagnosis_agent, registry=tool_registry),
    )
    loop.register_handler(
        AgentState.FINALIZING,
        create_finalizing_handler(
            hypothesis_agent,
            episodic_memory=episodic_memory,
            working_memory=working_memory,
        ),
    )

    # RETRYING handler: just transition back to previous state
    def handle_retrying(loop_: AgentLoop) -> StepResult:
        return StepResult(ok=True, data={"retry_completed": True})

    loop.register_handler(AgentState.RETRYING, handle_retrying)

    return loop
