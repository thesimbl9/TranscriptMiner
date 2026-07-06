"""fullscale_repair_test.py — 全量验证 v2 修复效果

对比 contrastive_connectives v1 vs v2 的全量(11,363 eps)提取+验证结果。
v2 使用 v1 的 retrieval_query（修复只改 extraction_instruction，不动检索）。

用法：
  python fullscale_repair_test.py
"""

import json
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

FULLPROJECT = Path(__file__).parent.resolve()
sys.path.insert(0, str(FULLPROJECT))

from agent_core.config import HISTORY_PATH, API_KEY


def load_latest_repair_spec(feature_name: str) -> dict | None:
    """从 history 加载指定特征的最新 repair_spec"""
    latest = None
    with open(HISTORY_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            obj = json.loads(line)
            if "_repair_spec" in obj:
                spec = obj["_repair_spec"]
                if spec.get("_repair_of") == feature_name:
                    latest = spec
    return latest


def load_v1_spec(feature_name: str) -> dict | None:
    """从 history 加载 v1 feature_spec"""
    with open(HISTORY_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            obj = json.loads(line)
            if "_repair_spec" in obj or "_patch_feature" in obj:
                continue
            spec = obj.get("feature_spec", {})
            if spec.get("feature_name") == feature_name:
                return spec
    return None


def main():
    print("=" * 72)
    print("  Full-Scale v1 vs v2 Validation Test")
    print("  contrastive_connectives: 11,363 episodes")
    print("=" * 72)

    feature_name = "contrastive_connectives"
    v2_name = feature_name + "_v2"

    # ── 加载 v1 原始结果 ──────────────────────────────────────────────────
    print(f"\n[1/5] 加载 v1 原始结果...")
    v1_spec = load_v1_spec(feature_name)
    if not v1_spec:
        print(f"  [FAIL] v1 spec 未找到")
        return 1
    print(f"  v1 retrieval_query: {v1_spec.get('retrieval_query', 'N/A')[:80]}")

    # ── 加载 v2 repair_spec ───────────────────────────────────────────────
    print(f"\n[2/5] 加载 v2 repair_spec...")
    v2_spec = load_latest_repair_spec(feature_name)
    if not v2_spec:
        print(f"  [FAIL] v2 repair_spec 未找到")
        return 1

    # 关键：用 v1 的 retrieval_query 替换 v2 中 LLM 私自修改的
    v2_orig_query = v2_spec.get("retrieval_query", "")
    v2_spec["retrieval_query"] = v1_spec.get("retrieval_query", v2_orig_query)
    print(f"  v2 retrieval_query (fixed to v1): {v2_spec['retrieval_query'][:80]}")
    print(f"  v2 extraction_instruction (前150字):")
    print(f"    {v2_spec.get('extraction_instruction', 'N/A')[:150]}...")

    # ── v1 全量提取（如果已有 CSV 则跳过）───────────────────────────────
    print(f"\n[3/5] v1 全量提取...")
    from agent_core.extraction_agent import extract_feature_global

    v1_csv = FULLPROJECT / "agent_output" / f"{feature_name}.csv"
    if v1_csv.exists():
        print(f"  v1 CSV 已存在: {v1_csv} ({v1_csv.stat().st_size / 1024:.0f} KB)")
        import pandas as pd
        v1_df = pd.read_csv(v1_csv)
        print(f"  loaded: {len(v1_df)} rows")
    else:
        print(f"  开始全量提取 v1 (~4-5 min)...")
        t0 = time.time()
        v1_df = extract_feature_global(
            v1_spec, api_key=API_KEY,
            sample_n=None, batch_size=50, max_workers=4, debug_n=0,
        )
        print(f"  v1 提取完成 ({time.time()-t0:.0f}s): {len(v1_df)} rows")

    # ── v2 全量提取 ──────────────────────────────────────────────────────
    print(f"\n[4/5] v2 全量提取 (~4-5 min)...")
    v2_csv = FULLPROJECT / "agent_output" / f"{v2_name}.csv"
    if v2_csv.exists():
        print(f"  v2 CSV 已存在: {v2_csv} ({v2_csv.stat().st_size / 1024:.0f} KB)")
        import pandas as pd
        v2_df = pd.read_csv(v2_csv)
        v2_df.columns = ["symbol", "earnings_date", v2_name]
        print(f"  loaded: {len(v2_df)} rows")
    else:
        print(f"  开始全量提取 v2 (~4-5 min)...")
        t0 = time.time()
        v2_df = extract_feature_global(
            v2_spec, api_key=API_KEY,
            sample_n=None, batch_size=50, max_workers=4, debug_n=0,
        )
        print(f"  v2 提取完成 ({time.time()-t0:.0f}s): {len(v2_df)} rows")

    if len(v2_df) == 0:
        print(f"  [FAIL] v2 extraction 返回空")
        return 1

    # ── 验证 v1 + v2 ──────────────────────────────────────────────────────
    print(f"\n[5/5] 验证 v1 + v2...")
    from agent_core.validation_agent import validate
    from agent_core.governance_agent import check

    t0 = time.time()
    v1_val = validate(v1_df, feature_name, use_lgbm=True, verbose=False)
    v1_gov = check(v1_val)
    print(f"  v1 验证完成 ({time.time()-t0:.0f}s)")

    t0 = time.time()
    v2_val = validate(v2_df, v2_name, use_lgbm=True, verbose=False)
    v2_gov = check(v2_val)
    print(f"  v2 验证完成 ({time.time()-t0:.0f}s)")

    # ── 对比 ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print(f"  全量 v1 vs v2 对比")
    print(f"{'=' * 72}")
    print(f"  {'指标':<28} {'v1 (原始)':<22} {'v2 (修复)':<22} {'变化':<16} {'评估'}")
    print(f"  {'-' * 28} {'-' * 22} {'-' * 22} {'-' * 16} {'-' * 8}")

    pairs = [
        ("IC", v1_gov.get("ic", 0), v2_gov.get("ic", 0), "+.4f"),
        ("|t-stat|", abs(v1_gov.get("t_stat", 0)), abs(v2_gov.get("t_stat", 0)), ".3f"),
        ("zero_ratio", v1_gov.get("zero_ratio", 0), v2_gov.get("zero_ratio", 0), ".1%"),
        ("direction_consistency", v1_gov.get("direction_consistency", 0), v2_gov.get("direction_consistency", 0), ".0%"),
        ("OOS test_coverage", v1_val.get("test_coverage_ratio", 0), v2_val.get("test_coverage_ratio", 0), ".1%"),
    ]

    zero_improved = False
    for label, v1_v, v2_v, fmt in pairs:
        delta = v2_v - v1_v
        if fmt == "+.4f":
            s1, s2, sd = f"{v1_v:+.4f}", f"{v2_v:+.4f}", f"{delta:+.4f}"
        elif fmt == ".3f":
            s1, s2, sd = f"{v1_v:.3f}", f"{v2_v:.3f}", f"{delta:+.3f}"
        elif fmt == ".1%":
            s1, s2, sd = f"{v1_v:.1%}", f"{v2_v:.1%}", f"{delta:+.1%}"
        else:
            s1, s2, sd = f"{v1_v:.0%}", f"{v2_v:.0%}", f"{delta:+.0%}"

        if "zero" in label.lower():
            ok = delta < 0
            if ok:
                zero_improved = True
        elif "t-stat" in label.lower() or label == "IC":
            ok = delta > 0
        elif "coverage" in label.lower() or "direction" in label.lower():
            ok = delta >= -0.02
        else:
            ok = delta > 0

        status = "[OK]" if ok else "[X]"
        print(f"  {label:<28} {s1:<22} {s2:<22} {sd:<16} {status}")

    # ── Governance ─────────────────────────────────────────────────────────
    print(f"\n  v1 PASS: {v1_gov['passed']}  failures: {v1_gov.get('failures', [])}")
    print(f"  v2 PASS: {v2_gov['passed']}  failures: {v2_gov.get('failures', [])}")

    # ── 总结 ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print(f"  全量验证结论")
    print(f"{'=' * 72}")

    if v2_gov['passed']:
        print(f"\n  *** PASS: v2 修复成功，全量验证通过！ ***")
        print(f"  修复流水线端到端闭环：G2-only FAIL -> Diagnosis -> Repair -> Extract -> PASS")
    else:
        failures = v2_gov.get('failures', [])
        g2_fail = any("G2" in f for f in failures)
        if zero_improved and not g2_fail:
            print(f"\n  v2 零值率已修复但其他门控失败")
            print(f"  修复方向正确，zero_ratio 改善显著")
        elif zero_improved and g2_fail:
            print(f"\n  v2 零值率改善但仍未达到 G2 门槛")
        else:
            print(f"\n  v2 修复未达预期")

    return 0


if __name__ == "__main__":
    sys.exit(main())
