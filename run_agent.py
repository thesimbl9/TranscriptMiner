"""
run_agent.py — EarningsSignal Agent 主循环

完整链路：
  HypothesisAgent → ExtractionAgent（全量）→ ValidationAgent → GovernanceAgent
                                                → FAIL → DiagnosisAgent → 写回 history → 下轮迭代

用法：
  python run_agent.py [--max-iter 10] [--dry-run] [--symbols AAPL MSFT]

  --dry-run:    只跑 HypothesisAgent（输出 feature_spec），不调 LLM API
  --max-iter:   最大迭代轮数（默认 10）
  --symbols:    只处理指定 symbol（调试用，默认全量）
  --years:      只处理指定年份，如 --years 2021 2022 2023
  --global:     使用全局检索模式（默认开启）
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agent_core.config import OUTPUT_DIR
OUTPUT_DIR.mkdir(exist_ok=True)

from agent_core.hypothesis_agent  import HypothesisAgent
from agent_core.extraction_agent  import extract_feature, extract_feature_global
from agent_core.validation_agent  import validate
from agent_core.governance_agent  import check
from agent_core.diagnosis_agent   import DiagnosisAgent


def run_loop(
    api_key: str,
    max_iter: int = 10,
    dry_run: bool = False,
    symbols: list[str] | None = None,
    years: list[int] | None = None,
    use_global: bool = False,
):
    agent      = HypothesisAgent(api_key=api_key)
    # 共享 _explored_names 引用：DiagnosisAgent 判断 _v2 是否已存在时使用
    diag_agent = DiagnosisAgent(
        history_path=agent.history_path,
        explored_names=agent._explored_names,
    )

    print("=" * 60)
    print("EarningsSignal Agent 启动")
    print(f"  max_iter={max_iter}  dry_run={dry_run}  global={use_global}")
    if symbols:
        print(f"  symbols={symbols}")
    if years:
        print(f"  years={years}")
    print(f"  种子特征数: {len(agent._seed_queue)}")
    print("=" * 60)

    for iteration in range(1, max_iter + 1):
        print(f"\n{'='*60}")
        print(f"[Iteration {iteration}/{max_iter}]")
        print("=" * 60)

        # ── Step 1: Hypothesis ──────────────────────────────────────
        feature_spec = agent.next_feature()
        print(f"\n[Step1] 特征: {feature_spec['feature_name']}")
        print(f"        定义: {feature_spec['definition']}")
        print(f"        query: {feature_spec['retrieval_query']}")
        print(f"        scope: {feature_spec['condition_scope']}")

        if dry_run:
            print("[DRY-RUN] 跳过 Extraction + Validation")
            continue

        # ── Step 2: 全量提取 ────────────────────────────────────────
        fname = feature_spec["feature_name"]
        csv_path = OUTPUT_DIR / f"{fname}.csv"

        if csv_path.exists():
            import pandas as pd
            feature_df = pd.read_csv(csv_path)
            feature_df["earnings_date"] = pd.to_datetime(feature_df["earnings_date"])
            print(f"\n[Step2] 已有缓存: {csv_path} ({len(feature_df)} 行)")
        else:
            print(f"\n[Step2] 全量提取: {fname}")
            feature_df = extract_feature_global(
                feature_spec=feature_spec,
                api_key=api_key,
                output_path=csv_path,
                symbols=symbols,
                years=years,
            )

        # 检索结果为空（如 sector 过滤后无 episodes）
        if len(feature_df) == 0:
            print(f"[SKIP] 检索结果为空，condition_scope 过于严格，跳过")
            gov_result = {
                "feature_name": fname, "passed": False,
                "failures": ["检索结果为空，condition_scope.sector 无匹配"],
                "feedback": f"特征 '{fname}' 检索结果为空，condition_scope.sector 可能过滤掉了所有 episodes。",
                "ic": 0.0, "t_stat": 0.0, "zero_ratio": 1.0, "direction_consistency": 0.0,
                "score_dist": {}, "zero_by_sector": {}, "zero_by_year": {},
            }
            agent.record_result(feature_spec, gov_result)
            report = {"iteration": iteration, "feature_spec": feature_spec,
                      "val_result": {}, "gov_result": gov_result, "level": "empty"}
            with open(OUTPUT_DIR / f"report_{fname}.json", "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            print(f"\n{'─'*40}")
            print(f"本轮结果: SKIP（检索为空）")
            continue

        # ── Step 3: Validation ──────────────────────────────────────
        print(f"\n[Step3] 验证特征: {fname}")
        val_result = validate(
            feature_df=feature_df,
            feature_name=fname,
            use_lgbm=True,
        )

        # ── Step 4: Governance ──────────────────────────────────────
        print(f"\n[Step4] 治理检查")
        gov_result = check(val_result, feature_spec=feature_spec)

        # ── Step 5: Diagnosis（仅 FAIL 时触发）──────────────────────
        if not gov_result["passed"]:
            print(f"\n[Step5] 失败诊断")
            diagnosis = diag_agent.diagnose(feature_spec, val_result, gov_result)
            gov_result["diagnosis"] = diagnosis

        agent.record_result(feature_spec, gov_result)

        report = {
            "iteration":     iteration,
            "feature_spec":  feature_spec,
            "val_result":    {k: v for k, v in val_result.items() if k != "season_ic"},
            "gov_result":    gov_result,
            "level":         "L3_full",
        }
        report_path = OUTPUT_DIR / f"report_{fname}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n[Report] 已保存: {report_path}")

        print(f"\n{'─'*40}")
        print(f"本轮结果: IC={val_result['ic']:+.4f}  t={val_result['t_stat']:+.3f}  "
              f"zero_ratio={val_result['zero_ratio']:.1%}  "
              f"{'PASS' if gov_result['passed'] else 'FAIL'}")

    # ── 最终摘要 ────────────────────────────────────────────────────────────
    summary = agent.summary()
    print(f"\n{'='*60}")
    print("Agent 运行完成")
    print(f"  总探索: {summary['total_explored']} 个特征")
    print(f"  PASS:  {summary['passed']} 个")
    print(f"  FAIL:  {summary['failed']} 个")
    if summary["passed_features"]:
        print(f"  已通过特征: {summary['passed_features']}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EarningsSignal Agent")
    parser.add_argument("--api-key",  required=False, default=None, help="API key (默认从 .env 读取)")
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--global",   dest="use_global", action="store_true")
    parser.add_argument("--symbols",  nargs="+", default=None)
    parser.add_argument("--years",    nargs="+", type=int, default=None)
    args = parser.parse_args()

    run_loop(
        api_key=args.api_key,
        max_iter=args.max_iter,
        dry_run=args.dry_run,
        symbols=args.symbols,
        years=args.years,
        use_global=args.use_global,
    )
