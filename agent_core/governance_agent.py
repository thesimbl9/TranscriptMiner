"""
governance_agent.py — Phase 1: 治理层

职责：对 ValidationResult 执行硬规则检查，决定特征是否 PASS。
不含任何 LLM 调用，不分析根因，不给改进建议。
FAIL 时由 DiagnosisAgent（独立第五层）负责诊断。

检查规则（全部通过才 PASS）：
  G2. zero_ratio < 阈值（按零值类型区分）
      - 均匀分布型（合法零值）：上限 0.70（管理层确实未提及，零值本身是有效信号）
      - 集中/未知型（逃避零值）：上限 0.45（LLM 对特定行业打分保守，需修复 instruction）
  G3. |t_stat| > 1.5
  G4. direction_consistency > 0.60（跨行业不系统性反转）
"""

import statistics
from typing import Any

ZERO_RATIO_MAX_UNIFORM      = 0.70   # 均匀分布型合法零值：放宽
ZERO_RATIO_MAX_CONCENTRATED = 0.45   # 集中/未知型逃避零值：严格
T_STAT_MIN                  = 1.5
DIRECTION_CONS_MIN          = 0.60


def _zero_type(validation_result: dict) -> str:
    """
    判断零值分布类型（与 DiagnosisAgent._zero_type 保持一致）：
      'uniform'         — 各行业零值率方差 < 0.02 且均值 ≤ 70%，合法零值（管理层确实未提及）
      'systemic_sparse' — 各行业方差 < 0.02 但均值 > 70%，所有行业一起死 → 管道级失败
      'concentrated'    — 某行业零值率显著偏高，逃避零值（instruction 对该行业适配差）
      'unknown'         — zero_by_sector 数据不足
    """
    zbs = validation_result.get("zero_by_sector", {})
    if len(zbs) < 3:
        return "unknown"
    vals  = list(zbs.values())
    var   = statistics.variance(vals)
    max_z = max(vals)
    min_z = min(vals)
    if var < 0.02:
        mean_zr = statistics.mean(vals)
        if mean_zr > 0.70:
            # 所有行业零值率都极高 → 不是"合法零值"，是管道级失败
            return "systemic_sparse"
        return "uniform"
    if max_z > 0.80 and min_z < 0.40:
        return "concentrated"
    return "unknown"


def check(
    validation_result: dict[str, Any],
    feature_spec: dict | None = None,
) -> dict[str, Any]:
    """
    对单个特征的验证结果执行治理检查。

    Args:
        validation_result: validation_agent.validate() 的返回值
        feature_spec:      原始 feature_spec（保留字段，便于上层传递，本层不使用）

    Returns:
        dict: {feature_name, passed, failures, feedback, ic, t_stat, zero_ratio,
               direction_consistency, score_dist, zero_by_sector, zero_by_year}
    """
    failures = []
    fn = validation_result.get("feature_name", "unknown")

    # G1: 测试期覆盖缺失（提取问题，非信号问题，优先判断）
    if validation_result.get("coverage_failure"):
        ratio = validation_result.get("test_coverage_ratio", 0.0)
        failures.append(
            f"G1_coverage: 测试期(2021-2023)特征覆盖率={ratio:.1%} < 5%，"
            f"提取阶段未覆盖测试期，结果不可信"
        )
        passed = False
        feedback = (
            f"特征 '{fn}' 测试期覆盖缺失（G1）。"
            f"condition_scope 或 retrieval_query 导致提取结果全落在训练期，"
            f"测试期(2021-2023)无有效样本。需放宽 condition_scope 或调整 retrieval_query。"
        )
        result = {
            "feature_name":          fn,
            "passed":                False,
            "failures":              failures,
            "feedback":              feedback,
            "ic":                    0.0,
            "t_stat":                0.0,
            "zero_ratio":            validation_result.get("zero_ratio"),
            "direction_consistency": 0.0,
            "score_dist":            validation_result.get("score_dist", {}),
            "zero_by_sector":        {},
            "zero_by_year":          {},
            "coverage_failure":      True,
            "test_coverage_ratio":   validation_result.get("test_coverage_ratio", 0.0),
        }
        print(f"[GovernanceAgent] {fn}: FAIL")
        print(f"  {failures[0]}")
        return result

    # G2: zero_ratio（按零值类型分档）
    zr = validation_result.get("zero_ratio", 1.0)
    zt = _zero_type(validation_result)
    g2_threshold = ZERO_RATIO_MAX_UNIFORM if zt == "uniform" else ZERO_RATIO_MAX_CONCENTRATED
    if zr >= g2_threshold:
        failures.append(
            f"G2_zero_ratio: {zr:.1%} >= {g2_threshold:.0%} (zero_type={zt})"
        )

    # G3: t_stat
    t = validation_result.get("t_stat", 0.0)
    if abs(t) < T_STAT_MIN:
        failures.append(
            f"G3_t_stat: |{t:.3f}| < {T_STAT_MIN}  IC={validation_result.get('ic', 0):+.4f}"
        )

    # G4: direction_consistency
    dc = validation_result.get("direction_consistency", 0.0)
    if dc < DIRECTION_CONS_MIN:
        ic = validation_result.get("ic", 0.0)
        bad_sectors = [
            f"{s}({v:+.3f})"
            for s, v in validation_result.get("per_sector_ic", {}).items()
            if (v < 0 and ic >= 0) or (v > 0 and ic < 0)
        ]
        failures.append(
            f"G4_direction: consistency={dc:.0%} < {DIRECTION_CONS_MIN:.0%}  "
            f"反向行业: {bad_sectors[:5]}"
        )

    passed = len(failures) == 0

    if passed:
        feedback = (
            f"特征 '{fn}' 通过全部治理检查。"
            f"IC={validation_result.get('ic', 0):+.4f}, "
            f"t={validation_result.get('t_stat', 0):+.3f}, "
            f"zero_ratio={validation_result.get('zero_ratio', 0):.1%}, "
            f"direction_consistency={validation_result.get('direction_consistency', 0):.0%}。"
        )
    else:
        feedback = (
            f"特征 '{fn}' 未通过治理检查（{len(failures)} 项失败）：\n"
            + "\n".join(f"  - {f}" for f in failures)
        )

    result = {
        "feature_name":          fn,
        "passed":                passed,
        "failures":              failures,
        "feedback":              feedback,
        "ic":                    validation_result.get("ic"),
        "t_stat":                validation_result.get("t_stat"),
        "zero_ratio":            validation_result.get("zero_ratio"),
        "direction_consistency": validation_result.get("direction_consistency"),
        "score_dist":            validation_result.get("score_dist", {}),
        "zero_by_sector":        validation_result.get("zero_by_sector", {}),
        "zero_by_year":          validation_result.get("zero_by_year", {}),
    }

    print(f"[GovernanceAgent] {fn}: {'PASS' if passed else 'FAIL'}")
    if not passed:
        for f in failures:
            print(f"  {f}")

    return result
