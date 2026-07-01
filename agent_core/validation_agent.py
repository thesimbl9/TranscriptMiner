"""
validation_agent.py — Phase 1: 信号验证

输入：feature_values DataFrame（symbol, earnings_date, <feature_name>）
操作：与 sp500_events 合并 → walk-forward IC 评估（与 ecagent v4.3 一致）
输出：ValidationResult dict

评估设计：
  - train: 2015-2019，valid: 2020，test: 2021-2023
  - 截面 IC = Pearson(pred, move_post)，每季度一个 IC 值
  - 报告：mean IC, NW t-stat, per_sector_ic, zero_ratio, season_ic
"""

import warnings
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from agent_core.config import SP500_EVENTS, TRAIN_YRS, VALID_YRS, TEST_YRS

BASE14 = [
    "gross_margin", "operating_margin", "net_margin", "cf_quality",
    "eps_yoy", "rev_yoy", "oi_yoy", "ni_yoy",
    "price_momentum_30d", "price_momentum_90d", "pct_from_52w_high_pt",
    "guidance_mentioned", "beat_mentioned", "sector",
]

LGBM_PARAMS = dict(
    objective="regression", metric="rmse",
    num_leaves=16, max_depth=3, learning_rate=0.02,
    subsample=0.7, colsample_bytree=0.7, min_child_samples=40,
    reg_alpha=0.5, reg_lambda=2.0, random_state=42, verbose=-1, n_jobs=-1,
)


def _newey_west_tstat(ics: np.ndarray, lags: int = 4) -> float:
    """Newey-West 修正 t-stat，lags=4（季度数据标准）。"""
    n = len(ics)
    if n < 2:
        return 0.0
    mu = np.mean(ics)
    resid = ics - mu
    # 方差
    gamma0 = np.dot(resid, resid) / n
    nw_var = gamma0
    for lag in range(1, lags + 1):
        w = 1 - lag / (lags + 1)
        gamma_l = np.dot(resid[lag:], resid[:-lag]) / n
        nw_var += 2 * w * gamma_l
    nw_var = max(nw_var, 1e-12)
    return float(mu / np.sqrt(nw_var / n))


def _ic_series(df: pd.DataFrame, pred_col: str) -> np.ndarray:
    """计算逐季 IC 序列（Pearson，截面归一化目标）。"""
    ics = []
    for (yr, qtr), g in df.groupby(["year", "quarter"]):
        sub = g[[pred_col, "move_post"]].dropna()
        if len(sub) < 5:
            continue
        ic = sub[pred_col].corr(sub["move_post"])
        if np.isfinite(ic):
            ics.append(ic)
    return np.array(ics)


def validate(
    feature_df: pd.DataFrame,
    feature_name: str,
    use_lgbm: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    验证单个特征的预测力。

    Args:
        feature_df:    包含 symbol, earnings_date, <feature_name> 的 DataFrame
        feature_name:  特征列名
        use_lgbm:      True=LightGBM（BASE14 + feature），False=仅用特征直接计算 IC
        verbose:       打印详情

    Returns:
        dict with keys: ic, t_stat, n_quarters, per_sector_ic, zero_ratio,
                        season_ic, pass_governance (bool)
    """
    # ── 加载基础数据 ──────────────────────────────────────────────────────────
    ev = pd.read_parquet(SP500_EVENTS)
    ev["earnings_date"] = pd.to_datetime(ev["earnings_date"])
    ev["target_cs"] = ev.groupby(["year", "quarter"])["move_post"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )

    feature_df = feature_df.copy()
    feature_df["earnings_date"] = pd.to_datetime(feature_df["earnings_date"])

    merged = ev.merge(
        feature_df[["symbol", "earnings_date", feature_name]],
        on=["symbol", "earnings_date"],
        how="inner",
    )
    merged = merged[merged["year"] >= 2015].copy()

    if verbose:
        print(f"[ValidationAgent] 特征: {feature_name}")
        print(f"[ValidationAgent] merge 后行数: {len(merged)}")

    # sector 编码
    sm = {s: i for i, s in enumerate(sorted(merged["sector"].dropna().unique()))}
    inv_sm = {v: k for k, v in sm.items()}
    merged["sector_enc"] = merged["sector"].map(sm).fillna(-1).astype(int)

    # zero_ratio（NaN + 0 视为无信号）
    zero_ratio = (merged[feature_name].isna() | (merged[feature_name] == 0)).mean()

    # winsorize（用 train 统计量）
    train_mask = merged["year"].isin(TRAIN_YRS)
    feat_cols  = BASE14[:-1] + ["sector_enc", feature_name]  # sector_enc 替换 sector
    for col in feat_cols:
        if col in ("sector_enc", "sector"):
            continue
        lo = float(merged.loc[train_mask, col].quantile(0.01))
        hi = float(merged.loc[train_mask, col].quantile(0.99))
        merged[col] = merged[col].clip(lo, hi)

    train = merged[merged["year"].isin(TRAIN_YRS) & merged["target_cs"].notna()].copy()
    valid = merged[merged["year"].isin(VALID_YRS) & merged["target_cs"].notna()].copy()
    test  = merged[merged["year"].isin(TEST_YRS)  & merged["target_cs"].notna()].copy()

    # ── 测试期覆盖率检查（缺失 = 提取阶段没覆盖到测试期，不是信号无效）────────
    test_coverage = (
        merged[merged["year"].isin(TEST_YRS)][feature_name]
        .notna()
        .sum()
    )
    total_test_rows = len(ev[ev["year"].isin(TEST_YRS)])
    test_coverage_ratio = test_coverage / total_test_rows if total_test_rows > 0 else 0.0

    if verbose:
        avg_qtr = test.groupby(["year", "quarter"]).size().mean() if len(test) > 0 else 0
        print(f"[ValidationAgent] train={len(train)}  valid={len(valid)}  test={len(test)}  avg/qtr={avg_qtr:.1f}")
        print(f"[ValidationAgent] 测试期覆盖率: {test_coverage}/{total_test_rows} ({test_coverage_ratio:.1%})")
        if test_coverage_ratio < 0.10:
            print(f"[ValidationAgent] WARN: 测试期覆盖率极低({test_coverage_ratio:.1%})，"
                  f"特征提取未覆盖 2021-2023，结果为提取缺失而非信号无效")

    # 测试期无有效样本 → 直接返回带覆盖诊断的特殊结果，不走 LightGBM
    if len(test) == 0 or test_coverage_ratio < 0.05:
        print(f"[ValidationAgent] FAIL: 测试期样本为空，跳过 LightGBM，返回 coverage_failure")
        return {
            "feature_name":          feature_name,
            "ic":                    0.0,
            "t_stat":                0.0,
            "n_quarters":            0,
            "zero_ratio":            round(float(zero_ratio), 3),
            "per_sector_ic":         {},
            "direction_consistency": 0.0,
            "season_ic":             {},
            "score_dist":            {str(k): round(float(v), 3) for k, v in
                                      merged[feature_name].dropna().value_counts(normalize=True).items()},
            "zero_by_sector":        {},
            "zero_by_year":          {},
            "coverage_failure":      True,   # 标记：失败原因是覆盖缺失，非信号无效
            "test_coverage_ratio":   round(test_coverage_ratio, 4),
        }

    # ── 评估模式 ──────────────────────────────────────────────────────────────
    if use_lgbm:
        all_feats = BASE14[:-1] + ["sector_enc", feature_name]
        cat_idx   = [i for i, f in enumerate(all_feats) if f == "sector_enc"]

        def _to_xy(d):
            return d[all_feats].fillna(0).values.astype(np.float32), d["target_cs"].values.astype(np.float32)

        X_tr, y_tr = _to_xy(train)
        X_vl, y_vl = _to_xy(valid)
        X_te, _    = _to_xy(test)

        dt = lgb.Dataset(X_tr, y_tr, feature_name=all_feats, categorical_feature=cat_idx)
        dv = lgb.Dataset(X_vl, y_vl, reference=dt, feature_name=all_feats, categorical_feature=cat_idx)
        m  = lgb.train(
            LGBM_PARAMS, dt, num_boost_round=200,
            valid_sets=[dv], valid_names=["valid"],
            callbacks=[lgb.log_evaluation(0)],
        )
        test = test.copy()
        test["pred"] = m.predict(X_te)
        pred_col = "pred"
    else:
        # 直接用特征值作为预测（单因子 IC）
        test = test.copy()
        test["pred"] = test[feature_name].fillna(0)
        pred_col = "pred"

    # ── IC 序列 & t-stat ──────────────────────────────────────────────────────
    ics = _ic_series(test, pred_col)
    ic_mean = float(np.mean(ics)) if len(ics) > 0 else 0.0
    t_stat  = _newey_west_tstat(ics)

    # ── 逐行业 IC ─────────────────────────────────────────────────────────────
    per_sector_ic = {}
    test["sector_name"] = test["sector_enc"].map(inv_sm)
    for sec, g in test.groupby("sector_name"):
        sub = g[["pred", "move_post"]].dropna()
        if len(sub) < 5:
            continue
        ic_sec = sub["pred"].corr(sub["move_post"])
        if np.isfinite(ic_sec):
            per_sector_ic[sec] = round(float(ic_sec), 4)

    # 方向一致性（同号比例）
    if per_sector_ic:
        signs = [1 if v > 0 else -1 for v in per_sector_ic.values()]
        dominant = 1 if ic_mean >= 0 else -1
        direction_consistency = sum(1 for s in signs if s == dominant) / len(signs)
    else:
        direction_consistency = 0.0

    # ── 逐季 IC ───────────────────────────────────────────────────────────────
    season_ic = {}
    for (yr, qtr), g in test.groupby(["year", "quarter"]):
        sub = g[["pred", "move_post"]].dropna()
        if len(sub) < 5:
            continue
        ic_q = sub["pred"].corr(sub["move_post"])
        if np.isfinite(ic_q):
            season_ic[f"{yr}Q{qtr}"] = round(float(ic_q), 4)

    # ── 打分分布诊断 ──────────────────────────────────────────────────────────
    score_col = merged[feature_name].dropna()
    score_dist = {}
    if len(score_col) > 0:
        vc = score_col.value_counts(normalize=True).sort_index()
        score_dist = {str(k): round(float(v), 3) for k, v in vc.items()}

    # 零值分布（哪些行业零值多）
    zero_by_sector = {}
    for sec, g in merged.groupby("sector"):
        zr = (g[feature_name].isna() | (g[feature_name] == 0)).mean()
        zero_by_sector[sec] = round(float(zr), 3)

    # 哪些年份零值多
    zero_by_year = {}
    merged["year_"] = pd.to_datetime(merged["earnings_date"]).dt.year
    for yr, g in merged.groupby("year_"):
        zr = (g[feature_name].isna() | (g[feature_name] == 0)).mean()
        zero_by_year[int(yr)] = round(float(zr), 3)

    result = {
        "feature_name":          feature_name,
        "ic":                    round(ic_mean, 4),
        "t_stat":                round(t_stat, 3),
        "n_quarters":            len(ics),
        "zero_ratio":            round(float(zero_ratio), 3),
        "per_sector_ic":         per_sector_ic,
        "direction_consistency": round(direction_consistency, 3),
        "season_ic":             season_ic,
        "score_dist":            score_dist,       # 打分分布（各分值占比）
        "zero_by_sector":        zero_by_sector,   # 各行业零值率
        "zero_by_year":          zero_by_year,     # 各年份零值率
    }

    if verbose:
        print(f"[ValidationAgent] IC={ic_mean:+.4f}  t={t_stat:+.3f}  n={len(ics)}")
        print(f"[ValidationAgent] zero_ratio={zero_ratio:.1%}  direction_consistency={direction_consistency:.0%}")
        print(f"[ValidationAgent] per_sector_ic: {per_sector_ic}")
        print(f"[ValidationAgent] score_dist: {score_dist}")
        # 零值率最高的前3个行业
        top_zero = sorted(zero_by_sector.items(), key=lambda x: -x[1])[:3]
        print(f"[ValidationAgent] zero率最高行业: {top_zero}")

    return result
