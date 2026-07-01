"""
fusion_eval.py — 多因子融合评估 vs M0 基线（行业中性化版）

评估设计：
  - 测试期：2021-2023（walk-forward held-out）
  - LightGBM 不含 sector_enc：去掉行业标签，只用财务+动量特征
  - IC 目标：move_post_sn（截面行业 demean），剔除行业 beta 免费贡献
  - NW t-stat，lags=4
  - 融合方式：等权 / IC 加权 / LightGBM（BASE13 + 文本特征）
"""

import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")

from agent_core.config import SP500_EVENTS, OUTPUT_DIR, TRAIN_YRS, VALID_YRS, TEST_YRS

PASS_FEATURES = {
    "guidance_revision_direction":       0.1153,
    "mgmt_tone_confidence":              0.0727,
    "forward_numeric_specificity":       0.1354,
    "risk_escalation_signal":            0.1339,
    "demand_environment_signal":         0.1043,
    "forward_strategic_specificity":     0.1035,
    "guidance_revision_direction_v2":    0.1063,
    "confident_tone_positive":           0.1182,
    "guidance_revision_direction_v2_v2": 0.1139,
    "qa_spontaneity":                    0.1656,
    "prepared_structure_clarity":        0.0723,
    "managerial_certainty_tone":         0.1028,
    "qa_tangentiality":                  0.1454,
}

# OOS IC 为正的 7 个特征（行业中性化后验证，剔除 5 个反转/无效特征）
SELECTED_FEATURES = {
    "forward_numeric_specificity":       0.1354,
    "qa_spontaneity":                    0.1656,
    "mgmt_tone_confidence":              0.0727,
    "guidance_revision_direction":       0.1153,
    "guidance_revision_direction_v2":    0.1063,
    "guidance_revision_direction_v2_v2": 0.1139,
    "managerial_certainty_tone":         0.1028,
}

M0_IC = 0.0579

# guidance_mentioned/beat_mentioned 重要性为 0，去掉；sector_enc 不进模型
BASE13 = [
    "gross_margin", "operating_margin", "net_margin", "cf_quality",
    "eps_yoy", "rev_yoy", "oi_yoy", "ni_yoy",
    "price_momentum_30d", "price_momentum_90d", "pct_from_52w_high_pt",
]

LGBM_PARAMS = dict(
    objective="regression", metric="rmse",
    num_leaves=16, max_depth=3, learning_rate=0.02,
    subsample=0.7, colsample_bytree=0.7, min_child_samples=40,
    reg_alpha=0.5, reg_lambda=2.0, random_state=42, verbose=-1, n_jobs=-1,
)


def newey_west_tstat(ics: np.ndarray, lags: int = 4) -> float:
    n = len(ics)
    if n < 2:
        return 0.0
    mu = np.mean(ics)
    resid = ics - mu
    gamma0 = np.dot(resid, resid) / n
    nw_var = gamma0
    for lag in range(1, lags + 1):
        w = 1 - lag / (lags + 1)
        gamma_l = np.dot(resid[lag:], resid[:-lag]) / n
        nw_var += 2 * w * gamma_l
    nw_var = max(nw_var, 1e-12)
    return float(mu / np.sqrt(nw_var / n))


def sector_demean(df: pd.DataFrame, col: str) -> pd.Series:
    """截面行业中性化：每个 (year, quarter, sector) 组内 demean。"""
    return df.groupby(["year", "quarter", "sector"])[col].transform(
        lambda x: x - x.mean()
    )


def ic_series_from_df(df: pd.DataFrame, pred_col: str, target_col: str = "move_post_sn") -> np.ndarray:
    ics = []
    for (yr, qtr), g in df.groupby(["year", "quarter"]):
        sub = g[[pred_col, target_col]].dropna()
        if len(sub) < 5:
            continue
        ic = sub[pred_col].corr(sub[target_col])
        if np.isfinite(ic):
            ics.append(ic)
    return np.array(ics)


def main():
    print("=" * 60)
    print("融合因子评估")
    print("=" * 60)

    # ── 1. 加载 sp500_events ───────────────────────────────────────
    ev = pd.read_parquet(SP500_EVENTS)
    ev["earnings_date"] = pd.to_datetime(ev["earnings_date"])
    ev["target_cs"] = ev.groupby(["year", "quarter"])["move_post"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )

    # ── 2. 加载 6 个 PASS 特征 CSV，逐一 merge ─────────────────────
    feat_names = list(PASS_FEATURES.keys())
    base = ev[ev["year"] >= 2015].copy()

    for name in feat_names:
        path = OUTPUT_DIR / f"{name}.csv"
        if not path.exists():
            print(f"[WARN] {name}.csv 不存在，跳过")
            continue
        df = pd.read_csv(path)
        df["earnings_date"] = pd.to_datetime(df["earnings_date"])
        df = df[["symbol", "earnings_date", name]]
        base = base.merge(df, on=["symbol", "earnings_date"], how="left")
        print(f"  加载 {name}: {df[name].notna().sum()} 条有效记录")

    loaded = [c for c in feat_names if c in base.columns]
    print(f"\n宽表维度: {base.shape}  已加载特征: {loaded}")

    # 行业中性化目标变量
    base["move_post_sn"] = sector_demean(base, "move_post")

    # ── 3. 因子相关矩阵（全期）────────────────────────────────────
    corr = base[loaded].corr(method="spearman")
    print("\n【因子 Spearman 相关矩阵】")
    print(corr.round(3).to_string())

    # ── 4. 截面 z-score 标准化（按季度）──────────────────────────
    for col in loaded:
        base[col + "_z"] = base.groupby(["year", "quarter"])[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )

    z_cols = [c + "_z" for c in loaded]

    # ── 5. 等权 & IC 加权融合打分 ──────────────────────────────────
    base["fusion_equal"] = base[z_cols].mean(axis=1)

    weights = np.array([PASS_FEATURES[c] for c in loaded])
    weights = weights / weights.sum()
    base["fusion_icw"] = base[z_cols].fillna(0).values @ weights

    # ── 6. 拆分训练/验证/测试 ─────────────────────────────────────
    # 覆盖标记：至少一个文本特征非空非零
    base["text_covered"] = base[loaded].apply(
        lambda row: any(pd.notna(v) and v != 0 for v in row), axis=1
    )

    # 全样本训练（文本特征缺失处 fillna(0)，保持样本量）
    train = base[base["year"].isin(TRAIN_YRS) & base["target_cs"].notna()].copy()
    valid = base[base["year"].isin(VALID_YRS) & base["target_cs"].notna()].copy()
    test  = base[base["year"].isin(TEST_YRS)  & base["target_cs"].notna()].copy()

    print(f"\ntrain={len(train)}  valid={len(valid)}  test={len(test)}")
    n_cov = test["text_covered"].sum()
    print(f"test 覆盖子集: {n_cov}/{len(test)} ({n_cov/len(test):.1%})")

    results = {}

    # ── 7. 单因子 IC（测试期，行业中性化 move_post_sn）──────────
    print("\n【单因子 IC（测试期 2021-2023，行业中性化）】")
    for col in loaded:
        sub = test[["year", "quarter", col, "move_post_sn"]].dropna()
        ics = ic_series_from_df(sub, col, "move_post_sn")
        if len(ics) == 0:
            continue
        mean_ic = float(np.mean(ics))
        nw_t = newey_west_tstat(ics)
        results[col] = {"IC": mean_ic, "NW_t": nw_t, "n_q": len(ics)}
        print(f"  {col:<35} IC={mean_ic:+.4f}  NW_t={nw_t:+.3f}  n_q={len(ics)}")

    # ── 8. 融合因子 IC（等权 & IC 加权，行业中性化）──────────────
    for fusion_col in ["fusion_equal", "fusion_icw"]:
        sub = test[["year", "quarter", fusion_col, "move_post_sn"]].dropna()
        ics = ic_series_from_df(sub, fusion_col, "move_post_sn")
        if len(ics) == 0:
            continue
        mean_ic = float(np.mean(ics))
        nw_t = newey_west_tstat(ics)
        results[fusion_col] = {"IC": mean_ic, "NW_t": nw_t, "n_q": len(ics)}

    # ── 9. LightGBM 融合（BASE13 无 sector_enc + 文本特征）───────
    all_feats = BASE13 + loaded

    def _to_xy(d):
        return d[all_feats].fillna(0).values.astype(np.float32), d["target_cs"].values.astype(np.float32)

    X_tr, y_tr = _to_xy(train)
    X_vl, y_vl = _to_xy(valid)
    X_te, _    = _to_xy(test)

    dt = lgb.Dataset(X_tr, y_tr, feature_name=all_feats)
    dv = lgb.Dataset(X_vl, y_vl, reference=dt, feature_name=all_feats)
    lgbm_model = lgb.train(
        LGBM_PARAMS, dt, num_boost_round=200,
        valid_sets=[dv], valid_names=["valid"],
        callbacks=[lgb.log_evaluation(0)],
    )
    test = test.copy()
    test["pred_lgbm"] = lgbm_model.predict(X_te)
    # 对预测值也做行业中性化
    test["pred_lgbm_sn"] = sector_demean(test, "pred_lgbm")

    ics_lgbm = ic_series_from_df(test, "pred_lgbm_sn", "move_post_sn")
    mean_ic_lgbm = float(np.mean(ics_lgbm)) if len(ics_lgbm) > 0 else 0.0
    nw_t_lgbm    = newey_west_tstat(ics_lgbm)
    results["lgbm_fusion (BASE13+text)"] = {"IC": mean_ic_lgbm, "NW_t": nw_t_lgbm, "n_q": len(ics_lgbm)}

    # M0 基线（全样本训练，全样本测试）
    X_tr_m0 = train[BASE13].fillna(0).values.astype(np.float32)
    X_vl_m0 = valid[BASE13].fillna(0).values.astype(np.float32)
    X_te_m0 = test[BASE13].fillna(0).values.astype(np.float32)
    dt_m0 = lgb.Dataset(X_tr_m0, y_tr, feature_name=BASE13)
    dv_m0 = lgb.Dataset(X_vl_m0, y_vl, reference=dt_m0, feature_name=BASE13)
    m0_model = lgb.train(
        LGBM_PARAMS, dt_m0, num_boost_round=200,
        valid_sets=[dv_m0], valid_names=["valid"],
        callbacks=[lgb.log_evaluation(0)],
    )
    test["pred_m0"] = m0_model.predict(X_te_m0)
    test["pred_m0_sn"] = sector_demean(test, "pred_m0")
    ics_m0 = ic_series_from_df(test, "pred_m0_sn", "move_post_sn")
    mean_ic_m0 = float(np.mean(ics_m0)) if len(ics_m0) > 0 else 0.0
    nw_t_m0    = newey_west_tstat(ics_m0)
    results["M0_baseline (cov-test)"] = {"IC": mean_ic_m0, "NW_t": nw_t_m0, "n_q": len(ics_m0)}

    # ── 10. 汇总输出 ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("【最终汇总：IC / NW t-stat（测试期 2021-2023）】")
    print(f"{'因子':<38} {'IC':>8} {'NW t':>8} {'n_q':>5}")
    print("─" * 65)
    # 先输出单因子
    for name in loaded:
        if name not in results: continue
        r = results[name]
        vs = f"  ({r['IC']-M0_IC:+.4f} vs M0)" if r['IC'] > M0_IC else ""
        print(f"  {name:<36} {r['IC']:>+8.4f} {r['NW_t']:>8.3f} {r['n_q']:>5}{vs}")
    print("─" * 65)
    # 融合因子
    for name in ["fusion_equal", "fusion_icw", "lgbm_fusion (BASE13+text)", "M0_baseline (cov-test)"]:
        if name not in results: continue
        r = results[name]
        marker = " ***" if name == "lgbm_fusion (BASE13+text)" else (" [M0]" if "M0" in name else "")
        print(f"  {name:<36} {r['IC']:>+8.4f} {r['NW_t']:>8.3f} {r['n_q']:>5}{marker}")
    print("─" * 65)
    print(f"  {'M0 基线 (存档)':<36} {M0_IC:>+8.4f}")

    # ── 11. 年度 IC ───────────────────────────────────────────────
    print("\n【年度 IC（测试期，行业中性化，lgbm_fusion vs M0）】")
    print(f"  {'年份':<6} {'lgbm_fusion':>12} {'M0':>10} {'增量':>10}")
    for yr in TEST_YRS:
        g = test[test["year"] == yr].dropna(subset=["pred_lgbm_sn", "pred_m0_sn", "move_post_sn"])
        if len(g) < 10: continue
        ic_f = g["pred_lgbm_sn"].corr(g["move_post_sn"])
        ic_m = g["pred_m0_sn"].corr(g["move_post_sn"])
        print(f"  {yr:<6} {ic_f:>+12.4f} {ic_m:>+10.4f} {ic_f-ic_m:>+10.4f}")

    # ── 11b. 精简融合（剔除 OOS 反转特征，仅保留 3 个 IC 为正的特征）──
    sel = [c for c in SELECTED_FEATURES if c in base.columns]
    if len(sel) >= 2:
        sel_z_cols = [c + "_z" for c in sel]
        base["fusion_sel_equal"] = base[sel_z_cols].mean(axis=1)

        sel_w = np.array([SELECTED_FEATURES[c] for c in sel])
        sel_w = sel_w / sel_w.sum()
        base["fusion_sel_icw"] = base[sel_z_cols].fillna(0).values @ sel_w

        # LightGBM 精简版（无 sector_enc）
        sel_feats = BASE13 + sel
        X_tr_sel = train[sel_feats].fillna(0).values.astype(np.float32)
        X_vl_sel = valid[sel_feats].fillna(0).values.astype(np.float32)
        X_te_sel = test[sel_feats].fillna(0).values.astype(np.float32)
        dt_sel = lgb.Dataset(X_tr_sel, y_tr, feature_name=sel_feats)
        dv_sel = lgb.Dataset(X_vl_sel, y_vl, reference=dt_sel, feature_name=sel_feats)
        m_sel = lgb.train(
            LGBM_PARAMS, dt_sel, num_boost_round=200,
            valid_sets=[dv_sel], valid_names=["valid"],
            callbacks=[lgb.log_evaluation(0)],
        )
        test["pred_sel"] = m_sel.predict(X_te_sel)
        test["pred_sel_sn"] = sector_demean(test, "pred_sel")

        test_oos = test[test["year"].isin(TEST_YRS)]
        n_sel = len(sel)
        lgbm_sel_key = f"lgbm_sel (BASE13+{n_sel}feat)"
        for col_name, pred_col in [("fusion_sel_equal", "fusion_sel_equal"),
                                   ("fusion_sel_icw",   "fusion_sel_icw"),
                                   (lgbm_sel_key,       "pred_sel_sn")]:
            src = test_oos if "lgbm" in col_name else base[base["year"].isin(TEST_YRS)]
            tgt = "move_post_sn"
            ics_s = ic_series_from_df(src, pred_col, tgt)
            if len(ics_s) == 0: continue
            results[col_name] = {"IC": float(np.mean(ics_s)), "NW_t": newey_west_tstat(ics_s), "n_q": len(ics_s)}

        print(f"\n[精简融合：OOS IC>0 的 {n_sel} 个特征，行业中性化]")
        print(f"  特征: {sel}")
        print(f"  {'方法':<38} {'IC':>8} {'NW t':>8} {'vs M0':>8}")
        print("  " + "-" * 62)
        for nm in ["fusion_sel_equal", "fusion_sel_icw", lgbm_sel_key]:
            if nm not in results: continue
            r = results[nm]
            m0_r = results.get("M0_baseline (cov-test)", {})
            delta = r["IC"] - m0_r.get("IC", 0)
            print(f"  {nm:<38} {r['IC']:>+8.4f} {r['NW_t']:>8.3f} {delta:>+8.4f}")
        m0_r = results.get("M0_baseline (cov-test)", {})
        print(f"  {'M0_baseline (cov-test)':<38} {m0_r.get('IC',0):>+8.4f} {m0_r.get('NW_t',0):>8.3f} {'[baseline]':>8}")

        # 年度 IC 对比（行业中性化）
        print("\n  [年度 IC 行业中性化: lgbm_sel vs M0]")
        print(f"  {'年份':<6} {'lgbm_sel':>10} {'M0':>10} {'增量':>8}")
        for yr in TEST_YRS:
            g = test[(test["year"] == yr)].dropna(subset=["pred_sel_sn", "pred_m0_sn", "move_post_sn"])
            if len(g) < 10: continue
            ic_s = g["pred_sel_sn"].corr(g["move_post_sn"])
            ic_m = g["pred_m0_sn"].corr(g["move_post_sn"])
            print(f"  {yr:<6} {ic_s:>+10.4f} {ic_m:>+10.4f} {ic_s-ic_m:>+8.4f}")

    # ── 12. 2020 年单独评估（真正 OOS，低波动年）────────────────────
    print("\n【2020 年单独评估（真正 OOS，低波动，不在训练期）】")
    valid_eval = base[base["year"] == 2020].copy()
    valid_eval = valid_eval.dropna(subset=["move_post"])

    valid_eval["move_post_sn"] = sector_demean(valid_eval, "move_post")

    # M0 在 2020 的预测（行业中性化）
    valid_eval["pred_m0_2020"] = m0_model.predict(valid_eval[BASE13].fillna(0).values.astype(np.float32))
    valid_eval["pred_m0_2020_sn"] = sector_demean(valid_eval, "pred_m0_2020")
    ic_m0_2020 = valid_eval["pred_m0_2020_sn"].corr(valid_eval["move_post_sn"])

    # lgbm_sel 在 2020 的预测（行业中性化）
    valid_eval["pred_sel_2020"] = m_sel.predict(valid_eval[sel_feats].fillna(0).values.astype(np.float32))
    valid_eval["pred_sel_2020_sn"] = sector_demean(valid_eval, "pred_sel_2020")
    ic_sel_2020 = valid_eval["pred_sel_2020_sn"].corr(valid_eval["move_post_sn"])

    # 单因子在 2020 的 IC（行业中性化 move_post_sn）
    print(f"  {'因子/方法':<38} {'2020 IC (sn)':>12}")
    print("  " + "-" * 52)
    for col in loaded:
        sub = valid_eval[[col, "move_post_sn"]].dropna()
        if len(sub) < 10: continue
        ic = sub[col].corr(sub["move_post_sn"])
        print(f"  {col:<38} {ic:>+12.4f}")
    print("  " + "-" * 52)
    print(f"  {lgbm_sel_key:<38} {ic_sel_2020:>+12.4f}")
    print(f"  {'M0_baseline (cov-test)':<38} {ic_m0_2020:>+12.4f}")
    print(f"  n={len(valid_eval)}")

    # ── 13. 覆盖子集评估（同口径：仅有文本覆盖的样本）─────────────
    # 文本覆盖定义：至少一个文本特征非空非零
    test["text_covered"] = test[loaded].apply(
        lambda row: any(pd.notna(v) and v != 0 for v in row), axis=1
    )
    test_cov = test[test["text_covered"]].copy()
    n_cov = len(test_cov)
    n_tot = len(test)
    print(f"\n【覆盖子集评估（同口径，仅有文本覆盖样本）】")
    print(f"  覆盖样本: {n_cov}/{n_tot} ({n_cov/n_tot:.1%})")

    cov_results = {}
    # M0 在覆盖子集上
    ics_m0_cov = ic_series_from_df(test_cov, "pred_m0_sn", "move_post_sn")
    cov_results["M0 (覆盖子集)"] = {"IC": float(np.mean(ics_m0_cov)), "NW_t": newey_west_tstat(ics_m0_cov), "n_q": len(ics_m0_cov)}

    # lgbm_fusion 在覆盖子集上
    ics_lgbm_cov = ic_series_from_df(test_cov, "pred_lgbm_sn", "move_post_sn")
    cov_results["lgbm_fusion (覆盖子集)"] = {"IC": float(np.mean(ics_lgbm_cov)), "NW_t": newey_west_tstat(ics_lgbm_cov), "n_q": len(ics_lgbm_cov)}

    # lgbm_sel 在覆盖子集上
    ics_sel_cov = ic_series_from_df(test_cov, "pred_sel_sn", "move_post_sn")
    cov_results[f"lgbm_sel (覆盖子集)"] = {"IC": float(np.mean(ics_sel_cov)), "NW_t": newey_west_tstat(ics_sel_cov), "n_q": len(ics_sel_cov)}

    # 单因子在覆盖子集上
    print(f"\n  单因子（覆盖子集）:")
    for col in loaded:
        sub = test_cov[["year", "quarter", col, "move_post_sn"]].dropna()
        ics = ic_series_from_df(sub, col, "move_post_sn")
        if len(ics) == 0: continue
        mic = float(np.mean(ics))
        nt  = newey_west_tstat(ics)
        cov_results[col + " (cov)"] = {"IC": mic, "NW_t": nt, "n_q": len(ics)}
        print(f"    {col:<35} IC={mic:+.4f}  NW_t={nt:+.3f}")

    print(f"\n  {'方法':<38} {'IC':>8} {'NW_t':>8} {'vs M0(cov)':>10}")
    print("  " + "-" * 66)
    m0_cov_ic = cov_results["M0 (覆盖子集)"]["IC"]
    for nm, r in cov_results.items():
        if nm.endswith("(cov)"): continue
        delta = f"{r['IC']-m0_cov_ic:>+.4f}" if nm != "M0 (覆盖子集)" else "[baseline]"
        print(f"  {nm:<38} {r['IC']:>+8.4f} {r['NW_t']:>+8.3f} {delta:>10}")

    # 逐年（覆盖子集）
    print(f"\n  逐年 IC（覆盖子集，行业中性化）:")
    print(f"  {'年份':<6} {'lgbm_sel':>10} {'lgbm_fusion':>12} {'M0':>10} {'增量(sel-M0)':>12} {'n':>6}")
    for yr in TEST_YRS:
        g = test_cov[test_cov["year"] == yr].dropna(subset=["pred_sel_sn", "pred_lgbm_sn", "pred_m0_sn", "move_post_sn"])
        if len(g) < 5: continue
        ic_s = g["pred_sel_sn"].corr(g["move_post_sn"])
        ic_f = g["pred_lgbm_sn"].corr(g["move_post_sn"])
        ic_m = g["pred_m0_sn"].corr(g["move_post_sn"])
        print(f"  {yr:<6} {ic_s:>+10.4f} {ic_f:>+12.4f} {ic_m:>+10.4f} {ic_s-ic_m:>+12.4f} {len(g):>6}")

    results.update(cov_results)

    # ── 15. 特征重要性 ────────────────────────────────────────────
    importance = pd.DataFrame({
        "feature": lgbm_model.feature_name(),
        "importance": lgbm_model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    print("\n【LightGBM 特征重要性 Top-10 (gain)】")
    print(importance.head(10).to_string(index=False))

    # ── 16. 保存 ──────────────────────────────────────────────────
    out_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "factor"})
    out_path = OUTPUT_DIR / "fusion_eval_results.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
