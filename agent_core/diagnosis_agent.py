"""
diagnosis_agent.py — Phase 1: 失败诊断层

职责：仅在 GovernanceAgent FAIL 时触发。
  1. 根据失败症状构造 RAG query（症状 → query 映射，不依赖 LLM 自由发挥）
  2. 检索 theory_index（与 HypothesisAgent 共用索引，但 query 语义不同）
  3. LLM + 文献证据 → 结构化诊断 {root_cause, fix, avoid, rag_refs}
  4. 将 diagnosis 写回 feature_history.jsonl（_patch_feature 记录）
  5. 判断是否值得修复（IC 强但仅 G2 失败）→ 若是，生成 repair_spec 并写入 history

与 HypothesisAgent 的 RAG 区别：
  HypothesisAgent：找新方向（"什么语言行为能预测股价"）
  DiagnosisAgent：解释失败（"这类信号为什么没有预测力"）

修复决策边界（DiagnosisAgent 持有，HypothesisAgent 不参与）：
  IC > 0.08 且 |t| > 2.0 且 failures 仅含 G2_zero_ratio → 值得修复
  → 写入 {"_repair_spec": feature_spec_dict} 到 history
  HypothesisAgent._load_history() 读取后放入 _repair_queue，优先于新生成
"""

import json
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI

# ── 全局配置（统一管理路径 & API key）──────────────────────────────────────────
from agent_core.config import (
    API_KEY as _API_KEY, MODEL as _MODEL, BASE_URL as _BASE_URL,
    FULLPROJECT, HISTORY_PATH,
)
_client   = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=_API_KEY,
            base_url=_BASE_URL,
            timeout=httpx.Timeout(60.0, connect=10.0, read=45.0),
        )
    return _client


# ── 症状 → RAG query 映射 ─────────────────────────────────────────────────────
# 规则化映射，不让 LLM 自由构造 query，避免检索方向偏移

def _build_rag_queries(val: dict, failures: list[str]) -> list[str]:
    """
    根据验证数据和治理失败项，构造针对"失败原因"的 RAG 检索 query。
    每种症状对应学术文献中讨论"该类信号失效原因"的子领域。
    """
    queries = []

    # G1: 测试期覆盖缺失（与信号质量无关，是条件过滤问题）
    if val.get("coverage_failure"):
        queries.append(
            "sample selection bias out-of-sample coverage earnings call signal condition scope"
        )
        return queries  # 覆盖缺失与信号理论无关，无需其他 query

    zr = val.get("zero_ratio", 0.0)
    t  = abs(val.get("t_stat", 0.0))
    dc = val.get("direction_consistency", 1.0)
    ic = abs(val.get("ic", 0.0))
    sd = val.get("score_dist", {})

    polar = (sd.get("2.0", sd.get("2", 0.0)) + sd.get("-2.0", sd.get("-2", 0.0)))

    # 零值高 + t 显著 → 覆盖率问题（信号有效但 LLM 打分保守）
    if zr >= 0.45 and t >= 1.5:
        queries.append(
            "sparse coverage LLM scoring conservative zero inflation valid signal earnings call"
        )
    # 零值高 + t 低 → instruction 歧义或信号本身无效
    elif zr >= 0.45:
        queries.append(
            "zero inflation ambiguous instruction LLM scoring threshold earnings transcript signal"
        )

    # t 低、zero 正常、IC 低 → 信号本身无预测力
    if t < 1.5 and zr < 0.45 and ic < 0.04:
        if polar > 0.65:
            queries.append(
                "bimodal score distribution granularity mismatch earnings call LLM rating"
            )
        else:
            queries.append(
                "language feature no predictive power stock return earnings call transcript weak signal"
            )

    # 方向不一致 → 行业异质性
    if dc < 0.60:
        queries.append(
            "sector heterogeneity earnings call tone signal reversal industry specific cross-sectional"
        )

    if not queries:
        queries.append(
            "earnings call transcript signal failure low IC t-stat diagnosis improvement"
        )

    return queries[:2]


def _retrieve_theory_for_diagnosis(queries: list[str]) -> tuple[str, list[dict]]:
    """
    用症状 query 检索 theory_index，返回 (上下文文字, refs列表)。
    检索失败时静默降级，返回空字符串和空列表。
    """
    try:
        from agent_core.hypothesis_agent import retrieve_theory
        seen: dict[tuple, dict] = {}
        for q in queries:
            for c in retrieve_theory(q, top_k=4):
                key = (c["paper_title"], c["page_num"])
                if key not in seen or c["score"] > seen[key]["score"]:
                    seen[key] = c

        chunks = sorted(seen.values(), key=lambda x: -x["score"])[:5]
        if not chunks:
            return "", []

        lines = []
        refs  = []
        for c in chunks:
            lines.append(f"[{c['paper_title'][:60]} p.{c['page_num']}]\n{c['text'][:350]}")
            refs.append({
                "paper": c["paper_title"],
                "page":  c["page_num"],
                "score": c["score"],
            })
        return "\n\n".join(lines), refs

    except Exception as e:
        print(f"[DiagnosisAgent] RAG 检索失败（降级到无文献模式）: {e}")
        return "", []


# ── LLM 诊断生成 ──────────────────────────────────────────────────────────────

def _call_llm_diagnosis(
    feature_name: str,
    feature_spec: dict,
    val: dict,
    failures: list[str],
    theory_ctx: str,
) -> dict:
    """
    调用 LLM 生成结构化诊断。
    返回 {root_cause, fix, avoid}，失败时返回规则生成的 fallback。
    """
    zr  = val.get("zero_ratio", 0)
    t   = val.get("t_stat", 0)
    ic  = val.get("ic", 0)
    dc  = val.get("direction_consistency", 0)
    sd  = val.get("score_dist", {})
    zbs = val.get("zero_by_sector", {})
    zby = val.get("zero_by_year", {})

    top_zero_sec  = sorted(zbs.items(), key=lambda x: -x[1])[:3]
    low_zero_sec  = sorted(zbs.items(), key=lambda x:  x[1])[:3]
    top_zero_year = sorted(zby.items(), key=lambda x: -x[1])[:3]

    # 零值类型分析（规则判断，传给 LLM 作为前置结论）
    zero_type_label = _zero_type(val)
    if zero_type_label == "uniform":
        zero_type_hint = (
            "零值类型：【均匀分布型】各行业零值率差异小，说明该特征信号本身覆盖率低，"
            "零值大多是合法零值（管理层确实未提及相关内容）。"
            "修复方向：不应全局放宽门槛，应考虑换特征定义或放宽 condition_scope。"
        )
    elif zero_type_label == "concentrated":
        high_sec = [s for s, v in zbs.items() if v > 0.80]
        zero_type_hint = (
            f"零值类型：【行业集中型】高零值行业={high_sec}，低零值行业={[s for s,v in low_zero_sec]}。"
            f"说明 instruction 对特定行业语言风格识别力弱，属于逃避零值。"
            f"修复方向：针对高零值行业补充 instruction 示例，不要全局放宽打分门槛。"
        )
    else:
        zero_type_hint = "零值类型：【未知】数据不足，请根据行业分布自行判断。"

    theory_section = f"\n## 相关学术文献（来自 theory_index，按相关度排序）\n{theory_ctx}\n" if theory_ctx else ""

    prompt = f"""你是量化研究员，正在诊断一个财报语言信号特征的失败原因。

## 特征信息
- 名称: {feature_name}
- 定义: {feature_spec.get('definition', 'N/A')}
- extraction_instruction: {feature_spec.get('extraction_instruction', 'N/A')[:300]}
- condition_scope: {feature_spec.get('condition_scope', {})}
{theory_section}
## 实测验证结果
- IC = {ic:+.4f}，t-stat = {t:+.3f}，zero_ratio = {zr:.1%}，direction_consistency = {dc:.0%}
- score_dist = {sd}
- 零值率最高行业（前3）: {top_zero_sec}
- 零值率最低行业（前3）: {low_zero_sec}
- 零值率最高年份（前3）: {top_zero_year}

## 零值类型前置判断（规则分析，请以此为基础给出修复建议）
{zero_type_hint}

## 治理检查失败项
{chr(10).join(f'- {f}' for f in failures)}

## 诊断任务
根据上方数据、零值类型判断和文献，给出针对性修复建议。
注意：零值有两种语义——合法零值（管理层确实未提及）和逃避零值（LLM 对模糊表述保守打0）。
请区分后再给建议，不要一律建议"放宽门槛"。

输出严格按以下格式（每项一行，不要其他文字）：
根因: <1句，说明为什么失败，并指明是合法零值还是逃避零值>
修复: <2-3句，针对零值类型给出具体修改建议>
避免: <1句，指出容易犯的错误方向>"""

    try:
        resp = _get_client().chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=350,
            temperature=0.2,
        )
        raw = resp.choices[0].message.content.strip()
        result = {}
        for line in raw.splitlines():
            if line.startswith("根因:"):
                result["root_cause"] = line[3:].strip()
            elif line.startswith("修复:"):
                result["fix"] = line[3:].strip()
            elif line.startswith("避免:"):
                result["avoid"] = line[3:].strip()
        # 字段补全
        result.setdefault("root_cause", raw[:100])
        result.setdefault("fix", "")
        result.setdefault("avoid", "")
        return result

    except Exception as e:
        print(f"[DiagnosisAgent] LLM 调用失败，使用规则 fallback: {e}")
        return {
            "root_cause": f"LLM诊断失败（{e}）；治理失败项：{'; '.join(failures)}",
            "fix": "参考 HypothesisAgent._diagnose_failure() 规则诊断。",
            "avoid": "",
        }


# ── 修复判断 ─────────────────────────────────────────────────────────────────

# 修复准入门槛（DiagnosisAgent 持有，HypothesisAgent 不参与）
_SALVAGE_IC_MIN  = 0.08
_SALVAGE_T_MIN   = 2.0


def _zero_type(val: dict) -> str:
    """
    判断零值类型：
      'uniform'     — 各行业零值率方差 < 0.02，说明信号本身覆盖率低（合法零值）
      'concentrated'— 某行业零值率显著偏高（逃避零值，instruction 对该行业适配差）
      'unknown'     — zero_by_sector 数据不足，无法判断
    """
    zbs = val.get("zero_by_sector", {})
    if len(zbs) < 3:
        return "unknown"
    import statistics
    vals = list(zbs.values())
    var  = statistics.variance(vals)
    max_z = max(vals)
    min_z = min(vals)
    if var < 0.02:
        return "uniform"
    if max_z > 0.80 and min_z < 0.40:
        return "concentrated"
    return "unknown"


def _is_salvageable(val: dict, gov: dict, explored_names: set[str]) -> bool:
    """
    判断一个 FAIL 特征是否值得生成修复版。
    条件：IC > 0.08 且 |t| > 2.0 且 failures 仅含 G2_zero_ratio
          且 _v2 未探索过 且 零值不是均匀分布型（合法零值不修复）。
    """
    if gov.get("coverage_failure"):
        return False
    ic       = abs(val.get("ic", 0.0))
    t        = abs(val.get("t_stat", 0.0))
    failures = gov.get("failures", [])
    fn       = gov.get("feature_name", "")
    only_g2  = bool(failures) and all("G2_zero_ratio" in f for f in failures)
    v2_name  = fn + "_v2"
    if not (ic > _SALVAGE_IC_MIN and t > _SALVAGE_T_MIN and only_g2 and v2_name not in explored_names):
        return False
    # 均匀型零值 = 合法零值，不应强制降低，跳过修复
    zt = _zero_type(val)
    if zt == "uniform":
        print(f"[DiagnosisAgent] 零值类型=均匀分布（合法零值），不生成 repair_spec")
        return False
    return True


def _generate_repair_spec(
    feature_spec: dict,
    val: dict,
    gov: dict,
    diagnosis: dict,
    client,
    model: str,
) -> dict:
    """
    基于 DiagnosisAgent 诊断生成修复版 feature_spec（_v2）。
    重点修改 extraction_instruction 降低零值率，其余字段保持不变。
    """
    fn      = feature_spec.get("feature_name", "unknown")
    v2_name = fn + "_v2"
    zr = val.get("zero_ratio", 0)
    ic = val.get("ic", 0)
    t  = val.get("t_stat", 0)
    fix_suggestion = diagnosis.get("fix", "")
    avoid          = diagnosis.get("avoid", "")
    root_cause     = diagnosis.get("root_cause", "")

    prompt = f"""你是量化研究员，正在修复一个财报信号特征的 extraction_instruction，目标是降低零值率。

## 原始特征
- 名称: {fn}
- 定义: {feature_spec.get('definition', '')}
- 原 extraction_instruction:
{feature_spec.get('extraction_instruction', '')}
- condition_scope: {feature_spec.get('condition_scope', {})}
- score_range: {feature_spec.get('score_range', [-2, 2])}

## 验证结果
- IC = {ic:+.4f}，t = {t:+.3f}（信号本身有效，值得修复）
- zero_ratio = {zr:.1%}（过高，需要降低到 30% 以下）

## DiagnosisAgent 诊断
- 根因: {root_cause}
- 修复建议: {fix_suggestion}
- 避免: {avoid}

## 你的任务
重写 extraction_instruction，使零值率降低到 30% 以下，同时保留原有信号方向。
要求：
1. 明确列出每个分值对应的具体词语/句式（不要用模糊描述）
2. 强调"只要有相关内容就给 ±1，不确定强弱时给 ±1 而非 0"
3. 不加"如果没有相关内容则输出0"的说明（减少保守倾向）
4. 不改变 definition、condition_scope、retrieval_query

严格输出以下 JSON（不要有其他文字）：
{{
  "feature_name": "{v2_name}",
  "definition": "{feature_spec.get('definition', '')}",
  "theory_basis": {json.dumps(feature_spec.get('theory_basis', {}), ensure_ascii=False)},
  "extraction_instruction": "<重写后的 instruction，3-5句>",
  "retrieval_query": "{feature_spec.get('retrieval_query', '')}",
  "expected_ic_direction": "{feature_spec.get('expected_ic_direction', '+')}",
  "condition_scope": {json.dumps(feature_spec.get('condition_scope', {}), ensure_ascii=False)},
  "top_k": {feature_spec.get('top_k', 15)},
  "score_range": {json.dumps(feature_spec.get('score_range', [-2, 2]))}
}}"""

    import time as _time
    for _attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a quantitative financial research assistant. Output only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=800,
                temperature=0.3,
            )
            break
        except Exception as _e:
            print(f"[DiagnosisAgent] repair_spec LLM 失败（attempt {_attempt+1}/3）: {_e}")
            if _attempt < 2:
                _time.sleep(10)
            else:
                raise
    raw = resp.choices[0].message.content.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    brace = raw.find("{")
    if brace > 0:
        raw = raw[brace:]

    spec = json.loads(raw.strip())
    spec["feature_name"]     = v2_name
    spec["_repair_of"]       = fn
    spec["_theory_cluster"]  = feature_spec.get("_theory_cluster", "repair")
    spec["_theory_rag_refs"] = feature_spec.get("_theory_rag_refs", [])
    spec.setdefault("top_k", feature_spec.get("top_k", 15))
    spec.setdefault("score_range", feature_spec.get("score_range", [-2, 2]))
    spec.setdefault("condition_scope", feature_spec.get("condition_scope", {}))
    return spec


# ── 写回 history ──────────────────────────────────────────────────────────────

def _patch_history(history_path: Path, feature_name: str, diagnosis: dict):
    """追加诊断 patch 记录（_patch_feature）。"""
    patch = {
        "_patch_feature": feature_name,
        "diagnosis": diagnosis,
    }
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(patch, ensure_ascii=False) + "\n")


def _write_repair_spec(history_path: Path, repair_spec: dict):
    """追加修复 spec 记录（_repair_spec），供 HypothesisAgent 读取并优先入队。"""
    record = {"_repair_spec": repair_spec}
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── 主入口 ────────────────────────────────────────────────────────────────────

class DiagnosisAgent:
    """
    失败诊断层。在 GovernanceAgent FAIL 后调用。

    用法：
        diag_agent = DiagnosisAgent(explored_names=agent._explored_names)
        diagnosis  = diag_agent.diagnose(feature_spec, val_result, gov_result)
        # diagnosis + 可能的 repair_spec 已自动写入 feature_history.jsonl

    修复决策由本层持有：IC>0.08 且 |t|>2.0 且 only_G2 → 生成 _repair_spec 写入 history。
    HypothesisAgent 读取 _repair_spec 并优先入队，不参与任何修复判断。
    """

    def __init__(
        self,
        history_path: Path | None = None,
        explored_names: set[str] | None = None,
    ):
        self.history_path   = history_path or str(HISTORY_PATH)
        # 共享 HypothesisAgent 的 _explored_names 引用，用于判断 _v2 是否已存在
        self._explored_names: set[str] = explored_names if explored_names is not None else set()

    def diagnose(
        self,
        feature_spec: dict,
        val_result: dict[str, Any],
        gov_result: dict[str, Any],
    ) -> dict:
        """
        对一个 FAIL 特征生成结构化诊断，并写回 history。
        若判断值得修复，额外生成并写入 _repair_spec。

        Returns:
            diagnosis dict: {root_cause, fix, avoid, rag_refs, queries_used}
        """
        feature_name = feature_spec.get("feature_name", "unknown")
        failures     = gov_result.get("failures", [])

        print(f"[DiagnosisAgent] 开始诊断: {feature_name}")

        # Step 1: 症状 → RAG query
        queries = _build_rag_queries(val_result, failures)
        print(f"[DiagnosisAgent] RAG queries: {queries}")

        # Step 2: 检索 theory_index
        theory_ctx, rag_refs = _retrieve_theory_for_diagnosis(queries)
        if rag_refs:
            print(f"[DiagnosisAgent] 检索到 {len(rag_refs)} 条文献片段")
        else:
            print("[DiagnosisAgent] 无文献检索结果，使用纯 LLM 诊断")

        # Step 3: LLM 生成结构化诊断
        llm_result = _call_llm_diagnosis(
            feature_name, feature_spec, val_result, failures, theory_ctx
        )

        diagnosis = {
            "root_cause":   llm_result.get("root_cause", ""),
            "fix":          llm_result.get("fix", ""),
            "avoid":        llm_result.get("avoid", ""),
            "rag_refs":     rag_refs,
            "queries_used": queries,
        }

        print(f"[DiagnosisAgent] 根因: {diagnosis['root_cause']}")
        print(f"[DiagnosisAgent] 修复: {diagnosis['fix']}")

        # Step 4: 写回 history（diagnosis patch）
        _patch_history(self.history_path, feature_name, diagnosis)
        print(f"[DiagnosisAgent] 诊断已写入 history: {feature_name}")

        # Step 5: 修复判断（DiagnosisAgent 持有，HypothesisAgent 不参与）
        if _is_salvageable(val_result, gov_result, self._explored_names):
            print(f"[DiagnosisAgent] 判定可修复（IC={abs(val_result.get('ic',0)):.3f}, "
                  f"t={abs(val_result.get('t_stat',0)):.2f}, only_G2）→ 生成 repair_spec")
            try:
                from openai import OpenAI
                import httpx
                client = OpenAI(
                    api_key=_API_KEY,
                    base_url=_BASE_URL,
                    timeout=httpx.Timeout(60.0, connect=10.0, read=45.0),
                )
                repair_spec = _generate_repair_spec(
                    feature_spec, val_result, gov_result, diagnosis, client, _MODEL
                )
                _write_repair_spec(self.history_path, repair_spec)
                # 同步更新 explored_names 防止重复生成
                self._explored_names.add(repair_spec["feature_name"])
                print(f"[DiagnosisAgent] repair_spec 已写入 history: {repair_spec['feature_name']}")
            except Exception as e:
                print(f"[DiagnosisAgent] repair_spec 生成失败（跳过）: {e}")

        return diagnosis
