"""
hypothesis_agent.py — Phase 1: 假设生成层

职责：
  1. 维护初始种子特征列表（10 个手写假设，覆盖 6 个信号维度）
  2. 种子用完后，基于 Theory-First 多簇 RAG + 历史反馈，用 LLM 生成下一轮假设
  3. 管理已探索特征库，避免重复，控制理论方向多样性

Theory-First 设计：
  - 预定义 6 个理论簇（语气/信息不对称/前瞻信息/主观性/管理层行为/技术分析）
  - 每簇 2 个查询，各取 top-3，合并去重后得 6-10 条来自不同论文的理论片段
  - LLM 先阅读理论 → 从中选一个推导特征定义，而非先想特征再找支撑
  - 记录已使用的理论簇，轮换覆盖，保证多样性

feature_spec 格式（标准）：
{
  "feature_name":            str,   # 唯一标识，snake_case
  "definition":              str,   # 特征的量化定义
  "theory_basis": {                 # 学术依据（Theory RAG 驱动）
    "source":      str,             # 论文名称 + 页码
    "excerpt":     str,             # 论文关键原句（英文）
    "implication": str,             # 该理论对预测方向的含义（中文）
  },
  "extraction_instruction":  str,   # LLM 提取时的具体指引
  "retrieval_query":         str,   # 用于向量检索的英文短语
  "expected_ic_direction":   "+"|"-",
  "condition_scope": {
    "section_type": list[str] | null,
    "speaker_role": list[str] | null,
    "sector":       str | null,
  },
  "top_k":             int,         # 检索 chunk 数，默认 15
  "score_range":       [int, int],  # LLM 打分范围，默认 [-2, 2]
  "_theory_cluster":   str,         # 本次使用的理论簇名称（审计用）
  "_theory_rag_refs":  list[dict],  # RAG 检索命中的原始论文列表（审计用）
}
"""

import json
import os
import re
from pathlib import Path

import faiss
import httpx
import numpy as np
import pandas as pd
import torch
from openai import OpenAI
from transformers import AutoModel, AutoTokenizer

# ── 全局配置（统一管理路径 & API key）──────────────────────────────────────────
from agent_core.config import (
    API_KEY as _API_KEY, MODEL as _MODEL, BASE_URL as _BASE_URL,
    PROJECT, MODEL_PATH, FULLPROJECT,
    THEORY_INDEX as THEORY_INDEX_PATH, THEORY_META as THEORY_META_PATH,
)

# ── Theory RAG 单例 ───────────────────────────────────────────────────────────
_theory_index = None
_theory_meta  = None
_tok          = None
_embed_model  = None
_device       = None


def _load_theory_index():
    global _theory_index, _theory_meta
    if _theory_index is None:
        if not THEORY_INDEX_PATH.exists():
            return None, None
        _theory_index = faiss.read_index(str(THEORY_INDEX_PATH))
        _theory_meta  = pd.read_parquet(THEORY_META_PATH)
    return _theory_index, _theory_meta


def _load_embed_model():
    global _tok, _embed_model, _device
    if _embed_model is None:
        _device      = "cuda" if torch.cuda.is_available() else "cpu"
        _tok         = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
        _embed_model = AutoModel.from_pretrained(str(MODEL_PATH), local_files_only=True).to(_device).eval()
    return _tok, _embed_model, _device


def retrieve_theory(query: str, top_k: int = 5) -> list[dict]:
    """从 theory_index 检索最相关的论文段落，返回 {paper_title, page_num, text, score}。"""
    index, meta = _load_theory_index()
    if index is None:
        return []
    tok, model, device = _load_embed_model()
    with torch.no_grad():
        enc   = tok([query], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        out   = model(**enc)
        vec   = torch.nn.functional.normalize(out.last_hidden_state[:, 0, :], p=2, dim=1)
        q_arr = vec.cpu().float().numpy()
    scores, idxs = index.search(q_arr, top_k)
    results = []
    for idx, score in zip(idxs[0], scores[0]):
        row = meta.iloc[idx]
        results.append({
            "paper_title": row["paper_title"],
            "page_num":    int(row["page_num"]),
            "text":        row["text"],
            "score":       round(float(score), 4),
        })
    return results


# ── 6 个理论簇：每簇 2 个查询，覆盖 20 篇论文的核心主题 ─────────────────────
# 每个簇对应一个独立的财报信号理论方向，轮换使用保证特征多样性
THEORY_CLUSTERS: dict[str, list[str]] = {
    "tone_sentiment": [
        "managerial tone optimism language sentiment earnings call stock return prediction",
        "positive negative words tone conference call transcript abnormal return",
    ],
    "information_asymmetry": [
        "information asymmetry disclosure transparency analyst forecast revision",
        "earnings surprise announcement abnormal return information content",
    ],
    "forward_guidance": [
        "management earnings guidance forecast specificity quantitative numeric future outlook",
        "forward looking statements revenue EPS guidance accuracy reliability signal",
    ],
    "qa_subjectivity": [
        "question answer session analyst earnings call evasiveness uncertainty hedge language",
        "subjectivity qualitative disclosure Q&A analyst interaction managerial response",
    ],
    "managerial_behavior": [
        "managerial overconfidence attribution self-serving bias tangent concealment",
        "CEO CFO language identity same company cross-period signal consistency",
    ],
    "alpha_discovery": [
        "LLM alpha factor discovery earnings transcript financial signal machine learning",
        "genetic algorithm self-evolving agent factor mining earnings call feature extraction",
    ],
}


# ── 种子假设：10个手写初始特征 ───────────────────────────────────────────────
SEED_FEATURES: list[dict] = [
    {
        "feature_name": "guidance_revision_direction",
        "definition": "管理层本季度对下季度/全年营收或EPS guidance的调整方向：上调为正，维持为0，下调为负",
        "extraction_instruction": (
            "阅读以下财报电话会议片段，判断管理层是否明确上调、维持或下调了未来的业绩指引（guidance）。"
            "重点关注具体数字变化、'raised'/'increased'/'lowered'/'withdrawn'等关键词。"
            "输出：2=明确强力上调, 1=小幅上调, 0=维持/无明确指引, -1=小幅下调, -2=明确强力下调。"
        ),
        "retrieval_query": "revenue guidance raised target next quarter growth outlook increased",
        "expected_ic_direction": "+",
        "condition_scope": {"section_type": ["prepared", "qa"], "speaker_role": ["mgmt"], "sector": None},
        "top_k": 15,
        "score_range": [-2, 2],
    },
    {
        "feature_name": "beat_quality_signal",
        "definition": "本季度实际业绩超预期的质量：不仅是beat/miss，还考虑超出幅度和管理层对beat的归因",
        "extraction_instruction": (
            "阅读以下片段，评估本季度业绩超出/不及预期的质量。"
            "高质量beat=超出幅度大且来自核心业务增长（非一次性项目）。"
            "低质量beat=仅靠削减成本/一次性收益。miss=不及预期。"
            "输出：2=高质量大幅beat, 1=小幅beat, 0=符合预期, -1=小幅miss, -2=大幅miss。"
        ),
        "retrieval_query": "exceeded expectations beat estimates earnings per share revenue above consensus",
        "expected_ic_direction": "+",
        "condition_scope": {"section_type": ["prepared"], "speaker_role": ["mgmt"], "sector": None},
        "top_k": 12,
        "score_range": [-2, 2],
    },
    {
        "feature_name": "mgmt_tone_confidence",
        "definition": "管理层在 prepared remarks 中的整体语气信心度：措辞是否积极、确定、前瞻",
        "extraction_instruction": (
            "阅读管理层的prepared remarks，评估整体语气的信心程度。"
            "高信心=使用'strong'/'confident'/'momentum'等积极措辞，提供具体数字，语气确定。"
            "低信心=频繁使用'uncertainty'/'headwind'/'challenging'，回避具体预测，措辞模糊。"
            "输出：2=非常积极自信, 1=偏积极, 0=中性, -1=偏谨慎, -2=非常悲观谨慎。"
        ),
        "retrieval_query": "strong momentum confident outlook demand growth execution excellent performance",
        "expected_ic_direction": "+",
        "condition_scope": {"section_type": ["prepared"], "speaker_role": ["mgmt"], "sector": None},
        "top_k": 20,
        "score_range": [-2, 2],
    },
    {
        "feature_name": "analyst_sentiment_gap",
        "definition": "Q&A 环节分析师提问的情绪 vs 管理层回答的情绪差异：分析师悲观但管理层乐观=正信号",
        "extraction_instruction": (
            "阅读Q&A环节的问答对话，分别评估分析师提问的情绪（负面关切/正面认可）"
            "和管理层回答的情绪（防御性/主动乐观）。"
            "计算 管理层情绪 - 分析师情绪 的差值。"
            "输出：2=分析师悲观但管理层明显更乐观（正向差异大），0=情绪一致，-2=管理层比分析师还悲观。"
        ),
        "retrieval_query": "analyst question concern risk management response confident positive outlook",
        "expected_ic_direction": "+",
        "condition_scope": {"section_type": ["qa"], "speaker_role": None, "sector": None},
        "top_k": 20,
        "score_range": [-2, 2],
    },
    {
        "feature_name": "forward_numeric_specificity",
        "definition": "管理层在财报中提供具体未来数字预测的程度：提供精确数字=高，只给定性描述=低",
        "extraction_instruction": (
            "阅读以下片段，评估管理层提供具体未来数值预测的程度。"
            "高分=给出具体营收/EPS/毛利率等数字目标，且目标上调或超出历史水平。"
            "低分=只有定性描述（'我们预计增长'），或拒绝提供指引，或数字明显低于预期。"
            "输出：2=提供详细具体数字且积极, 1=有数字但一般, 0=定性描述, -1=数字偏弱, -2=明确负面数字。"
        ),
        "retrieval_query": "expect revenue million earnings per share guidance range full year target specific",
        "expected_ic_direction": "+",
        "condition_scope": {"section_type": ["prepared", "qa"], "speaker_role": ["mgmt"], "sector": None},
        "top_k": 15,
        "score_range": [-2, 2],
    },
    {
        "feature_name": "risk_escalation_signal",
        "definition": "管理层提及新增风险因素的程度：本季度新出现或明显加剧的风险",
        "extraction_instruction": (
            "阅读以下片段，识别管理层提及的风险因素，特别是本季度新增或明显加剧的风险。"
            "常见风险：供应链、宏观环境、竞争加剧、成本压力、需求疲软、监管风险。"
            "输出：-2=多项严重新风险被明确提及, -1=有新风险但程度有限, 0=无明显新风险, "
            "1=过去的风险明显缓解, 2=风险大幅改善/消除。"
        ),
        "retrieval_query": "headwind risk uncertainty supply chain challenge pressure uncertain macro environment",
        "expected_ic_direction": "-",
        "condition_scope": {"section_type": ["prepared", "qa"], "speaker_role": ["mgmt"], "sector": None},
        "top_k": 15,
        "score_range": [-2, 2],
    },
    {
        "feature_name": "press_release_eps_surprise",
        "definition": "从press release中提取EPS实际值与预期的对比信号",
        "extraction_instruction": (
            "阅读以下earnings press release片段，提取EPS相关信息。"
            "关注：'diluted EPS'、'earnings per share'、'beat'、'exceeded'、'above expectations'等。"
            "判断EPS是否超出分析师预期，以及超出幅度。"
            "输出：2=大幅超预期(>10%), 1=小幅超预期, 0=符合预期, -1=小幅不及, -2=大幅不及。"
        ),
        "retrieval_query": "diluted earnings per share exceeded beat consensus estimate quarterly results",
        "expected_ic_direction": "+",
        "condition_scope": {"section_type": ["press_release"], "speaker_role": None, "sector": None},
        "top_k": 10,
        "score_range": [-2, 2],
    },
    {
        "feature_name": "demand_environment_signal",
        "definition": "管理层描述的需求环境：客户需求强劲/疲软，订单情况，pipeline质量",
        "extraction_instruction": (
            "阅读以下片段，评估管理层描述的需求环境。"
            "正面信号：'strong demand'/'robust pipeline'/'backlog growing'/'order book'/'record bookings'。"
            "负面信号：'demand softness'/'customers cautious'/'deal slippage'/'slower ramp'。"
            "输出：2=需求极强，可见度高; 1=需求健康; 0=需求平稳; -1=需求放缓; -2=需求明显疲软。"
        ),
        "retrieval_query": "strong demand robust pipeline backlog growing order book customer momentum",
        "expected_ic_direction": "+",
        "condition_scope": {"section_type": ["prepared", "qa"], "speaker_role": ["mgmt"], "sector": None},
        "top_k": 15,
        "score_range": [-2, 2],
    },
    {
        "feature_name": "tech_guidance_revision",
        "definition": "科技行业特定：管理层对软件/硬件/云业务的具体guidance修订方向",
        "extraction_instruction": (
            "阅读以下科技公司财报片段，评估管理层对核心技术业务（云/软件/AI/半导体）的guidance修订。"
            "关注：订阅收入guidance、ARR增长目标、AI相关收入展望、数据中心需求。"
            "输出：2=明确上调技术核心业务guidance, 1=小幅上调, 0=维持, -1=小幅下调, -2=明确下调。"
        ),
        "retrieval_query": "cloud revenue growth AI demand semiconductor guidance raised software subscription ARR",
        "expected_ic_direction": "+",
        "condition_scope": {"section_type": ["prepared", "qa"], "speaker_role": ["mgmt"], "sector": "Information Technology"},
        "top_k": 15,
        "score_range": [-2, 2],
    },
    {
        "feature_name": "qa_evasiveness",
        "definition": "Q&A环节管理层回避分析师问题的程度：直接回答=低分，回避/转移=高负分",
        "extraction_instruction": (
            "阅读Q&A片段，评估管理层回答分析师问题的直接程度。"
            "直接回答=给出具体数字或明确立场; 回避=使用'we don't guide on that'/'it's too early to say'/"
            "'we'll provide more color later'等措辞转移话题。"
            "输出：2=非常直接透明，主动提供超预期信息; 0=正常回答; -2=明显回避关键问题，信息不透明。"
        ),
        "retrieval_query": "analyst question we don't guide decline comment provide color too early say",
        "expected_ic_direction": "-",
        "condition_scope": {"section_type": ["qa"], "speaker_role": None, "sector": None},
        "top_k": 20,
        "score_range": [-2, 2],
    },
]


class HypothesisAgent:
    """
    管理特征假设的生成和迭代。

    用法：
        agent = HypothesisAgent(api_key="sk-...")
        spec = agent.next_feature()           # 获取下一个待测特征
        agent.record_result(spec, gov_result) # 记录治理结果
        spec = agent.next_feature()           # 自动生成改进假设（种子用完后）
    """

    def __init__(self, api_key: str | None = None, history_path: Path | None = None):
        self.api_key = api_key or _API_KEY
        if not self.api_key:
            raise RuntimeError(f"SILICONFLOW_API_KEY not set in {FULLPROJECT / '.env'}")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=_BASE_URL,
            timeout=httpx.Timeout(180.0, connect=10.0, read=120.0),
        )
        self.history_path = history_path or Path(__file__).parent.parent / "agent_core" / "feature_history.jsonl"
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

        self._explored: list[dict]  = []
        self._explored_names:  set[str] = set()
        self._repair_queue:    list[dict] = []  # repair_spec 待跑队列（DiagnosisAgent 生成）
        self._load_history()

        # 记录每个理论簇已被使用的次数，用于轮换策略
        self._cluster_usage: dict[str, int] = {k: 0 for k in THEORY_CLUSTERS}
        for r in self._explored:
            used = r["feature_spec"].get("_theory_cluster")
            if used and used in self._cluster_usage:
                self._cluster_usage[used] += 1

        self._seed_queue = [
            s for s in SEED_FEATURES
            if s["feature_name"] not in self._explored_names
        ]

    def _load_history(self) -> None:
        """
        加载 feature_history.jsonl，处理三类记录：
          - 普通 feature 记录 → self._explored
          - _patch_feature 记录 → 合并 diagnosis 到对应 feature 记录
          - _repair_spec 记录 → self._repair_queue（跳过已探索的）
        """
        if not self.history_path.exists():
            return
        records: list[dict] = []
        patches: dict[str, dict] = {}
        repair_specs: list[dict] = []

        with open(self.history_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "_patch_feature" in obj:
                    patches[obj["_patch_feature"]] = obj.get("diagnosis", {})
                elif "_repair_spec" in obj:
                    repair_specs.append(obj["_repair_spec"])
                else:
                    records.append(obj)

        # 合并 diagnosis patch
        for rec in records:
            fn = rec.get("feature_spec", {}).get("feature_name")
            if fn and fn in patches:
                rec["governance_result"]["diagnosis"] = patches[fn]

        self._explored      = records
        self._explored_names = {r["feature_spec"]["feature_name"] for r in records}

        # repair_spec 入队（跳过已探索的）
        for spec in repair_specs:
            fn = spec.get("feature_name", "")
            if fn and fn not in self._explored_names:
                self._repair_queue.append(spec)
                self._explored_names.add(fn)  # 防止重复入队

        if self._repair_queue:
            print(f"[HypothesisAgent] 从 history 加载 {len(self._repair_queue)} 个待修复特征")

    def next_feature(self, episodic_hints: str = "") -> dict:
        """
        返回下一个待测的 feature_spec，优先级：
          1. 种子队列（手写初始假设）
          2. DiagnosisAgent 生成的 repair_spec（_repair_queue，从 history 加载）
          3. Theory-First LLM 生成全新假设

        Args:
            episodic_hints: EpisodicMemory 检索到的跨会话历史教训，
                           由 Harness PLANNING handler 注入。

        修复决策完全由 DiagnosisAgent 持有，本层只取队列中的 spec。
        """
        # 存储 episodic_hints 供 _generate_new_feature 使用
        self._episodic_hints = episodic_hints

        if self._seed_queue:
            spec = self._seed_queue.pop(0)
            print(f"[HypothesisAgent] 使用种子特征: {spec['feature_name']}")
            return spec

        if self._repair_queue:
            spec = self._repair_queue.pop(0)
            print(f"[HypothesisAgent] 使用修复特征: {spec['feature_name']} (修复自: {spec.get('_repair_of', '?')})")
            return spec

        print("[HypothesisAgent] 种子用完，Theory-First LLM 生成新假设...")
        return self._generate_new_feature()

    def record_result(self, feature_spec: dict, governance_result: dict):
        """记录一个特征的测试结果到 history。"""
        record = {"feature_spec": feature_spec, "governance_result": governance_result}
        self._explored.append(record)
        self._explored_names.add(feature_spec["feature_name"])
        cluster = feature_spec.get("_theory_cluster")
        if cluster and cluster in self._cluster_usage:
            self._cluster_usage[cluster] += 1
        with open(self.history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[HypothesisAgent] 记录结果: {feature_spec['feature_name']} "
              f"{'PASS' if governance_result['passed'] else 'FAIL'}")

    # ── Theory-First 多簇 RAG ─────────────────────────────────────────────────

    def _select_cluster(self) -> str:
        """选择使用次数最少的理论簇（轮换策略），平衡各方向的覆盖。"""
        return min(self._cluster_usage, key=lambda k: self._cluster_usage[k])

    def _retrieve_theory_multi(self, cluster_name: str) -> tuple[list[dict], list[dict]]:
        """
        对选中的理论簇执行双查询检索，合并去重后返回 6-8 条高质量理论片段。
        同时返回 refs（审计用）。
        去重规则：同一论文同一页只保留一条（取 score 更高的）。
        """
        queries = THEORY_CLUSTERS[cluster_name]
        seen:    dict[tuple, dict] = {}  # (paper_title, page_num) → chunk

        for q in queries:
            chunks = retrieve_theory(q, top_k=4)
            for c in chunks:
                key = (c["paper_title"], c["page_num"])
                if key not in seen or c["score"] > seen[key]["score"]:
                    seen[key] = c

        # 按 score 降序，最多取 8 条
        merged = sorted(seen.values(), key=lambda x: -x["score"])[:8]
        refs   = [{"paper": c["paper_title"], "page": c["page_num"], "score": c["score"]}
                  for c in merged]
        return merged, refs

    # ── 失败模式诊断 ──────────────────────────────────────────────────────────

    @staticmethod
    def _diagnose_failure(fn: str, spec: dict, gov: dict) -> str:
        """
        自动识别 6 种失败模式，输出结构化诊断供 LLM 参考。

        实际遇到过的失败模式（来自 ecagent v4.3 + CM v1.5 实验经验）：

        [A] Instruction 歧义：LLM 不知道如何打分，大量输出 0
            症状：zero_ratio>50%，score_dist 集中在 "0"
            ecagent 案例：risk_shift 定义为"delta"，但 LLM 不清楚"delta"边界，
                          对所有已知风险的重复提及都打 0，导致几乎无信号

        [B] 分值粒度失配：打分分布双极化，但 IC 很低
            症状：score_dist 中 ±2 合计>65%，IC<0.04
            ecagent 案例：guidance_direction 在 [-1,1] 区间内区分 raised/maintained/lowered
                          本就只有三档，[-2,2] 中间两档几乎无人打，分布空洞

        [C] 信号本身无效：打分合理，但与收益无关
            症状：score_dist 均匀，zero_ratio<30%，IC≈0，t<1.5
            ecagent 案例：analyst_persistence（分析师重复追问频次）—— 追问次数
                          与股价完全无相关性，纯粹是行为计数而非信息量指标

        [D] 跨行业方向不一致：因子有行业依赖性
            症状：direction_consistency<60%，部分行业 IC 符号反转
            ecagent 案例：attribution_locus 在周期股（能源/材料）中正向，
                          在科技股中反向（科技公司"内部归因"反而说明过度自信）

        [E] 时序不稳定：早期年份大量零值，后期正常
            症状：zero_by_year 中 2015-2017 零值率>60%，2019+ 正常
            原因：早期 transcript 格式不规范、Q&A 分段不清晰，
                  导致 prepared/qa 分类错误，condition_scope 过滤掉太多

        [F] 跨 section 差值问题（qa_vs_prepared_delta 类）
            症状：score_dist 两端各有聚集（正负分都有但 IC 低），
                  section_type 同时包含 prepared 和 qa
            ecagent 案例（DESIGN_V4.md）：qa_vs_prepared_delta v1 把"情绪对比"当"信息增量"，
                          大量 AAL 案例 prepared 负底色、QA 有具体正面承诺，LLM 打负分但 ret 为正。
                          正确定义：QA 里是否出现了 prepared 未提及的新信息（而非情绪差异）。
            修复：如果要跨 section 打分，必须在 instruction 里明确定义"增量"而非"情绪对比"；
                  或拆成两个独立特征（prepared_tone + qa_tone），差值在 ValidationAgent 层计算。

        [G] 特征定义依赖外部数据（结构性问题）
            症状：IC 极低但 score_dist 合理，仔细看 definition 发现需要"相对买方共识"或"上季度数字"
            ecagent 案例（DESIGN_V4.md）：tone_vs_expectations 定义为"beat vs consensus"，
                          但 transcript 里没有买方共识数字，LLM 只能依赖公司自述，Agilent 案例中
                          45% 的 beat+正面tone 对应负 return，因为 guidance raise 幅度低于隐含预期。
            修复：特征定义只能依赖 transcript 内部信息；如需跨期比较（本季 vs 上季 guidance），
                  需要把上季 guidance 作为 context 注入 prompt，而不是让 LLM 自行判断。

        [H] Prompt 约束过度导致零值膨胀
            症状：IC 合理特征突然 zero_ratio 从 <10% 升至 >40%，发生在 instruction 被"优化"后
            ecagent 案例（DESIGN_V4.md）：v2 给每个特征加了 anchor cases（"0.0=显式重申"），
                          LLM 遇到不确定情况就逃到 0，evasiveness 零值率从 4%→57%。
            修复：instruction 只写一行定义 + "Use full range, do not default to 0"，
                  不加 anchor cases、不加边界条件示例，不加"如果没有相关内容则..."的说明。
        """
        ic  = gov.get("ic", 0)
        t   = gov.get("t_stat", 0)
        zr  = gov.get("zero_ratio", 0)
        dc  = gov.get("direction_consistency", 1.0)
        sd  = gov.get("score_dist", {})
        by_sector = gov.get("zero_by_sector", {})
        by_year   = gov.get("zero_by_year", {})
        section   = spec.get("condition_scope", {}).get("section_type") or []

        # 辅助计算
        top_score_ratio = max(sd.values()) if sd else 0.0
        zero_score_ratio = sd.get("0.0", sd.get("0", 0.0))
        polar_ratio = (sd.get("2.0", sd.get("2", 0.0)) +
                       sd.get("-2.0", sd.get("-2", 0.0)))
        bad_years   = sorted([yr for yr, zr_yr in by_year.items() if zr_yr > 0.60])
        early_bad   = [yr for yr in bad_years if int(yr) <= 2017]
        top_zero_sec = sorted(by_sector.items(), key=lambda x: -x[1])[:2]

        lines = [
            f"特征: {fn}  IC={ic:+.4f}  t={t:+.3f}  "
            f"zero={zr:.0%}  dir_consistency={dc:.0%}",
            f"  score_dist: {sd}",
        ]
        if top_zero_sec:
            lines.append(f"  高零值行业: {top_zero_sec}")
        if bad_years:
            lines.append(f"  高零值年份(>60%): {bad_years}")

        # ── 模式判断（按优先级，取第一个匹配）────────────────────────────────
        if gov.get("coverage_failure"):
            ratio = gov.get("test_coverage_ratio", 0.0)
            lines.append(
                f"  >> 诊断[G1] 测试期覆盖缺失：测试期(2021-2023)特征覆盖率={ratio:.1%}，"
                "提取结果全落在训练期，IC/t=0 是提取缺失而非信号无效。"
            )
            lines.append(
                "    修复：放宽 condition_scope（去掉 sector 或 section_type 限制）；"
                "或检查 retrieval_query 是否只匹配了早期财报的语言风格；"
                "绝对不要因 IC=0 判断该信号方向无效——先解决覆盖问题再评估。"
            )
        elif len(section) >= 2 and "prepared" in section and "qa" in section and abs(ic) < 0.03:
            lines.append(
                "  >> 诊断[F] 跨section差值问题：同时包含 prepared+qa 且 IC 极低。"
                "该特征可能要求 LLM 比较两个 section 的差异，超出单 chunk 打分能力。"
            )
            lines.append(
                "    修复：把特征拆成 prepared_tone 和 qa_tone 两个独立特征，"
                "各自单独打分，差值在 ValidationAgent 层计算。"
            )
        elif zr > 0.50 or (zero_score_ratio > 0.55):
            lines.append(
                "  >> 诊断[A] Instruction 歧义：LLM 大量输出 0，"
                "说明打分阈值过高或 instruction 未说清楚边界条件。"
                f"(zero_ratio={zr:.0%}, P(score=0)={zero_score_ratio:.0%})"
            )
            lines.append(
                "    修复：在 extraction_instruction 中明确写出每个分值对应的"
                "具体词语/句式示例；对于 delta 类特征，明确'什么算新增、什么算已知'；"
                "降低门槛（片段中有相关内容就给 ±1，无需极强信号才给 ±2）。"
            )
        elif polar_ratio > 0.65 and abs(ic) < 0.04:
            lines.append(
                "  >> 诊断[B] 分值粒度失配：打分极度集中在 ±2，但 IC 极低，"
                "说明 LLM 把量表当成二元标签在用，缺乏区分度。"
                f"(P(|score|=2)={polar_ratio:.0%})"
            )
            lines.append(
                "    修复：score_range 改为 [-1, 1]，或在 instruction 中"
                "明确区分强/弱的语言判断标准，避免每次都打极端值。"
            )
        elif early_bad and len(early_bad) >= 2:
            lines.append(
                f"  >> 诊断[E] 时序不稳定：{early_bad} 年零值率超 60%，"
                "说明早期 transcript 格式问题或 condition_scope 过窄导致覆盖缺失。"
            )
            lines.append(
                "    修复：放宽 section_type 或 speaker_role 过滤条件；"
                "或在 retrieval_query 中加入早期 transcript 的常见表达方式；"
                "考虑把 years 过滤到 2018+ 以回避格式不规范的早期数据。"
            )
        elif dc < 0.60:
            bad_sec = [s for s, v in by_sector.items()
                       if (v > 0 and ic < 0) or (v < 0 and ic > 0)][:3]
            lines.append(
                f"  >> 诊断[D] 行业方向不一致：direction_consistency={dc:.0%}，"
                f"反向行业: {bad_sec}。该信号有行业依赖性，不是普适因子。"
            )
            lines.append(
                "    修复：在 condition_scope.sector 限定单一行业（如 Information Technology）；"
                "或设计行业中性的打分标准（用行业内相对值而非绝对语气词）。"
            )
        elif abs(ic) < 0.02 and zr < 0.20 and abs(t) < 0.8:
            # IC 极低但 zero 不多 → 先检查是否依赖外部数据，再判断信号无效
            defn = spec.get("definition", "")
            external_keywords = ["consensus", "预期", "上季度", "市场预期", "分析师预期", "vs prior"]
            has_external = any(kw in defn for kw in external_keywords)
            if has_external:
                lines.append(
                    "  >> 诊断[G] 特征定义依赖外部数据：definition 中包含"
                    f"外部参照（'{[k for k in external_keywords if k in defn]}'），"
                    "但 transcript 内部没有这些信息，LLM 只能猜测，IC 必然极低。"
                )
                lines.append(
                    "    修复：把特征改成纯 transcript 内部可观测的信号；"
                    "如需跨期比较，需要把上季度数字作为 context 注入 prompt。"
                )
            else:
                lines.append(
                    "  >> 诊断[C] 信号本身无效：IC 极低且打分分布合理，"
                    "该语言特征对股价无预测力，放弃该信号方向。"
                )
        elif abs(t) < 1.5 and zr < 0.35 and top_score_ratio < 0.60:
            lines.append(
                "  >> 诊断[C] 信号本身无效：打分分布合理，但 IC 和 t-stat 都低，"
                "说明该语言特征对股价本身没有预测力（不是打分问题）。"
            )
            lines.append(
                "    修复：放弃该信号方向，选择完全不同的理论基础。"
                "注意：行为计数类特征（如分析师追问次数）通常属于此类。"
            )
        else:
            lines.append(
                "  >> 诊断[G-边缘] 各指标接近阈值，"
                "可尝试微调 condition_scope、retrieval_query 或适当增大 top_k。"
                "注意检查 instruction 是否过度约束导致零值膨胀（移除 anchor cases）。"
            )

        return "\n".join(lines)

    # ── 历史反馈归纳 ──────────────────────────────────────────────────────────

    def _summarize_history(self) -> str:
        """
        把历史探索结果归纳成：通过列表 + 各失败特征的结构化诊断。
        只取最近 12 条避免 prompt 过长。
        """
        if not self._explored:
            return "（尚无历史记录，这是第一轮 LLM 生成）"

        recent  = self._explored[-12:]
        passed  = [r for r in recent if r["governance_result"]["passed"]]
        failed  = [r for r in recent if not r["governance_result"]["passed"]]

        lines = []

        if passed:
            pass_strs = [
                f"{r['feature_spec']['feature_name']}"
                f"(IC={r['governance_result'].get('ic',0):+.3f}, "
                f"t={r['governance_result'].get('t_stat',0):+.2f})"
                for r in passed
            ]
            lines.append("[PASS] 已通过的特征（不要重复这些方向）:\n  " + ";  ".join(pass_strs))

        if failed:
            lines.append(f"\n[FAIL] 失败特征诊断（共 {len(failed)} 个，请针对性改进）:")
            for r in failed:
                gov = r["governance_result"]
                fn  = r["feature_spec"]["feature_name"]
                # 优先使用 DiagnosisAgent 的 RAG+LLM 诊断
                diag_obj = gov.get("diagnosis")
                if diag_obj and diag_obj.get("fix"):
                    ic  = gov.get("ic", 0)
                    t   = gov.get("t_stat", 0)
                    zr  = gov.get("zero_ratio", 0)
                    dc  = gov.get("direction_consistency", 0)
                    lines.append(
                        f"特征: {fn}  IC={ic:+.4f}  t={t:+.3f}  "
                        f"zero={zr:.0%}  dir={dc:.0%}\n"
                        f"  >> [DiagnosisAgent] 根因: {diag_obj.get('root_cause', '')}\n"
                        f"  >> 修复: {diag_obj.get('fix', '')}\n"
                        f"  >> 避免: {diag_obj.get('avoid', '')}"
                    )
                else:
                    # 降级到静态规则诊断（无 DiagnosisAgent 输出时）
                    diag = self._diagnose_failure(fn, r["feature_spec"], gov)
                    lines.append(diag)

        return "\n".join(lines) if lines else "（所有历史特征均已记录）"

    # ── 主生成函数 ────────────────────────────────────────────────────────────

    def _generate_new_feature(self) -> dict:
        """Theory-First：先从知识库选理论，再由 LLM 推导特征定义。"""

        # Step 1: 选理论簇（使用次数最少的方向）
        cluster = self._select_cluster()
        print(f"[HypothesisAgent] 选取理论簇: {cluster}  (已用次数: {self._cluster_usage})")

        # Step 2: 双查询检索，合并去重
        theory_chunks, theory_refs = self._retrieve_theory_multi(cluster)
        if theory_chunks:
            theory_lines = []
            for c in theory_chunks:
                ref = f"{c['paper_title'][:55]} (p.{c['page_num']})"
                theory_lines.append(f"[{ref}]\n{c['text'][:350]}")
                print(f"  → {ref}  score={c['score']:.3f}")
            theory_context = "\n\n".join(theory_lines)
        else:
            theory_context = "（theory_index 未找到，请基于财务学理论自行给出 theory_basis）"
            print("[HypothesisAgent] theory_index 未找到，跳过 RAG")

        # Step 3: 历史归纳
        history_digest = self._summarize_history()
        already_tried  = ", ".join(self._explored_names) if self._explored_names else "（无）"

        # Step 3.5: Episodic Memory 跨会话教训（Harness 注入）
        episodic_block = ""
        episodic_hints = getattr(self, '_episodic_hints', '')
        if episodic_hints:
            episodic_block = f"""

## 跨会话历史教训（Episodic Memory — 来自之前所有运行的经验）
{episodic_hints}

请务必在生成新特征时避开上述已记录的错误模式。"""
            print(f"[HypothesisAgent] 注入 EpisodicMemory 提示 ({len(episodic_hints)} chars)")

        # Step 4: Theory-First prompt
        prompt = f"""你是一名量化研究员，正在为财报电话会议信号发现系统设计新的特征假设。

## 本轮理论方向：{cluster}

以下是从学术文献库中检索到的相关理论发现（来自不同论文）：

{theory_context}

---

## 历史探索摘要
{history_digest}{episodic_block}

## 已探索过的特征名称（不要重复）
{already_tried}

---

## 零值率控制（ANTI-ZERO-INFLATION — 强制遵守）

**连续多轮特征因零值率过高而失败，以下规则为最高优先级：**

1. **禁止默认打0**：extraction_instruction 里绝对不要出现 "如无相关内容则输出0" / "default to 0" / "缺少信号时输出0"。改为要求 LLM 从整体语气和上下文推断置信度，always produce a non-zero judgment。
2. **禁止词汇清单匹配**：不要列出特定词汇（如 confident/sure/certain）。改用宽泛语义类别描述（如 "表达确定性" / "回避直接回答" / "积极或消极语气"）。
3. **强制用满分值范围**：extraction_instruction 末尾加："Score must use the full range [-2,2]. Assign +2 for the strongest positive signal, -2 for the strongest negative signal, ±1 for moderate signals. Reserve 0 only for genuinely balanced/neutral text that shows equal evidence in both directions."
4. **放宽 scope**：condition_scope 的 section_type 优先设为 ["prepared","qa"]（全文本），speaker_role 优先设为 null（不限角色），sector 优先设为 null（不限行业）。窄 scope 是高零值的主要原因。
5. **语义判断优于规则匹配**：要求 LLM 判断整体语义性质（tone/sentiment/commitment），而非查找特定词。如 "judge the overall confidence level" 而非 "count confident words"。

---

## 你的任务（严格按以下步骤）

**Step 1**：从上方学术文献中选择一个最具预测潜力的理论发现，记录其来源论文和关键原句。

**Step 2**：基于该理论，推导出一个可量化的财报信号特征：
- definition：该理论预测什么语言行为与股价有关？如何量化为[-2, 2]分？
- extraction_instruction：告诉打分LLM具体看什么语义特征，如何判断强/弱/中性
- retrieval_query：用英文关键词描述这类文本，用于向量检索

**Step 3**：严格对照历史诊断结果调整设计：
- 诊断[A] Instruction歧义 → 重写extraction_instruction，每个分值对应具体语义特征
- 诊断[B] 分值粒度失配 → score_range改为[-1,1]，instruction中区分强/弱判断标准
- 诊断[C] 信号无效 → 必须换理论方向，不要在同类语言特征上继续尝试
- 诊断[D] 行业方向不一致 → sector字段限定单一行业，或重新设计行业中性标准
- 诊断[E] 时序不稳定 → 放宽condition_scope，或years限定2018+
- 诊断[F] 跨section差值 → 把特征拆成两个单独信号（各自打分），而非要求LLM计算差值
- 诊断[G] 外部数据依赖 → 特征只能依赖transcript内部可观测信号；不要用"相对预期""相对上季度"等transcript里没有的参照

请严格输出以下JSON格式（不要有任何其他文字）：
{{
  "feature_name": "snake_case唯一名称（不超过5个单词，不与已探索重复）",
  "definition": "特征的量化定义（中文，1-2句，说明信号来源和方向）",
  "theory_basis": {{
    "source": "论文名称 (页码)",
    "excerpt": "论文中的关键原句（英文原文，≤80字）",
    "implication": "该理论推导出的预测方向及原因（中文，1句）"
  }},
  "extraction_instruction": "提取指引（中文，3-5句：先说看什么语义特征，再说怎么打满[-2,2]全范围，区分强/弱/中性信号）",
  "retrieval_query": "English phrase for semantic vector search (5-10 keywords)",
  "expected_ic_direction": "+" or "-",
  "condition_scope": {{
    "section_type": ["prepared"] or ["qa"] or ["prepared","qa"] or ["press_release"] or null,
    "speaker_role": ["mgmt"] or ["analyst"] or null,
    "sector": "GICS sector name or null"
  }},
  "top_k": 15,
  "score_range": [-2, 2]
}}"""

        import time as _time
        for _attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a quantitative financial research assistant. Output only valid JSON."},
                        {"role": "user",   "content": prompt},
                    ],
                    max_tokens=1000,
                    temperature=0.7,
                )
                break
            except Exception as _e:
                print(f"[HypothesisAgent] LLM 调用失败（attempt {_attempt+1}/3）: {_e}")
                if _attempt < 2:
                    _time.sleep(10)
                else:
                    raise
        raw = resp.choices[0].message.content.strip()

        # 提取 JSON（兼容 ```json``` 包裹 / 裸 JSON）
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        brace = raw.find("{")
        if brace > 0:
            raw = raw[brace:]

        try:
            spec = json.loads(raw.strip())
        except json.JSONDecodeError as e:
            raise ValueError(f"[HypothesisAgent] LLM 输出无法解析为 JSON: {e}\nraw={raw[:400]}")

        # ── 字段校验与修正 ────────────────────────────────────────────────────
        name = spec.get("feature_name", "")
        if not re.match(r'^[a-z][a-z0-9_]+$', name):
            fixed = re.sub(r'[^a-z0-9_]', '_', name.lower()).strip('_')
            print(f"[HypothesisAgent] feature_name 格式修正: '{name}' → '{fixed}'")
            spec["feature_name"] = fixed

        direction = spec.get("expected_ic_direction", "")
        if direction not in ("+", "-"):
            fixed_dir = "+" if str(direction).lower() in ("positive", "up", "1", "+1") else "-"
            print(f"[HypothesisAgent] expected_ic_direction 修正: '{direction}' → '{fixed_dir}'")
            spec["expected_ic_direction"] = fixed_dir

        if spec["feature_name"] in self._explored_names:
            spec["feature_name"] = spec["feature_name"] + "_v2"
            print(f"[HypothesisAgent] 重名，自动添加后缀: {spec['feature_name']}")

        spec.setdefault("top_k", 15)
        spec.setdefault("score_range", [-2, 2])
        spec.setdefault("condition_scope", {"section_type": None, "speaker_role": None, "sector": None})
        spec["_theory_cluster"] = cluster
        spec["_theory_rag_refs"] = theory_refs

        print(f"[HypothesisAgent] 生成新特征: {spec['feature_name']}")
        if spec.get("theory_basis"):
            print(f"  theory_basis: {spec['theory_basis'].get('source', '?')}")
            print(f"  implication:  {spec['theory_basis'].get('implication', '?')}")
        return spec

    def summary(self) -> dict:
        """返回当前探索状态摘要。"""
        passed = [r for r in self._explored if r["governance_result"]["passed"]]
        failed = [r for r in self._explored if not r["governance_result"]["passed"]]
        return {
            "total_explored":   len(self._explored),
            "passed":           len(passed),
            "failed":           len(failed),
            "seeds_remaining":  len(self._seed_queue),
            "cluster_usage":    dict(self._cluster_usage),
            "passed_features":  [r["feature_spec"]["feature_name"] for r in passed],
        }
