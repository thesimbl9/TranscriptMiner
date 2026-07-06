"""
extraction_agent.py — Phase 1: 特征提取

给定一个 feature_spec，执行：
  1. 加载 FAISS 索引 + metadata
  2. 用 retrieval_query 检索 Top-K 相关 chunk
  3. 按 condition_scope 过滤（sector / section_type / speaker_role）
  4. 批量拼 prompt → 调 LLM API 打分（batch_size 个 episode 一次调用）
  5. 返回 (symbol, earnings_date, feature_value) DataFrame

feature_spec 格式：
{
  "feature_name":            "tech_guidance_revision",
  "definition":              "管理层本季度对营收 guidance 的调整方向和力度",
  "extraction_instruction":  "阅读以下财报电话会议片段，判断管理层是否上调/维持/下调 guidance...",
  "retrieval_query":         "revenue guidance raised target next quarter growth",
  "expected_ic_direction":   "+",
  "condition_scope": {
    "section_type": ["prepared", "qa"],   # 可选：只检索 prepared / qa / press_release
    "speaker_role": ["mgmt"],             # 可选：只看 mgmt 发言
    "sector": null                        # 可选：字符串或 null（不限）
  },
  "top_k": 15,                            # 检索返回的 chunk 数，默认 15
  "score_range": [-2, 2]                  # LLM 输出分值范围，默认 [-2, 2]
}
"""

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import faiss
import httpx
import numpy as np
import pandas as pd
import torch
from openai import OpenAI
from transformers import AutoModel, AutoTokenizer

os.environ["HF_HUB_OFFLINE"]       = "1"
os.environ["TRANSFORMERS_OFFLINE"]  = "1"

# ── 全局配置（统一管理路径 & API key）──────────────────────────────────────────
from agent_core.config import (
    API_KEY as _API_KEY, MODEL as _MODEL, BASE_URL as _BASE_URL,
    PROJECT, FULLPROJECT, MODEL_PATH, STORE_DIR,
    FAISS_PATH, META_PATH, SP500_EVENTS,
)

# ── 单例缓存 ──────────────────────────────────────────────────────────────────
_meta       = None
_tokenizer  = None
_model      = None
_device     = None
_sector_map = None
_gpu_vecs   = None   # (947164, 1024) float16 tensor on GPU
_client     = None   # OpenAI client 单例，全程复用同一连接池


def _get_client(api_key: str | None = None, force_new: bool = False) -> OpenAI:
    global _client
    if force_new or _client is None:
        key = api_key or _API_KEY
        if not key:
            raise RuntimeError(f"API key not set in {FULLPROJECT / '.env'}")
        _client = OpenAI(
            api_key=key,
            base_url=_BASE_URL,
            timeout=httpx.Timeout(180.0, connect=10.0, read=120.0),
        )
    return _client


def _load_meta():
    global _meta
    if _meta is None:
        _meta = pd.read_parquet(META_PATH)
        _meta["earnings_date"] = pd.to_datetime(_meta["earnings_date"])
    return _meta


def _load_gpu_vecs():
    """从 FAISS 索引读取全量向量，转成 GPU float16 tensor（约 1.9GB 显存）。"""
    global _gpu_vecs
    if _gpu_vecs is None:
        if not FAISS_PATH.exists():
            raise FileNotFoundError(f"FAISS 索引不存在: {FAISS_PATH}")
        print("[Retrieval] 加载向量矩阵到 GPU...", flush=True)
        idx = faiss.read_index(str(FAISS_PATH))
        n   = idx.ntotal
        d   = idx.d
        buf = np.zeros((n, d), dtype=np.float32)
        idx.reconstruct_n(0, n, buf)
        _gpu_vecs = torch.from_numpy(buf).half().to("cuda")
        print(f"[Retrieval] GPU 向量矩阵就绪: {n}×{d}  显存≈{_gpu_vecs.nbytes/1024**3:.1f}GB", flush=True)
    return _gpu_vecs


def _load_model():
    global _tokenizer, _model, _device
    if _model is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
        _model = AutoModel.from_pretrained(str(MODEL_PATH), local_files_only=True)
        _model = _model.to(_device).eval()
    return _tokenizer, _model, _device


def _load_sector_map():
    global _sector_map
    if _sector_map is None:
        ev = pd.read_parquet(SP500_EVENTS, columns=["symbol", "sector"])
        _sector_map = ev.drop_duplicates("symbol").set_index("symbol")["sector"].to_dict()
    return _sector_map


def _encode_query(query: str) -> torch.Tensor:
    """返回 L2 归一化的 float16 GPU tensor (1, 1024)。"""
    tok, model, device = _load_model()
    with torch.no_grad():
        enc = tok([query], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        out = model(**enc)
        vec = torch.nn.functional.normalize(out.last_hidden_state[:, 0, :], p=2, dim=1)
    return vec.half()


# ── 检索（GPU matmul） ────────────────────────────────────────────────────────

def retrieve(
    query: str,
    top_k: int = 30,
    condition_scope: dict | None = None,
) -> pd.DataFrame:
    """
    用 GPU torch.matmul 计算余弦相似度，取 Top-K chunk。
    condition_scope 在检索后过滤（先多取余量，再过滤取 top_k）。
    """
    meta     = _load_meta()
    gpu_vecs = _load_gpu_vecs()
    q_vec    = _encode_query(query)

    # condition_scope 过滤会缩减结果集，必须预先多取足够余量。
    # 有 sector 过滤时科技股占 SP500 约 20%，需要 5× 余量；
    # 无 sector 过滤时 3× 余量够用。
    # global_top_k=3000 场景：有 sector 时实际取 15000，无 sector 取 9000——
    # GPU matmul 已经算出全量 scores，topk 只是排序截断，多取几乎无额外代价。
    has_sector = bool(condition_scope and condition_scope.get("sector"))
    multiplier = 5 if has_sector else 3
    fetch_k    = min(top_k * multiplier, len(gpu_vecs))

    with torch.no_grad():
        scores_all = torch.matmul(q_vec, gpu_vecs.T).squeeze(0)
        top_scores, top_idxs = scores_all.topk(fetch_k)

    idxs   = top_idxs.cpu().numpy()
    scores = top_scores.cpu().float().numpy()

    rows = meta.iloc[idxs].copy()
    rows["score"] = scores

    if condition_scope:
        st = condition_scope.get("section_type")
        sr = condition_scope.get("speaker_role")
        sc = condition_scope.get("sector")
        if st:
            rows = rows[rows["section_type"].isin(st)]
        if sr:
            rows = rows[rows["speaker_role"].isin(sr)]
        if sc:
            sector_map = _load_sector_map()
            rows = rows[rows["symbol"].map(sector_map) == sc]

    return rows.head(top_k).reset_index(drop=True)


# ── LLM 批量打分 ──────────────────────────────────────────────────────────────

_SCORE_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _parse_score(text: str, score_range: list[int]) -> float | None:
    nums = _SCORE_RE.findall(text.strip())
    if not nums:
        return None
    val = float(nums[0])
    lo, hi = score_range
    return float(np.clip(val, lo, hi))


def _build_batch_prompt(
    feature_spec: dict,
    batch: list[tuple[str, str, pd.DataFrame]],  # [(symbol, date_str, chunks), ...]
) -> str:
    """
    把多个 episode 拼成一个 prompt，要求 LLM 逐行输出分数。
    输出格式严格为：
      1: <score>
      2: <score>
      ...
    """
    definition             = feature_spec["definition"]
    extraction_instruction = feature_spec["extraction_instruction"]
    score_range            = feature_spec.get("score_range", [-2, 2])
    lo, hi                 = score_range

    episode_blocks = []
    for idx, (sym, date_str, chunks) in enumerate(batch, 1):
        excerpts = []
        for _, row in chunks.iterrows():
            tag = f"[{row['section_type']}/{row['speaker_role']}]"
            excerpts.append(f"  {tag} {row['text'][:300]}")
        excerpts_text = "\n".join(excerpts)
        episode_blocks.append(f"### Episode {idx}: {sym} {date_str}\n{excerpts_text}")

    episodes_text = "\n\n".join(episode_blocks)

    return f"""你是一名量化研究员，正在批量分析财报电话会议记录以提取量化信号。

## 特征定义
{definition}

## 提取指引
{extraction_instruction}

## 待评分的财报片段（共 {len(batch)} 个）
{episodes_text}

## 输出要求
对每个 Episode 输出一个整数分数，范围 [{lo}, {hi}]：
- {hi} = 极强正向信号（明确、强烈）
- 1 = 弱正向信号（有迹象但不强）
- 0 = 片段与本特征完全无关（慎用）
- -1 = 弱负向信号（有迹象但不强）
- {lo} = 极强负向信号（明确、强烈）

【重要】只要片段中有任何与本特征相关的内容，都应给出非零分（±1 或 ±2）。
仅当片段与本特征完全无关、没有任何可判断的线索时，才输出 0。
不确定强弱时，给 ±1 而非 0。

严格按以下格式逐行输出，不要有其他文字：
1: <score>
2: <score>
...{len(batch)}: <score>"""


_BATCH_LINE_RE = re.compile(r"^\s*(\d+)\s*:\s*([-+]?\d+(?:\.\d+)?)\s*$")


def _parse_batch_scores(
    text: str,
    batch_size: int,
    score_range: list[int],
) -> list[float | None]:
    """
    从 LLM 批量输出中解析每个 episode 的分数。
    未匹配到的 episode 返回 None。
    """
    lo, hi = score_range
    scores: dict[int, float] = {}
    for line in text.strip().splitlines():
        m = _BATCH_LINE_RE.match(line)
        if m:
            ep_idx = int(m.group(1))
            val    = float(np.clip(float(m.group(2)), lo, hi))
            scores[ep_idx] = val
    return [scores.get(i) for i in range(1, batch_size + 1)]


def score_batch_episodes(
    feature_spec: dict,
    batch: list[tuple[str, str, pd.DataFrame]],  # [(symbol, date_str, chunks), ...]
    api_key: str | None = None,
    max_retries: int = 3,
    _debug_sink: list | None = None,  # 若非 None，把 (prompt, raw, parsed) 追加进去
) -> list[float | None]:
    """
    一次 API 调用对 batch 内所有 episode 打分。
    返回与 batch 等长的分数列表，失败位置为 None。
    """
    # 过滤掉 chunks 为空的 episode（直接给 None，不放进 prompt）
    non_empty = [(i, item) for i, item in enumerate(batch) if not item[2].empty]
    if not non_empty:
        return [None] * len(batch)

    active_batch = [item for _, item in non_empty]
    prompt       = _build_batch_prompt(feature_spec, active_batch)
    score_range  = feature_spec.get("score_range", [-2, 2])
    client       = _get_client(api_key)

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=len(active_batch) * 8 + 32,
                temperature=0.0,
            )
            raw    = resp.choices[0].message.content.strip()
            parsed = _parse_batch_scores(raw, len(active_batch), score_range)

            if _debug_sink is not None:
                _debug_sink.append({
                    "prompt":  prompt,
                    "raw_llm": raw,
                    "parsed":  parsed,
                    "episodes": [(sym, date_str) for sym, date_str, _ in active_batch],
                })

            # 把结果映射回原始 batch 位置
            result = [None] * len(batch)
            for rank, (orig_idx, _) in enumerate(non_empty):
                result[orig_idx] = parsed[rank]
            return result

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                syms = [item[0] for item in active_batch]
                print(f"  [WARN] batch {syms[:3]}... LLM 失败: {e}")

    return [None] * len(batch)


# ── 主入口：全局检索 + 批量打分 ──────────────────────────────────────────────

def extract_feature_global(
    feature_spec: dict,
    api_key: str | None = None,
    output_path: str | Path | None = None,
    symbols: list[str] | None = None,
    years: list[int] | None = None,
    global_top_k: int = 3000,
    per_episode_top_k: int = 5,
    sample_n: int | None = None,
    sample_seed: int = 42,
    batch_size: int = 50,
    max_workers: int = 4,
    debug_n: int = 3,
) -> pd.DataFrame:
    """
    全局检索 + 批量并发打分。

    batch_size:        每次 API 调用处理的 episode 数（默认 50，约 7500 tokens/call；DeepSeek V4 Flash 64K context 无压力）
    per_episode_top_k: 每个 episode 送入 prompt 的 chunk 数（默认 5，每条截 300 字）
    max_workers:       并发线程数（默认 100；DeepSeek V4 Flash 官方并发上限 2500，SiliconFlow 实测留余量）
    sample_n:          快速筛选时的采样 episode 数（None=全量）
    debug_n:           调试模式：记录前 N 个 batch 的 prompt/LLM原始输出/解析结果到
                       agent_output/debug_<feature>.txt（0=关闭）
    """
    feature_name    = feature_spec["feature_name"]
    condition_scope = feature_spec.get("condition_scope", {})
    retrieval_query = feature_spec["retrieval_query"]

    _get_client(api_key, force_new=True)  # 每次提取重建 client，避免跨迭代连接池耗尽

    print(f"[ExtractionAgent] 全局检索模式: {feature_name}")
    print(f"[ExtractionAgent] query: {retrieval_query}  global_top_k={global_top_k}")

    # ── Step 1: 一次性全局检索 ───────────────────────────────────────────────
    all_chunks = retrieve(retrieval_query, top_k=global_top_k, condition_scope=condition_scope)

    if symbols:
        all_chunks = all_chunks[all_chunks["symbol"].isin(symbols)]
    if years:
        all_chunks = all_chunks[all_chunks["earnings_date"].dt.year.isin(years)]

    # ── Step 2: 按 episode 分组 ───────────────────────────────────────────────
    grouped       = all_chunks.groupby(["symbol", "earnings_date"])
    episodes_full = list(grouped.groups.keys())

    # 年份覆盖检查：测试期(2021-2023)覆盖率低于 30% 时发出 WARN
    ep_years = pd.Series([pd.Timestamp(d).year for _, d in episodes_full])
    test_ep_count  = (ep_years >= 2021).sum()
    total_ep_count = len(episodes_full)
    test_ep_ratio  = test_ep_count / total_ep_count if total_ep_count > 0 else 0.0
    expected_ratio = 3 / 9  # 测试期3年/全量9年 ≈ 33%
    print(f"[ExtractionAgent] 年份覆盖: 测试期(2021-2023) {test_ep_count}/{total_ep_count} "
          f"episodes ({test_ep_ratio:.1%}，期望≈{expected_ratio:.0%})", flush=True)
    if test_ep_ratio < 0.10:
        print(f"[ExtractionAgent] WARN: 测试期覆盖率极低({test_ep_ratio:.1%})！"
              f"retrieval_query 语义可能偏向训练期语言风格，"
              f"或 condition_scope 过滤后测试期 chunks 相似度偏低被 top-k 截断。"
              f"尝试增大 global_top_k 或放宽 condition_scope。", flush=True)

    # ── Step 3: 分层采样（按年份均匀抽取） ───────────────────────────────────
    if sample_n is not None and sample_n < len(episodes_full):
        rng   = np.random.default_rng(sample_seed)
        ep_df = pd.DataFrame(episodes_full, columns=["symbol", "earnings_date"])
        ep_df["year"] = pd.to_datetime(ep_df["earnings_date"]).dt.year
        sampled = (
            ep_df.groupby("year", group_keys=False)
            .apply(lambda g: g.sample(
                n=max(1, round(sample_n * len(g) / len(ep_df))),
                random_state=int(rng.integers(0, 9999)),
            ))
        )
        episodes = list(zip(sampled["symbol"], sampled["earnings_date"]))
        print(f"[ExtractionAgent] 分层采样: {len(episodes)}/{len(episodes_full)} episodes  (seed={sample_seed})")
    else:
        episodes = episodes_full

    # ── Step 4: 切分 batch ────────────────────────────────────────────────────
    batches: list[tuple[int, list]] = []   # (batch_start_idx, batch_items)
    for b_start in range(0, len(episodes), batch_size):
        b_eps = episodes[b_start : b_start + batch_size]
        items = []
        for sym, date in b_eps:
            chunks   = grouped.get_group((sym, date)).head(per_episode_top_k)
            date_str = str(date.date()) if hasattr(date, "date") else str(date)
            items.append((sym, date_str, chunks))
        batches.append((b_start, items))

    n_episodes = len(episodes)
    n_batches  = len(batches)
    print(f"[ExtractionAgent] 覆盖 episodes: {n_episodes}  "
          f"batch_size={batch_size}  max_workers={max_workers}  → {n_batches} 次 API 调用", flush=True)

    # 检索结果为空时直接返回空 DataFrame，避免后续 KeyError
    if n_episodes == 0:
        print(f"[ExtractionAgent] 警告: 检索结果为空，请检查 condition_scope 或 retrieval_query", flush=True)
        df_empty = pd.DataFrame(columns=["symbol", "earnings_date", feature_name])
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            df_empty.to_csv(output_path, index=False)
        return df_empty

    # ── Step 5: 并发批量打分 ──────────────────────────────────────────────────
    # results_map: {original_episode_index → score}
    results_map: dict[int, float | None] = {}
    completed = 0

    # debug_sinks: 前 debug_n 个 batch 各自一个 list，收集 (prompt, raw, parsed)
    debug_sinks: dict[int, list] = {}
    if debug_n > 0:
        for b_start, _ in batches[:debug_n]:
            debug_sinks[b_start] = []

    def _score_batch(b_start: int, items: list):
        sink = debug_sinks.get(b_start)  # None 表示不记录
        scores = score_batch_episodes(feature_spec, items, api_key=api_key, _debug_sink=sink)
        return b_start, items, scores

    # 单batch超时：batch_size越大API越慢，按100eps=120s线性外推，上限600s
    _batch_timeout = max(120, min(600, int(batch_size * 1.2)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_score_batch, b_start, items): b_start
                for b_start, items in batches}
        failed_batches = 0
        for fut in as_completed(futs):
            b_start = futs[fut]
            n_in_batch = min(batch_size, n_episodes - b_start)
            try:
                b_start, items, scores = fut.result(timeout=_batch_timeout)
            except Exception as e:
                failed_batches += 1
                print(f"  [WARN] batch b_start={b_start} ({n_in_batch} eps) 异常: {e}", flush=True)
                # 将该 batch 的所有 episode 标记为 None
                for offset in range(n_in_batch):
                    results_map.setdefault(b_start + offset, None)
                continue
            for offset, ((sym, date), val) in enumerate(zip(
                episodes[b_start : b_start + len(items)], scores
            )):
                results_map[b_start + offset] = val
            completed += len(items)
            filled = sum(1 for v in results_map.values() if v is not None)
            batch_idx = b_start // batch_size + 1
            print(f"  [batch {batch_idx}/{n_batches}] {completed}/{n_episodes} eps  filled={filled}  b_start={b_start}", flush=True)
        if failed_batches:
            print(f"  [WARN] {failed_batches}/{n_batches} batches 因异常失败，对应 episode 标记为 None", flush=True)

    # ── Step 6: 写调试文件 ────────────────────────────────────────────────────
    if debug_n > 0 and any(debug_sinks.values()):
        debug_dir = Path(output_path).parent if output_path else Path("agent_output")
        debug_path = debug_dir / f"debug_{feature_name}.txt"
        sep = "=" * 70
        lines = [
            f"DEBUG: {feature_name}",
            f"query: {retrieval_query}",
            f"condition_scope: {condition_scope}",
            f"batch_size={batch_size}  per_episode_top_k={per_episode_top_k}",
            sep,
        ]
        for b_start, records in sorted(debug_sinks.items()):
            for rec in records:
                eps_str = ", ".join(f"{s} {d}" for s, d in rec["episodes"])
                lines += [
                    f"",
                    f"[Batch b_start={b_start}]  episodes: {eps_str}",
                    f"",
                    f"--- PROMPT (送给 LLM) ---",
                    rec["prompt"],
                    f"",
                    f"--- LLM 原始输出 ---",
                    rec["raw_llm"],
                    f"",
                    f"--- 解析结果 ---",
                    str(list(zip([e[0] for e in rec["episodes"]], rec["parsed"]))),
                    sep,
                ]
        debug_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"[ExtractionAgent] 调试文件: {debug_path}")

    # 按原始顺序组装结果
    rows = []
    for i, (sym, date) in enumerate(episodes):
        rows.append({
            "symbol":        sym,
            "earnings_date": date,
            feature_name:    results_map.get(i),
        })

    df = pd.DataFrame(rows)
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])

    null_rate = df[feature_name].isna().mean()
    print(f"[ExtractionAgent] 完成  total={len(df)}  null_rate={null_rate:.1%}")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[ExtractionAgent] 结果已保存: {output_path}")

    return df


# ── 旧接口保留（extract_feature），不再主动使用 ────────────────────────────────

def extract_feature(
    feature_spec: dict,
    api_key: str | None = None,
    output_path: str | Path | None = None,
    symbols: list[str] | None = None,
    years: list[int] | None = None,
) -> pd.DataFrame:
    """兼容旧调用，内部转发到 extract_feature_global。"""
    return extract_feature_global(
        feature_spec=feature_spec,
        api_key=api_key,
        output_path=output_path,
        symbols=symbols,
        years=years,
    )
