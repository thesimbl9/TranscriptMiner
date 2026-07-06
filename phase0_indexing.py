"""
phase0_indexing.py — Phase 0: Transcript Vectorization & FAISS Indexing

Pipeline:
  1. 读取所有 batch_full_*.json（758 batches，337股 × 多季度）
  2. 按 speaker turn 切分 transcript -> chunks (prepared / qa)
  3. 读取 episodes.parquet，提取 press_release_ex991 -> chunks (press_release)
     边界：只取 337股 × 2015-2023 范围内有 ex991 的记录（~7229条，64%覆盖）
  4. bge-m3 本地离线编码 (batch_size=16, 8GB VRAM)
  5. 写 FAISS IndexFlatIP + metadata.parquet

metadata 字段：chunk_id, symbol, earnings_date, section_type, speaker_role, text
  section_type: "prepared" | "qa" | "press_release"
  speaker_role: "mgmt" | "analyst" | "operator" | "unknown" | "-"（press_release无说话人）

断点续传：已写入 metadata.parquet 的 chunk_id 自动跳过
"""

import json
import os
import re
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

# ── 路径配置（从 agent_core.config 继承，增补 phase0 专用路径）───────────────
from agent_core.config import (
    PROJECT, BATCH_DIR, EPISODES_PATH, SP500_EVENTS,
    STORE_DIR, FAISS_PATH, META_PATH, MODEL_PATH,
)
LOG_PATH = Path(__file__).parent / "logs" / "phase0_indexing.log"

# ── 模型配置 ──────────────────────────────────────────────────────────────────
BATCH_SIZE  = 16        # RTX 5070 Laptop 8GB VRAM
EMBED_DIM   = 1024      # bge-m3 output dim
MAX_CHUNK_TOKENS = 380  # 留20 token给特殊符号

# ── 切分参数 ──────────────────────────────────────────────────────────────────
MIN_CHUNK_CHARS = 100   # 过短的发言（<100字符）合并到下一个
MAX_CHUNK_CHARS = 1600  # 约400 token上限，超出则硬切
DROP_CHUNK_CHARS = 30   # 最终落盘前，低于此长度的 chunk 丢弃

os.environ["HF_HUB_OFFLINE"]      = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ── 工具函数 ──────────────────────────────────────────────────────────────────

def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def detect_section_type(text: str) -> str:
    """根据文本内容判断是 prepared 还是 qa section。"""
    qa_markers = ["question", "analyst", "q&a", "operator:", "please go ahead",
                  "your question", "thank you, operator"]
    lower = text.lower()
    for m in qa_markers:
        if m in lower:
            return "qa"
    return "prepared"


MGMT_TITLES = re.compile(
    r"\b(ceo|cfo|coo|cto|president|chairman|officer|vp|vice president|"
    r"treasurer|secretary|director|managing|chief|svp|evp)\b",
    re.IGNORECASE,
)

# Q&A 里 analyst 的标记前缀，格式为 "Q – Name:" 或 "Q - Name:"
QA_ANALYST_PREFIX = re.compile(r"^Q\s*[–\-]\s*", re.IGNORECASE)

ANALYST_FIRMS = re.compile(
    r"\b(analyst|research|securities|capital|morgan|goldman|jpmorgan|barclays|"
    r"citi|ubs|bofa|bank of america|deutsche|jefferies|wells|cowen|"
    r"nomura|hsbc|credit suisse|evercore|piper|raymond|bernstein|mizuho)\b",
    re.IGNORECASE,
)

def detect_speaker_role(speaker_line: str) -> str:
    """从说话人行判断是 mgmt 还是 analyst。

    transcript 格式：
      - Prepared: "Mike McMullen:" / "Mike McMullen - CEO:"
      - Q&A analyst: "Q – Jack Meehan:" / "Q - Tycho Peterson:"
      - Q&A mgmt:    "Mike McMullen:" (直接名字)
      - Operator:    "Operator:"
    """
    if not speaker_line:
        return "unknown"
    # "Q – name" 前缀 → analyst
    if QA_ANALYST_PREFIX.match(speaker_line):
        return "analyst"
    if "operator" in speaker_line.lower():
        return "operator"
    if ANALYST_FIRMS.search(speaker_line):
        return "analyst"
    if MGMT_TITLES.search(speaker_line):
        return "mgmt"
    return "mgmt"


SPEAKER_TURN_RE = re.compile(r"^([A-Z][A-Za-z\s,\-\.]{2,60}):[ \t]*\n?", re.MULTILINE)


def split_into_chunks(prepared: str, qa: str, symbol: str, earnings_date: str) -> list[dict]:
    """
    把 prepared + qa 按 speaker turn 切分为 chunks。
    每个 chunk 保证不超过 MAX_CHUNK_CHARS 字符，过短的合并到下一段。
    返回 list of dict，字段：symbol, earnings_date, section_type, speaker_role, text
    """
    chunks = []

    def process_section(text: str, default_section: str):
        if not text or not text.strip():
            return

        # 找所有说话人标记
        turns = []
        last_end = 0
        for m in SPEAKER_TURN_RE.finditer(text):
            if m.start() > last_end:
                # 上一段落结束到此处的孤立文本
                turns.append(("", text[last_end:m.start()]))
            turns.append((m.group(1).strip(), ""))
            last_end = m.end()

        # 重新遍历，拿每个说话人的实际内容
        raw_chunks = []
        positions = list(SPEAKER_TURN_RE.finditer(text))
        if not positions:
            # 无说话人标记，整段作为一个 chunk
            raw_chunks.append(("", text.strip()))
        else:
            for i, m in enumerate(positions):
                speaker = m.group(1).strip()
                start   = m.end()
                end     = positions[i + 1].start() if i + 1 < len(positions) else len(text)
                content = text[start:end].strip()
                if content:
                    raw_chunks.append((speaker, content))

        # 长段硬切 + 短段合并
        pending_speaker = ""
        pending_text    = ""
        section_type    = default_section

        for speaker, content in raw_chunks:
            sec = detect_section_type(content) if content else section_type
            role = detect_speaker_role(speaker)

            # 如果 content 超出限制，硬切
            while len(content) > MAX_CHUNK_CHARS:
                cut = content[:MAX_CHUNK_CHARS]
                # 尝试在句子边界切断
                last_period = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
                if last_period > MAX_CHUNK_CHARS // 2:
                    cut = content[:last_period + 1]

                if pending_text and (len(pending_text) + len(cut) > MAX_CHUNK_CHARS):
                    # 先flush pending
                    chunks.append({
                        "symbol":        symbol,
                        "earnings_date": earnings_date,
                        "section_type":  section_type,
                        "speaker_role":  detect_speaker_role(pending_speaker),
                        "text":          pending_text,
                    })
                    pending_text = ""
                    pending_speaker = ""

                chunks.append({
                    "symbol":        symbol,
                    "earnings_date": earnings_date,
                    "section_type":  sec,
                    "speaker_role":  role,
                    "text":          cut.strip(),
                })
                content = content[len(cut):].strip()
                section_type = sec

            if not content:
                continue

            # 短段合并
            if len(content) < MIN_CHUNK_CHARS:
                pending_text    += (" " if pending_text else "") + content
                pending_speaker = pending_speaker or speaker
                section_type    = sec
                continue

            # 正常段：先 flush pending 再开新段
            if pending_text:
                chunks.append({
                    "symbol":        symbol,
                    "earnings_date": earnings_date,
                    "section_type":  section_type,
                    "speaker_role":  detect_speaker_role(pending_speaker),
                    "text":          pending_text,
                })
                pending_text = ""
                pending_speaker = ""

            chunks.append({
                "symbol":        symbol,
                "earnings_date": earnings_date,
                "section_type":  sec,
                "speaker_role":  role,
                "text":          content,
            })
            section_type = sec

        # 最后的 pending
        if pending_text and len(pending_text) >= DROP_CHUNK_CHARS:
            chunks.append({
                "symbol":        symbol,
                "earnings_date": earnings_date,
                "section_type":  section_type,
                "speaker_role":  detect_speaker_role(pending_speaker),
                "text":          pending_text,
            })

    process_section(prepared, "prepared")
    process_section(qa,       "qa")
    return chunks


# ── Press Release 切分 ───────────────────────────────────────────────────────

PR_HEAD_CHARS = 20_000  # 只取前 20K chars（highlights/EPS/guidance 集中在开头）

# 行级噪声规则：只过滤短行（<120 chars）且匹配 SEC 格式头
PR_NOISE_LINE_RE = re.compile(
    r"^(EX-99\.1\s|Exhibit\s+99\.1|Table of Contents$|UNITED STATES SECURITIES|"
    r"Washington,\s*D\.C\.|FORM\s+8-K|CURRENT REPORT|Pursuant to Section|"
    r"Commission File Number|IRS Employer|CHECK THE APPROPRIATE|\(State or other)",
    re.IGNORECASE,
)

def _is_noise_line(line: str) -> bool:
    """单行噪声判断：SEC格式行 / 文件名行 / 纯数字表格行。"""
    s = line.strip()
    if not s:
        return True
    # SEC 格式头（短行）
    if len(s) < 120 and PR_NOISE_LINE_RE.match(s):
        return True
    # 文件名行：含 .htm / .pdf 后缀
    if re.search(r'\.(htm|pdf|htm[l]?)\b', s, re.IGNORECASE) and len(s) < 200:
        return True
    # 纯数字/表格行：字母占比 < 25%
    alpha = sum(1 for c in s if c.isalpha())
    if len(s) > 20 and alpha / len(s) < 0.25:
        return True
    # 财务报表行：以 $ 金额或数字列开头的密集数字行
    if re.match(r'^\s*[\$\d\(\-]', s) and s.count('$') + s.count(',') > 3:
        return True
    return False


def split_press_release(text: str, symbol: str, earnings_date: str) -> list[dict]:
    """
    把 ex991 新闻稿切分为 chunks，section_type="press_release"，speaker_role="-"。

    策略：
    - 只取前 20K chars（highlights / EPS / guidance 集中在开头）
    - 逐行过滤 SEC 噪声行，剩余行按空行重组段落
    - 短段合并，长段句子边界硬切，每 chunk 100-1600 chars
    """
    if not text or not text.strip():
        return []
    text = text[:PR_HEAD_CHARS]

    # 逐行过滤，然后按空行边界重组段落
    lines = text.splitlines()
    clean_lines = [l for l in lines if not _is_noise_line(l)]
    paragraphs = []
    current: list[str] = []
    for line in clean_lines:
        s = line.strip()
        if s:
            current.append(s)
        else:
            if current:
                paragraphs.append(" ".join(current))
                current = []
    if current:
        paragraphs.append(" ".join(current))

    chunks = []
    pending = ""

    def _append(txt: str):
        nonlocal pending
        chunks.append({
            "symbol": symbol, "earnings_date": earnings_date,
            "section_type": "press_release", "speaker_role": "-",
            "text": txt,
        })

    for para in paragraphs:
        if not para:
            continue
        # 长段硬切
        while len(para) > MAX_CHUNK_CHARS:
            cut = para[:MAX_CHUNK_CHARS]
            lp = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
            if lp > MAX_CHUNK_CHARS // 2:
                cut = para[:lp + 1]
            if pending and len(pending) + len(cut) > MAX_CHUNK_CHARS:
                if len(pending) >= DROP_CHUNK_CHARS:
                    _append(pending)
                pending = ""
            if len(cut) >= DROP_CHUNK_CHARS:
                _append(cut.strip())
            para = para[len(cut):].strip()
        if not para:
            continue
        # 短段合并
        if len(para) < MIN_CHUNK_CHARS:
            pending += (" " if pending else "") + para
            continue
        if pending:
            if len(pending) >= DROP_CHUNK_CHARS:
                _append(pending)
            pending = ""
        _append(para)

    if pending and len(pending) >= DROP_CHUNK_CHARS:
        _append(pending)
    return chunks


def load_press_release_chunks() -> list[dict]:
    """
    从 episodes.parquet 提取 press_release_ex991，
    边界严格限定在 337股 × 2015-2023（与 sp500_events inner join 对齐）。
    """
    ep_sp500 = pd.read_parquet(SP500_EVENTS)
    ep_sp500["earnings_date"] = pd.to_datetime(ep_sp500["earnings_date"])

    ep_full = pd.read_parquet(EPISODES_PATH)
    ep_full["earnings_date"] = pd.to_datetime(ep_full["earnings_date"])

    # inner join：只保留两表都有的 (symbol, earnings_date) 对
    merged = ep_sp500[["symbol", "earnings_date"]].merge(
        ep_full[["symbol", "earnings_date", "press_release_ex991"]],
        on=["symbol", "earnings_date"],
        how="inner",
    )

    # 过滤：ex991 非空 + 有实质内容（>500 chars）+ 年份 2015-2023
    merged["year"] = merged["earnings_date"].dt.year
    mask = (
        merged["press_release_ex991"].notna() &
        (merged["press_release_ex991"].str.len() > 500) &
        merged["year"].between(2015, 2023)
    )
    subset = merged[mask].copy()
    subset["earnings_date_str"] = subset["earnings_date"].dt.strftime("%Y-%m-%d")

    all_chunks = []
    for _, row in subset.iterrows():
        chunks = split_press_release(
            str(row["press_release_ex991"]),
            row["symbol"],
            row["earnings_date_str"],
        )
        all_chunks.extend(chunks)

    all_chunks = [c for c in all_chunks if len(c["text"]) >= DROP_CHUNK_CHARS]
    return all_chunks


# ── 主索引函数 ────────────────────────────────────────────────────────────────

def load_all_chunks(batch_files: list[Path]) -> list[dict]:
    """
    合并两路 chunks：
      1. transcript (prepared + qa)：来自 batch_full_*.json
      2. press_release (ex991)：来自 episodes.parquet，限定337股×2015-2023
    transcript 在前，press_release 在后，chunk_id 连续递增。
    """
    # --- 路径1：transcript ---
    transcript_chunks = []
    for bf in batch_files:
        records = json.loads(bf.read_text(encoding="utf-8"))
        for rec in records:
            chunks = split_into_chunks(
                rec.get("prepared_remarks", ""),
                rec.get("qa_section", ""),
                rec["symbol"],
                rec["earnings_date"],
            )
            transcript_chunks.extend(chunks)
    transcript_chunks = [c for c in transcript_chunks if len(c["text"]) >= DROP_CHUNK_CHARS]

    # --- 路径2：press_release ex991 ---
    pr_chunks = load_press_release_chunks()

    return transcript_chunks + pr_chunks


def load_model():
    """加载 bge-m3（本地权重目录），返回 (tokenizer, model, device)。"""
    log(f"加载 bge-m3 模型：{MODEL_PATH}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"  device: {device}")
    if device == "cuda":
        log(f"  GPU: {torch.cuda.get_device_name(0)}  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    model_dir = str(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True).to(device).eval()
    log("  模型加载完成")
    return tokenizer, model, device


@torch.no_grad()
def encode_batch(texts: list[str], tokenizer, model, device: str) -> np.ndarray:
    """编码一批文本，返回 L2-归一化的 (N, 1024) float32 矩阵。"""
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)
    out    = model(**enc)
    embeds = out.last_hidden_state[:, 0, :]  # [CLS] token
    embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)
    return embeds.cpu().float().numpy()


def build_index(chunks: list[dict], tokenizer, model, device: str,
                start_idx: int = 0) -> tuple[faiss.Index, list[dict]]:
    """
    对 chunks[start_idx:] 编码并追加到 FAISS 索引中。
    返回 (index, new_metadata_rows)
    """
    texts  = [c["text"] for c in chunks[start_idx:]]
    metas  = chunks[start_idx:]
    n      = len(texts)
    log(f"开始编码 {n} 个 chunk（从全量索引 #{start_idx} 开始）...")

    # 加载或新建 FAISS 索引
    if FAISS_PATH.exists() and start_idx > 0:
        index = faiss.read_index(str(FAISS_PATH))
        log(f"  已有索引：{index.ntotal} 向量，继续追加")
    else:
        index = faiss.IndexFlatIP(EMBED_DIM)
        log(f"  新建 IndexFlatIP，dim={EMBED_DIM}")

    new_rows = []
    t0 = time.time()

    for i in range(0, n, BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]
        batch_metas = metas[i:i + BATCH_SIZE]
        embeds = encode_batch(batch_texts, tokenizer, model, device)
        index.add(embeds)

        for j, (meta, vec) in enumerate(zip(batch_metas, embeds)):
            row = dict(meta)
            row["chunk_id"] = start_idx + i + j
            new_rows.append(row)

        if (i // BATCH_SIZE) % 50 == 0:
            done    = i + len(batch_texts)
            elapsed = time.time() - t0
            eta     = elapsed / max(done, 1) * (n - done)
            log(f"  {done}/{n} chunks  elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min")

        # 每 1000 chunk 落盘一次（防中断）
        if (i + BATCH_SIZE) % 1000 < BATCH_SIZE:
            faiss.write_index(index, str(FAISS_PATH))
            _flush_metadata(new_rows)
            log(f"  中间存档：{index.ntotal} 向量")

    return index, new_rows


def _flush_metadata(new_rows: list[dict]):
    """把新 metadata 行追加到 metadata.parquet。"""
    if not new_rows:
        return
    df_new = pd.DataFrame(new_rows)
    if META_PATH.exists():
        df_old = pd.read_parquet(META_PATH)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        df_all.drop_duplicates("chunk_id", inplace=True)
    else:
        df_all = df_new
    df_all.to_parquet(META_PATH, index=False)


# ── 断点续传逻辑 ──────────────────────────────────────────────────────────────

def get_resume_point(all_chunks: list[dict]) -> int:
    """返回应该从哪个 chunk_id 开始（断点续传）。"""
    if not META_PATH.exists():
        return 0
    df_meta = pd.read_parquet(META_PATH)
    if df_meta.empty:
        return 0
    done_count = len(df_meta)
    log(f"断点续传：已有 {done_count} / {len(all_chunks)} chunks")
    return done_count


# ── 入口 ──────────────────────────────────────────────────────────────────────

def run():
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("Phase 0: Transcript Vectorization")
    log("=" * 60)

    # 收集所有 batch 文件
    batch_files = sorted(BATCH_DIR.glob("batch_full_*.json"))
    log(f"找到 {len(batch_files)} 个 batch 文件")

    # 切分所有 transcript
    log("切分 transcript -> chunks ...")
    t0 = time.time()
    all_chunks = load_all_chunks(batch_files)
    log(f"  总 chunks: {len(all_chunks)}  耗时: {time.time()-t0:.1f}s")

    # 统计
    df_stat = pd.DataFrame(all_chunks)
    log(f"  prepared chunks:      {(df_stat['section_type']=='prepared').sum()}")
    log(f"  qa chunks:            {(df_stat['section_type']=='qa').sum()}")
    log(f"  press_release chunks: {(df_stat['section_type']=='press_release').sum()}")
    log(f"  mgmt chunks:          {(df_stat['speaker_role']=='mgmt').sum()}")
    log(f"  analyst chunks:       {(df_stat['speaker_role']=='analyst').sum()}")
    log(f"  avg text len:         {df_stat['text'].str.len().mean():.0f} chars")

    # 断点续传
    start_idx = get_resume_point(all_chunks)
    if start_idx >= len(all_chunks):
        log("所有 chunks 已编码完成，无需重新运行。")
        return

    # 加载模型
    tokenizer, model, device = load_model()

    # 编码 + 建索引
    index, new_rows = build_index(all_chunks, tokenizer, model, device, start_idx)

    # 最终写入
    faiss.write_index(index, str(FAISS_PATH))
    _flush_metadata(new_rows)

    log("=" * 60)
    log(f"完成！FAISS 索引：{index.ntotal} 向量")
    log(f"metadata.parquet：{META_PATH}")
    log(f"faiss.index：     {FAISS_PATH}")
    log("=" * 60)


if __name__ == "__main__":
    run()
