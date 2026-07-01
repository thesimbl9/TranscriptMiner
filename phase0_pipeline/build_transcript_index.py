"""
build_transcript_index.py — 财报电话会议转录文本向量化 & FAISS 索引构建

输入: earnings_call_data/*.parquet (HuggingFace earnings-call-data 数据集)
输出: vector_store/faiss.index + vector_store/metadata.parquet

用法:
  # 完整构建
  python build_transcript_index.py

  # 指定设备/批次
  python build_transcript_index.py --device cuda --batch-size 32

  # 只构建前 N 个 chunk（测试用）
  python build_transcript_index.py --max-rows 10000

数据来源:
  https://huggingface.co/datasets/RudrakshNanavaty/earnings-call-data
  下载后放到 earnings_call_data/ 目录下

Pipeline:
  1. 读取 Parquet 文件 → 提取 earnings_transcript + press_release_ex991
  2. Speaker turn 切分 + section 识别 (prepared / qa / press_release)
  3. BGE-M3 编码 (batch GPU/CPU)
  4. FAISS IndexFlatIP + metadata.parquet 落盘
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── 路径: 全部相对于本脚本所在目录 ──────────────────────────────────────────
ROOT      = Path(__file__).parent.resolve()
DATA_DIR  = ROOT / "earnings_call_data"
MODEL_DIR = ROOT / "model"
STORE_DIR = ROOT / "vector_store"
LOG_DIR   = ROOT / "logs"

FAISS_PATH = STORE_DIR / "faiss.index"
META_PATH  = STORE_DIR / "metadata.parquet"
LOG_PATH   = LOG_DIR / "build_transcript_index.log"

# ── 模型配置 ──────────────────────────────────────────────────────────────
BATCH_SIZE      = 16
EMBED_DIM       = 1024       # BGE-M3 output dim
MAX_CHUNK_TOKENS = 380

# ── 切分参数 ──────────────────────────────────────────────────────────────
MIN_CHUNK_CHARS  = 100
MAX_CHUNK_CHARS  = 1600
DROP_CHUNK_CHARS = 30

# ── Q&A 段落检测标记 ──────────────────────────────────────────────────────
QA_BOUNDARY_MARKERS = [
    r"(?i)^\s*questions?\s*(?:and|&)\s*answers?\s*$",
    r"(?i)^\s*q\s*&?\s*a\s*$",
    r"(?i)^\s*question[- ]and[- ]answer\s+session\s*$",
    r"(?i)we will now (?:begin|open|take|start).*(?:q(?:uestion)?[-\s]*&?[-\s]*a(?:nswer)?|question)",
    r"(?i)the operator will now.*(?:q(?:uestion)?[-\s]*&?[-\s]*a(?:nswer)?)",
    r"(?i)turn (?:the call|it) over.*(?:q(?:uestion)?[-\s]*&?[-\s]*a(?:nswer)?)",
]


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# 转录文本拆分
# ═══════════════════════════════════════════════════════════════════════════

def split_transcript_sections(transcript: str) -> tuple[str, str]:
    """将原始 transcript 拆分为 prepared_remarks 和 qa_section。

    检测 Q&A 边界标记（如 "Questions and Answers"），之前的为 prepared，
    之后的为 qa。如果找不到边界，整段视为 prepared。
    """
    if not transcript or not transcript.strip():
        return "", ""

    lines = transcript.splitlines()
    for i, line in enumerate(lines):
        for marker in QA_BOUNDARY_MARKERS:
            if re.search(marker, line):
                prepared = "\n".join(lines[:i]).strip()
                qa       = "\n".join(lines[i:]).strip()
                return prepared, qa

    # 找不到 Q&A 边界 → 全部为 prepared
    return transcript.strip(), ""


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

QA_ANALYST_PREFIX = re.compile(r"^Q\s*[–\-]\s*", re.IGNORECASE)

ANALYST_FIRMS = re.compile(
    r"\b(analyst|research|securities|capital|morgan|goldman|jpmorgan|barclays|"
    r"citi|ubs|bofa|bank of america|deutsche|jefferies|wells|cowen|"
    r"nomura|hsbc|credit suisse|evercore|piper|raymond|bernstein|mizuho)\b",
    re.IGNORECASE,
)


def detect_speaker_role(speaker_line: str) -> str:
    """从说话人行判断是 mgmt 还是 analyst。"""
    if not speaker_line:
        return "unknown"
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
    """把 prepared + qa 按 speaker turn 切分为 chunks。"""
    chunks = []

    def process_section(text: str, default_section: str):
        if not text or not text.strip():
            return

        positions = list(SPEAKER_TURN_RE.finditer(text))
        raw_chunks = []
        if not positions:
            raw_chunks.append(("", text.strip()))
        else:
            for i, m in enumerate(positions):
                speaker = m.group(1).strip()
                start   = m.end()
                end     = positions[i + 1].start() if i + 1 < len(positions) else len(text)
                content = text[start:end].strip()
                if content:
                    raw_chunks.append((speaker, content))

        pending_speaker = ""
        pending_text    = ""
        section_type    = default_section

        for speaker, content in raw_chunks:
            sec  = detect_section_type(content) if content else section_type
            role = detect_speaker_role(speaker)

            while len(content) > MAX_CHUNK_CHARS:
                cut = content[:MAX_CHUNK_CHARS]
                last_period = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
                if last_period > MAX_CHUNK_CHARS // 2:
                    cut = content[:last_period + 1]

                if pending_text and (len(pending_text) + len(cut) > MAX_CHUNK_CHARS):
                    chunks.append({
                        "symbol": symbol, "earnings_date": earnings_date,
                        "section_type": section_type,
                        "speaker_role": detect_speaker_role(pending_speaker),
                        "text": pending_text,
                    })
                    pending_text = ""
                    pending_speaker = ""

                chunks.append({
                    "symbol": symbol, "earnings_date": earnings_date,
                    "section_type": sec, "speaker_role": role,
                    "text": cut.strip(),
                })
                content = content[len(cut):].strip()
                section_type = sec

            if not content:
                continue

            if len(content) < MIN_CHUNK_CHARS:
                pending_text    += (" " if pending_text else "") + content
                pending_speaker = pending_speaker or speaker
                section_type    = sec
                continue

            if pending_text:
                chunks.append({
                    "symbol": symbol, "earnings_date": earnings_date,
                    "section_type": section_type,
                    "speaker_role": detect_speaker_role(pending_speaker),
                    "text": pending_text,
                })
                pending_text = ""
                pending_speaker = ""

            chunks.append({
                "symbol": symbol, "earnings_date": earnings_date,
                "section_type": sec, "speaker_role": role,
                "text": content,
            })
            section_type = sec

        if pending_text and len(pending_text) >= DROP_CHUNK_CHARS:
            chunks.append({
                "symbol": symbol, "earnings_date": earnings_date,
                "section_type": section_type,
                "speaker_role": detect_speaker_role(pending_speaker),
                "text": pending_text,
            })

    process_section(prepared, "prepared")
    process_section(qa,       "qa")
    return chunks


# ═══════════════════════════════════════════════════════════════════════════
# Press Release 切分
# ═══════════════════════════════════════════════════════════════════════════

PR_HEAD_CHARS = 20_000

PR_NOISE_LINE_RE = re.compile(
    r"^(EX-99\.1\s|Exhibit\s+99\.1|Table of Contents$|UNITED STATES SECURITIES|"
    r"Washington,\s*D\.C\.|FORM\s+8-K|CURRENT REPORT|Pursuant to Section|"
    r"Commission File Number|IRS Employer|CHECK THE APPROPRIATE|\(State or other)",
    re.IGNORECASE,
)


def _is_noise_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if len(s) < 120 and PR_NOISE_LINE_RE.match(s):
        return True
    if re.search(r'\.(htm|pdf|htm[l]?)\b', s, re.IGNORECASE) and len(s) < 200:
        return True
    alpha = sum(1 for c in s if c.isalpha())
    if len(s) > 20 and alpha / len(s) < 0.25:
        return True
    if re.match(r'^\s*[\$\d\(\-]', s) and s.count('$') + s.count(',') > 3:
        return True
    return False


def split_press_release(text: str, symbol: str, earnings_date: str) -> list[dict]:
    """把 ex991 新闻稿切分为 chunks。"""
    if not text or not text.strip():
        return []
    text = text[:PR_HEAD_CHARS]

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
        chunks.append({
            "symbol": symbol, "earnings_date": earnings_date,
            "section_type": "press_release", "speaker_role": "-",
            "text": txt,
        })

    for para in paragraphs:
        if not para:
            continue
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


# ═══════════════════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════════════════

def load_all_chunks(max_rows: int | None = None) -> list[dict]:
    """读取 earnings_call_data/*.parquet，切分为 chunks。"""

    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    if not parquet_files:
        log(f"[ERROR] earnings_call_data/ 中未找到 .parquet 文件")
        log(f"  请从以下地址下载数据集并放入该目录：")
        log(f"  https://huggingface.co/datasets/RudrakshNanavaty/earnings-call-data")
        sys.exit(1)

    log(f"找到 {len(parquet_files)} 个 Parquet 文件")

    all_chunks = []
    for pf in parquet_files:
        log(f"读取: {pf.name}")
        df = pd.read_parquet(pf)
        if max_rows:
            df = df.head(max_rows)

        required_cols = ["symbol", "earnings_date", "earnings_transcript"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            log(f"  [WARN] 缺少列 {missing}，跳过 {pf.name}")
            continue

        df["earnings_date"] = pd.to_datetime(df["earnings_date"])
        df["earnings_date_str"] = df["earnings_date"].dt.strftime("%Y-%m-%d")

        # 1) Transcript chunks
        transcript_count = 0
        for _, row in df.iterrows():
            transcript = str(row.get("earnings_transcript", "") or "")
            if not transcript.strip():
                continue
            prepared, qa = split_transcript_sections(transcript)
            chunks = split_into_chunks(
                prepared, qa,
                str(row["symbol"]), row["earnings_date_str"],
            )
            all_chunks.extend(chunks)
            transcript_count += len(chunks)

        log(f"  transcript chunks: {transcript_count}")

        # 2) Press release chunks
        if "press_release_ex991" in df.columns:
            pr_count = 0
            for _, row in df.iterrows():
                pr_text = str(row.get("press_release_ex991", "") or "")
                if not pr_text.strip() or len(pr_text) < 500:
                    continue
                chunks = split_press_release(
                    pr_text, str(row["symbol"]), row["earnings_date_str"],
                )
                all_chunks.extend(chunks)
                pr_count += len(chunks)
            log(f"  press_release chunks: {pr_count}")

    # 过滤过短的 chunk + 分配 chunk_id
    all_chunks = [c for c in all_chunks if len(c["text"]) >= DROP_CHUNK_CHARS]
    for i, c in enumerate(all_chunks):
        c["chunk_id"] = i

    return all_chunks


# ═══════════════════════════════════════════════════════════════════════════
# 模型加载
# ═══════════════════════════════════════════════════════════════════════════

def load_model(device_override: str | None = None):
    """加载 BGE-M3。

    优先级：
      1. 检查 model/ 目录是否有本地权重 → 离线加载
      2. 否则从 HuggingFace 自动下载 → 联网加载
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    # 判断本地是否有模型文件
    has_local = MODEL_DIR.exists() and any(MODEL_DIR.glob("*.safetensors")) or \
                (MODEL_DIR / "pytorch_model.bin").exists()

    if has_local:
        model_path = str(MODEL_DIR)
        log(f"本地模型: {model_path}")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    else:
        model_path = "BAAI/bge-m3"
        log(f"在线模型: {model_path} (首次将下载 ~2.2GB)")
        log(f"  如需离线使用，执行: git clone https://huggingface.co/BAAI/bge-m3 model/")

    device = device_override or ("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device: {device}")
    if device == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(0)}  "
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device).eval()
    log("模型加载完成")
    return tokenizer, model, device


# ═══════════════════════════════════════════════════════════════════════════
# 编码 & 建索引
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()  # type: ignore
def encode_batch(texts: list[str], tokenizer, model, device: str) -> np.ndarray:
    import torch
    enc = tokenizer(texts, padding=True, truncation=True, max_length=512,
                    return_tensors="pt").to(device)
    out    = model(**enc)
    embeds = out.last_hidden_state[:, 0, :]  # [CLS]
    embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)
    return embeds.cpu().float().numpy()


def build_index(chunks: list[dict], tokenizer, model, device: str,
                start_idx: int = 0, batch_size: int = BATCH_SIZE):
    import faiss

    texts = [c["text"] for c in chunks[start_idx:]]
    metas = chunks[start_idx:]
    n     = len(texts)
    log(f"开始编码 {n} 个 chunk (从 #{start_idx})...")

    if FAISS_PATH.exists() and start_idx > 0:
        index = faiss.read_index(str(FAISS_PATH))
        log(f"已有索引: {index.ntotal} 向量，继续追加")
    else:
        index = faiss.IndexFlatIP(EMBED_DIM)
        log(f"新建 IndexFlatIP, dim={EMBED_DIM}")

    new_rows = []
    t0 = time.time()

    for i in range(0, n, batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_metas = metas[i:i + batch_size]
        embeds = encode_batch(batch_texts, tokenizer, model, device)
        index.add(embeds)

        for j, (meta, vec) in enumerate(zip(batch_metas, embeds)):
            row = dict(meta)
            row["chunk_id"] = start_idx + i + j
            new_rows.append(row)

        if (i // batch_size) % 50 == 0:
            done    = i + len(batch_texts)
            elapsed = time.time() - t0
            eta     = elapsed / max(done, 1) * (n - done)
            log(f"  {done}/{n}  elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min")

        if (i + batch_size) % 1000 < batch_size:
            faiss.write_index(index, str(FAISS_PATH))
            _flush_metadata(new_rows)
            log(f"  中间存档: {index.ntotal} 向量")

    return index, new_rows


def _flush_metadata(new_rows: list[dict]):
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


def get_resume_point(all_chunks: list[dict]) -> int:
    if not META_PATH.exists():
        return 0
    df_meta = pd.read_parquet(META_PATH)
    if df_meta.empty:
        return 0
    done = len(df_meta)
    log(f"断点续传: 已有 {done} / {len(all_chunks)} chunks")
    return done


# ═══════════════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Earnings Call Transcript → FAISS Vector Index")
    parser.add_argument("--device",    default=None, help="cuda / cpu")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-rows",  type=int, default=None,
                        help="只处理前 N 行（测试用）")
    args = parser.parse_args()

    STORE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("Phase 0: Earnings Call Transcript Indexing")
    log(f"  数据目录: {DATA_DIR}")
    log(f"  模型目录: {MODEL_DIR}")
    log(f"  输出目录: {STORE_DIR}")
    log("=" * 60)

    # 切分
    log("切分 transcript → chunks ...")
    t0 = time.time()
    all_chunks = load_all_chunks(max_rows=args.max_rows)
    log(f"总 chunks: {len(all_chunks)}  耗时: {time.time()-t0:.1f}s")

    if not all_chunks:
        log("[ERROR] 未生成任何 chunk，请检查数据")
        sys.exit(1)

    # 统计
    df_stat = pd.DataFrame(all_chunks)
    for st in ["prepared", "qa", "press_release"]:
        if st in df_stat["section_type"].values:
            log(f"  {st}: {(df_stat['section_type']==st).sum()}")
    for sr in ["mgmt", "analyst", "operator"]:
        if sr in df_stat["speaker_role"].values:
            log(f"  {sr}: {(df_stat['speaker_role']==sr).sum()}")
    log(f"  avg text len: {df_stat['text'].str.len().mean():.0f} chars")

    # 断点续传
    start_idx = get_resume_point(all_chunks)
    if start_idx >= len(all_chunks):
        log("所有 chunks 已编码完成。")
        return

    # 模型 + 编码
    tokenizer, model, device = load_model(args.device)
    index, new_rows = build_index(
        all_chunks, tokenizer, model, device,
        start_idx=start_idx, batch_size=args.batch_size,
    )

    # 最终写入
    import faiss
    faiss.write_index(index, str(FAISS_PATH))
    _flush_metadata(new_rows)

    log("=" * 60)
    log(f"完成! FAISS 索引: {index.ntotal} 向量")
    log(f"  {FAISS_PATH}")
    log(f"  {META_PATH}")
    log("=" * 60)


if __name__ == "__main__":
    main()
