"""
phase0_theory_index.py — 建立学术文献 theory RAG 索引

输入：Fullproject/papers/*.pdf（20篇学术论文）
输出：vector_store/theory_index.faiss + theory_metadata.parquet

切分策略：
  - 按段落切分（以空行为边界）
  - MIN=150 chars，MAX=1200 chars
  - 每个 chunk 保留：paper_title, chunk_id, text, page_num
"""

import os
import re
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from pypdf import PdfReader
from transformers import AutoModel, AutoTokenizer

os.environ["HF_HUB_OFFLINE"]      = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from agent_core.config import (
    PROJECT, STORE_DIR, THEORY_INDEX as INDEX_PATH, THEORY_META as META_PATH, MODEL_PATH,
)
PAPERS_DIR = Path(__file__).parent / "papers"

BATCH_SIZE    = 16
EMBED_DIM     = 1024
MIN_CHUNK_CHARS = 150
MAX_CHUNK_CHARS = 1200

STORE_DIR.mkdir(exist_ok=True)


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── PDF 文本提取 ──────────────────────────────────────────────────────────────

def extract_pdf_text(pdf_path: Path) -> list[tuple[int, str]]:
    """提取 PDF 每页文本，返回 [(page_num, text), ...]"""
    pages = []
    try:
        reader = PdfReader(str(pdf_path))
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((i + 1, text))
    except Exception as e:
        log(f"  [WARN] 读取失败 {pdf_path.name}: {e}")
    return pages


# ── 文本清洗 ─────────────────────────────────────────────────────────────────

# 过滤纯引用行、页眉页脚、URL、arXiv 标识等噪声
NOISE_RE = re.compile(
    r"^(arXiv:|https?://|doi:|www\.|©|\d+\s*$|References$|REFERENCES$"
    r"|Abstract$|ABSTRACT$|Introduction$|Conclusion)",
    re.IGNORECASE,
)

def clean_text(text: str) -> str:
    """清洗 PDF 提取的原始文本。"""
    # 修复断行连字符
    text = re.sub(r"-\n([a-z])", r"\1", text)
    # 合并段内换行（非段落边界）
    text = re.sub(r"(?<!\n)\n(?!\n)(?![A-Z\d])", " ", text)
    # 多余空格
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def split_into_chunks(page_num: int, text: str, paper_title: str) -> list[dict]:
    """把单页文本切成段落 chunk。"""
    text = clean_text(text)
    # 按双换行（段落边界）切分
    paragraphs = re.split(r"\n{2,}", text)

    chunks = []
    pending = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # 过滤噪声行
        lines = [l for l in para.splitlines() if not NOISE_RE.match(l.strip())]
        para = " ".join(lines).strip()
        if len(para) < 50:
            continue

        # 太短的段落合并到下一段
        if len(para) < MIN_CHUNK_CHARS:
            pending = (pending + " " + para).strip()
            continue

        # 有 pending 先 flush
        if pending:
            combined = (pending + " " + para).strip()
            if len(combined) <= MAX_CHUNK_CHARS:
                chunks.append({
                    "paper_title": paper_title,
                    "page_num":    page_num,
                    "text":        combined,
                })
                pending = ""
                continue
            else:
                chunks.append({
                    "paper_title": paper_title,
                    "page_num":    page_num,
                    "text":        pending,
                })
                pending = ""

        # 正常段落
        if len(para) <= MAX_CHUNK_CHARS:
            chunks.append({
                "paper_title": paper_title,
                "page_num":    page_num,
                "text":        para,
            })
        else:
            # 长段落按句子边界硬切
            while len(para) > MAX_CHUNK_CHARS:
                cut = para[:MAX_CHUNK_CHARS]
                last = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
                if last > MAX_CHUNK_CHARS // 2:
                    cut = para[:last + 1]
                chunks.append({
                    "paper_title": paper_title,
                    "page_num":    page_num,
                    "text":        cut.strip(),
                })
                para = para[len(cut):].strip()
            if len(para) >= MIN_CHUNK_CHARS:
                chunks.append({
                    "paper_title": paper_title,
                    "page_num":    page_num,
                    "text":        para,
                })

    if pending and len(pending) >= MIN_CHUNK_CHARS:
        chunks.append({
            "paper_title": paper_title,
            "page_num":    page_num,
            "text":        pending,
        })

    return chunks


def load_all_chunks() -> list[dict]:
    """读取所有 PDF，切分成 chunks。"""
    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))
    log(f"发现 {len(pdf_files)} 篇论文")

    all_chunks = []
    for pdf_path in pdf_files:
        title = pdf_path.stem
        pages = extract_pdf_text(pdf_path)
        paper_chunks = []
        for page_num, text in pages:
            paper_chunks.extend(split_into_chunks(page_num, text, title))

        log(f"  {title[:60]}  →  {len(paper_chunks)} chunks")
        all_chunks.extend(paper_chunks)

    # 加 chunk_id
    for i, c in enumerate(all_chunks):
        c["chunk_id"] = i

    log(f"总 chunks: {len(all_chunks)}")
    return all_chunks


# ── 编码 & 建索引 ─────────────────────────────────────────────────────────────

def build_theory_index(chunks: list[dict]):
    log("加载 bge-m3 模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"  device: {device}")

    tok   = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
    model = AutoModel.from_pretrained(str(MODEL_PATH), local_files_only=True)
    model = model.to(device).eval()
    log("  模型加载完成")

    index = faiss.IndexFlatIP(EMBED_DIM)
    texts = [c["text"] for c in chunks]
    n     = len(texts)

    log(f"开始编码 {n} 个 chunks...")
    t0 = time.time()

    all_vecs = []
    for i in range(0, n, BATCH_SIZE):
        batch = texts[i: i + BATCH_SIZE]
        with torch.no_grad():
            enc = tok(
                batch, padding=True, truncation=True,
                max_length=512, return_tensors="pt"
            ).to(device)
            out = model(**enc)
            vecs = torch.nn.functional.normalize(
                out.last_hidden_state[:, 0, :], p=2, dim=1
            )
            all_vecs.append(vecs.cpu().float().numpy())

        if (i // BATCH_SIZE + 1) % 20 == 0:
            elapsed = (time.time() - t0) / 60
            log(f"  {i + len(batch)}/{n}  elapsed={elapsed:.1f}min")

    vectors = np.vstack(all_vecs)
    index.add(vectors)

    # 保存
    faiss.write_index(index, str(INDEX_PATH))
    meta = pd.DataFrame(chunks)
    meta.to_parquet(META_PATH, index=False)

    elapsed = (time.time() - t0) / 60
    size_mb = INDEX_PATH.stat().st_size / 1024 / 1024
    log(f"完成！theory_index: {index.ntotal} 向量  {size_mb:.1f}MB  耗时={elapsed:.1f}min")
    log(f"  → {INDEX_PATH}")
    log(f"  → {META_PATH}")


if __name__ == "__main__":
    log("=== Phase 0: Theory Index 构建 ===")
    chunks = load_all_chunks()

    if not chunks:
        log("[ERROR] 未找到任何 chunk，检查 papers/ 目录")
        sys.exit(1)

    # 打印分布
    df = pd.DataFrame(chunks)
    print("\n论文 chunk 数分布:")
    for title, cnt in df.groupby("paper_title").size().sort_values(ascending=False).items():
        print(f"  {cnt:4d}  {title[:70]}")
    print()

    build_theory_index(chunks)
