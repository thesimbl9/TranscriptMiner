"""
build_theory_index.py — 学术论文 Theory RAG 索引构建

输入: papers/*.pdf (学术论文)
输出: vector_store/theory_index.faiss + vector_store/theory_metadata.parquet

用法:
  python build_theory_index.py
  python build_theory_index.py --device cpu --batch-size 32

数据来源:
  将 20 篇学术论文 PDF 放入 papers/ 目录，详见 papers/README.txt
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── 路径 ──────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.resolve()
PAPERS_DIR = ROOT / "papers"
MODEL_DIR  = ROOT / "model"
STORE_DIR  = ROOT / "vector_store"

INDEX_PATH = STORE_DIR / "theory_index.faiss"
META_PATH  = STORE_DIR / "theory_metadata.parquet"

BATCH_SIZE       = 16
EMBED_DIM        = 1024
MIN_CHUNK_CHARS  = 150
MAX_CHUNK_CHARS  = 1200


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# PDF 文本提取
# ═══════════════════════════════════════════════════════════════════════════

def extract_pdf_text(pdf_path: Path) -> list[tuple[int, str]]:
    """提取 PDF 每页文本，返回 [(page_num, text), ...]"""
    from pypdf import PdfReader
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


# ═══════════════════════════════════════════════════════════════════════════
# 文本清洗 & 切分
# ═══════════════════════════════════════════════════════════════════════════

NOISE_RE = re.compile(
    r"^(arXiv:|https?://|doi:|www\.|©|\d+\s*$|References$|REFERENCES$"
    r"|Abstract$|ABSTRACT$|Introduction$|Conclusion)",
    re.IGNORECASE,
)


def clean_text(text: str) -> str:
    """清洗 PDF 提取的原始文本。"""
    text = re.sub(r"-\n([a-z])", r"\1", text)
    text = re.sub(r"(?<!\n)\n(?!\n)(?![A-Z\d])", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def split_into_chunks(page_num: int, text: str, paper_title: str) -> list[dict]:
    """把单页文本切成段落 chunk。"""
    text = clean_text(text)
    paragraphs = re.split(r"\n{2,}", text)

    chunks = []
    pending = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        lines = [l for l in para.splitlines() if not NOISE_RE.match(l.strip())]
        para = " ".join(lines).strip()
        if len(para) < 50:
            continue

        if len(para) < MIN_CHUNK_CHARS:
            pending = (pending + " " + para).strip()
            continue

        if pending:
            combined = (pending + " " + para).strip()
            if len(combined) <= MAX_CHUNK_CHARS:
                chunks.append({"paper_title": paper_title, "page_num": page_num,
                               "text": combined})
                pending = ""
                continue
            else:
                chunks.append({"paper_title": paper_title, "page_num": page_num,
                               "text": pending})
                pending = ""

        if len(para) <= MAX_CHUNK_CHARS:
            chunks.append({"paper_title": paper_title, "page_num": page_num,
                           "text": para})
        else:
            while len(para) > MAX_CHUNK_CHARS:
                cut = para[:MAX_CHUNK_CHARS]
                last = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
                if last > MAX_CHUNK_CHARS // 2:
                    cut = para[:last + 1]
                chunks.append({"paper_title": paper_title, "page_num": page_num,
                               "text": cut.strip()})
                para = para[len(cut):].strip()
            if len(para) >= MIN_CHUNK_CHARS:
                chunks.append({"paper_title": paper_title, "page_num": page_num,
                               "text": para})

    if pending and len(pending) >= MIN_CHUNK_CHARS:
        chunks.append({"paper_title": paper_title, "page_num": page_num,
                       "text": pending})

    return chunks


def load_all_chunks() -> list[dict]:
    """读取所有 PDF，切分成 chunks。"""
    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))
    if not pdf_files:
        log("[ERROR] papers/ 中未找到 PDF 文件")
        log("  请将 20 篇学术论文 PDF 放入 papers/ 目录，详见 papers/README.txt")
        sys.exit(1)

    log(f"发现 {len(pdf_files)} 篇论文")

    all_chunks = []
    for pdf_path in pdf_files:
        title = pdf_path.stem
        pages = extract_pdf_text(pdf_path)
        paper_chunks = []
        for page_num, text in pages:
            paper_chunks.extend(split_into_chunks(page_num, text, title))
        log(f"  {title[:60]:<62} → {len(paper_chunks):>4} chunks")
        all_chunks.extend(paper_chunks)

    for i, c in enumerate(all_chunks):
        c["chunk_id"] = i

    log(f"总 chunks: {len(all_chunks)}")
    return all_chunks


# ═══════════════════════════════════════════════════════════════════════════
# 模型加载
# ═══════════════════════════════════════════════════════════════════════════

def load_model(device_override: str | None = None):
    """加载 BGE-M3。优先本地 model/，其次 HF 自动下载。"""
    import torch
    from transformers import AutoModel, AutoTokenizer

    has_local = MODEL_DIR.exists() and (
        any(MODEL_DIR.glob("*.safetensors")) or
        (MODEL_DIR / "pytorch_model.bin").exists()
    )

    if has_local:
        model_path = str(MODEL_DIR)
        log(f"本地模型: {model_path}")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    else:
        model_path = "BAAI/bge-m3"
        log(f"在线模型: {model_path} (首次将下载 ~2.2GB)")

    device = device_override or ("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device).eval()
    log("模型加载完成")
    return tokenizer, model, device


# ═══════════════════════════════════════════════════════════════════════════
# 编码 & 建索引
# ═══════════════════════════════════════════════════════════════════════════

def build_theory_index(chunks: list[dict], tokenizer, model, device: str,
                       batch_size: int = BATCH_SIZE):
    import faiss
    import torch

    index = faiss.IndexFlatIP(EMBED_DIM)
    texts = [c["text"] for c in chunks]
    n     = len(texts)

    log(f"开始编码 {n} 个 chunks...")
    t0 = time.time()

    all_vecs = []
    for i in range(0, n, batch_size):
        batch = texts[i: i + batch_size]
        with torch.no_grad():
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=512, return_tensors="pt").to(device)
            out = model(**enc)
            vecs = torch.nn.functional.normalize(
                out.last_hidden_state[:, 0, :], p=2, dim=1)
            all_vecs.append(vecs.cpu().float().numpy())

        if (i // batch_size + 1) % 20 == 0:
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
    log(f"完成! theory_index: {index.ntotal} 向量  {size_mb:.1f}MB  "
        f"耗时={elapsed:.1f}min")
    log(f"  → {INDEX_PATH}")
    log(f"  → {META_PATH}")


# ═══════════════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Academic Papers → Theory RAG FAISS Index")
    parser.add_argument("--device",     default=None, help="cuda / cpu")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    STORE_DIR.mkdir(parents=True, exist_ok=True)

    log("=== Phase 0: Theory Index 构建 ===")
    log(f"  论文目录: {PAPERS_DIR}")
    log(f"  模型目录: {MODEL_DIR}")
    log(f"  输出目录: {STORE_DIR}")

    chunks = load_all_chunks()
    if not chunks:
        log("[ERROR] 未生成任何 chunk")
        sys.exit(1)

    # 分布统计
    df = pd.DataFrame(chunks)
    print("\n论文 chunk 数分布:")
    for title, cnt in df.groupby("paper_title").size().sort_values(
            ascending=False).items():
        print(f"  {cnt:4d}  {title[:70]}")
    print()

    tokenizer, model, device = load_model(args.device)
    build_theory_index(chunks, tokenizer, model, device,
                       batch_size=args.batch_size)


if __name__ == "__main__":
    main()
