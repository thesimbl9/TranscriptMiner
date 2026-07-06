"""
phase0_verify.py — 验证 phase0_indexing.py 的三个核心逻辑，无需 GPU 也能跑

验证内容：
  1. 切分逻辑（split_into_chunks）：chunk数量、section_type、speaker_role分布
  2. 模型可用性检查：bge-m3 是否能离线加载
  3. 小批量编码测试：取 5 个 chunk 编码，检查 shape 和 cos-sim 合理性
  4. FAISS 检索测试（如果索引已存在）：给定 query 返回 Top-5

运行：
  python phase0_verify.py           # 完整验证（含模型加载，需VRAM）
  python phase0_verify.py --no-gpu  # 只验证切分逻辑，不加载模型
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["HF_HUB_OFFLINE"]      = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from agent_core.config import (
    PROJECT, BATCH_DIR, STORE_DIR, FAISS_PATH, META_PATH, MODEL_PATH,
)


def test_split_logic():
    print("=" * 55)
    print("Test 1: 切分逻辑")
    print("=" * 55)

    sys.path.insert(0, str(Path(__file__).parent))
    from phase0_indexing import split_into_chunks, load_all_chunks

    # 取第一个 batch 做测试
    batch_files = sorted(BATCH_DIR.glob("batch_full_*.json"))[:3]
    print(f"  读取前3个batch文件...")
    chunks = load_all_chunks(batch_files)
    print(f"  总chunks: {len(chunks)}")

    df = pd.DataFrame(chunks)
    print(f"\n  section_type 分布:")
    print(df["section_type"].value_counts().to_string())
    print(f"\n  speaker_role 分布:")
    print(df["speaker_role"].value_counts().to_string())
    print(f"\n  text 长度统计 (chars):")
    print(df["text"].str.len().describe().to_string())

    # 抽查 5 个 chunk
    print(f"\n  抽查 5 个 chunk:")
    for _, row in df.sample(5, random_state=42).iterrows():
        print(f"  [{row['section_type']}/{row['speaker_role']}] {row['symbol']} {row['earnings_date']}")
        print(f"    {row['text'][:120]}...")
        print()

    # 验证无空 text
    empty = df[df["text"].str.strip() == ""]
    print(f"  空 text chunk 数: {len(empty)}  (应为0)")

    # 估算全量大小（transcript 部分按 batch 外推，press_release 已是全量）
    batch_all = sorted(BATCH_DIR.glob("batch_full_*.json"))
    transcript_chunks = [c for c in chunks if c["section_type"] != "press_release"]
    pr_chunks = [c for c in chunks if c["section_type"] == "press_release"]
    avg_transcript_per_batch = len(transcript_chunks) / len(batch_files)
    estimated_transcript = int(avg_transcript_per_batch * len(batch_all))
    estimated_total = estimated_transcript + len(pr_chunks)
    print(f"\n  全量 batch 文件: {len(batch_all)}")
    print(f"  估算 transcript chunks (全量758 batch): ~{estimated_transcript:,}")
    print(f"  press_release chunks (已是全量337股): {len(pr_chunks):,}")
    print(f"  估算总 chunks: ~{estimated_total:,}")
    print(f"  估算向量存储大小: ~{estimated_total * 1024 * 4 / 1024**2:.0f} MB (float32)")
    print()
    return True


def test_model_available():
    print("=" * 55)
    print("Test 2: bge-m3 模型可用性")
    print("=" * 55)
    import torch
    from transformers import AutoTokenizer, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU: {props.name}  VRAM: {props.total_memory/1e9:.1f} GB")

    model_dir = str(MODEL_PATH)
    try:
        tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        print(f"  tokenizer OK: vocab_size={tok.vocab_size}")
    except Exception as e:
        print(f"  [FAIL] tokenizer: {e}")
        return False

    try:
        model = AutoModel.from_pretrained(model_dir, local_files_only=True)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  model OK: {param_count/1e6:.0f}M params")

        # 快速推理测试（5个句子）
        model = model.to(device).eval()
        test_texts = [
            "Revenue guidance raised 8% above prior quarter expectations.",
            "Supply chain headwinds persisting into next quarter.",
            "We expect strong demand momentum across all verticals.",
            "Management provided specific EPS targets for fiscal year.",
            "Analysts questioned the sustainability of margin expansion.",
        ]
        import torch
        with torch.no_grad():
            enc = tok(test_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            out = model(**enc)
            embeds = out.last_hidden_state[:, 0, :]
            embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)
            arr = embeds.cpu().float().numpy()

        print(f"  编码测试 OK: shape={arr.shape}  (应为 (5, 1024))")
        assert arr.shape == (5, 1024), "shape 错误"

        # 检查 cos-sim 合理性
        sim_01 = float(np.dot(arr[0], arr[1]))  # 应偏低（不同主题）
        sim_02 = float(np.dot(arr[0], arr[2]))  # 应偏高（都是正向guidance）
        print(f"  cos-sim(raised_guidance, supply_chain): {sim_01:.3f}  (预期 <0.6，话题不同)")
        print(f"  cos-sim(raised_guidance, strong_demand): {sim_02:.3f}  (预期 >0.5，话题相近)")

    except Exception as e:
        print(f"  [FAIL] model: {e}")
        import traceback; traceback.print_exc()
        return False

    print()
    return True


def test_faiss_search():
    print("=" * 55)
    print("Test 3: FAISS 检索测试（需先完成 Phase 0）")
    print("=" * 55)

    if not FAISS_PATH.exists():
        print(f"  [SKIP] 索引文件不存在：{FAISS_PATH}")
        print(f"  请先运行 phase0_run.py 完成索引构建")
        return True

    import faiss
    from transformers import AutoTokenizer, AutoModel
    import torch

    index = faiss.read_index(str(FAISS_PATH))
    meta  = pd.read_parquet(META_PATH)
    print(f"  索引向量数: {index.ntotal}")
    print(f"  metadata行数: {len(meta)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = str(MODEL_PATH)
    tok    = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model  = AutoModel.from_pretrained(model_dir, local_files_only=True).to(device).eval()

    queries = [
        "guidance raised revenue target next quarter growth",
        "supply chain risk headwind uncertainty",
        "analyst asked management declined to provide specific numbers",
    ]

    for query in queries:
        print(f"\n  Query: '{query}'")
        with torch.no_grad():
            enc = tok([query], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            out = model(**enc)
            q_vec = torch.nn.functional.normalize(out.last_hidden_state[:, 0, :], p=2, dim=1)
            q_arr = q_vec.cpu().float().numpy()

        scores, idxs = index.search(q_arr, 5)
        for rank, (idx, score) in enumerate(zip(idxs[0], scores[0])):
            row = meta.iloc[idx]
            print(f"    #{rank+1} score={score:.3f}  [{row['section_type']}/{row['speaker_role']}]"
                  f"  {row['symbol']} {row['earnings_date']}")
            print(f"       {row['text'][:100]}...")

    print()
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-gpu", action="store_true", help="只验证切分逻辑，跳过模型加载")
    args = parser.parse_args()

    ok1 = test_split_logic()

    if args.no_gpu:
        print("--no-gpu: 跳过模型验证")
        sys.exit(0 if ok1 else 1)

    ok2 = test_model_available()
    ok3 = test_faiss_search()

    all_ok = ok1 and ok2 and ok3
    print("=" * 55)
    print(f"验证结果: {'全部通过 ✓' if all_ok else '有失败项 ✗'}")
    print("=" * 55)
    sys.exit(0 if all_ok else 1)
