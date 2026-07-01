# Phase 0: Earnings Call Transcript & Theory RAG Index Builder

构建 EarningsSignal Agent 所需的两个 FAISS 向量索引：
- **Transcript Index**：财报电话会议 ~95 万 chunk → 语义检索
- **Theory Index**：20 篇学术论文 → RAG 知识库

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

| 数据 | 目录 | 说明 |
|------|------|------|
| 财报转录文本 | `earnings_call_data/` | 见 `earnings_call_data/README.txt` |
| 学术论文 PDF | `papers/` | 见 `papers/README.txt` |
| BGE-M3 模型 | `model/`（可选） | 见 `model/README.txt` |

### 3. 运行

```bash
# 构建转录文本索引（~1-4小时，取决于 GPU）
python build_transcript_index.py

# 构建论文理论索引（~5分钟）
python build_theory_index.py

# 指定设备
python build_transcript_index.py --device cpu --batch-size 8
python build_theory_index.py --device cuda --batch-size 32

# 测试模式：只处理前 1000 行
python build_transcript_index.py --max-rows 1000
```

### 4. 输出

```
vector_store/
├── faiss.index              # 转录文本向量索引
├── metadata.parquet          # chunk 元数据 (symbol / date / section / speaker)
├── theory_index.faiss        # 论文向量索引
└── theory_metadata.parquet   # 论文 chunk 元数据 (title / page / text)
```

## 环境要求

- Python 3.9+
- 16GB RAM（CPU 模式）/ 8GB VRAM（GPU 模式）
- 磁盘空间：~6GB（向量索引 ~4GB + 模型 ~2.2GB）
- BGE-M3 模型：联网自动下载或手动放入 `model/` 目录

## 文件结构

```
phase0_pipeline/
├── build_transcript_index.py   # 转录文本 → FAISS
├── build_theory_index.py       # 论文 PDF → FAISS
├── requirements.txt
├── README.md                   # 本文件
├── model/
│   └── README.txt              # BGE-M3 下载说明
├── earnings_call_data/
│   └── README.txt              # 数据集下载说明
└── papers/
    └── README.txt              # 论文列表
```
