"""
agent_core/config.py — 项目全局配置（路径、API key、环境变量）

模块负责：
  - 发现项目根目录（基于本文件位置向上两级）
  - 加载 .env 文件并暴露所有 API 配置
  - 提供所有数据/模型/输出路径的集中管理
  - 替换全项目 7 个文件中的硬编码路径重复

使用方式：
  from agent_core.config import PROJECT, SP500_EVENTS, STORE_DIR, API_KEY, MODEL

注意：
  - 此模块在导入时执行，确保 .env 文件在项目根目录存在
  - 如果 .env 不存在，API_KEY 为空字符串，调用 LLM 的模块需自行检查
"""

import os
from pathlib import Path
from dotenv import load_dotenv


# ═══════════════════════════════════════════════════════════════════════════════════
# 项目根目录发现（无论从哪里导入，始终指向 Fullproject/）
# ═══════════════════════════════════════════════════════════════════════════════════
FULLPROJECT = Path(__file__).parent.parent.resolve()   # agent_core/ 的上一级 = Fullproject/
PROJECT     = FULLPROJECT.parent.resolve()               # Fullproject/ 的上一级 = 项目根


# ═══════════════════════════════════════════════════════════════════════════════════
# .env 加载 & API 配置
# ═══════════════════════════════════════════════════════════════════════════════════
_ENV_PATH = FULLPROJECT / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)

API_KEY  = os.environ.get("SILICONFLOW_API_KEY", "")
MODEL    = os.environ.get("SILICONFLOW_MODEL", "deepseek-ai/DeepSeek-V4-Flash")
BASE_URL = "https://api.siliconflow.cn/v1"

# HF 离线模式（安全约束）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Python 解释器路径（来自 .env 或默认）
PYTHON_EXE = os.environ.get("QUANT_PYTHON", "python")


# ═══════════════════════════════════════════════════════════════════════════════════
# 数据路径
# ═══════════════════════════════════════════════════════════════════════════════════
SP500_EVENTS = FULLPROJECT / "data" / "sp500_events.parquet"
EPISODES_PATH = PROJECT / "earnings-call-data" / "episodes.parquet"
BATCH_DIR    = PROJECT / "ecagent" / "transcript_batches"


# ═══════════════════════════════════════════════════════════════════════════════════
# 向量库 & 模型权重
# ═══════════════════════════════════════════════════════════════════════════════════
STORE_DIR      = FULLPROJECT / "vector_store"
FAISS_PATH     = STORE_DIR / "faiss.index"
META_PATH      = STORE_DIR / "metadata.parquet"
THEORY_INDEX   = STORE_DIR / "theory_index.faiss"
THEORY_META    = STORE_DIR / "theory_metadata.parquet"
MODEL_PATH     = PROJECT / "weights" / "bge-m3"


# ═══════════════════════════════════════════════════════════════════════════════════
# 输出 & 日志
# ═══════════════════════════════════════════════════════════════════════════════════
OUTPUT_DIR = FULLPROJECT / "agent_output"
LOG_DIR    = FULLPROJECT / "logs"
HISTORY_PATH = FULLPROJECT / "agent_core" / "feature_history.jsonl"


# ═══════════════════════════════════════════════════════════════════════════════════
# 常量（跨模块共享）
# ═══════════════════════════════════════════════════════════════════════════════════
TRAIN_YRS = list(range(2015, 2020))
VALID_YRS = [2020]
TEST_YRS  = list(range(2021, 2024))

# sp500_events 可用字段（从 validation_agent 迁移，在提取时预加载）
SP500_COLS = [
    "symbol", "sector", "earnings_date", "year", "quarter",
    "move_post", "target_cs",
]
