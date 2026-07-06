"""
harness/memory/ — 三层记忆系统

WorkingMemory  — 会话内保留最近 K 轮的成败摘要（自动压缩）
EpisodicMemory — 跨会话轨迹存储 + 检索（BGE-M3 编码 → FAISS 索引）
SemanticMemory — 复用现有 RAG（论文库 + 向量检索），见 agent_core/

解决的问题（失控模式 #3）：
  31 轮成败轨迹写入 feature_history.jsonl 但从不检索复用。
  Agent 每次新迭代从零开始，不知道"类似定义上次怎么失败的"。

Harness 解法：
  新任务 → EpisodicMemory 检索相似历史轨迹 → ContextConstructor 注入 →
  Agent 知道"类似定义方式历史上零值率偏高，考虑使用更细粒度的打分区间"
"""

from harness.memory.working import WorkingMemory
from harness.memory.episodic import EpisodicMemory

__all__ = ["WorkingMemory", "EpisodicMemory"]
