"""
harness/context.py — 动态上下文装配器

解决的问题（失控模式 #2）：
  Prompt 膨胀导致模型注意力稀释、零值逃避。
  V2 的 prompt 全量硬塞 8 特征定义 + 边界案例，模型注意力被稀释，
  零值率从 4% 升至 57%。不加信息模型缺约束，加了信息模型被淹没。

Harness 解法：
  三级动态装配——SystemContext（固定）+ TaskContext（当前任务）
  + SessionContext（WorkingMemory 摘要 + EpisodicMemory 检索）。
  三种可插拔压缩策略，在信息完整性和注意力集中度之间做动态权衡。

使用：
  ctx = ContextConstructor(strategy=ContextStrategy.SUMMARIZATION)
  prompt = ctx.assemble(system="你是一个量化研究员...",
                         task="分析以下特征定义...",
                         working_memory=wm.get_context(),
                         episodic_hints=em.retrieve_similar(spec))
"""

from __future__ import annotations

import re
from enum import Enum, auto
from typing import Any, Protocol


class ContextStrategy(Enum):
    """上下文压缩策略"""
    SLIDING_WINDOW       = auto()  # 滑窗截断：只保留最近 N 轮
    SUMMARIZATION        = auto()  # LLM 压缩：用 LLM 压缩历史再注入
    RETRIEVAL_AUGMENTED  = auto()  # 检索增强：从 Memory 检索相关内容注入


class TokenCounter(Protocol):
    """Token 计数接口（可替换为 tiktoken 实现）"""
    def count(self, text: str) -> int: ...


class SimpleTokenCounter:
    """简单 token 估算（英文：4 char ≈ 1 token，中文：1 char ≈ 1.5 token）"""
    def count(self, text: str) -> int:
        if not text:
            return 0
        # 粗略估算：混合中英文
        chinese_chars = len(re.findall(r'[一-鿿]', text))
        other_chars = len(text) - chinese_chars
        return int(chinese_chars * 1.5 + other_chars / 4)


# 默认 token 计数器
_default_counter = SimpleTokenCounter()


class ContextConstructor:
    """动态上下文装配器

    职责：
      - 按策略动态装配 System / Task / Session 三级上下文
      - 控制上下文窗口大小，防止膨胀
      - 在信息完整性和注意力集中度之间做权衡

    三种装配策略：
      SLIDING_WINDOW       — 滑窗截断，简单但可能丢关键信息
      SUMMARIZATION        — LLM 压缩历史，信息密度高但需要额外 LLM 调用
      RETRIEVAL_AUGMENTED  — 从 Memory 检索相关上下文，精准但依赖检索质量
    """

    def __init__(
        self,
        strategy: ContextStrategy = ContextStrategy.SLIDING_WINDOW,
        max_tokens: int = 8000,            # 总上下文 token 上限
        system_ratio: float = 0.15,        # SystemContext 最多占 15%
        task_ratio: float = 0.50,          # TaskContext 最多占 50%
        session_ratio: float = 0.35,       # SessionContext 最多占 35%
        sliding_window_n: int = 5,         # 滑窗策略：保留最近 N 轮
        counter: TokenCounter | None = None,
    ):
        self.strategy = strategy
        self.max_tokens = max_tokens
        self.system_ratio = system_ratio
        self.task_ratio = task_ratio
        self.session_ratio = session_ratio
        self.sliding_window_n = sliding_window_n
        self.counter = counter or _default_counter

        # 策略实现映射
        self._strategies = {
            ContextStrategy.SLIDING_WINDOW: self._assemble_sliding_window,
            ContextStrategy.SUMMARIZATION: self._assemble_with_history,
            ContextStrategy.RETRIEVAL_AUGMENTED: self._assemble_with_retrieval,
        }

        # 历史压缩回调（由外部注入 LLM 压缩函数）
        self._summarizer: Any = None

    def set_summarizer(self, fn: Any) -> None:
        """注入 LLM 压缩函数：fn(history_text: str, max_tokens: int) -> str"""
        self._summarizer = fn

    # ── 主装配方法 ─────────────────────────────────────────────────

    def assemble(
        self,
        system: str = "",
        task: str = "",
        working_memory: str = "",
        episodic_hints: str = "",
        extra: dict[str, str] | None = None,
    ) -> str:
        """装配完整上下文

        Args:
            system: 系统角色定义（固定，不可压缩）
            task: 当前任务描述（feature_spec + 本轮 RAG）
            working_memory: WorkingMemory 提供的最近 N 轮摘要
            episodic_hints: EpisodicMemory 检索到的相似历史提示
            extra: 其他要注入的上下文块

        Returns:
            装配后的完整 prompt 文本
        """
        budget = {
            "system": int(self.max_tokens * self.system_ratio),
            "task": int(self.max_tokens * self.task_ratio),
            "session": int(self.max_tokens * self.session_ratio),
        }

        # SystemContext：截断到预算
        system_block = self._truncate(system, budget["system"])

        # TaskContext：优先级最高，尽可能保留
        task_block = self._truncate(task, budget["task"])

        # SessionContext：按策略装配
        session_text = self._strategies[self.strategy](
            working_memory=working_memory,
            episodic_hints=episodic_hints,
        )
        session_block = self._truncate(session_text, budget["session"])

        # 组装
        parts = []
        if system_block:
            parts.append(system_block)
        if task_block:
            parts.append(task_block)
        if session_block:
            parts.append(f"[Session Context]\n{session_block}")
        if extra:
            for k, v in extra.items():
                truncated = self._truncate(v, budget.get(k, 500))
                if truncated:
                    parts.append(f"[{k}]\n{truncated}")

        return "\n\n---\n\n".join(parts)

    def token_count(self, text: str) -> int:
        return self.counter.count(text)

    # ── 策略实现 ───────────────────────────────────────────────────

    def _assemble_sliding_window(self, working_memory: str, episodic_hints: str) -> str:
        """滑窗策略：直接拼接最近 N 轮的摘要"""
        if episodic_hints:
            return f"{working_memory}\n\n[Similar Historical Traces]\n{episodic_hints}"
        return working_memory

    def _assemble_with_history(self, working_memory: str, episodic_hints: str) -> str:
        """压缩策略：调用 LLM 压缩（如果有 summarizer）"""
        parts = []
        if working_memory:
            if self._summarizer:
                parts.append(self._summarizer(working_memory, 500))
            else:
                parts.append(working_memory)
        if episodic_hints:
            parts.append(f"[Relevant Past Experience]\n{episodic_hints}")
        return "\n\n".join(parts)

    def _assemble_with_retrieval(self, working_memory: str, episodic_hints: str) -> str:
        """检索增强策略：优先展示检索到的相关内容"""
        if episodic_hints:
            return (
                f"[Retrieved Relevant Context — Prioritize This]\n{episodic_hints}\n\n"
                f"[Recent Session — For Reference]\n{working_memory}"
            )
        return working_memory

    # ── 工具方法 ───────────────────────────────────────────────────

    def _truncate(self, text: str, max_tokens: int) -> str:
        """截断文本到 max_tokens 以内"""
        if not text or max_tokens <= 0:
            return ""
        if self.counter.count(text) <= max_tokens:
            return text
        # 粗暴截断：按比例截取前半部分
        ratio = max_tokens / self.counter.count(text)
        cutoff = int(len(text) * ratio * 0.9)  # 留 10% 余量
        return text[:cutoff] + "\n\n[...truncated...]"

    def tokens_used(self, *texts: str) -> int:
        """计算多段文本的总 token 数"""
        return sum(self.counter.count(t) for t in texts)

    def budget_breakdown(self, system: str, task: str, session: str) -> dict:
        """返回上下文预算分配明细"""
        return {
            "system_tokens": self.counter.count(system),
            "task_tokens": self.counter.count(task),
            "session_tokens": self.counter.count(session),
            "total_tokens": self.counter.count(system) + self.counter.count(task) + self.counter.count(session),
            "max_tokens": self.max_tokens,
            "utilization": (
                (self.counter.count(system) + self.counter.count(task) + self.counter.count(session))
                / self.max_tokens
            ),
        }
