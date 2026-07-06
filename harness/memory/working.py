"""
harness/memory/working.py — 工作记忆（会话内短期记忆）

职责：
  - 保留最近 K 轮的执行摘要
  - 缓冲区满时自动压缩旧轮次（合并摘要）
  - 为 ContextConstructor 提供 SessionContext 的 Working Memory 部分

设计：
  - 环形缓冲区，容量固定
  - 存储 StepTrace.summary() 的简短摘要
  - get_context() 以时间倒序输出（最近的在前）
"""

from __future__ import annotations

from dataclasses import dataclass, field

from harness.tracer import StepTrace


@dataclass
class WorkingMemoryEntry:
    """工作记忆中的单条记录"""
    iteration: int
    feature_name: str
    outcome: str           # PASS / FAIL / SKIP
    key_metrics: str       # 精简指标字符串，如 "IC=+0.12 t=2.1 zr=32%"
    summary: str           # StepTrace.summary() 合并后的摘要


class WorkingMemory:
    """会话内工作记忆

    保留最近 K 轮成败摘要。当缓冲区满时，压缩最旧的 N/2 条记录
    为一条合并摘要，释放空间。
    """

    def __init__(self, max_entries: int = 10, compress_at: int | None = None):
        """
        Args:
            max_entries: 最大保留条目数
            compress_at: 触发压缩的阈值（默认 = max_entries）
        """
        self.max_entries = max_entries
        self.compress_at = compress_at or max_entries
        self._entries: list[WorkingMemoryEntry] = []
        self._compressed_count: int = 0  # 已压缩的轮次数

    # ── 写入 ───────────────────────────────────────────────────────

    def add(self, iteration: int, feature_name: str, outcome: str,
            key_metrics: str = "", trace_summary: str = "") -> None:
        """添加一条执行记录"""
        entry = WorkingMemoryEntry(
            iteration=iteration,
            feature_name=feature_name,
            outcome=outcome,
            key_metrics=key_metrics,
            summary=trace_summary,
        )
        self._entries.append(entry)

        # 检查是否需要压缩
        if len(self._entries) > self.compress_at:
            self._compact()

    def add_from_traces(self, iteration: int, feature_name: str,
                        outcome: str, traces: list[StepTrace]) -> None:
        """从 StepTrace 列表构建记录"""
        key_metrics = ""
        trace_summary = " | ".join(t.summary() for t in traces[-5:])  # 最后 5 步

        # 尝试从 trace 中提取关键指标
        for t in traces:
            obs = t.observation
            if "IC=" in obs:
                # 提取 IC/t-stat/zero_ratio
                import re
                ic_match = re.search(r'IC=([+-]?\d+\.\d+)', obs)
                t_match = re.search(r't=([+-]?\d+\.\d+)', obs)
                zr_match = re.search(r'zero_ratio=(\d+\.?\d*)%?', obs)
                parts = []
                if ic_match: parts.append(f"IC={ic_match.group(1)}")
                if t_match: parts.append(f"t={t_match.group(1)}")
                if zr_match: parts.append(f"zr={zr_match.group(1)}%")
                key_metrics = " ".join(parts)
                break

        self.add(iteration, feature_name, outcome, key_metrics, trace_summary)

    # ── 压缩 ───────────────────────────────────────────────────────

    def _compact(self) -> None:
        """压缩最旧的 N/2 条记录为一条合并摘要"""
        n = max(len(self._entries) // 2, 2)
        old = self._entries[:n]
        self._entries = self._entries[n:]

        # 合并摘要
        features = [e.feature_name for e in old]
        outcomes = [e.outcome for e in old]
        pass_count = outcomes.count("PASS")
        fail_count = outcomes.count("FAIL")

        merged = WorkingMemoryEntry(
            iteration=0,  # 0 表示压缩记录
            feature_name=",".join(features[:5]) + ("..." if len(features) > 5 else ""),
            outcome=f"COMPRESSED({pass_count}P/{fail_count}F)",
            key_metrics="",
            summary=f"[Compressed {n} entries] "
                    f"Features: {', '.join(features[:3])}... "
                    f"Results: {pass_count} PASS, {fail_count} FAIL. "
                    f"Typical failure pattern: {self._extract_pattern(old)}",
        )
        self._entries.insert(0, merged)  # 压缩记录插到最前面
        self._compressed_count += n

    @staticmethod
    def _extract_pattern(entries: list[WorkingMemoryEntry]) -> str:
        """从一组记录中提取失败模式摘要"""
        fail_features = [e for e in entries if e.outcome == "FAIL"]
        if not fail_features:
            return "no failures"
        # 简单模式：统计最常见的失败关键词
        keywords = {}
        for e in fail_features:
            for kw in ["zero_ratio", "t_stat", "direction", "coverage"]:
                if kw in e.key_metrics.lower() or kw in e.summary.lower():
                    keywords[kw] = keywords.get(kw, 0) + 1
        if keywords:
            top = max(keywords, key=keywords.get)
            return f"most common issue: {top} ({keywords[top]}/{len(fail_features)} features)"
        return "no clear pattern"

    # ── 读取 ───────────────────────────────────────────────────────

    def get_context(self, n: int | None = None) -> str:
        """获取工作记忆上下文

        Args:
            n: 返回最近 N 条（默认全部）

        Returns:
            格式化的工作记忆文本，可直接注入 ContextConstructor
        """
        entries = self._entries[-n:] if n else self._entries
        if not entries:
            return ""

        lines = ["[Working Memory — Recent Iterations]"]
        if self._compressed_count > 0:
            lines.append(f"(+ {self._compressed_count} compressed earlier iterations)")

        for e in reversed(entries):  # 最近的在前面
            metrics_str = f" [{e.key_metrics}]" if e.key_metrics else ""
            lines.append(
                f"  Iter{e.iteration}: {e.feature_name} → {e.outcome}{metrics_str}"
            )

        return "\n".join(lines)

    def get_recent_failures(self, n: int = 3) -> list[WorkingMemoryEntry]:
        """获取最近 N 个失败的记录"""
        failures = [e for e in self._entries if e.outcome == "FAIL"]
        return failures[-n:]

    def get_recent_pass(self, n: int = 3) -> list[WorkingMemoryEntry]:
        """获取最近 N 个通过的记录"""
        passed = [e for e in self._entries if e.outcome == "PASS"]
        return passed[-n:]

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def pass_rate(self) -> float:
        total = sum(1 for e in self._entries if e.outcome in ("PASS", "FAIL"))
        if total == 0:
            return 0.0
        return sum(1 for e in self._entries if e.outcome == "PASS") / total

    def clear(self) -> None:
        self._entries.clear()
        self._compressed_count = 0
