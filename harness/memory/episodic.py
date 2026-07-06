"""
harness/memory/episodic.py — 情景记忆（跨会话轨迹存储与检索）

解决的问题（失控模式 #3）：
  31 轮成败轨迹写入 feature_history.jsonl 但从不检索复用。
  相似定义方式的特征反复踩同样的坑。
  人工分析 history 才能发现模式（"guidance_revision_direction 经过 3 代迭代"）。

Harness 解法：
  每轮结束后整条轨迹编码存入库 → 新特征生成前检索最相似的 3 条历史轨迹 →
  如果 3 条中有 2 条因为零值率集中型被 G2 拦截 →
  ContextConstructor 自动注入："注意：类似定义方式历史上零值率偏高，
  考虑使用更细粒度的打分区间"

设计：
  - 编码：复用 BGE-M3（与 SemanticMemory 共用 encoder）
  - 存储：FAISS IndexFlatIP + metadata parquet
  - 检索：输入 feature_spec 的 definition 文本 → top-K 相似历史轨迹
  - 增量：支持运行中动态添加新轨迹
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class EpisodeRecord:
    """一条完整的跨会话轨迹记录"""
    episode_id: str                           # 唯一标识
    feature_name: str
    definition: str                           # feature_spec.definition
    retrieval_query: str                      # 用于检索的 query
    condition_scope: dict = field(default_factory=dict)
    # 结果
    outcome: str = ""                         # PASS / FAIL / SKIP
    ic: float = 0.0
    t_stat: float = 0.0
    zero_ratio: float = 0.0
    direction_consistency: float = 0.0
    # 失败详情
    failures: list[str] = field(default_factory=list)  # G1/G2/G3/G4 失败项
    diagnosis_root_cause: str = ""            # DiagnosisAgent 的根因
    diagnosis_fix: str = ""                   # DiagnosisAgent 的修复建议
    # 元数据
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    iteration: int = 0

    @property
    def summary_text(self) -> str:
        """用于编码的摘要文本——聚焦于特征定义和失败模式"""
        parts = [
            f"Feature: {self.feature_name}",
            f"Definition: {self.definition}",
            f"Outcome: {self.outcome}",
        ]
        if self.outcome == "FAIL":
            parts.append(f"Failures: {', '.join(self.failures)}")
            if self.diagnosis_root_cause:
                parts.append(f"Root Cause: {self.diagnosis_root_cause}")
            if self.diagnosis_fix:
                parts.append(f"Fix: {self.diagnosis_fix}")
        else:
            parts.append(f"IC={self.ic:+.4f} t={self.t_stat:+.3f}")
        return " | ".join(parts)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EpisodeRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class EpisodicMemory:
    """跨会话情景记忆

    职责：
      - 存储完整执行轨迹（spec → extraction → val → gov → diagnosis → outcome）
      - 检索历史上类似特征的轨迹
      - 为 ContextConstructor 提供 "相似任务的历史教训"

    注意：
      - 需要 BGE-M3 encoder 已加载（外部注入，与 SemanticMemory 共用）
      - 索引在初始化时从已有数据构建，支持增量更新
    """

    def __init__(
        self,
        store_dir: Path | str,
        encoder: Any = None,                   # BGE-M3 encoder（外部注入）
        embedding_dim: int = 1024,             # BGE-M3 输出维度
        max_records: int = 1000,               # 最大存储记录数
    ):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._records_path = self.store_dir / "episodic_records.jsonl"
        self._index_path = self.store_dir / "episodic_index.npy"
        self._meta_path = self.store_dir / "episodic_meta.json"

        self.encoder = encoder
        self.embedding_dim = embedding_dim
        self.max_records = max_records

        # 内存中的数据结构
        self._records: list[EpisodeRecord] = []
        self._embeddings: np.ndarray | None = None  # [N, dim]

        # 加载已有数据
        self._load()

    # ── 存储 ───────────────────────────────────────────────────────

    def store(self, record: EpisodeRecord) -> None:
        """存储一条轨迹（如果 encoder 可用则同时更新索引）"""
        self._records.append(record)

        # 持久化到 JSONL
        with open(self._records_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

        # 编码并更新索引
        if self.encoder is not None:
            try:
                vec = self._encode(record.summary_text)
                if self._embeddings is None:
                    self._embeddings = vec.reshape(1, -1)
                else:
                    self._embeddings = np.vstack([self._embeddings, vec])
                self._save_index()
            except Exception:
                pass  # 编码失败不影响存储

        # 超过最大记录数时删除最旧的
        if len(self._records) > self.max_records:
            self._records = self._records[-self.max_records:]
            if self._embeddings is not None and len(self._embeddings) > self.max_records:
                self._embeddings = self._embeddings[-self.max_records:]

    def store_from_iteration(
        self,
        feature_name: str,
        definition: str,
        retrieval_query: str,
        condition_scope: dict,
        outcome: str,
        ic: float, t_stat: float, zero_ratio: float,
        direction_consistency: float = 0.0,
        failures: list[str] | None = None,
        diagnosis_root_cause: str = "",
        diagnosis_fix: str = "",
        iteration: int = 0,
    ) -> EpisodeRecord:
        """便捷方法：从迭代结果构建并存储轨迹"""
        record = EpisodeRecord(
            episode_id=f"{feature_name}_{int(time.time())}",
            feature_name=feature_name,
            definition=definition,
            retrieval_query=retrieval_query,
            condition_scope=condition_scope,
            outcome=outcome,
            ic=ic, t_stat=t_stat, zero_ratio=zero_ratio,
            direction_consistency=direction_consistency,
            failures=failures or [],
            diagnosis_root_cause=diagnosis_root_cause,
            diagnosis_fix=diagnosis_fix,
            iteration=iteration,
        )
        self.store(record)
        return record

    # ── 检索 ───────────────────────────────────────────────────────

    def retrieve_similar(self, query_text: str, k: int = 3) -> list[EpisodeRecord]:
        """检索与 query 最相似的 K 条历史轨迹

        Args:
            query_text: 查询文本（通常是 feature_spec.definition）
            k: 返回数量

        Returns:
            按相似度排序的历史轨迹列表
        """
        if self.encoder is None or self._embeddings is None or len(self._records) == 0:
            # 无 encoder 或无索引时，退化为关键词匹配
            return self._keyword_search(query_text, k)

        try:
            q_vec = self._encode(query_text)
            q_vec = q_vec.reshape(1, -1)
            # 余弦相似度（向量已归一化时等价于内积）
            scores = np.dot(self._embeddings, q_vec.T).flatten()
            top_k = min(k, len(scores))
            top_indices = np.argsort(scores)[-top_k:][::-1]
            return [self._records[i] for i in top_indices if scores[i] > 0.3]
        except Exception:
            return self._keyword_search(query_text, k)

    def retrieve_by_pattern(self, failure_type: str, k: int = 3) -> list[EpisodeRecord]:
        """检索特定失败模式的轨迹

        Args:
            failure_type: G2_zero_ratio / G3_t_stat / G4_direction 等
            k: 返回数量
        """
        matches = [
            r for r in self._records
            if r.outcome == "FAIL" and any(failure_type in f for f in r.failures)
        ]
        return matches[-k:]

    def retrieve_by_outcome(self, outcome: str = "PASS", k: int = 3) -> list[EpisodeRecord]:
        """检索特定结果的轨迹"""
        matches = [r for r in self._records if r.outcome == outcome]
        return matches[-k:]

    # ── 统计 ───────────────────────────────────────────────────────

    def failure_pattern_stats(self, n_recent: int = 20) -> dict:
        """最近 N 条记录的失败模式统计"""
        recent = self._records[-n_recent:]
        stats = {"total": len(recent), "pass": 0, "fail": 0}
        failure_counts: dict[str, int] = {}

        for r in recent:
            if r.outcome == "PASS":
                stats["pass"] += 1
            elif r.outcome == "FAIL":
                stats["fail"] += 1
                for f in r.failures:
                    key = f.split(":")[0] if ":" in f else f[:20]
                    failure_counts[key] = failure_counts.get(key, 0) + 1

        stats["top_failure_types"] = sorted(
            failure_counts.items(), key=lambda x: x[1], reverse=True
        )[:3]
        stats["pass_rate"] = stats["pass"] / stats["total"] if stats["total"] > 0 else 0
        return stats

    def generate_hint(self, query_text: str, k: int = 3) -> str:
        """检索相似轨迹并生成可注入 ContextConstructor 的提示

        这是 EpisodicMemory 的核心价值：
        "注意：类似定义方式历史上零值率偏高，考虑使用更细粒度的打分区间"
        """
        similar = self.retrieve_similar(query_text, k)
        if not similar:
            return ""

        lines = ["[Episodic Memory — Lessons from Similar Past Features]"]

        # 统计相似特征的失败模式
        fail_count = sum(1 for r in similar if r.outcome == "FAIL")
        g2_fail = sum(1 for r in similar if any("G2" in f for f in r.failures))
        g3_fail = sum(1 for r in similar if any("G3" in f for f in r.failures))

        if fail_count >= 2:
            lines.append(f"WARNING: {fail_count}/{len(similar)} similar past features FAILED.")
            if g2_fail >= 2:
                lines.append("-> Common issue: High zero-ratio. Consider finer-grained scoring intervals.")
            if g3_fail >= 2:
                lines.append("-> Common issue: Weak signal (low |t-stat|). Consider narrowing condition_scope.")

        for r in similar:
            icon = "[PASS]" if r.outcome == "PASS" else "[FAIL]"
            lines.append(f"  {icon} {r.feature_name}: {r.outcome} "
                        f"(IC={r.ic:+.3f}, zr={r.zero_ratio:.0%})")
            if r.diagnosis_fix:
                lines.append(f"     Fix applied: {r.diagnosis_fix[:120]}")

        return "\n".join(lines)

    # ── 内部方法 ───────────────────────────────────────────────────

    def _encode(self, text: str) -> np.ndarray:
        """用 BGE-M3 编码文本（平均池化）"""
        if self.encoder is None:
            raise RuntimeError("No encoder configured for EpisodicMemory")

        import torch
        with torch.no_grad():
            inputs = self.encoder["tokenizer"](
                text, return_tensors="pt", truncation=True,
                max_length=512, padding=True,
            )
            # 移到 GPU（如果可用）
            device = next(self.encoder["model"].parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = self.encoder["model"](**inputs)
            # 平均池化
            vec = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            # 归一化（用于内积 = 余弦相似度）
            vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-8)
            return vec[0]

    def _keyword_search(self, query: str, k: int = 3) -> list[EpisodeRecord]:
        """回退方案：基于关键词匹配（无 encoder 时使用）"""
        keywords = set(query.lower().split())
        scored = []
        for r in self._records:
            text = (r.definition + " " + r.feature_name + " " +
                    " ".join(r.failures)).lower()
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:k]]

    def _save_index(self) -> None:
        """保存 embeddings 到磁盘"""
        if self._embeddings is not None:
            np.save(self._index_path, self._embeddings)
            meta = {"count": len(self._records), "dim": self.embedding_dim,
                    "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}
            with open(self._meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f)

    def _load(self) -> None:
        """从磁盘加载已有数据"""
        # 加载 records
        if self._records_path.exists():
            records = []
            with open(self._records_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(EpisodeRecord.from_dict(json.loads(line)))
                        except (json.JSONDecodeError, KeyError):
                            continue
            # 只保留最近的 max_records
            self._records = records[-self.max_records:]

        # 加载 embeddings
        if self._index_path.exists():
            try:
                self._embeddings = np.load(self._index_path)
                # 如果 records 被截断，对齐 embeddings
                if len(self._embeddings) > len(self._records):
                    self._embeddings = self._embeddings[-len(self._records):]
            except Exception:
                self._embeddings = None

    @property
    def record_count(self) -> int:
        return len(self._records)

    @property
    def pass_rate(self) -> float:
        total = sum(1 for r in self._records if r.outcome in ("PASS", "FAIL"))
        if total == 0:
            return 0.0
        return sum(1 for r in self._records if r.outcome == "PASS") / total
