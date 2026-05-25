from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from schema import TableSchema


@dataclass(frozen=True)
class Relationship:
    left_table: str
    left_column: str
    right_table: str
    right_column: str
    score: float
    reason: str


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / denom


def _is_id_like(col: str) -> bool:
    c = col.lower().strip()
    return c == "id" or c.endswith("_id") or c.endswith("id")


def _elbow_eps(sorted_kdist: np.ndarray) -> float:
    """Find eps at the knee of the sorted k-distance curve via max perpendicular distance from diagonal."""
    n = len(sorted_kdist)
    if n < 3:
        return float(sorted_kdist[-1])
    x = np.arange(n, dtype=float)
    y = sorted_kdist.astype(float)
    x1, y1, x2, y2 = x[0], y[0], x[-1], y[-1]
    dx, dy = x2 - x1, y2 - y1
    denom = np.sqrt(dx * dx + dy * dy)
    if denom < 1e-10:
        return float(y[-1])
    perp = np.abs(dy * (x - x1) - dx * (y - y1)) / denom
    return float(y[int(np.argmax(perp))])


def build_column_corpus(schema: Dict[str, TableSchema]) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Returns:
      - texts: list of natural language strings describing each column
      - keys: (table, column) pairs aligned with texts
    """
    texts: List[str] = []
    keys: List[Tuple[str, str]] = []
    for tname, ts in schema.items():
        for c in ts.columns:
            sample_hint = ""
            if c.sample_values:
                sample_hint = " examples: " + ", ".join([s for s in c.sample_values if s][:3])
            texts.append(f"table {tname}, column {c.name}, type {c.pandas_dtype}.{sample_hint}")
            keys.append((tname, c.name))
    return texts, keys


def infer_relationships(
    schema: Dict[str, TableSchema],
    embedder,
) -> List[Relationship]:
    """
    Relationship inference:
    - FAISS embeddings as base for neighbor search and scoring
    - k-distance elbow (kneedle method) auto-tunes DBSCAN eps
    - DBSCAN clusters semantically similar columns across tables
    - Hard cap at top 20 results by score
    """
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError("FAISS is required for relationship inference. Install `faiss-cpu`.") from e

    from sklearn.cluster import DBSCAN

    texts, keys = build_column_corpus(schema)
    if len(keys) < 2:
        return []

    emb = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    emb = emb.astype("float32")
    emb = _l2_normalize(emb)

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    # Auto-tune eps via k-distance elbow using FAISS nearest-neighbor distances
    k_nn = min(5, len(keys) - 1)
    faiss_scores, _ = index.search(emb, k_nn + 1)  # +1 to skip self at position 0
    k_elbow = min(4, k_nn)
    # Cosine distance = 1 - cosine_similarity (embeddings are L2-normalized so IP = cosine sim)
    kdist_col = (1.0 - faiss_scores[:, k_elbow]).clip(0.0, 2.0)
    sorted_kdist = np.sort(kdist_col)
    eps = max(_elbow_eps(sorted_kdist), 0.01)

    # DBSCAN clustering on embedding space
    labels = DBSCAN(eps=eps, min_samples=2, metric="cosine").fit_predict(emb)

    # Generate candidate pairs from within-cluster, cross-table column pairs
    clusters: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        if label >= 0:  # -1 = noise
            clusters[label].append(idx)

    rels: List[Relationship] = []
    seen: set = set()

    for members in clusters.values():
        for pi, i in enumerate(members):
            t1, c1 = keys[i]
            for j in members[pi + 1:]:
                t2, c2 = keys[j]
                if t1 == t2:
                    continue

                s = float(np.dot(emb[i], emb[j]))  # cosine similarity (normalized vectors)
                reason_parts = ["semantic similarity"]

                if _is_id_like(c1) and _is_id_like(c2) and c1.lower() == c2.lower():
                    s = min(0.999, s + 0.05)
                    reason_parts.append("id-like columns")
                if c1.lower() == c2.lower():
                    s = min(0.999, s + 0.07)
                    reason_parts.append("same column name")
                if c1.lower() == f"{t2.lower()}_id" or c2.lower() == f"{t1.lower()}_id":
                    s = min(0.999, s + 0.07)
                    reason_parts.append("foreign-key naming pattern")

                a = (t1, c1, t2, c2)
                b = (t2, c2, t1, c1)
                if a in seen or b in seen:
                    continue
                seen.add(a)

                rels.append(
                    Relationship(
                        left_table=t1,
                        left_column=c1,
                        right_table=t2,
                        right_column=c2,
                        score=s,
                        reason=", ".join(reason_parts),
                    )
                )

    rels.sort(key=lambda r: r.score, reverse=True)
    if len(rels) > 1:
        scores_arr = np.sort([r.score for r in rels])  # ascending
        diffs = np.diff(scores_arr)
        elbow_idx = int(np.argmax(diffs))
        threshold = float(scores_arr[elbow_idx + 1])   # first score above the biggest jump
        rels = [
            r for r in rels
            if r.score >= threshold or r.left_column.lower() == r.right_column.lower()
        ]
    return rels[:20]


def format_relationships_for_prompt(relationships: Sequence[Relationship]) -> str:
    if not relationships:
        return "None detected."
    lines: List[str] = []
    for r in relationships:
        lines.append(
            f"- {r.left_table}.{r.left_column} ↔ {r.right_table}.{r.right_column} (score {r.score:.2f}; {r.reason})"
        )
    return "\n".join(lines)
