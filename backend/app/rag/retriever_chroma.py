"""
Retriever module for querying ChromaDB vector store.

This module provides:
- Standard single-query retrieval from ChromaDB
- RAG-Fusion retrieval (multi-query expansion + Reciprocal Rank Fusion)
"""

from __future__ import annotations

import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

from .storage import get_storage
from .embedder import get_embedder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# RAG-Fusion helpers
# ---------------------------------------------------------------------

_RRF_K: int = int(os.getenv("RAG_FUSION_RRF_K", "60"))
_FUSION_ENABLED_DEFAULT: bool = os.getenv("RAG_FUSION_ENABLED", "1").strip() not in {"0", "false", "False", "no", "NO"}


def _normalize_text_for_key(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t[:1000]


def _make_result_key(res: Dict[str, Any]) -> str:
    rid = res.get("id")
    if isinstance(rid, str) and rid:
        return f"id:{rid}"

    meta = res.get("metadata", {}) or {}
    source = str(meta.get("source") or meta.get("file_path") or meta.get("file_name") or "")
    text = _normalize_text_for_key(res.get("text", "") or "")
    return f"st:{source}::{text}"


def _generate_fusion_queries(query: str, max_queries: int = 4) -> List[str]:
    """
    Deterministic query expansion (no LLM) to keep the system self-contained.
    Produces a small set of alternative phrasings to improve recall.
    """
    q = (query or "").strip()
    if not q:
        return []

    base = q
    variants: List[str] = [base]

    # Light cleaning
    q_compact = re.sub(r"\s+", " ", base)

    # French-oriented expansions (works fine for English too)
    variants.append(f"résumé {q_compact}")
    variants.append(f"détails {q_compact}")

    # If it looks like a troubleshooting question, add an "error" focused variant
    if re.search(r"\b(erreur|error|exception|traceback|stacktrace|failed|échec)\b", q_compact, flags=re.IGNORECASE):
        variants.append(f"cause racine {q_compact}")
    else:
        variants.append(f"expliquer {q_compact}")

    # Deduplicate while preserving order
    seen = set()
    out: List[str] = []
    for v in variants:
        vv = v.strip()
        if vv and vv not in seen:
            out.append(vv)
            seen.add(vv)

    return out[: max(1, max_queries)]


def _single_query_chroma(
    query: str,
    k: int,
    where_filter: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Execute one Chroma query and format results in the same shape used by the API.
    """
    embedder = get_embedder()
    storage = get_storage()

    try:
        query_embedding = embedder.embed_query(query)
    except Exception as e:
        logger.error(f"Failed to embed query: {e}")
        raise

    try:
        raw = storage.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            where=where_filter if where_filter else None,
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise

    formatted: List[Dict[str, Any]] = []
    if raw and raw.get("documents"):
        documents = raw["documents"][0] or []
        metadatas = (raw.get("metadatas") or [[]])[0] or []
        distances = (raw.get("distances") or [[]])[0] or []
        ids = (raw.get("ids") or [[]])[0] or []

        for i in range(len(documents)):
            formatted.append(
                {
                    "id": ids[i] if i < len(ids) else None,
                    "text": documents[i],
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                    "score": float(distances[i]) if i < len(distances) else 0.0,
                }
            )

    return formatted


def _rrf_fuse(
    per_query_results: List[List[Dict[str, Any]]],
    final_k: int,
    rrf_k: int = _RRF_K,
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion.
    We keep the output schema unchanged. We set 'score' to a fused score where
    smaller is better (negative RRF sum) to remain compatible with existing sorting
    assumptions in the rest of the codebase.
    """
    if not per_query_results:
        return []

    fused: Dict[str, Dict[str, Any]] = {}
    rrf_scores: Dict[str, float] = {}

    for results in per_query_results:
        for rank, res in enumerate(results, start=1):
            key = _make_result_key(res)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (rrf_k + rank))

            # Keep one representative result (first seen), but try to preserve richer metadata
            if key not in fused:
                fused[key] = dict(res)
            else:
                # Merge metadata if missing in the stored one
                cur_meta = fused[key].get("metadata") or {}
                new_meta = res.get("metadata") or {}
                if isinstance(cur_meta, dict) and isinstance(new_meta, dict):
                    merged = dict(cur_meta)
                    for mk, mv in new_meta.items():
                        if mk not in merged or merged.get(mk) in (None, "", 0):
                            merged[mk] = mv
                    fused[key]["metadata"] = merged

    # Convert to list and attach fused score
    out: List[Dict[str, Any]] = []
    for key, res in fused.items():
        rrf = rrf_scores.get(key, 0.0)
        res_out = dict(res)
        # Make "smaller is better" to match the existing distance semantics.
        res_out["score"] = -float(rrf)
        out.append(res_out)

    out.sort(key=lambda x: float(x.get("score", 0.0)))
    return out[: max(0, int(final_k))]


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def query_chroma(
    query: str,
    k: int = 5,
    project_id: Optional[str] = None,
    content_type: Optional[str] = None,
    language: Optional[str] = None,
    use_fusion: Optional[bool] = None,
    fusion_max_queries: int = 4,
    fusion_overfetch_multiplier: int = 2,
    **filters,
) -> List[Dict[str, Any]]:
    """
    Query ChromaDB for similar documents.

    Supports standard retrieval and RAG-Fusion.

    Args:
        query: Query string
        k: Number of results to return
        project_id: Filter by project ID
        content_type: Filter by content type ("code" or "text")
        language: Filter by language
        use_fusion: If True, enable multi-query expansion + RRF fusion.
                    If None, uses env RAG_FUSION_ENABLED (default enabled).
        fusion_max_queries: Maximum number of expanded queries (including the original).
        fusion_overfetch_multiplier: Per-query retrieval size multiplier.
        **filters: Additional metadata filters

    Returns:
        List of result dictionaries with text, metadata, and score
    """
    q = (query or "").strip()
    if not q:
        return []

    # Build where filter (same behavior as before)
    where_filter: Dict[str, Any] = {}
    if project_id:
        where_filter["project_id"] = project_id
    if content_type:
        where_filter["content_type"] = content_type
    if language:
        where_filter["language"] = language
    where_filter.update(filters)

    enabled = _FUSION_ENABLED_DEFAULT if use_fusion is None else bool(use_fusion)

    if not enabled:
        logger.info(f"Querying (single): '{q}' (k={k})")
        results = _single_query_chroma(q, k=k, where_filter=where_filter if where_filter else None)
        logger.info(f"✓ Found {len(results)} results")
        return results

    # RAG-Fusion path
    expanded = _generate_fusion_queries(q, max_queries=fusion_max_queries)
    if not expanded:
        return []

    per_query_k = max(1, int(k) * max(1, int(fusion_overfetch_multiplier)))

    logger.info(
        f"Querying (fusion): '{q}' (final_k={k}, queries={len(expanded)}, per_query_k={per_query_k})"
    )

    per_query_results: List[List[Dict[str, Any]]] = []
    for sub_q in expanded:
        try:
            sub_results = _single_query_chroma(sub_q, k=per_query_k, where_filter=where_filter if where_filter else None)
            per_query_results.append(sub_results)
        except Exception as e:
            logger.warning(f"Fusion sub-query failed: '{sub_q}': {e}")

    fused = _rrf_fuse(per_query_results, final_k=k, rrf_k=_RRF_K)
    logger.info(f"✓ Found {len(fused)} results (fusion)")
    return fused


def query_by_project(
    query: str,
    project_id: str,
    k: int = 5,
) -> List[Dict[str, Any]]:
    """Query documents from a specific project only."""
    return query_chroma(query, k=k, project_id=project_id)


def query_code_only(
    query: str,
    k: int = 5,
    language: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Query only code chunks."""
    return query_chroma(query, k=k, content_type="code", language=language)


def query_docs_only(
    query: str,
    k: int = 5,
) -> List[Dict[str, Any]]:
    """Query only document chunks (PDF, DOCX, PPTX, etc.)."""
    return query_chroma(query, k=k, content_type="text")


def get_context_for_generation(
    query: str,
    k: int = 5,
    **filters,
) -> str:
    """
    Get formatted context string for LLM generation.
    """
    results = query_chroma(query, k=k, **filters)

    if not results:
        logger.warning("No results found for query")
        return ""

    context_parts: List[str] = []
    for i, result in enumerate(results, 1):
        metadata = result.get("metadata", {}) or {}
        source = metadata.get("file_name", "unknown")
        score = float(result.get("score", 0.0))
        context_parts.append(
            f"[Context chunk {i} - Source: {source}, Score: {score:.3f}]\n"
            f"{result.get('text', '')}\n"
        )

    return "\n".join(context_parts)
 