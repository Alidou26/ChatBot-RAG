from __future__ import annotations

import json
from typing import Dict, Any, List

from .retriever_chroma import query_chroma
from .generator import generate_answer


class JinaRAG:
    """
    Orchestrateur principal du pipeline RAG.

    Rôle :
    - Interroger ChromaDB (retrieval)
    - Évaluer la qualité du contexte (Self-RAG – gating)
    - Générer une réponse uniquement si le contexte est suffisant
    - Formater les sources pour le frontend (Streamlit / API)
    """

    # ------------------------------------------------------------------
    # Heuristiques simples de Self-RAG (sans LLM, déterministes)
    # ------------------------------------------------------------------
    MIN_CHUNKS = 2            # nombre minimal de chunks requis
    MAX_MEAN_SCORE = 0.85     # score moyen maximal autorisé (distance Chroma)
    MIN_TOTAL_CHARS = 300     # taille minimale cumulée du contexte

    def _is_context_sufficient(
        self,
        question: str,
        results: List[Dict[str, Any]]
    ) -> bool:
        """
        Vérifie si le contexte récupéré est suffisamment pertinent
        pour autoriser une génération par le LLM.

        Objectif :
        - Éviter les réponses hors contexte
        - Éviter les hallucinations
        - Forcer un comportement "je ne sais pas" quand nécessaire
        """

        if len(results) < self.MIN_CHUNKS:
            return False

        scores = [float(r.get("score", 1.0)) for r in results]
        mean_score = sum(scores) / len(scores)

        total_chars = sum(len(r.get("text", "") or "") for r in results)

        if mean_score > self.MAX_MEAN_SCORE:
            return False

        if total_chars < self.MIN_TOTAL_CHARS:
            return False

        return True

    # ------------------------------------------------------------------
    # Méthode principale appelée par l’API
    # ------------------------------------------------------------------
    def query(
        self,
        question: str,
        top_k: int = 5,
        **filters: Any
    ) -> Dict[str, Any]:
        """
        Pipeline complet RAG :
        1. Retrieval via ChromaDB
        2. Vérification Self-RAG (gating)
        3. Génération contrôlée
        4. Formatage des sources
        """

        # Sécurité basique : question vide
        if not question or not question.strip():
            return {"answer": "Question vide.", "sources": []}

        # ------------------------------------------------------------------
        # 1) Retrieval : recherche sémantique dans ChromaDB
        # ------------------------------------------------------------------
        results = query_chroma(question, k=top_k, **filters)

        # ------------------------------------------------------------------
        # 2) Self-RAG : décider si on a le droit de générer
        # ------------------------------------------------------------------
        if not self._is_context_sufficient(question, results):
            return {
                "answer": (
                    "Les documents disponibles ne permettent pas de répondre "
                    "de manière fiable à cette question."
                ),
                "sources": []
            }

        # ------------------------------------------------------------------
        # 3) Génération LLM (contexte jugé suffisant)
        # ------------------------------------------------------------------
        answer = generate_answer(question, results)

        # ------------------------------------------------------------------
        # 4) Formatage des sources pour le frontend
        # ------------------------------------------------------------------
        formatted_sources: List[Dict[str, Any]] = []

        for res in results:
            meta = res.get("metadata", {}) or {}
            text = ""

            # Cas standard : texte directement retourné par Chroma
            if res.get("text"):
                text = res["text"]

            # Cas fallback : texte stocké dans _node_content (LlamaIndex)
            elif "_node_content" in meta:
                try:
                    node_data = json.loads(meta["_node_content"])
                    text = node_data.get("text", "") or ""
                except Exception:
                    text = ""

            # Normalisation des métadonnées (Streamlit-safe)
            formatted_meta: Dict[str, Any] = {
                k: (v if isinstance(v, (str, int, float)) else str(v))
                for k, v in meta.items()
            }

            formatted_meta["score"] = f"{float(res.get('score', 0.0)):.3f}"
            formatted_meta["text"] = text

            formatted_sources.append(formatted_meta)

        return {
            "answer": answer,
            "sources": formatted_sources
        }
