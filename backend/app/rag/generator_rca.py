"""
Générateur de réponses utilisant Ollama LLM pour les résultats de retrieval.

Ce module implémente une génération contrôlée :
- basée uniquement sur le contexte récupéré
- avec une vérification post-génération (Self-RAG) pour limiter les hallucinations
"""

from __future__ import annotations

from typing import List, Dict, Any
import re

# Import de la configuration LLM
from ..config import (
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
)

# Import Ollama (LLM local)
try:
    from llama_index.llms.ollama import Ollama
    _LLM_AVAILABLE = True
except ImportError:
    _LLM_AVAILABLE = False


# ---------------------------------------------------------------------
# Self-RAG post-génération (grounding check)
# ---------------------------------------------------------------------

_MIN_OVERLAP_TOKENS = 3  # seuil minimal de mots communs réponse <--> contexte


def _is_answer_grounded(answer: str, results: List[Dict[str, Any]]) -> bool:
    """
    Vérifie que la réponse générée est bien ancrée dans le contexte récupéré.

    Heuristique simple, déterministe et explicable :
    - on extrait les tokens significatifs du contexte
    - on vérifie qu'un minimum de ces tokens apparaît dans la réponse

    Objectif :
    - éviter les réponses génériques ou hors base documentaire
    - renforcer le comportement "answer only if grounded"
    """

    if not answer or not results:
        return False

    # Concaténer le texte du contexte
    context_text = " ".join(r.get("text", "") or "" for r in results)
    if not context_text:
        return False

    # Normalisation simple
    def tokenize(text: str) -> set[str]:
        text = text.lower()
        text = re.sub(r"[^a-zà-ÿ0-9\s]", " ", text)
        tokens = {t for t in text.split() if len(t) > 3}
        return tokens

    context_tokens = tokenize(context_text)
    answer_tokens = tokenize(answer)

    if not context_tokens or not answer_tokens:
        return False

    overlap = context_tokens.intersection(answer_tokens)

    return len(overlap) >= _MIN_OVERLAP_TOKENS


# ---------------------------------------------------------------------
# Génération principale
# ---------------------------------------------------------------------

def generate_answer(query: str, results: List[Dict[str, Any]]) -> str:
    """
    Génère une réponse en utilisant un LLM local (Ollama) à partir
    des résultats de retrieval.

    La génération est strictement contrainte au contexte fourni.
    Une vérification post-génération empêche les réponses non ancrées
    dans les documents.
    """

    # Vérification de la disponibilité du LLM
    if not _LLM_AVAILABLE:
        raise ImportError(
            "Le module llama_index.llms.ollama n'est pas disponible. "
            "Veuillez installer le package requis avec: "
            "pip install llama-index-llms-ollama"
        )

    # Aucun contexte disponible
    if not results:
        return (
            "Je suis désolé, mais je ne peux pas répondre à cette question "
            "avec les documents disponibles."
        )

    # ------------------------------------------------------------------
    # Construction du contexte
    # ------------------------------------------------------------------
    context_parts: List[str] = []

    for res in results:
        text: str = res.get("text", "") or ""
        metadata: Dict[str, Any] = res.get("metadata", {}) or {}
        source: str = metadata.get("file_name", metadata.get("source", "inconnu"))

        snippet = text.strip().replace("\n", " ")
        snippet = re.sub(r"\s+", " ", snippet)
        snippet = snippet[:5000]

        context_parts.append(
            f"Source: {source}\nContenu: {snippet}"
        )

    context_str = "\n\n".join(context_parts)

    # ------------------------------------------------------------------
    # Configuration LLM
    # ------------------------------------------------------------------
    base_url = LLM_BASE_URL or "http://localhost:11434"
    model_name = LLM_MODEL or "llama3.1:latest"
    temperature = LLM_TEMPERATURE or 0
    timeout = LLM_TIMEOUT or 180.0

    try:
        llm = Ollama(
            model=model_name,
            base_url=base_url,
            temperature=temperature,
            request_timeout=timeout,
        )

        system_prompt = (
        "Tu es un assistant expert en Root Cause Analysis (RCA).\n"
        "Ton rôle est d'analyser le code, les logs, les erreurs, et les messages système.\n"
        "RÈGLES STRICTES :\n"
        "1. Tu dois identifier la cause exacte de l'erreur donnée.\n"
        "2. Utilise UNIQUEMENT les informations présentes dans le contexte.\n"
        "3. Si un problème provient d'une variable None, d'une mauvaise initialisation ou d'un appel incorrect, tu dois l'expliquer clairement.\n"
        "4. Fournis :\n"
        "   - un résumé du problème\n"
        "   - les causes racines exactes\n"
        "   - les actions correctives immédiates\n"
        "   - les actions préventives\n"
        )

        prompt = (
            f"{system_prompt}\n\n"
            f"CONTEXTE:\n{context_str}\n\n"
            f"QUESTION: {query.strip()}\n\n"
            f"RÉPONSE:"
        )

        response = llm.complete(prompt)
        answer = str(response).strip()

        if not answer:
            raise ValueError("Le LLM a retourné une réponse vide")

        # ------------------------------------------------------------------
        # Self-RAG post-génération : grounding check
        # ------------------------------------------------------------------
        if not _is_answer_grounded(answer, results):
            return (
                "Les documents disponibles ne permettent pas de répondre "
                "de manière fiable à cette question."
            )

        return answer

    except Exception as e:
        error_msg = (
            f"Erreur lors de la génération avec le LLM: {str(e)}\n"
            f"Vérifiez que:\n"
            f"1. Ollama est actif sur {base_url}\n"
            f"2. Le modèle '{model_name}' est disponible\n"
            f"3. Le service Ollama fonctionne correctement"
        )
        raise Exception(error_msg) from e
