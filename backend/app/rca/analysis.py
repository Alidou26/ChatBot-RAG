"""
Root Cause Analysis – Chroma-only version

Analyse la cause racine d'un problème en se basant UNIQUEMENT
sur les données déjà indexées dans ChromaDB (issues, wiki,
code, fichiers de config, etc.).

Pipeline :
1. Récupération des chunks pertinents via query_chroma
2. Re-ranking "feature engineering" selon :
   - issue vs code vs autres docs
   - présence de mots-clés d'erreur / stacktrace
3. Génération d'une réponse structurée avec generate_answer.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
import re

from ..rag.retriever_chroma import query_chroma
from ..rag.generator_rca import generate_answer


# -------------------------------------------------------------------
# 1. Détection de features (feature engineering "maison")
# -------------------------------------------------------------------

ERROR_KEYWORDS = [
    "error", "erreur", "exception", "stack trace", "stacktrace",
    "traceback", "failed", "échec", "echec", "cannot", "crash",
    "timeout", "segmentation fault", "invalid", "unexpected"
]

CODE_EXTENSIONS = {".py", ".cpp", ".c", ".hpp", ".h", ".js", ".ts", ".java", ".cs", ".yml", ".yaml", ".json"}


def _has_error_signal(text: str) -> bool:
    """Retourne True si le chunk ressemble à un message d'erreur / stacktrace."""
    t = text.lower()
    return any(kw in t for kw in ERROR_KEYWORDS)


def _is_issue_chunk(metadata: Dict[str, Any]) -> bool:
    """Heuristique : chunk issu d'un fichier d'issue Redmine."""
    file_name = str(metadata.get("file_name", "")).lower()
    return file_name.startswith("issue_") or "issue" in file_name


def _is_code_chunk(metadata: Dict[str, Any]) -> bool:
    """Heuristique : chunk de code ou fichier de config."""
    content_type = str(metadata.get("content_type", "")).lower()
    file_type = str(metadata.get("file_type", "")).lower()
    file_name = str(metadata.get("file_name", "")).lower()

    if content_type == "code":
        return True

    if file_type in CODE_EXTENSIONS:
        return True

    # fallback : heuristique sur le nom du fichier
    return any(file_name.endswith(ext) for ext in CODE_EXTENSIONS)


def _rerank_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Re-rank des résultats en fonction de features métiers :

    - chunks issus d'issues : bonus
    - chunks contenant une erreur/stacktrace : bonus
    - chunks de code/config : léger bonus

    On part du score (distance) renvoyé par Chroma (plus petit = plus pertinent),
    et on applique des "bonus" (on diminue le score).
    """
    reranked = []

    for res in results:
        base_score = float(res.get("score", 0.0))
        text = res.get("text", "") or ""
        metadata = res.get("metadata", {}) or {}

        issue_flag = _is_issue_chunk(metadata)
        error_flag = _has_error_signal(text)
        code_flag = _is_code_chunk(metadata)

        # Bonus négatifs (on réduit la distance si feature intéressante)
        bonus = 0.0
        if issue_flag:
            bonus -= 0.20  # chunk lié à un ticket : très important
        if error_flag:
            bonus -= 0.15  # chunk avec stacktrace/message d'erreur
        if code_flag:
            bonus -= 0.05  # code associé

        new_score = base_score + bonus

        res_copy = dict(res)
        res_copy["score"] = new_score
        reranked.append(res_copy)

    # Tri par score croissant (plus petit = plus pertinent)
    reranked.sort(key=lambda x: x.get("score", 0.0))
    return reranked


# -------------------------------------------------------------------
# 2. Fonction principale : analyse_root_cause
# -------------------------------------------------------------------

def analyse_root_cause(
    issue_description: str,
    project_id: Optional[str] = None,
    top_k: int = 8,
) -> Dict[str, Any]:
    """
    Analyse la cause racine en utilisant UNIQUEMENT ChromaDB.

    Args:
        issue_description: description libre du problème (stacktrace,
                           message utilisateur, comportement observé…)
        project_id: (optionnel) pour filtrer sur un projet Redmine
                    déjà ingéré (ex: "solent-rca-test")
        top_k: nombre de chunks à utiliser pour l'analyse

    Returns:
        {
          "answer": str,
          "sources": [ { ...meta + text + score... }, ... ]
        }
    """
    description = (issue_description or "").strip()
    if not description:
        return {
            "answer": "Aucune description fournie.",
            "sources": []
        }

    # ----------------------------------------------------------
    # 1) Interrogation de ChromaDB
    #    -> on demande un peu plus de résultats pour reranking
    # ----------------------------------------------------------
    chroma_results = query_chroma(
        query=description,
        k=top_k * 2,
        project_id=project_id
    )

    if not chroma_results:
        return {
            "answer": (
                "Aucune information pertinente trouvée dans la base de "
                "connaissances pour ce problème. "
                "Assurez-vous que les tickets, fichiers et wiki ont bien "
                "été ingérés pour ce projet."
            ),
            "sources": []
        }

    # ----------------------------------------------------------
    # 2) Re-ranking "feature engineering"
    # ----------------------------------------------------------
    ranked_results = _rerank_results(chroma_results)[:top_k]

    # ----------------------------------------------------------
    # 3) Construction d'une requête orientée RCA + solutions
    # ----------------------------------------------------------
    # NOTE : aujourd'hui, generate_answer concatène surtout les passages
    # pertinents. Si vous pluggez plus tard un vrai LLM derrière,
    # ce prompt guidera la structure de la réponse.
    rca_query = (
        "Tu es un assistant spécialisé en Root Cause Analysis (RCA) pour "
        "des systèmes logiciels industriels.\n\n"
        "À partir des extraits de tickets, fichiers et code fournis dans le "
        "contexte, fais une analyse structurée du problème suivant.\n\n"
        "Pour ta réponse, fournis les sections suivantes :\n"
        "1. Résumé du problème (en une ou deux phrases)\n"
        "2. Causes racines probables (liste numérotée)\n"
        "3. Actions correctives concrètes (ce qu'il faut faire maintenant)\n"
        "4. Actions préventives / améliorations durables\n\n"
        f"Problème décrit par l'utilisateur : {description}"
    )

    # ----------------------------------------------------------
    # 4) Génération de la réponse à partir des chunks re-rankés
    # ----------------------------------------------------------
    answer = generate_answer(rca_query, ranked_results)

    # ----------------------------------------------------------
    # 5) Formatage des sources pour l'API/frontend
    # ----------------------------------------------------------
    formatted_sources: List[Dict[str, Any]] = []

    for res in ranked_results:
        meta = res.get("metadata", {}) or {}
        # On garde les méta + on ajoute le texte + le score
        formatted_meta: Dict[str, Any] = {
            k: (v if isinstance(v, (str, int, float)) else str(v))
            for k, v in meta.items()
        }
        formatted_meta["score"] = f"{float(res.get('score', 0.0)):.3f}"
        formatted_meta["text"] = res.get("text", "")

        formatted_sources.append(formatted_meta)

    return {
        "answer": answer,
        "sources": formatted_sources
    }
