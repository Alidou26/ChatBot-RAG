"""Paramètres de configuration pour le backend.

Ce module définit les constantes utilisées dans toute l’application,
avec des valeurs par défaut adaptées à un usage local.  Chaque
paramètre peut être surchargé via une variable d’environnement.

Les constantes sont exposées directement pour simplifier l’accès
depuis les autres modules.
"""

from __future__ import annotations

import os


# -----------------------------------------------------------------------------
# Large language model configuration
# -----------------------------------------------------------------------------
# Adresse du serveur LLM local. 
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "http://localhost:11434")

# Nom du modèle servi par l'endpoint LLM (Ollama).
LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen2.5:0.5b")

# Température par défaut pour l’échantillonnage du LLM
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", 0))

# Délai maximal (en s) pour les appels au LLM 
LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", 180))

# -----------------------------------------------------------------------------
# Miscellaneous
# -----------------------------------------------------------------------------
# Nombre maximal de passages retournés lors d’une recherche 
MAX_SEARCH_RESULTS: int = int(os.getenv("KNOWLEDGE_MAX_RESULTS", 5))

# -----------------------------------------------------------------------------
# Chroma and LlamaIndex configuration
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# New RAG indexing and retrieval configuration
# -----------------------------------------------------------------------------
# Dossier où ChromaDB persiste ses données.  ``CHROMA_DB_DIR`` est un alias pour compatibilité.
CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./chroma_db")
CHROMA_DB_DIR: str = CHROMA_DB_PATH  # alias for legacy code

# Nom de la collection Chroma utilisée pour stocker les embeddings.  Par défaut, ``redmine_projects``.
CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "redmine_projects")


# Modèle d'embedding (version locale via HuggingFace / SentenceTransformers)
EMBEDDING_MODEL = "jinaai/jina-embeddings-v3"

# Dimension de sortie du modèle d’embedding (Jina v3 : 1024 par défaut).
EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", 1024))

# Taille de lot pour l’encodage des textes 
EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", 32))


# Tailles des morceaux pour les nouveaux chunkers : code (petit) et texte (plus grand).  Le chevauchement indique le nombre de caractères à superposer.
CODE_CHUNK_SIZE: int = int(os.getenv("CODE_CHUNK_SIZE", 800))
TEXT_CHUNK_SIZE: int = int(os.getenv("TEXT_CHUNK_SIZE", 1000))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 100))

# Extensions de fichiers prises en charge par les parseurs.  Ajoutez d’autres extensions ici si nécessaire.
SUPPORTED_CODE_EXTENSIONS = {
    ".cpp", ".hpp", ".h", ".cc", ".cxx",
    ".py", ".java", ".js", ".ts"
}
SUPPORTED_DOC_EXTENSIONS = {".pdf", ".docx", ".pptx"}
# Extensions de texte brut prises en charge.  Cela sert d’indication pour le TextParser.
SUPPORTED_TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".wiki", ".rst",
    ".json", ".csv", ".tsv", ".yaml", ".yml",
    ".ini", ".cfg", ".conf", ".toml",
    ".xml", ".properties", ".cmake", ".gitignore",
    ".sh", ".bat"
}
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}

# Mappage des extensions de code vers l’identifiant de langue utilisé par Tree‑sitter.
LANGUAGE_MAPPING = {
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".h": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".py": "python",
    ".java": "java",
    ".js": "javascript",
    ".ts": "typescript",
}

__all__ = [
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_TEMPERATURE",
    "LLM_TIMEOUT",
    "REDMINE_BASE_URL",
    "MAX_SEARCH_RESULTS",
    "CHROMA_DB_PATH",
    "CHROMA_DB_DIR",
    "CHROMA_COLLECTION_NAME",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSION",
    "EMBEDDING_BATCH_SIZE",
    "CODE_CHUNK_SIZE",
    "TEXT_CHUNK_SIZE",
    "SUPPORTED_CODE_EXTENSIONS",
    "SUPPORTED_DOC_EXTENSIONS",
    "SUPPORTED_TEXT_EXTENSIONS",
    "SUPPORTED_IMAGE_EXTENSIONS",
    "LANGUAGE_MAPPING",
]