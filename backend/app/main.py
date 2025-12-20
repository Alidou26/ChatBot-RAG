"""FastAPI application exposing knowledge management endpoints.

This module defines the REST API for the AI assistant backend. It
provides endpoints for ingesting new documents, querying the knowledge
base, analysing root causes and fetching Redmine issues. The API
follows simple JSON conventions to facilitate easy integration with
front-end clients or command line tools.

Run the application with::

    uvicorn backend.app.main:app --reload

The ``--reload`` flag enables hot reloading during development.
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb import Settings
chromadb.api.client.SharedClient = None  # désactive tout fallback
Settings.default_embedding_function = None

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field

from .config import (
    MAX_SEARCH_RESULTS,
)
from .ingestion.chroma_ingester import ingest_to_chroma  
from .rag.jina_rag import JinaRAG
from .rag.indexer import get_indexer
from .ingestion.dataset_ingester import ingest_archive
from .redmine.client import ingest_full_redmine_project
from .rca.analysis import analyse_root_cause 


app = FastAPI(title="AI Knowledge Assistant API")

rag_engine: Optional[JinaRAG] = None

# === Dossier persistant pour conserver les vrais noms des fichiers ===
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class QueryRequest(BaseModel):
    query: str = Field(..., description="Question à poser au chatbot")
    top_k: Optional[int] = Field(None, description="Nombre maximum de passages à considérer")


class RootCauseRequest(BaseModel):
    description: str
    top_k: Optional[int] = Field(5, description="Nombre de sources à utiliser")



def _reload_rag_after_ingestion() -> None:
    """Force reload complet du RAG avec nettoyage des caches."""
    global rag_engine
    
    print("[RELOAD] Vidage du cache ChromaDB...")
    chromadb.api.client.SharedClient = None
    
    print("[RELOAD] Force reload du storage (singleton)...")
    from .rag.storage import get_storage
    get_storage(force_reload=True)
    
    print("[RELOAD] Force reload de l'embedder (singleton)...")
    from .rag.embedder import get_embedder
    get_embedder(force_reload=True)  

    print("[RELOAD] Garbage collection...")
    import gc
    gc.collect()
    
    print("[RELOAD] Rechargement de l'index LlamaIndex...")
    from .ingestion.index_loader import load_index
    load_index()
    
    print("[RELOAD] Création d'une NOUVELLE instance JinaRAG...")
    rag_engine = JinaRAG()
    
    print("[RELOAD] RAG complètement rechargé et prêt")




@app.on_event("startup")
def _startup() -> None:
    """Initialise le moteur RAG lors du démarrage."""
    global rag_engine
    try:
        rag_engine = JinaRAG()
    except Exception:
        rag_engine = None


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _safe_filename(name: str) -> str:
    """Nettoie le nom de fichier (basique), enlève les séparateurs et garde le basename."""
    return Path(name).name.replace("\x00", "")  # supprime null bytes, etc.

def _save_upload_with_original_name(upload: UploadFile) -> Path:
    """
    Sauvegarde l'upload sous son nom d'origine dans ./uploads/
    (en évitant les collisions en ajoutant un suffixe si nécessaire).
    Retourne le chemin *persistant* final.
    """
    base = _safe_filename(upload.filename or "")
    if not base:
        raise HTTPException(status_code=400, detail="Nom de fichier vide")

    dest = UPLOAD_DIR / base
    # Évite d'écraser un fichier existant : ajoute un suffixe (1), (2), ...
    if dest.exists():
        stem, suf = dest.stem, dest.suffix
        i = 1
        while True:
            candidate = UPLOAD_DIR / f"{stem} ({i}){suf}"
            if not candidate.exists():
                dest = candidate
                break
            i += 1

    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)

    return dest.resolve()


# -------------------------------------------------------------------
# Endpoints d’ingestion
# -------------------------------------------------------------------
@app.post("/ingest/files", summary="Ingestion de fichiers")
async def ingest_files(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    """Ingest uploaded files into the knowledge base.

    Correction importante : on conserve les fichiers avec leur VRAI nom
    dans ./uploads/ et on passe ces chemins persistants à l'indexeur.
    """
    if not files:
        raise HTTPException(status_code=400, detail="Aucun fichier fourni")

    persisted_paths: List[Path] = []
    for upload in files:
        try:
            persisted = _save_upload_with_original_name(upload)
            persisted_paths.append(persisted)
        finally:
            # s'assure de fermer le file-like d'UploadFile
            try:
                await upload.close()
            except Exception:
                pass

    # Indexation via le DocumentIndexer (métadonnées file_path/file_name propres)
    indexer = get_indexer(collection_name="redmine_projects")
    metadata = {"project_id": "default", "project_name": "default"}

    initial_count = indexer.chroma_collection.count()
    index = indexer.index_files(persisted_paths, metadata)
    new_count = indexer.chroma_collection.count()
    count = max(new_count - initial_count, 0)

    _reload_rag_after_ingestion()

    # Le moteur RAG lit la même collection, pas besoin de le recréer
    return {"status": "ok", "ingested_chunks": count, "files": [str(p) for p in persisted_paths]}


@app.post("/ingest/urls", summary="Ingestion de pages web")
async def ingest_urls(urls: List[str]) -> Dict[str, Any]:
    """Ingest one or more web pages given by their URLs."""
    if not urls:
        raise HTTPException(status_code=400, detail="Aucune URL fournie")
    # Legacy URL ingestion retained for backwards compatibility; still uses chroma_ingester
    count = ingest_to_chroma([], urls=urls, drop_existing=False)

    _reload_rag_after_ingestion()
    
    return {"status": "ok", "ingested_chunks": count, "urls": urls}


@app.post("/ingest/archive", summary="Ingestion d'archives (.tar.gz ou .zip)")
async def ingest_archive_endpoint(
    file: UploadFile = File(...),
    project_id: str = "default",
    project_name: str = "default"
) -> Dict[str, Any]:
    """Ingest a compressed archive of documents and code.

    On garde un fichier temporaire pour l'archive elle-même (OK),
    mais l’extracteur/ingesteur interne utilisera les vrais chemins
    des fichiers extraits → file_path/file_name propres dans les sources.
    """
    filename = file.filename or ""
    suffixes = Path(filename).suffixes
    # Accept .tar.gz (suffixes[-2:] == ['.tar','.gz']) or .zip (suffixes[-1] == '.zip')
    if not (
        (len(suffixes) >= 2 and suffixes[-2:] == ['.tar', '.gz'])
        or (len(suffixes) >= 1 and suffixes[-1].lower() == '.zip')
    ):
        raise HTTPException(status_code=400, detail="Le fichier doit être une archive .tar.gz ou .zip")

    # Create temporary file with correct suffix (OK pour l'archive)
    suffix = "".join(suffixes)
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as buffer:
            buffer.write(await file.read())
        chunks_added = ingest_archive(
            Path(tmp_path),
            project_id,
            project_name,
            collection_name="redmine_projects"
        )

        _reload_rag_after_ingestion()

        return {"status": "ok", "ingested_chunks": chunks_added}
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# -------------------------------------------------------------------
# Endpoints de requête
# -------------------------------------------------------------------
@app.post("/query", summary="Poser une question")
async def ask_question(request: QueryRequest) -> Dict[str, Any]:
    """Répondre à une question en utilisant RAG et LlamaIndex."""
    if rag_engine is None:
        raise HTTPException(status_code=404, detail="Aucun moteur RAG disponible. Veuillez ingérer des documents d'abord.")
    k = request.top_k or MAX_SEARCH_RESULTS
    result = rag_engine.query(request.query, top_k=k)

    # Normalisation de sécurité côté API : si une source ressemble à un chemin temporaire,
    # on remplace par le file_name (lisible) ; et on s'assure d'avoir un file_name
    # d'après file_path si besoin.
    fixed_sources: List[Dict[str, Any]] = []
    for m in result.get("sources", []) or []:
        meta = dict(m)
        src = meta.get("source") or meta.get("file_path") or ""
        fn = meta.get("file_name")

        if isinstance(src, str):
            if ("/var/folders/" in src) or src.startswith("/tmp"):
                if fn:
                    meta["source"] = fn
            elif "://" not in src:
                # chemin local → garantir un file_name lisible
                meta["file_name"] = fn or Path(src).name

        fixed_sources.append(meta)

    return {"answer": result.get("answer", ""), "sources": fixed_sources}


@app.post("/root_cause", summary="Analyse de cause racine avec ChromaDB uniquement")
async def root_cause(request: RootCauseRequest) -> Dict[str, Any]:
    """
    Analyse la cause racine d’un problème en utilisant UNIQUEMENT
    les données déjà indexées dans ChromaDB.
    """

    description = (request.description or "").strip()

    if not description:
        raise HTTPException(
            status_code=400,
            detail="La description du problème est vide."
        )
    
     # K provenant du frontend
    user_k = request.top_k or MAX_SEARCH_RESULTS

    # 1. Appeler le moteur RCA
    try:
        
        rca = analyse_root_cause(
            issue_description=description,
            project_id=None,     # compatible si None
            top_k=user_k
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur interne RCA : {str(e)}"
        )

    answer = rca.get("answer", "")
    sources = rca.get("sources", [])

    # -------------------------------------------
    # 2. Normalisation des sources pour Streamlit
    # (même comportement que /query)
    # -------------------------------------------
    cleaned_sources: List[Dict[str, Any]] = []

    for src in sources:
        meta = dict(src)

        source_path = meta.get("source") or meta.get("file_path") or ""
        file_name = meta.get("file_name")

        # Nettoyage des chemins temporaires
        if isinstance(source_path, str):
            # Chemins macOS (/var/folders/…), Linux (/tmp), Windows temp
            if ("/var/folders/" in source_path) or source_path.startswith("/tmp") or "AppData\\Local\\Temp" in source_path:
                if file_name:
                    meta["source"] = file_name
            else:
                # Sinon extraire juste le nom du fichier
                if not file_name and "://" not in source_path:
                    meta["file_name"] = Path(source_path).name

        cleaned_sources.append(meta)

    # -----------------------------------------------------------
    #  3. Retour API
    # -----------------------------------------------------------
    return {
        "answer": answer,
        "sources": cleaned_sources
    }

@app.post("/redmine/ingest/{project_id}", summary="Ingestion complète d'un projet Redmine")
async def ingest_redmine_project(project_id: str) -> Dict[str, Any]:
    """
    Ingeste TOUTES les données d'un projet Redmine dans ChromaDB :
    - issues + journaux + champs personnalisés
    - wiki pages + pièces jointes
    - fichiers du projet
    - news, versions, trackers, catégories, membres
    Le projet est d’abord exporté sous forme d’archive (.tar.gz),
    puis ingéré via ingest_archive().
    """
    try:
        total_chunks = ingest_full_redmine_project(project_id)

        _reload_rag_after_ingestion()

        if total_chunks == 0:
            raise HTTPException(
                status_code=500,
                detail=f"Aucune donnée n'a pu être ingérée pour le projet '{project_id}'."
            )

        return {
            "status": "success",
            "project_id": project_id,
            "chunks_indexed": total_chunks,
            "message": f"Ingestion réussie pour le projet '{project_id}' avec {total_chunks} chunks."
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur durant l’ingestion du projet '{project_id}': {str(e)}"
        )


@app.get("/health", summary="Vérification d'état")
async def healthcheck() -> Dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "ok"}
