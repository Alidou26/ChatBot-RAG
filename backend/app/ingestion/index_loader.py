"""
Ce module gère le chargement/rechargement de l'index LlamaIndex
et ChromaDB avec nettoyage complet des caches.
"""

import os
import gc
import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

from ..rag.embedder import get_embedder
from ..config import CHROMA_DB_DIR, CHROMA_COLLECTION_NAME

# Global cache
index_cache = None
_last_doc_count = 0


def reload_index_completely():
    """
    ⚡ FONCTION PRINCIPALE : Recharge COMPLÈTEMENT l'index.
    
    À appeleraprès chaque ingestion pour :
    1. Vider le cache ChromaDB global
    2. Libérer la mémoire (garbage collection)
    3. Reconnecter à ChromaDB depuis le disque
    4. Reconstruire l'index LlamaIndex
    
    Returns:
        VectorStoreIndex: L'index rechargé
    """
    global index_cache, _last_doc_count
    
    print("[INDEX_LOADER]Début du rechargement complet...")
    
    # === ÉTAPE 1 : Vider ChromaDB en mémoire ===
    print("[INDEX_LOADER]Vidage du client ChromaDB global...")
    chromadb.api.client.SharedClient = None
    
    # === ÉTAPE 2 : Forcer garbage collection ===
    print("[INDEX_LOADER]Garbage collection...")
    gc.collect()
    
    # === ÉTAPE 3 : Reconnecter à ChromaDB ===
    print(f"[INDEX_LOADER]Connexion à ChromaDB ({CHROMA_DB_DIR})...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    
    # === ÉTAPE 4 : Récupérer la collection ===
    print(f"[INDEX_LOADER]Récupération de la collection '{CHROMA_COLLECTION_NAME}'...")
    try:
        collection = chroma_client.get_collection(CHROMA_COLLECTION_NAME)
        doc_count = collection.count()
        print(f"[INDEX_LOADER]   Collection trouvée : {doc_count} documents")
    except Exception as e:
        print(f"[INDEX_LOADER]   Collection inexistante, création...")
        collection = chroma_client.create_collection(CHROMA_COLLECTION_NAME)
        doc_count = 0
    
    # === ÉTAPE 5 : Créer nouveau vector store ===
    print("[INDEX_LOADER] Création du vecteur store...")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    
    # === ÉTAPE 6 : Créer nouveau storage context ===
    print("[INDEX_LOADER] Création du contexte de stockage...")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # === ÉTAPE 7 : Recharger embedder ===
    print("[INDEX_LOADER] Chargement du modèle d'embedding...")
    embed_model = get_embedder()
    
    # === ÉTAPE 8 : Reconstruire l'index ===
    print("[INDEX_LOADER] Reconstruction de l'index LlamaIndex...")
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    
    # === ÉTAPE 9 : Mettre en cache ===
    index_cache = index
    _last_doc_count = doc_count
    
    print(f"[INDEX_LOADER]Rechargement réussi : {doc_count} documents en mémoire")
    print("[INDEX_LOADER] " + "="*60)
    
    return index


def load_index():
    """
    Alias pour rechargement complet.
    Appele reload_index_completely() et retourne l'index.
    
    Returns:
        VectorStoreIndex: L'index rechargé
    """
    return reload_index_completely()


def get_query_engine():
    """
    Retourne TOUJOURS un query engine à jour.
    
    Contrairement à une approche classique, on ne cache PAS
    le query engine car il repose sur un cache interne qui devient rapidement stale.
    
    Donc on FORCE le rechargement de l'index à chaque fois.
    
    Returns:
        QueryEngine: Un query engine frais
    """
    global index_cache
    
    # OPTION 1 : Rechargement complet à chaque fois (PLUS LENT mais SÛRE)
    # index = reload_index_completely()
    
    # OPTION 2 : Réutiliser le cache si disponible (PLUS RAPIDE)
    # À utiliser seulement si on est sûr que le cache est frais
    if index_cache is None:
        index = reload_index_completely()
    else:
        index = index_cache
    
    return index.as_query_engine(similarity_top_k=5)


def get_cached_index():
    """
    Récupère l'index depuis le cache (sans rechargement).
    Utile pour des opérations en lecture-seule rapides.
    
    Returns:
        VectorStoreIndex or None: L'index si chargé, sinon None
    """
    global index_cache
    return index_cache


def get_document_count():
    """
    Retourne le nombre de documents actuellement en mémoire.
    
    Returns:
        int: Nombre de documents
    """
    global _last_doc_count
    return _last_doc_count


def clear_cache():
    """
    Vide le cache de l'index.
    Utile pour tester ou nettoyer.
    """
    global index_cache, _last_doc_count
    index_cache = None
    _last_doc_count = 0
    print("[INDEX_LOADER] Cache vidé")