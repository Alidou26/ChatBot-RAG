"""
ChromaDB storage module for managing vector embeddings and metadata.
"""
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

from ..config import CHROMA_DB_PATH, CHROMA_COLLECTION_NAME

logger = logging.getLogger(__name__)


class ChromaDBManager:
    """Manager for ChromaDB operations."""
    
    def __init__(
        self,
        persist_directory: str = CHROMA_DB_PATH,
        collection_name: str = CHROMA_COLLECTION_NAME
    ):
        """Initialize ChromaDB manager."""
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        
        logger.info(f"Initializing ChromaDB at: {self.persist_directory}")
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Vider le cache Chroma AVANT de créer le client
        chromadb.api.client.SharedClient = None
        
        # Initialize ChromaDB client
        try:
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(anonymized_telemetry=False, is_persistent=True)
            )
            logger.info(" ChromaDB client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise
        
        # Get or create collection
        self._init_collection()
    
    def _init_collection(self) -> None:
        """Initialize or get the collection with appropriate metadata."""
        try:
            # NE PAS recréer self.client ici
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "description": "Redmine technical documents and code"
                }
            )
            count = self.collection.count()
            logger.info(f" Collection '{self.collection_name}' ready ({count} documents)")
        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")
            raise
    
    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> int:
        """Add documents with embeddings to the collection."""
        if not (len(documents) == len(embeddings) == len(metadatas)):
            raise ValueError(
                f"Length mismatch: documents={len(documents)}, "
                f"embeddings={len(embeddings)}, metadatas={len(metadatas)}"
            )
        
        if not documents:
            logger.warning("No documents to add")
            return 0
        
        if ids is None:
            ids = [self._generate_id(doc, idx) for idx, doc in enumerate(documents)]
        
        try:
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f" Added {len(documents)} documents to collection")
            return len(documents)
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Query the collection for similar documents."""
        try:
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                where_document=where_document
            )
            
            logger.info(f" Query returned {len(results.get('ids', [[]])[0])} results")
            return results
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise
    
    def get_by_ids(self, ids: List[str]) -> Dict[str, Any]:
        """Retrieve documents by their IDs."""
        try:
            return self.collection.get(ids=ids)
        except Exception as e:
            logger.error(f"Failed to get documents by IDs: {e}")
            raise
    
    def delete_by_ids(self, ids: List[str]) -> None:
        """Delete documents by their IDs."""
        try:
            self.collection.delete(ids=ids)
            logger.info(f" Deleted {len(ids)} documents")
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise
    
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f" Deleted collection '{self.collection_name}'")
            self._init_collection()
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise
    
    def count(self) -> int:
        """Get the number of documents in the collection."""
        return self.collection.count()
    
    def _generate_id(self, content: str, index: int) -> str:
        """Generate a unique ID for a document."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:8]
        return f"doc_{index}_{content_hash}"
    
    @staticmethod
    def create_metadata(
        project_id: str,
        project_name: str,
        source: str,
        file_name: str,
        file_type: str,
        content_type: str,
        language: str,
        chunk_index: int,
        total_chunks: int,
        **extra_metadata
    ) -> Dict[str, Any]:
        """Create a standardized metadata dictionary."""
        metadata = {
            'project_id': str(project_id),
            'project_name': project_name,
            'source': source,
            'file_name': file_name,
            'file_type': file_type,
            'content_type': content_type,
            'language': language,
            'chunk_index': chunk_index,
            'total_chunks': total_chunks,
        }
        
        metadata.update(extra_metadata)
        return metadata


# ===== SINGLETON GLOBAL =====
_storage_instance = None

def get_storage(force_reload: bool = False) -> ChromaDBManager:
    global _storage_instance
    
    if _storage_instance is None or force_reload:
        if force_reload:
            chromadb.api.client.SharedClient = None
            import gc
            gc.collect()
        _storage_instance = ChromaDBManager()
    
    return _storage_instance  # RETOURNE TOUJOURS LA MÊME INSTANCE
