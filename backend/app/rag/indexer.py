"""
Document indexer using LlamaIndex framework.

Pipeline: Parse → Chunk → Embed → Index with LlamaIndex
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import os

try:
    from llama_index.core import (
        VectorStoreIndex,
        Document,
        StorageContext,
        Settings
    )
    from llama_index.vector_stores.chroma import ChromaVectorStore
    import chromadb
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    LLAMA_INDEX_AVAILABLE = False

from .parsers import ParserFactory
from .chunkers import ChunkerFactory
from ..config import CHROMA_DB_PATH, EMBEDDING_MODEL,TEXT_CHUNK_SIZE, CHUNK_OVERLAP


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentIndexer:
    """
    Document indexer using LlamaIndex framework.
    
    Integrates custom parsers/chunkers with LlamaIndex's indexing capabilities.
    
    Features:
    - Custom parsing (PDF, DOCX, PPTX, code)
    - Custom chunking (semantic for code, text-based for documents)
    - LlamaIndex vector store integration
    - ChromaDB backend
    - Jina v3 embeddings
    
    Examples:
        >>> indexer = DocumentIndexer()
        >>> files = [Path("project.cpp"), Path("doc.pdf")]
        >>> metadata = {'project_id': '123', 'project_name': 'MyProject'}
        >>> index = indexer.index_files(files, metadata)
        >>> print(f" Index created with {len(files)} files")
    """
    
    def __init__(
        self,
        collection_name: str = "redmine_documents",
        persist_dir: str = None,
        embedding_model: str = None
    ):
        """
        Initialize document indexer.
        
        Args:
            collection_name: ChromaDB collection name
            persist_dir: Directory to persist ChromaDB
            embedding_model: Embedding model name (default: from config)
        """
        if not LLAMA_INDEX_AVAILABLE:
            raise ImportError(
                "LlamaIndex not installed. "
                "Install with: pip install llama-index llama-index-embeddings-huggingface "
                "llama-index-vector-stores-chroma"
            )
        
        self.collection_name = collection_name
        self.persist_dir = persist_dir or CHROMA_DB_PATH
        self.embedding_model_name = embedding_model or EMBEDDING_MODEL
        
        # Initialize parsers and chunkers
        self.parser_factory = ParserFactory()
        self.chunker_factory = ChunkerFactory()


        #Utiliser le singleton au lieu de créer un client directement
        from .storage import get_storage
        storage = get_storage()

        self.chroma_client = storage.client
        
        self.chroma_collection = storage.client.get_or_create_collection(
              name=collection_name
        )
        
        # Setup LlamaIndex components
        self._setup_llamaindex()
        
        logger.info(f"  DocumentIndexer initialized")
        logger.info(f"  Collection: {collection_name}")
        logger.info(f"  Persist dir: {self.persist_dir}")
        logger.info(f"  Embedding model: {self.embedding_model_name}")
    
    def _setup_llamaindex(self):
        """Setup LlamaIndex global settings and vector store.

        LlamaIndex uses a global ``Settings`` object to configure the
        embedding model and chunking behaviour for node parsing.  When
        indexing large code files, the per‑chunk metadata can easily
        exceed the default chunk size (1024), which triggers a
        ``ValueError`` inside LlamaIndex's ``SentenceSplitter``.  To
        mitigate this we increase the global chunk size and overlap.  We
        also wrap the embedding model configuration in a try/except to
        provide a graceful fallback if the requested model cannot be
        loaded (e.g. due to missing dependencies).
        """
       # === NEW LOCAL EMBEDDER (SentenceTransformers) ===

        from .embedder import get_embedder
        Settings.embed_model = get_embedder()

        # Increase the default chunk size to accommodate larger metadata.
        # The metadata for a chunk can be a couple of thousand characters if
        # it contains information about many code units. Setting this to
        # 4096 avoids ValueError while still producing reasonable chunk
        # sizes.  ``chunk_overlap`` is kept modest (100 characters) to
        # preserve some context between chunks without exploding the
        # number of nodes.

        Settings.chunk_size = TEXT_CHUNK_SIZE
        Settings.chunk_overlap = CHUNK_OVERLAP

        # Create vector store
        self.vector_store = ChromaVectorStore(
            chroma_collection=self.chroma_collection
        )

        # Create storage context
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

        logger.info(
            " LlamaIndex components configured (chunk_size=%s, overlap=%s)",
            Settings.chunk_size,
            Settings.chunk_overlap,
        )
    
    def index_files(
        self,
        files: List[Path],
        metadata: Dict[str, Any]
    ) -> VectorStoreIndex:
        """
        Index a list of files using LlamaIndex.
        
        Args:
            files: List of file paths to index
            metadata: Common metadata for all files (project_id, project_name, etc.)
            
        Returns:
            LlamaIndex VectorStoreIndex instance
            
        Examples:
            >>> indexer = DocumentIndexer()
            >>> files = [Path("test.py"), Path("doc.pdf")]
            >>> meta = {'project_id': '1', 'project_name': 'Test'}
            >>> index = indexer.index_files(files, meta)
            >>> index is not None
            True
        """
        if not files:
            logger.warning("No files to index")
            return None
        
        logger.info(f"Starting indexing of {len(files)} files...")
        
        all_documents = []
        successful_files = 0
        failed_files = []
        
        for file_path in tqdm(files, desc="Processing files"):
            try:
                documents = self._process_single_file(file_path, metadata)
                all_documents.extend(documents)
                successful_files += 1
                logger.info(f" {file_path.name}: {len(documents)} chunks")
                
            except Exception as e:
                logger.error(f" Failed to process {file_path}: {e}")
                failed_files.append((file_path, str(e)))
                # Continue with next file
        
        if not all_documents:
            logger.error("No documents to index")
            return None
        
        # Create index from all documents
        logger.info(f"Creating LlamaIndex from {len(all_documents)} chunks...")
        
        index = VectorStoreIndex.from_documents(
            all_documents,
            storage_context=self.storage_context,
            show_progress=True
        )
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info(f"Indexing Complete!")
        logger.info(f"{'='*60}")
        logger.info(f" Successfully indexed: {successful_files}/{len(files)} files")
        logger.info(f" Total chunks created: {len(all_documents)}")
        logger.info(f" Index type: {type(index).__name__}")
        
        if failed_files:
            logger.warning(f"\n⚠ Failed files ({len(failed_files)}):")
            for file_path, error in failed_files:
                logger.warning(f"  - {file_path.name}: {error}")
        
        return index
    
    def _process_single_file(
        self,
        file_path: Path,
        common_metadata: Dict[str, Any]
    ) -> List[Document]:
        """
        Process a single file through custom pipeline.
        
        Args:
            file_path: Path to file
            common_metadata: Common metadata (project info, etc.)
            
        Returns:
            List of LlamaIndex Document objects
        """
        logger.debug(f"Processing: {file_path}")
        
        # Step 1: Parse with custom parsers
        parser = self.parser_factory.get_parser(file_path)
        if parser is None:
            raise ValueError(f"No parser available for {file_path}")
        
        parsed_content = parser.parse(file_path)
        logger.debug(f"   Parsed: {len(parsed_content.text)} chars")
        
        # Step 2: Chunk with custom chunkers
        chunker = self.chunker_factory.get_chunker(parsed_content)
        if chunker is None:
            raise ValueError(f"No chunker available for {file_path}")
        
        chunks = chunker.chunk(parsed_content)
        logger.debug(f"   Chunked: {len(chunks)} chunks")
        
        # Step 3: Convert to LlamaIndex documents
        llama_documents = []
        
        for i, chunk in enumerate(chunks):
            # Merge chunk metadata with common metadata
            # Start with common metadata and file info
            chunk_metadata = {
                **common_metadata,
                'file_path': str(file_path),
                'file_name': file_path.name,
                'chunk_index': i,
                'content_type': parsed_content.content_type,
                'language': parsed_content.language,
            }
            # Merge chunk-specific metadata but drop heavy fields to keep
            # metadata strings reasonably small. In particular, the code
            # parser stores a list of all semantic units under the key
            # "units", which can be thousands of characters and causes
            # problems for LlamaIndex when splitting nodes. Also drop
            # any inline code text stored in metadata (e.g. 'text').
            for key, value in chunk.metadata.items():
                if key in {'units', 'text', 'signature_lines'}:
                    continue
                chunk_metadata[key] = value
            
            # Create LlamaIndex Document
            doc = Document(
                text=chunk.text,
                metadata=chunk_metadata,
                id_=f"{file_path.stem}_{i}"
            )
            
            llama_documents.append(doc)
        
        return llama_documents
    
    def load_index(self) -> Optional[VectorStoreIndex]:
        """
        Load existing index from storage.
        
        Returns:
            VectorStoreIndex if exists, None otherwise
        """
        try:
            index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                storage_context=self.storage_context
            )
            logger.info(" Loaded existing index from storage")
            return index
        except Exception as e:
            logger.warning(f"Could not load existing index: {e}")
            return None
    
    def query(
        self,
        query_text: str,
        top_k: int = 5,
        filters: Dict[str, Any] = None
    ) -> Any:
        """
        Query the index.
        
        Args:
            query_text: Query string
            top_k: Number of results to return
            filters: Metadata filters
            
        Returns:
            Query response
        """
        index = self.load_index()
        if index is None:
            logger.error("No index available. Please index files first.")
            return None
        
        query_engine = index.as_query_engine(
            similarity_top_k=top_k,
            filters=filters
        )
        
        response = query_engine.query(query_text)
        return response
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get indexer statistics.
        
        Returns:
            Dictionary with stats
        """
        return {
            'collection_name': self.collection_name,
            'persist_dir': self.persist_dir,
            'embedding_model': self.embedding_model_name,
            'total_documents': self.chroma_collection.count(),
            'parsers_available': len(self.parser_factory._parsers),
            'chunkers_available': 2  # TextChunker + CodeChunker
        }
    
    def clear_collection(self):
        
        from .storage import get_storage
        storage = get_storage(force_reload=True)
        storage.client.delete_collection(self.collection_name)
        self.chroma_collection = storage.client.get_or_create_collection(
        name=self.collection_name
    )
        logger.info(f" Cleared collection: {self.collection_name}")


# Singleton instance
_indexer_instance: Optional[DocumentIndexer] = None


def get_indexer(
    collection_name: str = "redmine_documents",
    reset: bool = False
) -> DocumentIndexer:
    """
    Get or create indexer singleton.
    
    Args:
        collection_name: ChromaDB collection name
        reset: Force recreation of indexer
        
    Returns:
        DocumentIndexer instance
    """
    global _indexer_instance
    
    if _indexer_instance is None or reset:
        _indexer_instance = DocumentIndexer(
            collection_name=collection_name
        )
    
    return _indexer_instance
