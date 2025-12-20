from __future__ import annotations

import os
import shutil
import json
import tempfile
from typing import List, Iterable, Dict, Optional
from pathlib import Path

from PIL import Image  # type: ignore
import pytesseract  # type: ignore

from llama_index.core import StorageContext, Document as LlamaDocument# type: ignore
from llama_index.core.node_parser import SentenceSplitter  # type: ignore
from llama_index.vector_stores.chroma import ChromaVectorStore  # type: ignore
from llama_index.readers.web import BeautifulSoupWebReader  # type: ignore
from llama_index.readers.file import PyMuPDFReader, PandasCSVReader


from ..config import (
    CHROMA_DB_DIR,
    CHROMA_COLLECTION_NAME,
    TEXT_CHUNK_SIZE,
    CHUNK_OVERLAP
)


def _read_image(file_path: str) -> Optional[str]:
    try:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        print(f"[ingestion] OCR image error for {file_path}: {e}")
        return None

def _pdf_ocr_all(path: str) -> str:
    # Convert every page to image(s) then OCR (simple approach)
    try:
        import fitz
        doc = fitz.open(path)
        texts: List[str] = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img_bytes = pix.pil_tobytes(format="PNG")
            from io import BytesIO
            image = Image.open(BytesIO(img_bytes))
            page_text = pytesseract.image_to_string(image)
            texts.append(page_text or "")
        combined = "\n".join(texts).strip()
        print(f"[ingestion] PDF OCR OK: {len(combined)} chars for {path}")
        return combined
    except Exception as e:
        print(f"[ingestion] PDF OCR full error {path}: {e}")
        return ""

def _make_llama_docs_from_files(file_paths: Iterable[str]) -> List[LlamaDocument]:
    from llama_index.core import Document as LlamaDocument  # type: ignore
    llama_docs: List[LlamaDocument] = []
    for path_str in file_paths:
        path = Path(path_str)
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        # IMPORTANT : forcer source et file_name dans les métadonnées dès l’entrée
        metadata = {
            "file_path": str(path.resolve()),
            "file_name": path.name,
            "source": str(path.resolve()),  # <- clé affichée côté “sources”
        }
        try:
            if ext == ".pdf":
                # extraction texte classique
                txt1 = ""
                try:
                    reader = PyMuPDFReader()
                    docs = reader.load_data([str(path)])
                    for d in docs:
                        d.metadata.update(metadata)
                        # Forcer également ici (les readers posent parfois une source tmp)
                        d.metadata["source"] = metadata["file_path"]
                        d.metadata["file_name"] = metadata["file_name"]
                    if docs:
                        llama_docs.extend(docs)
                        txt1 = "".join(d.text for d in docs)
                except Exception as e:
                    print(f"[ingestion] PyMuPDFReader failed {path}: {e}")

                # extraction fallback/ocr
                txt2 = _pdf_ocr_all(str(path))
                # choisir le plus long
                content = txt1.strip() if len(txt1) >= len(txt2) else txt2.strip()
                if content:
                    llama_docs.append(LlamaDocument(text=content, metadata=metadata))
                else:
                    print(f"[ingestion] PDF skipped after extractions: {path}")

            elif ext == ".csv":
                reader = PandasCSVReader()
                docs = reader.load_data([str(path)])
                for d in docs:
                    d.metadata.update(metadata)
                    d.metadata["source"] = metadata["file_path"]
                    d.metadata["file_name"] = metadata["file_name"]
                llama_docs.extend(docs)

            elif ext == ".json":
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    try:
                        data = json.load(f)
                        text = json.dumps(data, ensure_ascii=False, indent=2)
                    except Exception:
                        f.seek(0)
                        text = f.read()
                llama_docs.append(LlamaDocument(text=text, metadata=metadata))

            elif ext in {".docx", ".pptx", ".txt", ".md", ".rst"}:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                llama_docs.append(LlamaDocument(text=text, metadata=metadata))

            elif ext in {".png", ".jpg", ".jpeg", ".bmp", ".gif"}:
                text = _read_image(str(path))
                if text:
                    llama_docs.append(LlamaDocument(text=text, metadata=metadata))

            else:
                # tentative extraction texte
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                if text:
                    llama_docs.append(LlamaDocument(text=text, metadata=metadata))
        except Exception as e:
            print(f"[ingestion] Skipping {path}: {e}")
            continue
    return llama_docs

def ingest_to_chroma(file_paths: Iterable[str], urls: Optional[Iterable[str]] = None, drop_existing: bool = False) -> int:
    """Ajouter des documents et pages web à la collection Chroma."""
    llama_docs = _make_llama_docs_from_files(file_paths)

    if urls:
        try:
            web_reader = BeautifulSoupWebReader()
            url_docs = web_reader.load_data(urls=list(urls))
            for doc in url_docs:
                url_val = doc.metadata.get("source") or doc.metadata.get("url") or ""
                if url_val:
                    doc.metadata["url"] = url_val
                doc.metadata["source"] = url_val
                doc.metadata["file_name"] = url_val or "url"
            llama_docs.extend(url_docs)
        except Exception as e:
            print(f"[ingestion] WebReader error: {e}")

    if not llama_docs:
        return 0

    splitter = SentenceSplitter(chunk_size=TEXT_CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(llama_docs)

    # === Normalisation métadonnées au niveau node ===
    from pathlib import Path as _P
    for node in nodes:
        fp = node.metadata.get("file_path") or node.metadata.get("source")
        if fp:
            node.metadata["source"] = fp
            node.metadata["file_name"] = _P(fp).name if "://" not in fp else fp

    # Filtrage des nodes
    filtered_nodes: List = []
    for node in nodes:
        if not node.text or not node.text.strip():
            continue
        if len(node.text.split()) > 1000:
            continue
        filtered_nodes.append(node)

    persist_dir = CHROMA_DB_DIR
    collection_name = CHROMA_COLLECTION_NAME
    
    if drop_existing and os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    #Utiliser get_storage() au lieu de créer un client directement
    from ..rag.storage import get_storage
    storage = get_storage(force_reload=True)  # Force reload après drop_existing
    
    chroma_collection = storage.collection
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Embedding model
    from ..rag.embedder import get_embedder
    embed_model = get_embedder(force_reload=True)  # Force reload aussi

    from llama_index.core import VectorStoreIndex
    if chroma_collection.count() == 0:
        index = VectorStoreIndex(
            filtered_nodes,
            storage_context=storage_context,
            embed_model=embed_model,
        )
    else:
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
            embed_model=embed_model
        )
        index.insert_nodes(filtered_nodes)

    return len(filtered_nodes)

