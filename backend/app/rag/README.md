# Pipeline RAG - Documentation Technique

## Vue d'ensemble

Pipeline RAG complet pour l'indexation et la récupération de documents techniques provenant de Redmine. Le système utilise des composants modulaires pour le parsing, le chunking, l'embedding et le stockage vectoriel.

**Technologies principales:**
- Embeddings: Jina AI v3 (1024 dimensions)
- Base vectorielle: ChromaDB avec HNSW
- Framework d'indexation: LlamaIndex
- Parsing: Docling (documents) + Tree-sitter (code)

## Architecture

```
backend/app/rag/
├── config.py               # Configuration centrale
├── indexer.py              # Orchestrateur LlamaIndex
├── embedder.py             # Embeddings Jina v3
├── storage.py              # Gestionnaire ChromaDB
├── retriever_chroma.py     # Récupération avec filtres
├── parsers/                # Extraction de contenu
│   ├── __init__.py         # Factory
│   ├── base_parser.py      # Interface ParsedContent
│   ├── pdf_parser.py       # Docling PDF
│   ├── docx_parser.py      # Docling DOCX
│   ├── pptx_parser.py      # Docling PPTX
│   └── code_parser.py      # Tree-sitter (C++/Python/Java/JS)
└── chunkers/               # Découpage intelligent
    ├── __init__.py         # Factory
    ├── base_chunker.py     # Interface Chunk
    ├── text_chunker.py     # LangChain RecursiveCharacterTextSplitter
    └── code_chunker.py     # Chunking sémantique par unité de code
```

## Composants du système

### 1. Parsers (Extraction de contenu)

**Rôle:** Extraire le texte structuré depuis différents formats de fichiers.

**Implémentation:**
- **pdf_parser.py, docx_parser.py, pptx_parser.py**: Utilisent Docling (>=1.0.0) pour une extraction de haute qualité avec préservation de la structure (tableaux, listes, hiérarchie).
- **code_parser.py**: Utilise Tree-sitter pour l'analyse syntaxique de code source (C++, Python, Java, JavaScript). Détecte les unités sémantiques (fonctions, classes, méthodes) et extrait le code avec contexte.

**ParserFactory:** Sélection automatique du parser approprié selon l'extension du fichier.

**Sortie:** Objet `ParsedContent` contenant le texte extrait, le type de contenu, le langage et les métadonnées.

### 2. Chunkers (Découpage intelligent)

**Rôle:** Diviser le contenu en chunks optimaux pour l'embedding et la récupération.

**Stratégies:**
- **TextChunker**: Utilise `RecursiveCharacterTextSplitter` de LangChain avec séparateurs hiérarchiques (paragraphes → phrases → mots). Taille configurable avec chevauchement pour préserver le contexte.
- **CodeChunker**: Découpage sémantique basé sur les unités de code détectées par Tree-sitter. Chaque fonction/classe devient un chunk avec ses métadonnées (nom, type, ligne de début/fin).

**ChunkerFactory:** Sélectionne automatiquement le chunker selon le type de contenu (code vs texte).

**Sortie:** Liste d'objets `Chunk` contenant le texte, les métadonnées spécifiques et la position dans le document source.

### 3. Embedder (Encodage vectoriel)

**Rôle:** Transformer le texte en représentations vectorielles pour la recherche sémantique.

**Modèle:** Jina AI v3 (`jinaai/jina-embeddings-v3`)
- Dimension: 1024
- Encodage asymétrique avec instructions de tâche
- `retrieval.passage`: Pour encoder les documents à stocker
- `retrieval.query`: Pour encoder les requêtes de recherche

**Caractéristiques:**
- Support GPU/CPU automatique
- Traitement par batch pour performance
- Normalisation L2 pour similarité cosinus
- Singleton pattern pour réutilisation

### 4. Storage (Stockage vectoriel)

**Rôle:** Gérer le stockage persistant des embeddings et métadonnées dans ChromaDB.

**Fonctionnalités:**
- Création et gestion de collections
- Insertion batch avec génération d'IDs uniques (hash MD5)
- Recherche vectorielle avec filtres de métadonnées
- Support des opérations CRUD complètes
- Persistance automatique sur disque

**ChromaDB:** Utilise l'index HNSW pour recherche approximative rapide avec haute précision.

### 5. Indexer (Orchestration)

**Rôle:** Coordonner le pipeline complet d'indexation via LlamaIndex.

**Pipeline d'indexation:**
1. Récupération du parser approprié (ParserFactory)
2. Parsing du fichier → ParsedContent
3. Récupération du chunker approprié (ChunkerFactory)
4. Chunking du contenu → Liste de Chunks
5. Conversion en Documents LlamaIndex
6. Création du VectorStoreIndex avec ChromaVectorStore

**Intégration LlamaIndex:**
- Configure `Settings.embed_model` avec HuggingFaceEmbedding (Jina v3)
- Utilise `ChromaVectorStore` comme backend
- Crée `StorageContext` pour persistance
- Génère `VectorStoreIndex` pour requêtes structurées

**Gestion des métadonnées:** Fusion des métadonnées communes (projet) avec métadonnées spécifiques (fichier, chunk, type de contenu).

### 6. Retriever ChromaDB (Récupération)

**Rôle:** Rechercher les documents pertinents avec filtrage avancé.

**Fonctionnement:**
1. Embedding de la requête avec task="retrieval.query"
2. Application des filtres de métadonnées (projet, type de contenu, langage)
3. Recherche vectorielle par similarité cosinus
4. Sélection des top-k résultats
5. Formatage avec scores de pertinence

**Filtres disponibles:**
- `project_id`: Filtrer par projet Redmine
- `content_type`: "code" ou "text"
- `language`: "cpp", "python", "java", "javascript"
- Filtres personnalisés additionnels

**Fonctions utilitaires:**
- `query_by_project()`: Recherche dans un projet spécifique
- `query_code_only()`: Recherche uniquement dans le code
- `query_docs_only()`: Recherche uniquement dans les documents
- `get_context_for_generation()`: Contexte formaté pour LLM

### 7. Configuration (config.py)

**Centralisation des paramètres:**
- Chemins de stockage (ChromaDB, données)
- Configuration des embeddings (modèle, dimensions, batch size)
- Paramètres de chunking (tailles, chevauchement)
- Extensions de fichiers supportées
- Mapping langages pour Tree-sitter
- Configuration de logging

## Flux de données

**Indexation:**
```
Fichier → Parser → ParsedContent → Chunker → Chunks → 
Indexer (LlamaIndex) → Embeddings (Jina v3) → 
ChromaDB (VectorStore) → Index persistant
```

**Récupération:**
```
Requête utilisateur → Embedder (task=query) → 
Query vector → ChromaDB (similarité + filtres) → 
Top-k chunks pertinents → Contexte pour génération
```

## Points techniques importants

**Encodage asymétrique:** Jina v3 encode différemment les documents (passage) et les requêtes (query) pour optimiser la récupération.

**Factory Pattern:** Sélection automatique et transparente des parsers/chunkers selon le type de fichier.

**Singleton Pattern:** Réutilisation des instances d'embedder, storage et indexer pour économiser les ressources.

**Gestion d'erreurs:** Continuation du traitement même si certains fichiers échouent, avec logging détaillé.

**Métadonnées enrichies:** Chaque chunk conserve sa traçabilité complète (projet, fichier, position, type, langage).

## Configuration

Fichier `backend/app/config.py` :

```python
# Embedding
EMBEDDING_MODEL = "jinaai/jina-embeddings-v3"
EMBEDDING_DIMENSION = 1024
EMBEDDING_BATCH_SIZE = 32

# ChromaDB
CHROMA_DB_PATH = "./chroma_db"
CHROMA_COLLECTION_NAME = "redmine_projects"

# Chunking
CODE_CHUNK_SIZE = 800
TEXT_CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# Formats supportés
SUPPORTED_CODE_EXTENSIONS = {".cpp", ".hpp", ".h", ".py", ".java", ".js"}
SUPPORTED_DOC_EXTENSIONS = {".pdf", ".docx", ".pptx"}
```

## Pipeline détaillé

### 1. Parsing

**Parsers disponibles** :
- `PDFParser` : Docling (préserve structure, tableaux, métadonnées)
- `DOCXParser` : Docling pour Word
- `PPTXParser` : Docling pour PowerPoint
- `CodeParser` : Tree-sitter (C++, Python, Java, JS)
- `ImageParser` : Stub (vision à implémenter)

**Sortie** : `ParsedContent` avec text, content_type, language, metadata

### 2. Chunking

**TextChunker** :
- LangChain `RecursiveCharacterTextSplitter`
- Séparateurs : paragraphes → lignes → phrases → mots
- chunk_size=1000, overlap=100

**CodeChunker** :
- Découpage sémantique par fonction/classe
- Préserve commentaires et signatures
- chunk_size=800

### 3. Embedding

**Jina v3** :
- Modèle : `jinaai/jina-embeddings-v3`
- Dimension : 1024
- Task-specific : `retrieval.passage` pour docs, `retrieval.query` pour requêtes
- Batch processing (batch_size=32)
- Normalisation L2 pour similarité cosinus

### 4. Stockage

**ChromaDB** :
- PersistentClient avec path `./chroma_db`
- Collection : `redmine_projects`
- Métadonnées par chunk :
  ```python
  {
    'project_id': '123',
    'project_name': 'MonProjet',
    'source': '/path/to/file.cpp',
    'file_name': 'file.cpp',
    'file_type': '.cpp',
    'content_type': 'code',
    'language': 'cpp',
    'chunk_index': 0,
    'total_chunks': 5,
    # Métadonnées spécifiques au code
    'unit_type': 'function',
    'unit_name': 'parse_file',
    'signature': 'bool parse_file(const std::string& path)'
  }
  ```

## Métadonnées extraites

### Pour le code (Tree-sitter)

- `unit_type` : function, class, method
- `unit_name` : Nom de la fonction/classe
- `signature` : Signature complète
- `start_line`, `end_line` : Position dans le fichier
- `has_comments` : Présence de commentaires

### Pour les documents (Docling)

- `parser` : docling_pdf, docling_docx, docling_pptx
- `page_count` : Nombre de pages (PDF)
- `slide_count` : Nombre de slides (PPTX)
- `title`, `author` : Métadonnées du document

## Gestion d'erreurs

Le pipeline continue même en cas d'erreur sur un fichier :

```python
# Logs détaillés
✓ file1.cpp: 12 chunks
✓ doc.pdf: 8 chunks
✗ Failed to index corrupt.pdf: File corrupted
✓ code.py: 5 chunks

============================================================
Indexing Complete!
============================================================
✓ Successfully indexed: 3/4 files
✓ Total chunks created: 25
⚠ Failed files (1):
  - corrupt.pdf: File corrupted
```

## Statistiques

```python
stats = indexer.get_stats()
print(stats)
# {
#   'total_documents': 150,
#   'collection_name': 'redmine_projects',
#   'embedding_model': 'jinaai/jina-embeddings-v3',
#   'embedding_dimension': 1024
# }
```

## Nettoyage

```python
# Supprimer tous les documents d'un projet
indexer.clear_project(project_id='123')

# Supprimer toute la collection
indexer.storage.delete_collection()
```

## TODO / Améliorations futures

- [ ] Implémenter `ImageParser` avec vision model (GPT-4V, LLaVA)
- [ ] Support CSV/JSON avec parsers spécifiques
- [ ] Intégration API Redmine (Rayene a fait ça)
- [ ] Cache d'embeddings pour éviter recalcul
- [ ] Monitoring et métriques (latence, taux d'erreur)
