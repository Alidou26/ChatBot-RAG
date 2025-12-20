# Guide d'installation et d'exécution du projet

Ce document décrit les étapes nécessaires pour installer et exécuter le projet en local sur macOS et Windows.  
Le système implémente un pipeline RAG avancé intégrant RAG-Fusion (multi-requêtes + RRF) et Self-RAG (gating du contexte).

---

## 1. Cloner le dépôt GitHub

### Cloner le dépôt
```bash
git clone https://github.com/AchrafEssaleh/AI-Assistant-for-Technical-Knowledge-Management.git

Accéder au dossier du projet

cd AI-Assistant-for-Technical-Knowledge-Management
```
Le développement et l'exécution se font directement depuis la branche main.
## 2. Ouvrir le projet

    Ouvrir le dossier dans Visual Studio Code

    Vérifier que Python 3.10 ou supérieur est installé

## 3. Créer un environnement virtuel
macOS / Linux

python3 -m venv venv

Windows

python -m venv venv

## 4. Activer l'environnement virtuel
macOS / Linux

source venv/bin/activate

Windows

venv\Scripts\activate

## 5. Installer les dépendances
Backend

pip install -r backend/requirements.txt

Frontend

pip install -r frontend/requirements.txt

## 6. Installer Ollama
macOS

https://ollama.com/download/mac
Windows

https://ollama.com/download/windows
## 7. Télécharger le modèle LLM

Le pipeline est compatible avec tout modèle Ollama.
Modèle recommandé et validé :

ollama pull qwen2.5:3b

## 8. Configuration de l'environnement

Créer un fichier .env à la racine du projet :

LLM_BASE_URL=http://127.0.0.1:11434
LLM_MODEL=qwen2.5:3b
LLM_TEMPERATURE=0
LLM_TIMEOUT=180

RAG_FUSION_ENABLED=1
RAG_FUSION_RRF_K=60

## 9. Lancer le serveur Ollama

ollama serve

Le serveur Ollama doit rester actif pendant toute l'exécution.
## 10. Lancer l'API FastAPI

Dans un nouveau terminal (venv activé) :

uvicorn backend.app.main:app --reload

## 11. Lancer l'interface Streamlit

Dans un autre terminal (venv activé) :

streamlit run frontend/streamlit_app.py

## 12. Accès aux services

    API FastAPI : http://127.0.0.1:8000

Documentation OpenAPI : http://127.0.0.1:8000/docs

Interface utilisateur Streamlit : http://localhost:8501
## 13. Fonctionnalités RAG implémentées
- RAG-Fusion

    Expansion déterministe de requêtes 

    Recherche multi-requêtes dans ChromaDB

    Fusion des résultats par Reciprocal Rank Fusion (RRF)

    Déduplication stricte des chunks

- Self-RAG

    Gating déterministe avant génération

    Vérification du nombre de chunks

    Vérification du score moyen de similarité

    Vérification du volume de contexte disponible

    Refus explicite de génération si le contexte est insuffisant

Ces mécanismes garantissent des réponses contrôlées, traçables et ancrées exclusivement dans les documents indexés.