"""
Ingestion complète Redmine → Archive (.tar.gz) → ChromaDB via ingest_archive

Étapes :
1. Récupérer toutes les données du projet Redmine (issues, wiki, files, etc.)
2. Reconstruire un dossier projet contenant :
   - project_info.txt
   - trackers.txt, versions.txt, memberships.txt, categories.txt, news.txt
   - issues/issue_X.txt + pièces jointes
   - wiki/*.txt + pièces jointes
   - project_files/ (fichiers de projet Redmine)
3. Compresser ce dossier en archive (.tar.gz)
4. Appeler ingest_archive(archive_path, project_id, project_name, collection_name)
"""

import os
import json
import shutil
import requests
import tempfile
from pathlib import Path
from typing import List, Dict, Any

from .fetch_redmine import fetch_full_project_data
from ..ingestion.dataset_ingester import ingest_archive  
from ..config import CHROMA_COLLECTION_NAME


# -----------------------------------------------
# HELPERS
# -----------------------------------------------

def _save_text(content: str, folder: Path, filename: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def _download(url: str, dest: Path, api_key: str, filename: str) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    local = dest / filename
    resp = requests.get(url, headers={"X-Redmine-API-Key": api_key}, timeout=30)
    resp.raise_for_status()
    with open(local, "wb") as f:
        f.write(resp.content)
    return local


# -----------------------------------------------
# MAIN INGESTION FUNCTION
# -----------------------------------------------

def ingest_full_redmine_project(project_id: str) -> int:
    print(f"\n=== FULL REDMINE INGESTION (archive) : {project_id} ===")

    # 1) Fetch full data
    data = fetch_full_project_data(project_id, status_id="*")
    if not data:
        print("Impossible de charger le projet.")
        return 0

    proj = data.get("project", {})
    proj_name = proj.get("name", project_id)
    api_key = os.getenv("REDMINE_API_KEY", "")

    # 2) Dossier temporaire pour construire l'arborescence du projet
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        project_root = tmp_dir / project_id
        project_root.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------
        # 2.2 Ingest project metadata
        # ------------------------------------------------------
        txt = (
            f"Projet ID: {proj.get('id')}\n"
            f"Nom: {proj_name}\n"
            f"Description: {proj.get('description')}\n"
            f"Homepage: {proj.get('homepage')}\n"
            f"Créé le: {proj.get('created_on')}\n"
        )
        _save_text(txt, project_root / "project", "project_info.txt")

        # ------------------------------------------------------
        # 2.3 Trackers
        # ------------------------------------------------------
        trackers = data.get("trackers", [])
        if trackers:
            txt = "Trackers:\n\n"
            for t in trackers:
                txt += f"- {t.get('id')}: {t.get('name')}\n"
            _save_text(txt, project_root / "trackers", "trackers.txt")

        # ------------------------------------------------------
        # 2.4 Versions
        # ------------------------------------------------------
        versions = data.get("versions", [])
        if versions:
            txt = "Versions:\n\n"
            for v in versions:
                txt += (
                    f"- {v.get('id')}: {v.get('name')} "
                    f"(status={v.get('status')}, due={v.get('due_date')})\n"
                )
            _save_text(txt, project_root / "versions", "versions.txt")

        # ------------------------------------------------------
        # 2.5 Memberships
        # ------------------------------------------------------
        members = data.get("memberships", [])
        if members:
            txt = "Members:\n\n"
            for m in members:
                user = m.get("user", {})
                roles = m.get("roles", [])
                role_name = roles[0].get("name") if roles else "Unknown"
                txt += f"- {user.get('name')} (role: {role_name})\n"
            _save_text(txt, project_root / "memberships", "memberships.txt")

        # ------------------------------------------------------
        # 2.6 Issue categories
        # ------------------------------------------------------
        categories = data.get("issue_categories", [])
        if categories:
            txt = "Issue Categories:\n\n"
            for c in categories:
                txt += f"- {c.get('id')}: {c.get('name')}\n"
            _save_text(txt, project_root / "categories", "categories.txt")

        # ------------------------------------------------------
        # 2.7 News
        # ------------------------------------------------------
        news = data.get("news", [])
        if news:
            txt = "News:\n\n"
            for n in news:
                txt += (
                    f"Title: {n.get('title')}\n"
                    f"Author: {n.get('author', {}).get('name')}\n"
                    f"Description: {n.get('description')}\n"
                    f"--\n"
                )
            _save_text(txt, project_root / "news", "news.txt")

        # ------------------------------------------------------
        # 2.8 Project files (fichiers Redmine "Files" du projet)
        # ------------------------------------------------------
        for f in data.get("files", []):
            url = f.get("content_url") or f.get("download_url")
            name = f.get("filename")
            if url and name:
                try:
                    _download(url, project_root / "project_files", api_key, name)
                except Exception as e:
                    print(f"Unable to download project file {name}: {e}")

        # ------------------------------------------------------
        # 2.9 Issues + attachments + journals + custom fields
        # ------------------------------------------------------
        for issue in data.get("issues", []):
            iid = issue["id"]

            text = (
                f"Issue ID: {iid}\n"
                f"Subject: {issue.get('subject')}\n"
                f"Description:\n{issue.get('description')}\n\n"
            )

            # custom fields
            for cf in issue.get("custom_fields", []):
                text += f"CustomField {cf.get('name')}: {cf.get('value')}\n"

            # journals
            for j in issue.get("journals", []):
                notes = j.get("notes", "")
                if notes:
                    text += f"[Journal] {notes}\n"

            issue_folder = project_root / "issues"
            _save_text(text, issue_folder, f"issue_{iid}.txt")

            # attachments
            for att in issue.get("attachments", []):
                url = att.get("content_url")
                name = att.get("filename")
                if url and name:
                    try:
                        _download(
                            url,
                            project_root / "issues" / f"issue_{iid}",
                            api_key,
                            name
                        )
                    except Exception as e:
                        print(f"Attachment download failed: {name} → {e}")

        # ------------------------------------------------------
        # 2.10 Wiki pages + attachments
        # ------------------------------------------------------
        for page in data.get("wiki_pages", []):
            title = page.get("title", "untitled")
            text = page.get("text", "")

            safe_title = title.replace(" ", "_")
            _save_text(
                f"WIKI PAGE: {title}\n\n{text}",
                project_root / "wiki",
                f"{safe_title}.txt"
            )

            for att in page.get("attachments", []):
                url = att.get("content_url")
                name = att.get("filename")
                if url and name:
                    try:
                        _download(
                            url,
                            project_root / "wiki" / safe_title,
                            api_key,
                            name
                        )
                    except Exception as e:
                        print(f"Wiki attachment failed {name}: {e}")

        # ------------------------------------------------------
        # 3) Création de l’archive (.tar.gz) du dossier projet_root
        # ------------------------------------------------------
        archive_base = tmp_dir / f"{project_id}_redmine_dump"
        # crée : {archive_base}.tar.gz
        archive_path_str = shutil.make_archive(
            base_name=str(archive_base),
            format="gztar",          
            root_dir=str(tmp_dir),
            base_dir=project_root.name,
        )
        archive_path = Path(archive_path_str)

        print(f"Archive Redmine créée : {archive_path}")

        # ------------------------------------------------------
        # 4) Ingestion via ingest_archive (ta fonction existante)
        # ------------------------------------------------------
        ingested_chunks = ingest_archive(
            archive_path=archive_path,
            project_id=project_id,
            project_name=proj_name,
            collection_name=CHROMA_COLLECTION_NAME or "redmine_projects",
        )

        print(
            f" Ingestion terminée – {ingested_chunks} nouveaux chunks ajoutés "
            f"pour le projet {project_id}."
        )
        return ingested_chunks
