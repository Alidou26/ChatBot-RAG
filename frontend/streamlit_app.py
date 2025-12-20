import streamlit as st
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configuration API
API_URL = "http://127.0.0.1:8000"
TIMEOUT = 1200  # 20 minutes


# ------------------------------------------------------------------------
# CONFIG STREAMLIT
# ------------------------------------------------------------------------
st.set_page_config(
    page_title="Assistant RAG SOLENT",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------
# STYLES
# ------------------------------------------------------------------------
st.markdown(
    """
<style>
/* Espacement des tabs */
.stTabs [data-baseweb="tab"] {
    padding-top: 18px !important;
    padding-bottom: 18px !important;
    margin-right: 100px !important;
}
.stTabs [data-baseweb="tab-list"] {
    margin-bottom: 5px !important;
}

/* Radio buttons */
div[role='radiogroup'] > label {
    padding-top: 15px !important;
    padding-bottom: 15px !important;
}

/* Chat bubbles slightly wider */
div[data-testid="stChatMessage"] {
    padding-top: 6px;
    padding-bottom: 6px;
}

/* Small badges */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 12px;
    line-height: 18px;
    border: 1px solid rgba(255,255,255,0.15);
}
.badge-ok { background: rgba(16,185,129,0.15); }
.badge-warn { background: rgba(245,158,11,0.15); }
.badge-off { background: rgba(239,68,68,0.15); }
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------
def check_backend_connection() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def is_no_answer(answer: str) -> bool:
    """
    Heuristic to detect fallback / refusal answers.
    We rely on backend self-rag to return empty sources, but this protects the UI.
    """
    a = (answer or "").strip().lower()
    if not a:
        return True
    patterns = [
        "ne permettent pas de répondre",
        "ne peux pas répondre",
        "pas suffisamment",
        "je suis désolé",
        "je ne sais pas",
        "question vide",
        "impossible",
        "pas de réponse",
        "ne permet pas",
    ]
    return any(p in a for p in patterns)


def extract_source_label(src: Dict[str, Any]) -> str:
    """
    Streamlit-safe source display label.
    """
    if not isinstance(src, dict):
        return "N/A"
    return str(src.get("file_name") or src.get("source") or src.get("file_path") or "N/A")


def extract_text(src: Dict[str, Any]) -> str:
    if not isinstance(src, dict):
        return ""
    return (
        src.get("text")
        or src.get("content")
        or src.get("chunk")
        or ""
    )


# ------------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------------
with st.sidebar:
    st.header("Navigation")

    page = st.radio(
        "Select",
        ["Chat RAG", "Analyse RCA", "Ingestion de données"],
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    st.subheader("État du backend")
    if check_backend_connection():
        st.success("Backend connecté")
    else:
        st.error("Backend déconnecté")
    st.divider()

    st.caption("API")
    st.code(API_URL, language="text")


# =============================================================================
#                               PAGE 1 — CHAT RAG
# =============================================================================
if page == "Chat RAG":
    st.title("Chat RAG")
    st.write("Posez vos questions. Les sources ne sont affichées que si une réponse fiable est possible.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ------------------------ AFFICHAGE HISTORIQUE ------------------------
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Display sources only if explicitly allowed and non-empty
            results = message.get("results")
            if message["role"] == "assistant" and isinstance(results, list) and len(results) > 0:
                with st.expander(f"{len(results)} passages pertinents"):
                    for i, r in enumerate(results, 1):
                        st.markdown(f"Passage {i} — Source: `{extract_source_label(r)}`")
                        text = extract_text(r)
                        with st.expander("Voir texte complet"):
                            st.code(text, language="text")

    # Paramètre K
    col1, col2 = st.columns([3, 1])
    with col2:
        k = st.slider("Nombre de passages", 1, 10, 3)

    # ------------------------ INPUT USER ------------------------
    if prompt := st.chat_input("Ex: Comment fonctionne l'indexation ?"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()

            with st.spinner("Recherche RAG..."):
                try:
                    response = requests.post(
                        f"{API_URL}/query",
                        json={"query": prompt, "top_k": k},
                        timeout=TIMEOUT,
                    )

                    if response.status_code == 200:
                        data = response.json()
                        answer = (data.get("answer") or "").strip()
                        sources = data.get("sources") or []

                        # Decide if we show sources:
                        # - If backend returns no sources, we do not show them.
                        # - If answer is a refusal / no-answer, we do not show them even if present.
                        show_sources = isinstance(sources, list) and len(sources) > 0 and not is_no_answer(answer)

                        if show_sources:
                            placeholder.markdown(
                                f'<span class="badge badge-ok">Réponse basée sur les documents</span>',
                                unsafe_allow_html=True,
                            )
                            st.markdown(answer)
                        else:
                            placeholder.markdown(
                                f'<span class="badge badge-warn">Contexte insuffisant</span>',
                                unsafe_allow_html=True,
                            )
                            st.markdown(answer if answer else "Les documents disponibles ne permettent pas de répondre de manière fiable à cette question.")

                        # Save assistant message.
                        # IMPORTANT: only store "results" if show_sources == True
                        msg: Dict[str, Any] = {"role": "assistant", "content": answer}
                        if show_sources:
                            msg["results"] = sources
                        st.session_state.messages.append(msg)

                    else:
                        placeholder.error(f"Erreur API {response.status_code}")

                except Exception as e:
                    placeholder.error(f"Erreur : {str(e)}")

        st.rerun()

    if st.session_state.messages:
        if st.button("Effacer l'historique"):
            st.session_state.messages = []
            st.rerun()


# =============================================================================
#                           PAGE 2 — ANALYSE RCA
# =============================================================================
elif page == "Analyse RCA":
    st.title("Analyse RCA — Root Cause Analysis")

    if "rca_messages" not in st.session_state:
        st.session_state.rca_messages = []

    for msg in st.session_state.rca_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            results = msg.get("results")
            if msg["role"] == "assistant" and isinstance(results, list) and len(results) > 0:
                with st.expander("Sources utilisées"):
                    for i, r in enumerate(results, 1):
                        st.markdown(f"Source {i} — `{extract_source_label(r)}`")
                        text = extract_text(r)
                        with st.expander("Voir texte"):
                            st.code(text)

    col1, col2 = st.columns([3, 1])
    with col2:
        k_rca = st.slider("Nombre de sources", 1, 10, 5)

    if problem := st.chat_input("Décrivez le problème..."):
        st.session_state.rca_messages.append({"role": "user", "content": problem})

        with st.chat_message("assistant"):
            placeholder = st.empty()

            with st.spinner("Analyse RCA..."):
                try:
                    response = requests.post(
                        f"{API_URL}/root_cause",
                        json={"description": problem, "top_k": k_rca},
                        timeout=TIMEOUT,
                    )

                    if response.status_code == 200:
                        data = response.json()
                        answer = (data.get("answer") or "").strip()
                        sources = data.get("sources") or []

                        show_sources = isinstance(sources, list) and len(sources) > 0 and not is_no_answer(answer)

                        if show_sources:
                            placeholder.markdown(
                                f'<span class="badge badge-ok">Analyse basée sur les sources</span>',
                                unsafe_allow_html=True,
                            )
                            st.markdown(answer)
                        else:
                            placeholder.markdown(
                                f'<span class="badge badge-warn">Sources insuffisantes</span>',
                                unsafe_allow_html=True,
                            )
                            st.markdown(answer if answer else "Les documents disponibles ne permettent pas de produire une analyse RCA fiable.")

                        msg: Dict[str, Any] = {"role": "assistant", "content": answer}
                        if show_sources:
                            msg["results"] = sources
                        st.session_state.rca_messages.append(msg)

                    else:
                        placeholder.error(f"Erreur API {response.status_code}")

                except Exception as e:
                    placeholder.error(f"Erreur : {str(e)}")

        st.rerun()

    if st.session_state.rca_messages:
        if st.button("Effacer l'historique RCA"):
            st.session_state.rca_messages = []
            st.rerun()


# =============================================================================
#                               PAGE 3 — INGESTION
# =============================================================================
elif page == "Ingestion de données":
    st.title("Ingestion de données")
    st.write("Ajoutez des documents dans ChromaDB.")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Fichiers", "URLs", "Archives", "Redmine (Projet complet)"]
    )

    # -------------------- TAB 1 : FICHIERS --------------------
    with tab1:
        st.header("Ingestion de fichiers")
        files = st.file_uploader(
            "Sélectionnez les fichiers",
            type=["pdf", "txt", "json", "docx", "pptx"],
            accept_multiple_files=True,
        )

        if st.button("Ingérer fichiers", use_container_width=True):
            if not files:
                st.warning("Aucun fichier sélectionné.")
            else:
                with st.spinner("Ingestion en cours..."):
                    payload = [("files", (f.name, f.getbuffer(), f.type)) for f in files]
                    try:
                        r = requests.post(
                            f"{API_URL}/ingest/files",
                            files=payload,
                            timeout=TIMEOUT,
                        )
                        if r.status_code == 200:
                            st.success("Ingestion réussie.")
                        else:
                            st.error(f"Erreur API : {r.status_code}")
                    except Exception as e:
                        st.error(str(e))

    # -------------------- TAB 2 : URL --------------------
    with tab2:
        st.header("Ingestion URL")
        url_text = st.text_area("Entrez l'url (une seule)")

        if st.button("Ingérer URL"):
            urls = [u.strip() for u in url_text.split("\n") if u.strip()]
            if not urls:
                st.warning("Aucune URL fournie.")
            else:
                with st.spinner("Ingestion URL..."):
                    try:
                        r = requests.post(
                            f"{API_URL}/ingest/urls",
                            json=urls,
                            timeout=TIMEOUT,
                        )
                        if r.status_code == 200:
                            st.success("URLs ingérées avec succès.")
                        else:
                            st.error(f"Erreur API : {r.status_code}")
                    except Exception as e:
                        st.error(str(e))

    # -------------------- TAB 3 : ARCHIVES --------------------
    with tab3:
        st.header("Ingestion archive ZIP / TAR")

        col1, col2 = st.columns(2)
        project_id = col1.text_input("ID projet", value="default")
        project_name = col2.text_input("Nom projet", value="default")

        archive = st.file_uploader("Archive", type=["zip", "tar", "gz"])

        if st.button("Ingérer archive", use_container_width=True):
            if not archive:
                st.warning("Sélectionnez une archive.")
            else:
                with st.spinner("Ingestion de l’archive..."):
                    try:
                        files_payload = [("file", (archive.name, archive.getbuffer(), archive.type))]
                        r = requests.post(
                            f"{API_URL}/ingest/archive",
                            files=files_payload,
                            data={"project_id": project_id, "project_name": project_name},
                            timeout=TIMEOUT,
                        )
                        if r.status_code == 200:
                            st.success("Archive ingérée.")
                        else:
                            st.error(f"Erreur API : {r.status_code}")
                    except Exception as e:
                        st.error(str(e))

    # -------------------- TAB 4 : REDMINE --------------------
    with tab4:
        st.header("Ingestion projet Redmine")
        st.write("Ingère : issues, wikis, fichiers, news, versions, membres...")

        project = st.text_input("ID projet Redmine", placeholder="ex: solent-rca-test")

        if st.button("Ingérer projet Redmine", use_container_width=True):
            if not project.strip():
                st.warning("Entrez un ID projet.")
            else:
                with st.spinner("Ingestion complète..."):
                    try:
                        r = requests.post(
                            f"{API_URL}/redmine/ingest/{project}",
                            timeout=TIMEOUT,
                        )
                        if r.status_code == 200:
                            data = r.json()
                            st.success(f"{data.get('chunks_indexed', 0)} chunks ingérés.")
                        else:
                            st.error(f"Erreur API : {r.status_code}")
                    except Exception as e:
                        st.error(str(e))
