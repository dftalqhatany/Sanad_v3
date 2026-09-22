"""Sanad — the user interface shell.

    navigation + language  ->  one service page  ->  this module's service calls  ->  the Sanad API

This is the only frontend module that touches the API client, so the mapping from "what the user
asked for" to "what the service is asked to do" lives in exactly one place. No agent, orchestrator,
parser or RAG module is imported anywhere in this folder, and no page decides anything about the
answers it displays.

Run it with:

    python -m api                                   # service  (http://127.0.0.1:8000)
    PYTHONPATH=. streamlit run frontend/app.py      # this UI  (http://localhost:8501)
"""

from __future__ import annotations

import os
from typing import Any

import streamlit as st

from frontend import pages, theme
from frontend.client import DEFAULT_BASE_URL, SanadApiClient
from frontend.i18n import DEFAULT_LANGUAGE, LANGUAGES, t

# What each service asks the backend to do. These values belong to the API contract and are never
# shown to the user, who chooses a service instead.
SERVICE_TASKS = {
    "contract": "contract_analysis",
    "cv": "cv_analysis",
    "comparison": "contract_comparison",
    "salary": "salary_benchmark",
}
OFFLINE_CONFIG = {
    "allowed_file_types": [".pdf", ".docx"], "max_file_size_mb": 20, "max_files": 6, "max_contracts": 5,
    "comparison_priorities": [], "answer_generation_enabled": False, "salary_benchmarking_enabled": False,
}
NAV = (("home", "nav.home", "home"), ("ask", "nav.ask", "ask"), ("analyze", "nav.analyze", "analyze"),
       ("compare", "nav.compare", "compare"), ("salary", "nav.salary", "salary"))
SECONDARY_NAV = (("help", "nav.help", "book"), ("settings", "nav.settings", "settings"))

theme.page_config()
lang = st.session_state.setdefault("lang", DEFAULT_LANGUAGE)
theme.apply(lang)

# --------------------------------------------------------------------------- service connection
base_url = st.session_state.setdefault("api_url", os.environ.get("SANAD_API_URL", DEFAULT_BASE_URL))
client = SanadApiClient(base_url=base_url)
config_response = client.config()
online = bool(config_response.ok and config_response.result)
config: dict[str, Any] = config_response.result if online else dict(OFFLINE_CONFIG)


# --------------------------------------------------------------------------- service calls
def ask_question(question: str):
    """A question about the labor law, with no document attached."""
    return client.ask(question)


def analyze_contract(uploads, *, target_job: str | None = None):
    """One contract, optionally with a CV, checked against the regulations."""
    return client.analyze(uploads, task=SERVICE_TASKS["contract"], target_job_json=target_job)


def analyze_cv(uploads, *, target_job: str | None = None):
    return client.analyze(uploads, task=SERVICE_TASKS["cv"], target_job_json=target_job)


def compare_contracts(uploads, *, priorities=()):
    return client.analyze(uploads, task=SERVICE_TASKS["comparison"], priorities=list(priorities))


def benchmark_salary(uploads, *, job_title: str = "", city: str = "", years: str = ""):
    return client.analyze(uploads, task=SERVICE_TASKS["salary"],
                          salary={"salary_job_title": job_title, "salary_location": city,
                                  "years_experience": years})


def go(page: str) -> None:
    st.session_state["page"] = page
    st.rerun()


# --------------------------------------------------------------------------- navigation
theme.sidebar_logo()
theme.brand_block(t("app.name", lang), t("app.tagline", lang))

page = st.session_state.setdefault("page", "home")
theme.nav_label(t("nav.section", lang))
for key, label, icon_name in NAV:
    if st.sidebar.button(t(label, lang), key=f"nav_{key}", use_container_width=True,
                         type="primary" if page == key else "secondary"):
        page = st.session_state["page"] = key

st.sidebar.markdown("---")
for key, label, icon_name in SECONDARY_NAV:
    if st.sidebar.button(t(label, lang), key=f"nav_{key}", use_container_width=True,
                         type="primary" if page == key else "secondary"):
        page = st.session_state["page"] = key

theme.nav_label(t("nav.language", lang))
language_columns = st.sidebar.columns(len(LANGUAGES))
for column, (code, label) in zip(language_columns, LANGUAGES.items()):
    if column.button(label, key=f"lang_{code}", use_container_width=True,
                     type="primary" if lang == code else "secondary") and code != lang:
        st.session_state["lang"] = code
        st.rerun()

# --------------------------------------------------------------------------- page
context = pages.Ctx(
    lang=lang, config=config, online=online, go=go, ask=ask_question, analyze_contract=analyze_contract,
    analyze_cv=analyze_cv, compare=compare_contracts, benchmark=benchmark_salary,
)

if page == "settings":
    pages.settings_page(context, DEFAULT_BASE_URL)
else:
    pages.PAGES.get(page, pages.home)(context)

theme.footer(t("common.disclaimer", lang))
