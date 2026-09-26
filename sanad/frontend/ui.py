"""Result rendering for the Sanad interface.

Every value shown here comes from the service response, passed through `frontend/view.py`. Nothing
in this module decides, scores, ranks, re-orders by quality or fills a gap: a missing value is shown
as missing. The only things added are the layout, the semantic colour of a state, and the labels in
the chosen interface language.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from frontend import theme, view
from frontend.i18n import t

# A finding's state -> (tone, mark, label key). The states are the ones the service returns.
FINDING_STATES = {
    "compliant": ("good", "✓", "status.compliant"),
    "non_compliant": ("bad", "✕", "status.non_compliant"),
    "requires_review": ("caution", "⚠", "status.attention"),
    "ambiguous": ("caution", "❓", "status.ambiguous"),
    "insufficient_evidence": ("neutral", "—", "status.insufficient"),
    "not_found": ("neutral", "—", "status.not_found"),
    "not_applicable": ("neutral", "N/A", "status.document_fact"),
    "error": ("caution", "⚠", "status.check_failed"),
}
# Legal Compliance dashboard: which statuses get their own group, and in what order (Part 9 - a
# compliant/non-compliant verdict is read first, then what needs a human, then what could not be
# evaluated at all). Never invented: a status this dict does not name simply gets no group header.
COMPLIANCE_GROUPS = ("compliant", "non_compliant", "requires_review", "ambiguous", "not_applicable", "not_found")
# The overall state of a request -> (tone, label key).
REQUEST_STATES = {
    "success": ("good", "status.completed"),
    "partial": ("caution", "status.partial"),
    "insufficient_evidence": ("caution", "status.insufficient"),
    "invalid_input": ("bad", "status.rejected"),
    "rag_error": ("bad", "status.rag_error"),
    "analysis_error": ("bad", "status.failed"),
}
COMPATIBILITY_TONES = {"met": "good", "partially met": "caution", "not met": "bad"}
# The states a reader has to do something about, most urgent first. Display order only.
ACTION_ORDER = ("non_compliant", "requires_review", "error", "compliant")


# --------------------------------------------------------------------------- shared pieces
def request_banner(result: dict[str, Any], lang: str) -> dict[str, Any]:
    """The one-line outcome of the request, in the service's own words."""
    summary = view.overview(result)
    tone, label = REQUEST_STATES.get(summary["status"], ("neutral", "status.completed"))
    theme.banner(tone, t(label, lang), summary["summary"])
    return summary


def documents_panel(result: dict[str, Any], lang: str) -> None:
    rows = view.document_rows(result)
    if not rows:
        return
    with st.expander(f"{t('common.documents', lang)} ({len(rows)})", expanded=False):
        st.dataframe(rows, use_container_width=True, hide_index=True)


def caveats_panel(result: dict[str, Any], lang: str) -> None:
    notes = view.caveats(result)
    if not notes:
        return
    with st.expander(f"{t('common.notes', lang)} ({len(notes)})", expanded=False):
        st.markdown('<ul class="sanad-list">' + "".join(f"<li>{note}</li>" for note in notes) + "</ul>",
                    unsafe_allow_html=True)


def raw_panel(result: dict[str, Any], lang: str) -> None:
    if not st.session_state.get("show_raw"):
        return
    with st.expander(t("settings.raw", lang), expanded=False):
        st.json(result)


def unavailable_state(result: dict[str, Any], lang: str) -> bool:
    """True when the whole request failed. The reason is shown in words, never as an error code."""
    status = result.get("status")
    if status == "rag_error":
        theme.empty_state(t("state.rag_down", lang), t("state.rag_down_body", lang), "shield")
    elif status in ("invalid_input", "analysis_error"):
        theme.empty_state(t("state.error_title", lang), _reason(result), "shield")
    else:
        return False
    _technical_details(result, lang)
    return True


def _reason(result: dict[str, Any]) -> str:
    """The service's own sentence about what went wrong, without the internal code in front of it."""
    errors = result.get("errors") or []
    message = errors[0].get("message") if errors else ""
    return message or result.get("summary", "")


def _technical_details(result: dict[str, Any], lang: str) -> None:
    """Codes and agent names are for whoever is running the service, not for the person using it."""
    if not st.session_state.get("show_raw"):
        return
    details = [f"{e.get('code')}: {e.get('message')}" for e in result.get("errors", [])]
    if details:
        with st.expander(t("settings.developer", lang)):
            st.markdown('<ul class="sanad-list">' + "".join(f"<li>{item}</li>" for item in details) + "</ul>",
                        unsafe_allow_html=True)


# --------------------------------------------------------------------------- regulatory answer
def answer_panel(result: dict[str, Any], lang: str) -> None:
    answer = view.regulatory_answer(result)
    if not answer:
        return
    theme.section(t("ask.answer_title", lang))
    if answer["answer"]:
        st.markdown(answer["answer"])
    elif not answer["evidence"]:
        # nothing was retrieved at all: say so in the user's language rather than showing an empty panel
        theme.banner("neutral", t("evidence.no_context", lang))
    else:
        theme.banner("neutral", t("ask.no_answer", lang))
    if answer["message"]:
        st.caption(answer["message"])
    if answer["evidence"]:
        theme.section(t("common.sources", lang))
        for item in answer["evidence"]:
            # "Similarity" = how closely this article matches the question. It is NOT a statement about
            # how correct the answer is, so it is never labelled accuracy or confidence.
            percent = item.get("similarity_percentage")
            label = item["citation"]
            if percent is not None:
                label = f'{label}  ·  {t("evidence.similarity", lang)}: {percent}%'
            with st.expander(label):
                _article(item.get("arabic_text"), item.get("english_text"), lang)


def _article(arabic: str | None, english: str | None, lang: str) -> None:
    if arabic:
        st.markdown(f'<p class="sanad-drop__hint">{t("common.arabic_text", lang)}</p>'
                    f'<div class="sanad-quote sanad-quote--ar">{arabic}</div>', unsafe_allow_html=True)
    if english:
        st.markdown(f'<p class="sanad-drop__hint" style="margin-top:12px">{t("common.english_text", lang)}</p>'
                    f'<div class="sanad-quote">{english}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- contract analysis
# NOTE ON analysis_panel BELOW: it renders `view.finding_rows`, the Stage 4 FIELD-LEVEL findings.
# Those are informational only and can never carry a compliant/non-compliant verdict (the field-level
# reading is unconditionally demoted to "requires_review", by an enforced backend validator, even when
# an interpreter's own explanation reads as compliant - see agents/contract_analysis.py). This
# function is therefore never used as the Legal Compliance dashboard any more (see
# legal_compliance_panel below, which is the one wired into the Analyze Contract tabs): showing
# analysis_panel's statuses as "the" compliance result is exactly the "explanation says compliant,
# status says needs attention" contradiction Sanad must not show. It is kept, unused by the dashboard,
# only as a still-tested pure function (`view.finding_rows` has direct test coverage) and a home for
# the per-field retrieved-evidence view.
def analysis_panel(result: dict[str, Any], lang: str) -> None:
    analysis = view.contract_analysis(result)
    if not analysis:
        return
    theme.section(t("analyze.compliance_title", lang))
    summary = analysis.get("overall_summary", "")
    if summary and summary not in (result.get("summary") or ""):
        st.write(summary)

    counts = analysis.get("status_counts") or {}
    if counts:
        _state_summary(counts, lang)

    rows = view.finding_rows(analysis)
    findings = analysis.get("findings", [])
    if not rows:
        theme.empty_state(t("state.no_results", lang), t("ui.no_findings", lang))
        return

    # Display order only: what the reader has to act on first, then the terms that were merely
    # recorded. No finding is hidden, reworded, merged or re-assessed.
    pairs = list(zip(rows, findings))
    ranked = [p for p in pairs if p[1].get("status") in ACTION_ORDER]
    ranked.sort(key=lambda pair: ACTION_ORDER.index(pair[1].get("status")))
    rest = [p for p in pairs if p[1].get("status") not in ACTION_ORDER]

    theme.section(t("analyze.findings_title", lang))
    for row, finding in ranked:
        _finding_card(row, finding, lang)
    if rest:
        with st.expander(t("analyze.other_findings", lang, n=len(rest))):
            for row, finding in rest:
                _finding_card(row, finding, lang)

    evidence = view.evidence_items(analysis)
    if evidence:
        with st.expander(t("ui.retrieved_articles", lang, n=len(evidence))):
            for item in evidence:
                st.markdown(f"**{item['citation']}**")
                _article(item.get("arabic_text"), item.get("english_text"), lang)
                st.divider()


# --------------------------------------------------------------------------- legal compliance (authoritative)
def legal_compliance_panel(result: dict[str, Any], lang: str) -> None:
    """The Legal Compliance tab: a compliance dashboard built ONLY from `view.legal_compliance_findings`
    - the deterministic clause_findings, never a field-level or LLM opinion (see that function's
    docstring). Grouped by status so a reader sees compliant items, then what needs review, then what
    could not be evaluated; a status is never inferred from explanation wording.
    """
    theme.section(t("analyze.tab_compliance", lang))
    rows = view.legal_compliance_findings(result)
    if not rows:
        theme.empty_state(t("state.no_results", lang), t("analyze.no_compliance_data", lang))
        return

    _state_summary(view.legal_compliance_counts(result), lang)

    by_status: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_status.setdefault(row["status"], []).append(row)

    for status in COMPLIANCE_GROUPS:
        group = by_status.get(status)
        if not group:
            continue
        tone, mark, label = FINDING_STATES.get(status, ("neutral", "—", "status.unavailable"))
        theme.spacer(6)
        st.markdown(f"**{mark} {t(label, lang)}**")
        for row in group:
            _compliance_row(row, lang)


def _compliance_row(row: dict[str, Any], lang: str) -> None:
    """One clause: its own name and words, the article Sanad checked it against, and why."""
    title = str(row.get("clause_name") or (row.get("field") or "").replace("_", " ")).strip().capitalize()
    article = row.get("article_number")
    article_line = (f'<span class="sanad-finding__articles">{t("ui.regulation", lang)}: {article}</span>'
                    if article else "")
    value = row.get("contract_value")
    value_line = (f'<p class="sanad-finding__text"><strong>{t("analyze.contract_says", lang)}:</strong> {value}</p>'
                  if value and value != "-" else "")
    st.markdown(
        f'<div class="sanad-finding"><div class="sanad-finding__head">'
        f'<span class="sanad-finding__title">{title}</span>{article_line}</div>'
        f'{value_line}'
        f'<p class="sanad-finding__text">{row.get("explanation", "")}</p></div>',
        unsafe_allow_html=True,
    )
    evidence = [e for e in row.get("evidence", []) if e.get("citation")]
    if evidence:
        with st.expander(t("ui.show_evidence", lang)):
            for item in evidence:
                st.markdown(f"**{item['citation']}**")
                _article(item.get("arabic_text"), item.get("english_text"), lang)


def _state_summary(counts: dict[str, int], lang: str) -> None:
    """The service's own per-state counts, shown as pills. Nothing is added up or scored."""
    pills = []
    for state, (tone, mark, label) in FINDING_STATES.items():
        number = counts.get(state)
        if number:
            pills.append(theme.pill(f"{mark} {t(label, lang)} · {number}", tone))
    if pills:
        st.markdown('<div style="display:flex;flex-wrap:wrap;gap:8px;margin:2px 0 6px">'
                    + "".join(pills) + "</div>", unsafe_allow_html=True)


def _finding_card(row: dict[str, Any], finding: dict[str, Any], lang: str) -> None:
    tone, mark, label = FINDING_STATES.get(finding.get("status", ""), ("neutral", "—", "status.unavailable"))
    title = str(row.get("Field", "")).strip().capitalize()
    articles = row.get("Articles") or "-"
    article_line = (f'<span class="sanad-finding__articles">{t("ui.regulation", lang)}: {articles}</span>'
                    if articles != "-" else "")
    st.markdown(
        f'<div class="sanad-finding"><div class="sanad-finding__head">'
        f'{theme.pill(f"{mark} {t(label, lang)}", tone)}'
        f'<span class="sanad-finding__title">{title}</span>{article_line}</div>'
        f'<p class="sanad-finding__text">{row.get("Explanation", "")}</p></div>',
        unsafe_allow_html=True,
    )
    contract_says = row.get("Contract says")
    source_text = row.get("Source text")
    quotes = [q for q in ((finding.get("interpretation") or {}).get("evidence_quotes") or []) if q.get("quote")]
    if not (contract_says and contract_says != "-") and not source_text and not quotes:
        return
    with st.expander(t("ui.show_evidence", lang)):
        if contract_says and contract_says != "-":
            page = row.get("Page")
            suffix = f" · {t('analyze.page', lang)} {page}" if page and page != "-" else ""
            st.markdown(f'<p class="sanad-drop__hint">{t("analyze.contract_says", lang)}{suffix}</p>'
                        f'<div class="sanad-quote">{contract_says}</div>', unsafe_allow_html=True)
        if source_text:
            st.markdown(f'<div class="sanad-quote sanad-quote--ar">{source_text}</div>', unsafe_allow_html=True)
        for quote in quotes:
            css = "sanad-quote sanad-quote--ar" if quote.get("language") == "ar" else "sanad-quote"
            st.markdown(f'<p class="sanad-drop__hint" style="margin-top:12px">{t("ui.regulation", lang)}</p>'
                        f'<div class="{css}">{quote["quote"]}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- contract analysis: dashboard
# Value placeholders view.py hands back for a field that has nothing, or something other than a plain
# found value, to show - never a language decision made in view.py, only made here at render time.
_VALUE_PLACEHOLDERS = {
    "__not_found__": "common.not_found",
    "__not_applicable__": "common.not_applicable",
    "__ambiguous__": "common.ambiguous",
}


def _display(value: str, lang: str) -> str:
    key = _VALUE_PLACEHOLDERS.get(value)
    return t(key, lang) if key else value


def contract_dashboard(result: dict[str, Any], lang: str) -> None:
    """The Analyze Contract results: one dashboard, five sections reached through the tabs below.

    Nothing here decides a status, a value or a ranking - every section reads what
    `agents.contract_analysis.ContractAnalysisAgent` (via the existing API) already returned,
    including the salary benchmark, which is already part of that same result.
    """
    analysis = view.contract_analysis(result)
    if not analysis:
        # A CV-only analysis, or a failed one: nothing to show as a contract dashboard.
        request_banner(result, lang)
        cv_panel(result, lang)
        return

    documents = view.document_rows(result)
    if documents:
        st.markdown(f'<div class="sanad-file">{theme.icon("file", 18, theme.GREEN_600)}'
                    f'<span class="sanad-file__name">{documents[0]["File"]}</span></div>',
                    unsafe_allow_html=True)
    request_banner(result, lang)

    cv = view.cv_analysis(result)
    labels = [t("analyze.tab_overview", lang), t("analyze.tab_compliance", lang),
              t("analyze.tab_compensation", lang), t("analyze.tab_salary_benchmark", lang),
              t("analyze.tab_findings", lang)]
    if cv:
        labels.append(t("analyze.tab_cv", lang))
    tabs = st.tabs(labels)

    with tabs[0]:
        overview_panel(result, lang)
    with tabs[1]:
        legal_compliance_panel(result, lang)
    with tabs[2]:
        compensation_panel(result, lang)
    with tabs[3]:
        salary_panel(result, lang, heading=False)
    with tabs[4]:
        findings_panel(result, lang)
    if cv:
        with tabs[5]:
            cv_panel(result, lang)


def overview_panel(result: dict[str, Any], lang: str) -> None:
    """Contract Overview: the default screen - a summary dashboard, not a wizard step."""
    theme.section(t("analyze.tab_overview", lang))
    _contract_information_card(result, lang)
    theme.spacer(10)

    columns = st.columns(3, gap="large")
    with columns[0]:
        _compliance_summary_card(result, lang)
    with columns[1]:
        _compensation_summary_card(result, lang)
    with columns[2]:
        _salary_benchmark_summary_card(result, lang)

    theme.spacer(10)
    _findings_summary_card(result, lang)


def _contract_information_card(result: dict[str, Any], lang: str) -> None:
    theme.section(t("analyze.contract_info_title", lang))
    rows = view.contract_overview_rows(result)
    if not rows:
        theme.empty_state(t("state.no_results", lang), t("ui.no_findings", lang))
        return
    columns = st.columns(2, gap="large")
    for index, row in enumerate(rows):
        with columns[index % 2]:
            theme.metric(row["label"], _display(row["display"], lang))


# A dominant tone/mark for the compact Legal Compliance glance, worst-first: one non-compliant
# clause outweighs any number of compliant ones, matching the order a reader should worry about.
_COMPLIANCE_GLANCE = (
    ("non_compliant", "bad", "✕", "status.non_compliant"),
    ("requires_review", "caution", "!", "status.attention"),
    ("ambiguous", "caution", "?", "status.ambiguous"),
    ("compliant", "good", "✓", "status.compliant"),
)


def _compliance_summary_card(result: dict[str, Any], lang: str) -> None:
    """Overview's compact card: a single glance mark plus the counts behind it - the same
    authoritative rows as the Legal Compliance tab (view.legal_compliance_findings), just
    summarised, so the two can never disagree."""
    label = t("analyze.legal_compliance_card", lang)
    counts = view.legal_compliance_counts(result)
    if not counts:
        theme.metric(label, "-", t("analyze.no_compliance_data", lang))
        return
    tone, mark, label_key = next(((tone, mark, key) for status, tone, mark, key in _COMPLIANCE_GLANCE
                                  if counts.get(status)), ("neutral", "—", "status.unavailable"))
    note = " · ".join(f"{n} {t(FINDING_STATES[s][2], lang).lower()}" for s, n in counts.items() if s in FINDING_STATES)
    st.markdown(
        f'<div class="sanad-metric"><span class="sanad-metric__label">{label}</span>'
        f'<div style="margin-top:8px" class="sanad-glance">'
        f'<span class="sanad-glance__mark sanad-glance__mark--{tone}">{mark}</span>'
        f'<span class="sanad-glance__text"><span class="sanad-glance__title">{t(label_key, lang)}</span>'
        f'<span class="sanad-glance__note">{note}</span></span></div></div>',
        unsafe_allow_html=True,
    )


def _compensation_summary_card(result: dict[str, Any], lang: str) -> None:
    rows = {row["field"]: row for row in view.compensation_summary(result)}
    headline = rows.get("total_salary")
    if not headline or headline["display"] == "__not_found__":
        headline = rows.get("salary")
    if not headline or headline["display"] == "__not_found__":
        theme.metric(t("analyze.compensation_card", lang), t("common.not_found", lang))
        return
    theme.metric(t("analyze.compensation_card", lang), _display(headline["display"], lang), headline["label"])


def _salary_benchmark_summary_card(result: dict[str, Any], lang: str) -> None:
    salary = view.salary_view(result)
    if not salary:
        theme.metric(t("analyze.salary_benchmark_card", lang), t("common.not_found", lang))
        return
    if salary["status"] == "not_configured":
        theme.metric(t("analyze.salary_benchmark_card", lang), t("salary.unavailable", lang))
        return
    if salary["market_range"] != "no market data":
        theme.metric(t("analyze.salary_benchmark_card", lang), salary["market_range"], salary.get("period", ""))
    else:
        theme.metric(t("analyze.salary_benchmark_card", lang), t("salary.no_range", lang))


def _findings_summary_card(result: dict[str, Any], lang: str) -> None:
    theme.section(t("analyze.findings_card", lang))
    findings = view.findings_summary(result)
    columns = st.columns(3, gap="large")
    groups = (("key_findings", "analyze.key_findings", "analyze.no_key_findings"),
             ("requires_review", "analyze.requires_review", "analyze.no_review_items"),
             ("missing_information", "analyze.missing_information", "analyze.no_missing_info"))
    for column, (key, title_key, empty_key) in zip(columns, groups):
        with column:
            st.markdown(f"**{t(title_key, lang)}**")
            items = findings[key]
            if not items:
                st.caption(t(empty_key, lang))
                continue
            st.markdown('<ul class="sanad-list">' + "".join(
                f'<li>{(item.get("field") or "").replace("_", " ")}</li>' for item in items[:5]) + "</ul>",
                unsafe_allow_html=True)
            if len(items) > 5:
                st.caption(f"+{len(items) - 5}")


def compensation_panel(result: dict[str, Any], lang: str) -> None:
    """Compensation tab: every salary component the contract states, kept separate, as a compact
    label/value table rather than a stack of large cards - Total Compensation is set off with a rule
    once every component above it has had its own row, never mixed into a single figure."""
    theme.section(t("analyze.tab_compensation", lang))
    rows = view.compensation_summary(result, detailed=True)
    if not rows:
        theme.empty_state(t("state.no_results", lang), t("analyze.no_compensation_data", lang))
        return
    total_index = next((i for i, row in enumerate(rows) if row["field"] == "total_salary"), None)
    theme.kv_table([(row["label"], _display(row["display"], lang)) for row in rows], total_index=total_index)

    evidence_rows = [row for row in rows if row.get("source_text") or row.get("page")]
    if evidence_rows:
        with st.expander(t("ui.show_evidence", lang)):
            for row in evidence_rows:
                page = f" · {t('analyze.page', lang)} {row['page']}" if row.get("page") else ""
                st.markdown(f"**{row['label']}**{page}")
                source_text = view.safe_evidence_text(row.get("source_text"))
                if row.get("source_text") and source_text is None:
                    # Genuine encoding corruption (see view.safe_evidence_text) - the page reference
                    # above is still real and useful; the quote itself is not, so it is left out
                    # rather than shown as unreadable replacement glyphs.
                    st.caption(t("ui.source_unavailable", lang))
                elif source_text:
                    st.markdown(f'<div class="sanad-quote sanad-quote--ar">{source_text}</div>',
                                unsafe_allow_html=True)


def findings_panel(result: dict[str, Any], lang: str) -> None:
    """Findings tab: key findings, review items and missing information, plus the service's own notes."""
    theme.section(t("analyze.tab_findings", lang))
    findings = view.findings_summary(result)

    theme.section(t("analyze.key_findings", lang))
    if not findings["key_findings"]:
        st.caption(t("analyze.no_key_findings", lang))
    for item in findings["key_findings"]:
        tone, mark, label = FINDING_STATES.get(item["status"], ("neutral", "—", "status.unavailable"))
        title = (item.get("clause_name") or (item.get("field") or "").replace("_", " ")).strip().capitalize()
        st.markdown(
            f'<div class="sanad-finding"><div class="sanad-finding__head">'
            f'{theme.pill(f"{mark} {t(label, lang)}", tone)}<span class="sanad-finding__title">{title}</span></div>'
            f'<p class="sanad-finding__text">{item.get("explanation", "")}</p></div>', unsafe_allow_html=True)

    theme.section(t("analyze.requires_review", lang))
    if not findings["requires_review"]:
        st.caption(t("analyze.no_review_items", lang))
    else:
        st.markdown('<ul class="sanad-list">' + "".join(
            f'<li><strong>{(item.get("field") or "").replace("_", " ").capitalize()}</strong>: '
            f'{item.get("explanation", "")}</li>' for item in findings["requires_review"]) + "</ul>",
            unsafe_allow_html=True)

    theme.section(t("analyze.missing_information", lang))
    if not findings["missing_information"]:
        st.caption(t("analyze.no_missing_info", lang))
    else:
        st.markdown('<ul class="sanad-list">'
                    + "".join(f'<li>{(item.get("field") or "").replace("_", " ").capitalize()}</li>'
                             for item in findings["missing_information"]) + "</ul>", unsafe_allow_html=True)

    theme.section(t("analyze.recommendations", lang))
    notes = view.caveats(result)
    if not notes:
        st.caption(t("analyze.no_review_items", lang))
    else:
        st.markdown('<ul class="sanad-list">' + "".join(f"<li>{note}</li>" for note in notes) + "</ul>",
                    unsafe_allow_html=True)


# --------------------------------------------------------------------------- CV compatibility
def cv_panel(result: dict[str, Any], lang: str) -> None:
    """Kept visually separate from contract compliance: they answer different questions."""
    cv = view.cv_analysis(result)
    if not cv:
        return
    theme.section(t("analyze.cv_section", lang))
    st.write(cv.get("overall_summary", ""))
    rows = view.compatibility_rows(cv)
    if rows:
        st.markdown(f"**{t('ui.requirements', lang)}**")
        for row in rows:
            status = str(row.get("Status", "")).strip()
            tone = COMPATIBILITY_TONES.get(status.lower(), "neutral")
            evidence = row.get("CV evidence")
            st.markdown(
                f'<div class="sanad-finding"><div class="sanad-finding__head">'
                f'{theme.pill(status or "-", tone)}'
                f'<span class="sanad-finding__title">{row.get("Requirement", "")}</span>'
                f'<span class="sanad-finding__articles">{row.get("Importance", "")}</span></div>'
                f'<p class="sanad-finding__text">{row.get("Explanation", "")}</p>'
                + (f'<div class="sanad-quote">{evidence}</div>' if evidence else "")
                + "</div>",
                unsafe_allow_html=True,
            )
    else:
        st.caption(t("ui.no_requirements", lang))
    observations = [o.get("message") for o in cv.get("observations", []) if o.get("message")]
    if observations:
        with st.expander(f"{t('analyze.cv_observations', lang)} ({len(observations)})"):
            st.markdown('<ul class="sanad-list">' + "".join(f"<li>{item}</li>" for item in observations) + "</ul>",
                        unsafe_allow_html=True)


# --------------------------------------------------------------------------- comparison
# Display order for Compare Contracts (Part 2 redesign): Legal Compliance and Compensation and Salary
# Benchmark are each rendered from their own dedicated, per-contract function (never the generic
# dimension table below - see _compliance_comparison / _compensation_comparison /
# _salary_benchmark_comparison), so this list only drives the remaining, genuinely tabular sections.
COMPARISON_GROUP_LABELS = (
    ("compliance", "compare.group_compliance"),
    ("benefits", "compare.group_benefits"),
    ("working_conditions", "compare.group_working_conditions"),
    ("contract_terms", "compare.group_contract_terms"),
    ("cv_compatibility", "compare.group_cv_compatibility"),
)
# "Allowances" is left out of the generic Benefits table: Compensation (below) already shows housing,
# transportation and other allowances as their own separate rows, in full - showing the same amounts
# again here as one concatenated line would just repeat it in a less readable form.
_BENEFITS_DIMENSIONS_HANDLED_ELSEWHERE = {"Allowances"}
RISK_SEVERITY_LABELS = {
    "high": ("bad", "compare.risk_group_high"),
    "medium": ("caution", "compare.risk_group_medium"),
    "info": ("neutral", "compare.risk_group_info"),
}


def _short_name(name: Any, limit: int = 22) -> str:
    """Fallback column-header-friendly version of a contract's label, used only when the naming
    convention below (view.comparison_contract_names, "Contract A - <job title>") has nothing to work
    with. The full name is never lost - it is always shown once, in the legend line above the tables -
    this only shortens what repeats in every row after that."""
    text = str(name)
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _contract_summary_cards(comparison: dict[str, Any], lang: str) -> None:
    """Part 12: one card per compared contract - its name, job title, location and headline pay - so
    a reader knows what is being compared before reaching a single row of the tables below."""
    cards = view.comparison_contract_cards(comparison)
    if not cards:
        return
    theme.section(t("compare.contracts_title", lang))
    columns = st.columns(len(cards), gap="medium")
    for column, card in zip(columns, cards):
        with column:
            pay = _display(card["salary_display"], lang)
            note = _display(card["location"], lang) + (f" · {pay}" if card["salary_display"] != "__not_found__" else "")
            st.markdown(
                f'<div class="sanad-metric"><span class="sanad-metric__label">{card["short_name"]}</span>'
                f'<span class="sanad-metric__value" style="font-size:1.02rem">{_display(card["job_title"], lang)}</span>'
                f'<span class="sanad-metric__note">{note}</span></div>',
                unsafe_allow_html=True,
            )
            st.caption(card["full_name"])
    theme.spacer(10)


_QUICK_COMPARE_PLACEHOLDERS = {
    "__not_found__": "common.not_found",
    "__not_configured__": "compare.not_available_short",
    "__no_range__": "compare.not_available_short",
}


def _quick_compare_strip(comparison: dict[str, Any], contract_names: dict[str, dict[str, str]], lang: str) -> None:
    """Part 2 (Compare Contracts redesign): the top-of-page compact summary strip - Compensation /
    Annual Leave / Probation / Salary Benchmark, both contracts' values side by side, before any
    section below. Reuses `view.comparison_quick_compare`, which itself reuses the exact same
    functions the sections below call - so nothing here can disagree with the detail underneath it."""
    rows = view.comparison_quick_compare(comparison)
    if not rows:
        return
    row_labels = {"compensation": t("compare.group_compensation", lang), "annual_leave": t("priority.annual_leave", lang),
                 "probation": t("priority.probation_period", lang), "salary_benchmark": t("compare.group_salary_benchmark", lang)}
    contract_ids = list(next(iter(rows))["values"].keys()) if rows else []
    table = []
    for row in rows:
        display_row = {"": row_labels.get(row["key"], row["label"])}
        for cid in contract_ids:
            value = row["values"].get(cid, "__not_found__")
            key = _QUICK_COMPARE_PLACEHOLDERS.get(value)
            display_row[contract_names.get(cid, {}).get("short", cid)] = t(key, lang) if key else value
        table.append(display_row)
    st.markdown(f"**{t('compare.at_a_glance', lang)}**")
    st.dataframe(table, use_container_width=True, hide_index=True)
    theme.spacer(10)


def _compensation_comparison(comparison: dict[str, Any], contract_names: dict[str, dict[str, str]], lang: str) -> None:
    """Part 2: Compensation as one clean table - Basic Salary/Housing Allowance/Transportation/Other
    Allowances/Total Compensation/Net Salary as separate rows, contracts as columns - reusing each
    contract's own `compensation_summary` (see `view.comparison_compensation_table`). Basic salary and
    total compensation are always separate rows, never compared against each other."""
    rows = view.comparison_compensation_table(comparison)
    if not any(row.get(cid) != "__not_found__" for row in rows for cid in contract_names):
        return
    st.markdown(f"**{t('compare.group_compensation', lang)}**")
    table = []
    for row in rows:
        display_row = {"Component": row["label"]}
        for cid, names in contract_names.items():
            display_row[names["short"]] = _display(row.get(cid, "__not_found__"), lang)
        table.append(display_row)
    st.dataframe(table, use_container_width=True, hide_index=True)
    theme.spacer(10)


def _compliance_comparison(comparison: dict[str, Any], contract_names: dict[str, dict[str, str]], lang: str) -> None:
    """Part 15: Legal Compliance shown as one block per contract (its own status counts, read from its
    own clause_findings - see view.comparison_compliance_by_contract), then a plain-language list of
    the fields the contracts actually differ on - never a dimension-per-row table, and never the
    comparison agent's own 'compliance' dimension (which cannot see non-compliant clauses - see that
    function's docstring). Headers use the "Contract A - <job title>" naming convention (falling back to
    the block's own label when a contract has no name info) for consistency with the cards above."""
    blocks = view.comparison_compliance_by_contract(comparison)
    if not any(block["rows"] for block in blocks):
        return

    def short(cid: str, fallback: str) -> str:
        return contract_names.get(cid, {}).get("short") or fallback

    st.markdown(f"**{t('compare.group_compliance', lang)}**")
    columns = st.columns(len(blocks), gap="medium")
    for column, block in zip(columns, blocks):
        with column:
            st.markdown(f"**{short(block['contract_id'], block['label'])}**")
            _state_summary(block["counts"], lang)
    differences = view.comparison_compliance_differences(blocks)
    if differences:
        with st.expander(t("compare.key_differences", lang)):
            for diff in differences:
                parts = "; ".join(
                    f"{short(d['contract_id'], d['label'])}: "
                    f"{t(FINDING_STATES.get(d['status'], ('neutral', '—', 'status.unavailable'))[2], lang)}"
                    for d in diff["per_contract"]
                )
                st.markdown(f"**{diff['clause_name']}** — {parts}")
    else:
        st.caption(t("compare.no_differences", lang))
    theme.spacer(10)


def _salary_benchmark_comparison(comparison: dict[str, Any], contract_names: dict[str, dict[str, str]], lang: str) -> None:
    """Part 16: Salary Benchmark shown as one block per contract - each contract's own benchmark,
    exactly as the single-contract Salary Benchmark tab would show it (view.comparison_salary_by_contract
    reuses view.salary_view directly) - rather than a row in the ranked dimension tables, since a market
    comparison is informational, never a ranking between the contracts themselves."""
    blocks = view.comparison_salary_by_contract(comparison)
    if not any(block["salary"] for block in blocks):
        return
    st.markdown(f"**{t('compare.group_salary_benchmark', lang)}**")
    st.caption(t("compare.salary_benchmark_informational", lang))
    columns = st.columns(len(blocks), gap="medium")
    for column, block in zip(columns, blocks):
        with column:
            st.markdown(f"**{contract_names.get(block['contract_id'], {}).get('short') or block['label']}**")
            salary = block["salary"]
            if not salary:
                st.caption(t("common.not_found", lang))
                continue
            if salary["status"] == "not_configured":
                st.caption(t("salary.unavailable", lang))
                continue
            if salary["market_range"] != "no market data":
                theme.metric(t("salary.range_title", lang), salary["market_range"], salary.get("period", ""))
                position = POSITION_KEYS.get(salary["position"] or "")
                suffix = f" · {t(position, lang)}" if position else ""
                st.caption(f'{t("salary.contract_salary", lang)}: {salary["contract_salary"]}{suffix}')
            else:
                st.caption(t("salary.no_range", lang))
    theme.spacer(10)


def comparison_panel(result: dict[str, Any], lang: str) -> None:
    comparison = result.get("comparison")
    if not comparison:
        return

    _contract_summary_cards(comparison, lang)

    contract_names = view.comparison_contract_names(comparison)
    full_names = [c.get("label") or c["contract_id"] for c in comparison.get("contracts", [])]
    short_by_full = {info["full"]: info["short"] for info in contract_names.values()}
    short_names = {full: short_by_full.get(full, _short_name(full)) for full in full_names}
    if any(short != full for full, short in short_names.items()):
        st.caption(" · ".join(f"**{short}** = {full}" for full, short in short_names.items() if short != full))

    _quick_compare_strip(comparison, contract_names, lang)

    recommendation = view.recommendation_view(comparison)
    theme.section(t("compare.recommendation", lang))
    if recommendation["status"] == "preferred_contract":
        theme.banner("good", t("compare.preferred", lang, name=recommendation["preferred"]),
                     recommendation["explanation"])
    else:
        theme.banner("neutral", t("compare.no_preference", lang), recommendation["explanation"])
    if recommendation["factors"]:
        st.markdown(f"**{t('compare.why', lang)}**")
        st.dataframe([{"Dimension": f["Dimension"], "Favours": f["Favours"], "Why": f["Why"]}
                     for f in recommendation["factors"]], use_container_width=True, hide_index=True)
    if recommendation["trade_offs"]:
        st.markdown(f"**{t('compare.trade_offs', lang)}**")
        st.markdown('<ul class="sanad-list">'
                    + "".join(f"<li>{item}</li>" for item in recommendation["trade_offs"]) + "</ul>",
                    unsafe_allow_html=True)
    if recommendation["basis"]:
        st.caption(recommendation["basis"])

    # Section order (Part 2 redesign): Legal Compliance, Compensation, Salary Benchmark, Benefits,
    # Working Conditions, Contract Terms - each shown separately, contracts side by side within it.
    theme.section(t("compare.table", lang))
    _compliance_comparison(comparison, contract_names, lang)
    _compensation_comparison(comparison, contract_names, lang)
    _salary_benchmark_comparison(comparison, contract_names, lang)

    sections = view.comparison_sections(comparison)
    for group, label_key in COMPARISON_GROUP_LABELS:
        if group == "compliance":
            continue  # rendered above, from each contract's own authoritative data - see
                      # _compliance_comparison; the comparison agent's own "compliance" dimension is
                      # never shown here, so it is never duplicated or contradicted
        rows = sections.get(group)
        if group == "benefits":
            rows = [row for row in (rows or []) if row["Dimension"] not in _BENEFITS_DIMENSIONS_HANDLED_ELSEWHERE]
        if not rows:
            continue
        st.markdown(f"**{t(label_key, lang)}**")
        # "Category" (this section's own header, just above), "Ranked" (raw ranking terminology) and
        # "Leads" (which contract wins this row - the comparison is deliberately never presented as
        # ranking contracts, see compare.no_preference and Part 13/14) are left out of the visible
        # table; "Explanation" moves into an expander below so a table cell is never a paragraph.
        display_rows, explanations = [], []
        for row in rows:
            display_row = {"Dimension": row["Dimension"]}
            for full, short in short_names.items():
                if full in row:
                    display_row[short] = row[full]
            display_rows.append(display_row)
            if row.get("Explanation"):
                explanations.append((row["Dimension"], row["Explanation"]))
        st.dataframe(display_rows, use_container_width=True, hide_index=True)
        if explanations:
            with st.expander(t("compare.why", lang)):
                for dimension, explanation in explanations:
                    st.markdown(f"**{dimension}:** {explanation}")
        theme.spacer(6)

    risk_groups = view.risk_groups(comparison)
    if risk_groups:
        theme.section(t("compare.risks", lang))
        # Grouped cards by severity, not one flat table - and never the internal finding_id (see
        # view.risk_groups). Highest severity first; a reader sees "Needs urgent attention - 2" before
        # ever reading a single row.
        for group in risk_groups:
            tone, label_key = RISK_SEVERITY_LABELS.get(group["severity"], ("neutral", "compare.risk_group_info"))
            st.markdown(theme.pill(f"{t(label_key, lang)} · {len(group['items'])}", tone), unsafe_allow_html=True)
            for item in group["items"]:
                what = item["field"] if item["field"] not in (None, "", "-") else item["type"]
                st.markdown(
                    f'<div class="sanad-finding"><div class="sanad-finding__head">'
                    f'<span class="sanad-finding__title">{item["contract"]} · {what}</span>'
                    f'</div><p class="sanad-finding__text">{item["message"]}</p></div>',
                    unsafe_allow_html=True,
                )


# --------------------------------------------------------------------------- salary
POSITION_KEYS = {
    "below the observed market range": "salary.position_below",
    "inside the observed market range": "salary.position_inside",
    "above the observed market range": "salary.position_above",
}


def salary_panel(result: dict[str, Any], lang: str, *, heading: bool = True) -> None:
    salary = view.salary_view(result)
    if not salary:
        return
    if heading:
        theme.section(t("salary.range_title", lang))

    if salary["status"] == "not_configured":
        theme.empty_state(t("salary.unavailable", lang), salary["message"], "salary")
        return

    has_range = salary["market_range"] != "no market data"
    if has_range:
        columns = st.columns(3)
        with columns[0]:
            theme.metric(t("salary.range_title", lang), salary["market_range"], salary["period"])
        with columns[1]:
            theme.metric(t("salary.contract_salary", lang), salary["contract_salary"])
        with columns[2]:
            position = POSITION_KEYS.get(salary["position"] or "")
            theme.metric(t("salary.position", lang), t(position, lang) if position else "-",
                         salary["basis"])
    else:
        theme.banner("caution", t("salary.no_range", lang), salary["message"])

    details = st.columns(3)
    with details[0]:
        theme.metric(t("salary.job_title", lang), salary["job_title"])
    with details[1]:
        theme.metric(t("salary.city", lang), salary["location"])
    with details[2]:
        theme.metric(t("salary.data_quality", lang), salary["data_quality"])

    if salary["sources"]:
        theme.section(t("common.sources", lang))
        for source in salary["sources"]:
            st.markdown(
                f'<div class="sanad-source"><a href="{source["url"]}" target="_blank">{source["name"]}</a>'
                f'<span class="sanad-source__tier">{source["tier"]} · {t("salary.retrieved", lang)} '
                f'{source["retrieved_at"]}</span></div>',
                unsafe_allow_html=True,
            )
    if salary["evidence"]:
        with st.expander(f"{t('salary.evidence', lang)} ({len(salary['evidence'])})"):
            st.dataframe(salary["evidence"], use_container_width=True, hide_index=True)
    # The benchmark's own limitations are already collected by caveats_panel, which every result
    # page shows once - they are deliberately not repeated here.
