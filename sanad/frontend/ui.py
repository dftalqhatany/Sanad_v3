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
    "insufficient_evidence": ("neutral", "—", "status.insufficient"),
    "not_found": ("neutral", "—", "status.not_found"),
    "not_applicable": ("neutral", "—", "status.document_fact"),
    "error": ("caution", "⚠", "status.check_failed"),
}
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
    else:
        theme.banner("neutral", t("ask.no_answer", lang))
    if answer["message"]:
        st.caption(answer["message"])
    if answer["evidence"]:
        theme.section(t("common.evidence", lang))
        for item in answer["evidence"]:
            with st.expander(item["citation"]):
                _article(item.get("arabic_text"), item.get("english_text"), lang)


def _article(arabic: str | None, english: str | None, lang: str) -> None:
    if arabic:
        st.markdown(f'<p class="sanad-drop__hint">{t("common.arabic_text", lang)}</p>'
                    f'<div class="sanad-quote sanad-quote--ar">{arabic}</div>', unsafe_allow_html=True)
    if english:
        st.markdown(f'<p class="sanad-drop__hint" style="margin-top:12px">{t("common.english_text", lang)}</p>'
                    f'<div class="sanad-quote">{english}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- contract analysis
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
def comparison_panel(result: dict[str, Any], lang: str) -> None:
    comparison = result.get("comparison")
    if not comparison:
        return
    recommendation = view.recommendation_view(comparison)
    theme.section(t("compare.recommendation", lang))
    if recommendation["status"] == "preferred_contract":
        theme.banner("good", t("compare.preferred", lang, name=recommendation["preferred"]),
                     recommendation["explanation"])
    else:
        theme.banner("neutral", t("compare.no_preference", lang), recommendation["explanation"])
    if recommendation["factors"]:
        st.markdown(f"**{t('compare.why', lang)}**")
        st.dataframe(recommendation["factors"], use_container_width=True, hide_index=True)
    if recommendation["trade_offs"]:
        st.markdown(f"**{t('compare.trade_offs', lang)}**")
        st.markdown('<ul class="sanad-list">'
                    + "".join(f"<li>{item}</li>" for item in recommendation["trade_offs"]) + "</ul>",
                    unsafe_allow_html=True)
    if recommendation["basis"]:
        st.caption(recommendation["basis"])

    theme.section(t("compare.table", lang), t("compare.table_hint", lang))
    st.dataframe(view.comparison_rows(comparison), use_container_width=True, hide_index=True)

    risks = view.risk_rows(comparison)
    if risks:
        theme.section(t("compare.risks", lang))
        st.dataframe(risks, use_container_width=True, hide_index=True)


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
