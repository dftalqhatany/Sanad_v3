"""The Sanad services, one function per screen.

Each page collects what the user wants in plain language and hands it to the service callables on
`Ctx`, which are implemented in `frontend/app.py` (the only module that talks to the API client).
No page knows a route name, an agent name or an internal task value.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

import streamlit as st

from frontend import theme, ui
from frontend.i18n import LANGUAGES, t

Upload = tuple[str, bytes, str]


@dataclass
class Ctx:
    """Everything a page is allowed to do, handed in from the shell."""

    lang: str
    config: dict[str, Any]
    online: bool
    go: Callable[[str], None]
    ask: Callable[[str], Any]
    analyze_contract: Callable[..., Any]
    analyze_cv: Callable[..., Any]
    compare: Callable[..., Any]
    benchmark: Callable[..., Any]

    def s(self, key: str, **values) -> str:
        return t(key, self.lang, **values)

    @property
    def max_mb(self) -> float:
        return float(self.config.get("max_file_size_mb") or 20)

    @property
    def max_contracts(self) -> int:
        return int(self.config.get("max_contracts") or 5)

    @property
    def file_types(self) -> list[str]:
        return [str(item).lstrip(".") for item in self.config.get("allowed_file_types", [".pdf", ".docx"])]


# --------------------------------------------------------------------------- shared helpers
def _container(key: str):
    try:
        return st.container(key=key)
    except TypeError:  # older Streamlit: the card keeps its content, loses its frame
        return st.container()


def _offline_notice(ctx: Ctx) -> bool:
    if ctx.online:
        return False
    theme.empty_state(ctx.s("state.offline"), ctx.s("state.offline_body"), "shield")
    return True


def _uploader(ctx: Ctx, key: str, title: str, hint: str = ""):
    theme.dropzone_label(title, hint or f"{ctx.s('common.supported_files')} · "
                                       f"{ctx.s('common.file_limit', size=f'{ctx.max_mb:g}')}")
    return st.file_uploader(title, type=ctx.file_types, key=key, label_visibility="collapsed")


def _as_upload(item, role: str) -> Upload:
    return (item.name, item.getvalue(), role)


def _target_job(title: str, skills: str) -> str | None:
    """The optional role a CV is checked against, as the service expects it."""
    if not title.strip():
        return None
    wanted = [skill.strip() for skill in skills.split(",") if skill.strip()]
    return json.dumps({"title": title.strip(), "required_skills": wanted})


def _result_footer(ctx: Ctx, result: dict[str, Any], label_key: str, reset: Callable[[], None], *,
                   show_caveats: bool = True) -> None:
    if show_caveats:
        ui.caveats_panel(result, ctx.lang)
    ui.documents_panel(result, ctx.lang)
    ui.raw_panel(result, ctx.lang)
    if st.button(ctx.s(label_key), type="secondary"):
        reset()
        st.rerun()


def _store(key: str, response) -> bool:
    """Keep a response for the next rerun. Returns True when there is something to render."""
    st.session_state[key] = response
    return response is not None


# --------------------------------------------------------------------------- home
# Sanad has exactly three primary workflows. Salary benchmarking is not a fourth one: it is reused
# inside Analyze Contract (Compensation) and inside Compare Contracts (a compared dimension) - see
# frontend/ui.py - so it has no standalone card here and no entry of its own in the sidebar.
SERVICES = (
    ("ask", "ask", "home.ask_title", "home.ask_desc", "home.ask_cta"),
    ("analyze", "analyze", "home.analyze_title", "home.analyze_desc", "home.analyze_cta"),
    ("compare", "compare", "home.compare_title", "home.compare_desc", "home.compare_cta"),
)


def home(ctx: Ctx) -> None:
    theme.hero(ctx.s("home.hero_title"), ctx.s("app.tagline"), ctx.s("home.hero_sub"))
    theme.section(ctx.s("home.services"), ctx.s("home.trust"))
    columns = st.columns(len(SERVICES), gap="large")
    for column, (page, icon_name, title, desc, cta) in zip(columns, SERVICES):
        with column, _container(f"svc_{page}"):
            theme.card_body(icon_name, ctx.s(title), ctx.s(desc))
            if st.button(ctx.s(cta), key=f"go_{page}", type="primary", use_container_width=True):
                ctx.go(page)


# --------------------------------------------------------------------------- ask
SUGGESTIONS = ("ask.q1", "ask.q2", "ask.q3", "ask.q4")


def ask(ctx: Ctx) -> None:
    theme.page_header(ctx.s("ask.title"), ctx.s("ask.sub"), "ask")
    if _offline_notice(ctx):
        return

    st.markdown(f"**{ctx.s('ask.suggestions')}**")
    for row in (SUGGESTIONS[:2], SUGGESTIONS[2:]):
        columns = st.columns(2)
        for column, key in zip(columns, row):
            if column.button(ctx.s(key), key=f"sugg_{key}", use_container_width=True):
                st.session_state["ask_q"] = ctx.s(key)
    theme.spacer(8)

    question = st.text_area(ctx.s("ask.input_label"), key="ask_q", height=110,
                            placeholder=ctx.s("ask.placeholder"))
    st.caption(ctx.s("ask.contract_note"))
    if st.button(ctx.s("ask.cta"), type="primary"):
        if not question.strip():
            theme.banner("caution", ctx.s("ask.empty"))
        else:
            placeholder = st.empty()
            with placeholder.container():
                theme.progress_panel(ctx.s("ask.cta"), [ctx.s("analyze.p3"), ctx.s("analyze.p4")])
            _store("ask_result", ctx.ask(question.strip()))
            placeholder.empty()

    response = st.session_state.get("ask_result")
    if response is None:
        return
    if not response.ok and response.result is None:
        theme.empty_state(ctx.s("state.offline"), response.message, "shield")
        return
    result = response.result
    if ui.unavailable_state(result, ctx.lang):
        return
    ui.request_banner(result, ctx.lang)
    ui.answer_panel(result, ctx.lang)
    ui.caveats_panel(result, ctx.lang)
    ui.raw_panel(result, ctx.lang)


# --------------------------------------------------------------------------- analyze
def _reset_analyze() -> None:
    for key in ("an_step", "an_mode", "an_result", "an_contract", "an_cv"):
        st.session_state.pop(key, None)


def analyze(ctx: Ctx) -> None:
    if st.session_state.get("an_mode") == "cv_only":
        _cv_only(ctx)
        return

    theme.page_header(ctx.s("analyze.title"), ctx.s("analyze.sub"), "analyze")
    if _offline_notice(ctx):
        return

    response = st.session_state.get("an_result")
    step = 3 if response is not None else (2 if st.session_state.get("an_contract") is not None else 1)
    theme.steps([ctx.s("analyze.step1"), ctx.s("analyze.step2"), ctx.s("analyze.step3")], step)

    if response is not None:
        _analysis_results(ctx, response)
        return

    contract = _uploader(ctx, "an_contract", ctx.s("analyze.upload_label"), ctx.s("analyze.upload_hint"))
    if contract is None:
        theme.spacer(6)
        theme.empty_state(ctx.s("state.no_file"), ctx.s("state.no_file_body"), "upload")
        theme.spacer(10)
        if st.button(ctx.s("analyze.cv_only_switch"), type="tertiary"):
            st.session_state["an_mode"] = "cv_only"
            st.rerun()
        return

    theme.file_chip(contract.name, len(contract.getvalue()) / 1024)

    theme.section(ctx.s("analyze.choose_title"))
    choice = st.session_state.get("an_mode")
    columns = st.columns(2, gap="large")
    with columns[0], _container("opt_only"):
        theme.card_body("shield", ctx.s("analyze.only_title"), ctx.s("analyze.only_desc"))
        if st.button(ctx.s("analyze.only_title"), key="pick_only",
                     type="primary" if choice == "contract" else "secondary", use_container_width=True):
            st.session_state["an_mode"] = "contract"
            st.rerun()
    with columns[1], _container("opt_cv"):
        theme.card_body("file", ctx.s("analyze.cv_title"), ctx.s("analyze.cv_desc"))
        if st.button(ctx.s("analyze.cv_title"), key="pick_cv",
                     type="primary" if choice == "cv" else "secondary", use_container_width=True):
            st.session_state["an_mode"] = "cv"
            st.rerun()

    if choice is None:
        return

    cv = None
    job_title = skills = ""
    if choice == "cv":
        theme.spacer(10)
        cv = _uploader(ctx, "an_cv", ctx.s("analyze.cv_upload_label"))
        if cv is not None:
            theme.file_chip(cv.name, len(cv.getvalue()) / 1024)
        fields = st.columns(2)
        job_title = fields[0].text_input(f"{ctx.s('analyze.role_label')} ({ctx.s('common.optional')})",
                                         help=ctx.s("analyze.role_help"), key="an_role")
        skills = fields[1].text_input(f"{ctx.s('analyze.skills_label')} ({ctx.s('common.optional')})", key="an_skills")
        if cv is None:
            theme.banner("neutral", ctx.s("analyze.cv_upload_label"), ctx.s("state.no_file_body"))
            return

    theme.spacer(8)
    label = "analyze.cta_cv" if choice == "cv" else "analyze.cta"
    if not st.button(ctx.s(label), type="primary"):
        return

    uploads = [_as_upload(contract, "contract")]
    if cv is not None:
        uploads.append(_as_upload(cv, "cv"))
    placeholder = st.empty()
    with placeholder.container():
        theme.progress_panel(ctx.s("analyze.progress"),
                             [ctx.s("analyze.p1"), ctx.s("analyze.p2"), ctx.s("analyze.p3"), ctx.s("analyze.p4")])
    response = ctx.analyze_contract(uploads, target_job=_target_job(job_title, skills))
    placeholder.empty()
    st.session_state["an_result"] = response
    st.rerun()


def _analysis_results(ctx: Ctx, response) -> None:
    if not response.ok and response.result is None:
        theme.empty_state(ctx.s("state.offline"), response.message, "shield")
        if st.button(ctx.s("action.start_over"), type="secondary"):
            _reset_analyze()
            st.rerun()
        return
    result = response.result
    if not ui.unavailable_state(result, ctx.lang):
        # The dashboard below already shows the request outcome (Contract Overview's status pill),
        # the compliance/compensation/salary-benchmark summary, and the service's own notes (the
        # Findings tab) - so the caveats panel in the shared footer would only repeat them.
        ui.contract_dashboard(result, ctx.lang)
    _result_footer(ctx, result, "analyze.new", _reset_analyze, show_caveats=False)


def _cv_only(ctx: Ctx) -> None:
    """The CV check on its own - the same service, without a contract."""
    theme.page_header(ctx.s("analyze.cv_only_title"), ctx.s("analyze.cv_only_sub"), "file")
    if st.button(ctx.s("analyze.back_to_contract"), type="tertiary"):
        _reset_analyze()
        st.rerun()
    if _offline_notice(ctx):
        return

    response = st.session_state.get("an_result")
    if response is not None:
        _analysis_results(ctx, response)
        return

    cv = _uploader(ctx, "an_cv", ctx.s("analyze.cv_upload_label"))
    if cv is not None:
        theme.file_chip(cv.name, len(cv.getvalue()) / 1024)
    fields = st.columns(2)
    job_title = fields[0].text_input(ctx.s("analyze.role_label"), key="cv_role")
    skills = fields[1].text_input(f"{ctx.s('analyze.skills_label')} ({ctx.s('common.optional')})", key="cv_skills")
    if cv is None:
        theme.spacer(6)
        theme.empty_state(ctx.s("state.no_file"), ctx.s("state.no_file_body"), "upload")
        return
    if st.button(ctx.s("analyze.cv_only_cta"), type="primary"):
        placeholder = st.empty()
        with placeholder.container():
            theme.progress_panel(ctx.s("analyze.progress"), [ctx.s("analyze.p1"), ctx.s("analyze.p4")])
        st.session_state["an_result"] = ctx.analyze_cv([_as_upload(cv, "cv")],
                                                       target_job=_target_job(job_title, skills))
        placeholder.empty()
        st.rerun()


# --------------------------------------------------------------------------- compare
def _reset_compare() -> None:
    for key in [k for k in list(st.session_state) if k.startswith("cmp_")]:
        st.session_state.pop(key, None)


def compare(ctx: Ctx) -> None:
    theme.page_header(ctx.s("compare.title"), ctx.s("compare.sub"), "compare")
    if _offline_notice(ctx):
        return

    response = st.session_state.get("cmp_result")
    chosen = sum(1 for key in st.session_state if key.startswith("cmp_file_") and st.session_state[key] is not None)
    step = 3 if response is not None else (2 if chosen >= 2 else 1)
    theme.steps([ctx.s("compare.step1"), ctx.s("compare.step2"), ctx.s("compare.step3")], step)

    if response is not None:
        _comparison_results(ctx, response)
        return

    slots = st.session_state.setdefault("cmp_slots", 2)
    files = []
    for index in range(slots):
        item = _uploader(ctx, f"cmp_file_{index}", ctx.s("compare.slot", n=index + 1))
        if item is not None:
            theme.file_chip(item.name, len(item.getvalue()) / 1024)
            files.append(item)
        theme.spacer(6)

    buttons = st.columns([1, 1, 3])
    if slots < ctx.max_contracts and buttons[0].button(ctx.s("compare.add"), type="secondary"):
        st.session_state["cmp_slots"] = slots + 1
        st.rerun()
    if slots > 2 and buttons[1].button(ctx.s("compare.remove_last"), type="tertiary"):
        st.session_state.pop(f"cmp_file_{slots - 1}", None)
        st.session_state["cmp_slots"] = slots - 1
        st.rerun()
    st.caption(ctx.s("compare.max_note", n=ctx.max_contracts))

    if len(files) < 2:
        theme.spacer(8)
        theme.empty_state(ctx.s("compare.min_note"), ctx.s("state.no_file_body"), "compare")
        return
    st.session_state["cmp_ready"] = True

    theme.section(ctx.s("compare.priorities_title"))
    st.caption(ctx.s("compare.priorities_help"))
    options = list(ctx.config.get("comparison_priorities", []))
    priorities = st.multiselect(ctx.s("compare.priorities_order"), options=options, key="cmp_priorities",
                                format_func=lambda value: t(f"priority.{value}", ctx.lang)
                                if t(f"priority.{value}", ctx.lang) != f"priority.{value}"
                                else str(value).replace("_", " "))

    theme.spacer(8)
    if not st.button(ctx.s("compare.cta"), type="primary"):
        return
    placeholder = st.empty()
    with placeholder.container():
        theme.progress_panel(ctx.s("compare.progress"),
                             [ctx.s("analyze.p1"), ctx.s("analyze.p2"), ctx.s("analyze.p3"), ctx.s("analyze.p4")])
    st.session_state["cmp_result"] = ctx.compare([_as_upload(item, "contract") for item in files],
                                                 priorities=priorities)
    placeholder.empty()
    st.rerun()


def _comparison_results(ctx: Ctx, response) -> None:
    if not response.ok and response.result is None:
        theme.empty_state(ctx.s("state.offline"), response.message, "shield")
        if st.button(ctx.s("action.start_over"), type="secondary"):
            _reset_compare()
            st.rerun()
        return
    result = response.result
    if not ui.unavailable_state(result, ctx.lang):
        if result.get("comparison"):
            ui.request_banner(result, ctx.lang)
            ui.comparison_panel(result, ctx.lang)
            ui.cv_panel(result, ctx.lang)
        else:
            # Only one contract was actually comparable: show its own full, correctly-sourced
            # analysis dashboard (same Contract Overview / Legal Compliance / Compensation / Salary
            # Benchmark / Findings tabs as Analyze Contract, plus CV Compatibility when present) -
            # never the old flat analysis_panel, which reads field-level findings only.
            ui.contract_dashboard(result, ctx.lang)
    _result_footer(ctx, result, "compare.new", _reset_compare)


# Salary Benchmark has no standalone page any more: it is reused inside Analyze Contract's own
# result dashboard (a "Salary Benchmark" section within Compensation - see
# ui.contract_dashboard / ui.salary_panel) and inside Compare Contracts (a compared dimension - see
# ui.comparison_panel). Its backend agent, provider, configuration and API task are all unchanged;
# only this dedicated wizard page and its sidebar/home entry points were removed.


# --------------------------------------------------------------------------- help and settings
def help_page(ctx: Ctx) -> None:
    theme.page_header(ctx.s("help.title"), ctx.s("help.sub"), "book")
    columns = st.columns(2, gap="large")
    with columns[0], _container("svc_can"):
        theme.card_body("shield", ctx.s("help.can_title"), "")
        st.markdown('<ul class="sanad-list">'
                    + "".join(f"<li>{ctx.s(key)}</li>" for key in ("help.can_1", "help.can_2", "help.can_3", "help.can_4"))
                    + "</ul>", unsafe_allow_html=True)
    with columns[1], _container("svc_cannot"):
        theme.card_body("book", ctx.s("help.cannot_title"), "")
        st.markdown('<ul class="sanad-list">'
                    + "".join(f"<li>{ctx.s(key)}</li>" for key in ("help.cannot_1", "help.cannot_2", "help.cannot_3"))
                    + "</ul>", unsafe_allow_html=True)
    theme.section(ctx.s("help.files_title"))
    st.markdown(f"<p class='sanad-drop__hint'>{ctx.s('common.supported_files')} · "
                f"{ctx.s('common.file_limit', size=f'{ctx.max_mb:g}')} · "
                f"{ctx.s('compare.max_note', n=ctx.max_contracts)}</p>", unsafe_allow_html=True)
    theme.section(ctx.s("help.privacy_title"))
    st.markdown(f"<p class='sanad-drop__hint'>{ctx.s('help.privacy_body')}</p>", unsafe_allow_html=True)


def settings_page(ctx: Ctx, default_url: str) -> None:
    theme.page_header(ctx.s("settings.title"), ctx.s("settings.sub"), "settings")
    theme.section(ctx.s("settings.language"))
    current = list(LANGUAGES).index(ctx.lang)
    chosen = st.radio(ctx.s("settings.language"), options=list(LANGUAGES), index=current, horizontal=True,
                      format_func=lambda code: LANGUAGES[code], label_visibility="collapsed")
    if chosen != ctx.lang:
        st.session_state["lang"] = chosen
        st.rerun()

    theme.section(ctx.s("settings.developer"), ctx.s("settings.developer_note"))
    st.session_state.setdefault("api_url", default_url)
    st.text_input(ctx.s("settings.service_address"), key="api_url")
    tone, label = ("good", "settings.connected") if ctx.online else ("bad", "settings.disconnected")
    answers = "settings.answers_on" if ctx.config.get("answer_generation_enabled") else "settings.answers_off"
    paid = "settings.salary_on" if ctx.config.get("salary_benchmarking_enabled") else "settings.salary_off"
    st.markdown(
        theme.pill(ctx.s(label), tone) + " " + theme.pill(ctx.s(answers), "neutral") + " "
        + theme.pill(ctx.s(paid), "neutral"),
        unsafe_allow_html=True,
    )
    theme.spacer(10)
    st.checkbox(ctx.s("settings.show_raw"), key="show_raw")


PAGES = {"home": home, "ask": ask, "analyze": analyze, "compare": compare, "help": help_page}
