"""Sanad design system for the user interface.

Presentation only: the palette, the stylesheet, the icons and the layout primitives the pages are
built from. This module imports no backend code, opens no socket and makes no decision about the
content it displays.

Colours: every green and gold below was sampled from the official logo
(`frontend/assets/sanad-logo.png`). The lighter tints are derived from those same hues so that dark
brand text reaches WCAG AA on them. DANGER is the single colour outside the logo palette; a
non-compliant state needs a red and the artwork has none, so it is muted towards the brand.

The logo is used exactly as delivered: same file, same proportions, no recolouring, no rotation, no
cropping, no opacity tricks. Size it with `height` or `width` only.
"""

from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).resolve().parent / "assets"
LOGO_PATH = ASSETS / "sanad-logo.png"

# --------------------------------------------------------------------------- palette
GREEN_900 = "#00150D"  # deepest shadow of the ribbon - navigation
GREEN_800 = "#002514"
GREEN_700 = "#013D2A"  # core body green - primary brand
GREEN_600 = "#014E2B"  # lit face - primary action
GREEN_500 = "#05603A"
GREEN_400 = "#0C6F43"  # brightest edge
GREEN_100 = "#E3F0E9"  # derived surface tint
GREEN_50 = "#F2F8F5"  # derived page tint

GOLD_700 = "#5C3C05"
GOLD_600 = "#865E14"
GOLD_500 = "#926D1D"  # primary gold
GOLD_400 = "#9E7A27"  # decorative only: too light for text on white
GOLD_300 = "#C9A84E"  # derived tint, for gold text and rules on dark green
GOLD_100 = "#F2E9D2"  # derived surface tint

INK = "#0E1B14"
INK_MUTED = "#4A5B52"
ON_DARK = "#F4F8F5"
ON_DARK_MUTED = "#C3D8CB"
SURFACE = "#FFFFFF"
BORDER = "#E2ECE6"
DANGER = "#9B2C22"
DANGER_SOFT = "#FBEFED"
NEUTRAL = "#6B7A72"
NEUTRAL_SOFT = "#EFF3F1"

# Contrast ratios in use, all verified against WCAG AA (4.5:1 for body text):
#   ON_DARK on GREEN_900 17.6   ON_DARK on GREEN_700 11.5   ON_DARK_MUTED on GREEN_900 12.1
#   GOLD_300 on GREEN_900 8.3   GOLD_300 on GREEN_700 5.4   INK on SURFACE 17.7
#   INK_MUTED on SURFACE 7.2    GREEN_700 on SURFACE 12.3   GREEN_600 on SURFACE 9.9
#   GOLD_600 on SURFACE 5.8     GOLD_700 on GOLD_100 8.3    DANGER on SURFACE 7.6
# Forbidden by rule, because they fail: GOLD_400 as text on white (4.0 - borders only), and
# GOLD_500 as text on GOLD_100 (3.9 - use GOLD_700 on that surface).

# --------------------------------------------------------------------------- icons
ICONS = {
    "ask": ("M12 3a9 9 0 0 0-9 9 8.7 8.7 0 0 0 1.2 4.4L3 21l4.8-1.2A9 9 0 1 0 12 3z"
            "M9.4 9.6a2.6 2.6 0 0 1 5 .9c0 1.7-2.5 2.2-2.5 3.9M12 17.2h.01"),
    "analyze": ("M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z M14 3v5h5"
                "M9 13h3M9 17h6"),
    "compare": "M4 5h6v14H4zM14 5h6v14h-6zM7 9h0M17 9h0",
    "salary": ("M12 2v20M17 6.5C17 4.6 14.8 3.5 12 3.5S7 4.6 7 6.5s2.2 2.8 5 3.4 5 1.5 5 3.4-2.2 3-5 3"
               "-5-1.1-5-3"),
    "shield": "M12 3l7 3v5c0 4.6-3 8.3-7 10-4-1.7-7-5.4-7-10V6z M9.2 12.2l2 2 3.6-3.9",
    "book": "M4 5.5A2.5 2.5 0 0 1 6.5 3H19v15H6.5A2.5 2.5 0 0 0 4 20.5zM19 18v3H6.5",
    "home": "M4 10.5 12 4l8 6.5V20a1 1 0 0 1-1 1h-4v-6H9v6H5a1 1 0 0 1-1-1z",
    "settings": ("M12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6z"
                 "M19.4 13.5a7.6 7.6 0 0 0 0-3l1.8-1.4-1.9-3.3-2.2.9a7.6 7.6 0 0 0-2.6-1.5L14.1 2h-4.2"
                 "l-.4 2.2A7.6 7.6 0 0 0 6.9 5.7l-2.2-.9L2.8 8.1l1.8 1.4a7.6 7.6 0 0 0 0 3l-1.8 1.4 1.9 3.3"
                 " 2.2-.9a7.6 7.6 0 0 0 2.6 1.5l.4 2.2h4.2l.4-2.2a7.6 7.6 0 0 0 2.6-1.5l2.2.9 1.9-3.3z"),
    "upload": "M12 16V4M7.5 8.5 12 4l4.5 4.5M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3",
    "file": "M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8zM14 3v5h5",
}


def icon(name: str, size: int = 24, color: str = "currentColor", stroke: float = 1.6) -> str:
    """An inline SVG icon. Returns markup, so it can be embedded in any HTML block."""
    path = ICONS.get(name, ICONS["file"])
    return (f'<svg class="sanad-icon" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" '
            f'aria-hidden="true"><path d="{path}"/></svg>')


# --------------------------------------------------------------------------- logo
@lru_cache(maxsize=1)
def logo_data_uri() -> str:
    """The logo as a data URI, so it can be placed inside the HTML blocks below."""
    try:
        return "data:image/png;base64," + base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    except OSError:
        return ""


def page_config(title: str = "Sanad") -> None:
    try:
        st.set_page_config(page_title=title, page_icon=str(LOGO_PATH), layout="wide",
                           initial_sidebar_state="expanded")
    except Exception:  # an unreadable asset must never stop the interface from starting
        st.set_page_config(page_title=title, layout="wide", initial_sidebar_state="expanded")


def sidebar_logo() -> None:
    """The logo above the navigation, at its original proportions."""
    if not LOGO_PATH.exists():
        return
    try:
        st.logo(str(LOGO_PATH), size="large", link=None)
    except Exception:  # older Streamlit builds have no st.logo
        st.sidebar.image(str(LOGO_PATH), width=64)


def brand_block(name: str, tagline: str, with_image: bool = False) -> None:
    """Wordmark and tagline in the navigation. The logo itself is placed by `sidebar_logo()`."""
    image = f'<img src="{logo_data_uri()}" alt="">' if with_image else ""
    st.sidebar.markdown(
        f'<div class="sanad-brand">{image}'
        f'<div><span class="sanad-brand__name">{name}</span>'
        f'<span class="sanad-brand__tag">{tagline}</span></div></div>',
        unsafe_allow_html=True,
    )


def nav_label(text: str) -> None:
    st.sidebar.markdown(f'<p class="sanad-navlabel">{text}</p>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- layout primitives
def hero(title: str, tagline: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="sanad-hero">
          <img class="sanad-hero__mark" src="{logo_data_uri()}" alt="">
          <h1 class="sanad-hero__title">{title}</h1>
          <p class="sanad-hero__tagline">{tagline}</p>
          <p class="sanad-hero__body">{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = "", icon_name: str | None = None) -> None:
    mark = f'<span class="sanad-page__icon">{icon(icon_name, 22, GREEN_600)}</span>' if icon_name else ""
    sub = f'<p class="sanad-page__sub">{subtitle}</p>' if subtitle else ""
    st.markdown(f'<div class="sanad-page">{mark}<div><h1>{title}</h1>{sub}</div></div>', unsafe_allow_html=True)


def section(title: str, note: str = "") -> None:
    suffix = f'<span class="sanad-section__note">{note}</span>' if note else ""
    st.markdown(f'<div class="sanad-section"><h2>{title}</h2>{suffix}</div>', unsafe_allow_html=True)


def steps(labels: list[str], current: int) -> None:
    """A numbered progress rail across the top of a workflow. `current` is 1-based."""
    items = []
    for number, label in enumerate(labels, 1):
        state = "done" if number < current else ("now" if number == current else "next")
        mark = "&#10003;" if state == "done" else str(number)
        items.append(f'<li class="sanad-steps__item sanad-steps__item--{state}">'
                     f'<span class="sanad-steps__dot">{mark}</span><span>{label}</span></li>')
    st.markdown(f'<ol class="sanad-steps">{"".join(items)}</ol>', unsafe_allow_html=True)


def card_body(icon_name: str, title: str, body: str) -> None:
    """The visual part of a service card. Put a button under it in the same container."""
    st.markdown(
        f'<div class="sanad-card__body"><span class="sanad-card__icon">{icon(icon_name, 26, GREEN_600)}</span>'
        f'<h3>{title}</h3><p>{body}</p></div>',
        unsafe_allow_html=True,
    )


def banner(tone: str, title: str, message: str = "") -> None:
    """A status banner. tone: good | caution | bad | neutral."""
    safe = tone if tone in ("good", "caution", "bad", "neutral") else "neutral"
    text = f'<span class="sanad-banner__text">{message}</span>' if message else ""
    st.markdown(f'<div class="sanad-banner sanad-banner--{safe}">'
                f'<span class="sanad-banner__title">{title}</span>{text}</div>', unsafe_allow_html=True)


def empty_state(title: str, body: str = "", icon_name: str = "file", with_logo: bool = False) -> None:
    mark = (f'<img class="sanad-empty__mark" src="{logo_data_uri()}" alt="">' if with_logo
            else f'<span class="sanad-empty__icon">{icon(icon_name, 28, GREEN_400)}</span>')
    text = f"<p>{body}</p>" if body else ""
    st.markdown(f'<div class="sanad-empty">{mark}<h3>{title}</h3>{text}</div>', unsafe_allow_html=True)


def file_chip(name: str, size_kb: float) -> None:
    st.markdown(f'<div class="sanad-file">{icon("file", 20, GREEN_600)}'
                f'<span class="sanad-file__name">{name}</span>'
                f'<span class="sanad-file__size">{size_kb:,.0f} KB</span></div>', unsafe_allow_html=True)


def pill(label: str, tone: str = "neutral") -> str:
    """Inline status pill markup. tone: good | caution | bad | neutral."""
    return f'<span class="sanad-pill sanad-pill--{tone}">{label}</span>'


def metric(label: str, value: str, note: str = "") -> None:
    extra = f'<span class="sanad-metric__note">{note}</span>' if note else ""
    st.markdown(f'<div class="sanad-metric"><span class="sanad-metric__label">{label}</span>'
                f'<span class="sanad-metric__value">{value}</span>{extra}</div>', unsafe_allow_html=True)


def progress_panel(title: str, items: list[str], done: bool = False) -> None:
    """The loading state. Steps are only ticked once the whole request has come back."""
    mark = "&#10003;" if done else "&#9675;"
    rows = "".join(f'<li class="{"is-done" if done else ""}"><span>{mark}</span>{item}</li>' for item in items)
    st.markdown(f'<div class="sanad-progress"><p class="sanad-progress__title">{title}</p>'
                f'<ul>{rows}</ul></div>', unsafe_allow_html=True)


def footer(text: str) -> None:
    st.markdown(f'<div class="sanad-footer"><img src="{logo_data_uri()}" alt=""><span>{text}</span></div>',
                unsafe_allow_html=True)


def spacer(height: int = 18) -> None:
    st.markdown(f'<div style="height:{height}px"></div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- stylesheet
BASE_CSS = f"""
<style>
:root {{
  --g900:{GREEN_900}; --g800:{GREEN_800}; --g700:{GREEN_700}; --g600:{GREEN_600};
  --g500:{GREEN_500}; --g400:{GREEN_400}; --g100:{GREEN_100}; --g50:{GREEN_50};
  --gold700:{GOLD_700}; --gold600:{GOLD_600}; --gold500:{GOLD_500}; --gold400:{GOLD_400};
  --gold300:{GOLD_300}; --gold100:{GOLD_100};
  --ink:{INK}; --muted:{INK_MUTED}; --on-dark:{ON_DARK}; --on-dark-muted:{ON_DARK_MUTED};
  --surface:{SURFACE}; --border:{BORDER}; --danger:{DANGER}; --danger-soft:{DANGER_SOFT};
  --neutral:{NEUTRAL}; --neutral-soft:{NEUTRAL_SOFT};
  --r-lg:16px; --r-md:12px; --r-sm:9px;
  --shadow:0 1px 2px rgba(0,21,13,.05), 0 10px 26px rgba(0,21,13,.055);
  --shadow-lift:0 2px 6px rgba(0,21,13,.07), 0 18px 40px rgba(0,21,13,.10);
}}

/* ------------------------------------------------------------------ page shell */
[data-testid="stAppViewContainer"] {{ background: var(--g50); }}
[data-testid="stHeader"] {{ background: transparent; height: 0; }}
[data-testid="stToolbar"] {{ display: none; }}
#MainMenu, footer {{ visibility: hidden; }}
[data-testid="stAppViewContainer"] .block-container {{ padding: 2.4rem 2.6rem 3.5rem; max-width: 1180px; }}
html, body, [data-testid="stAppViewContainer"] {{ color: var(--ink); }}
[data-testid="stAppViewContainer"] a {{ color: var(--g600); text-underline-offset: 3px; }}
[data-testid="stAppViewContainer"] a:hover {{ color: var(--gold600); }}
hr {{ border-color: var(--border); }}
.sanad-icon {{ display: block; }}

/* ------------------------------------------------------------------ navigation */
[data-testid="stSidebar"] {{ background: var(--g900); border-right: 1px solid var(--g800); }}
[data-testid="stSidebar"] > div:first-child {{ padding-top: .6rem; }}
[data-testid="stSidebar"] img[data-testid="stLogo"] {{ height: 40px; width: auto; margin: 8px 0 0 6px; }}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] li,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{ color: var(--on-dark) !important; }}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{ color: var(--on-dark-muted) !important; }}
[data-testid="stSidebar"] hr {{ border-color: rgba(244,248,245,.12); }}
[data-testid="stSidebar"] input, [data-testid="stSidebar"] textarea {{
  background: rgba(244,248,245,.06) !important; color: var(--on-dark) !important;
  border: 1px solid rgba(244,248,245,.18) !important;
}}
.sanad-brand {{ display: flex; align-items: center; gap: 12px; padding: 4px 6px 2px; margin-bottom: 14px; }}
.sanad-brand img {{ height: 42px; width: auto; }}
.sanad-brand__name {{ display: block; color: var(--on-dark); font-size: 1.16rem; font-weight: 700; line-height: 1.1; }}
.sanad-brand__tag {{ display: block; color: var(--gold300); font-size: .76rem; margin-top: 3px; }}
.sanad-navlabel {{ color: var(--on-dark-muted) !important; font-size: .7rem; font-weight: 700;
  letter-spacing: .16em; text-transform: uppercase; margin: 22px 6px 8px; }}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap: .28rem; }}
[data-testid="stSidebar"] .stButton > button {{
  width: 100%; border-radius: var(--r-sm); padding: .48rem .85rem; font-weight: 500; font-size: .95rem;
  border: 1px solid transparent; background: transparent; color: var(--on-dark);
}}
/* Streamlit centres a button's inner block; navigation reads better aligned to the text edge */
[data-testid="stSidebar"] .stButton > button > div {{ width: 100%; justify-content: flex-start; }}
[data-testid="stSidebar"] .stButton > button p {{ text-align: start; width: 100%; }}
[data-testid="stSidebar"] .sanad-lang .stButton > button > div,
[data-testid="stSidebar"] [class*="st-key-lang_"] .stButton > button > div {{ justify-content: center; }}
[data-testid="stSidebar"] .stButton > button:hover {{ background: rgba(244,248,245,.08); color: var(--on-dark); }}
/* these two beat the light-surface rules further down, whatever the order */
[data-testid="stSidebar"] .stButton > button[kind="secondary"],
[data-testid="stSidebar"] .stButton > button[kind="tertiary"] {{
  background: transparent; color: var(--on-dark); border-color: transparent;
}}
[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover,
[data-testid="stSidebar"] .stButton > button[kind="tertiary"]:hover {{
  background: rgba(244,248,245,.10); color: var(--on-dark); border-color: transparent;
}}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {{
  background: var(--g600); color: #FFF; border-color: var(--g500); font-weight: 600;
}}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {{ background: var(--g500); }}
.sanad-lang {{ display: flex; gap: 8px; }}

/* ------------------------------------------------------------------ typography */
[data-testid="stAppViewContainer"] h1 {{ font-size: 1.95rem; font-weight: 700; letter-spacing: -.015em;
  color: var(--g700); margin: 0; }}
[data-testid="stAppViewContainer"] h2 {{ font-size: 1.22rem; font-weight: 650; color: var(--g700); }}
[data-testid="stAppViewContainer"] h3 {{ font-size: 1.02rem; font-weight: 650; color: var(--g700); }}
.sanad-page {{ display: flex; align-items: flex-start; gap: 14px; margin-bottom: 22px; }}
.sanad-page__icon {{ flex: 0 0 auto; background: var(--g100); border-radius: var(--r-md); padding: 10px;
  display: inline-flex; margin-top: 2px; }}
.sanad-page__sub {{ margin: 6px 0 0; color: var(--muted); font-size: .97rem; max-width: 76ch; }}
.sanad-section {{ display: flex; align-items: baseline; gap: 12px; margin: 30px 0 12px;
  padding-bottom: 9px; border-bottom: 1px solid var(--border); }}
.sanad-section h2 {{ margin: 0 !important; }}
.sanad-section h2::before {{ content: ""; display: inline-block; width: 8px; height: 8px; margin-inline-end: 10px;
  border-radius: 2px; background: var(--gold500); transform: translateY(-2px); }}
.sanad-section__note {{ color: var(--muted); font-size: .85rem; }}

/* ------------------------------------------------------------------ hero and cards */
.sanad-hero {{ text-align: center; padding: 44px 24px 34px; }}
.sanad-hero__mark {{ height: 84px; width: auto; display: inline-block; margin-bottom: 18px; }}
.sanad-hero__title {{ margin: 0 !important; font-size: 2.5rem; letter-spacing: -.02em; }}
.sanad-hero__tagline {{ margin: 8px 0 0; color: var(--gold600); font-size: 1.08rem; font-weight: 600; }}
.sanad-hero__body {{ margin: 14px auto 0; color: var(--muted); max-width: 62ch; font-size: 1rem; }}

div[class*="st-key-svc_"], div[class*="st-key-opt_"] {{
  background: var(--surface); border: 1px solid var(--border); border-radius: var(--r-lg);
  padding: 22px 22px 16px; box-shadow: var(--shadow); height: 100%;
  transition: box-shadow .16s ease, border-color .16s ease, transform .16s ease;
}}
div[class*="st-key-svc_"]:hover, div[class*="st-key-opt_"]:hover {{
  box-shadow: var(--shadow-lift); border-color: var(--g100); transform: translateY(-2px);
}}
.sanad-card__body h3 {{ margin: 14px 0 6px !important; }}
.sanad-card__body p {{ margin: 0 0 14px; color: var(--muted); font-size: .92rem; min-height: 42px; }}
.sanad-card__icon {{ display: inline-flex; padding: 10px; border-radius: var(--r-md); background: var(--g100); }}

/* ------------------------------------------------------------------ steps rail */
.sanad-steps {{ list-style: none; display: flex; flex-wrap: wrap; gap: 26px; margin: 0 0 26px; padding: 0; }}
.sanad-steps__item {{ display: flex; align-items: center; gap: 10px; font-size: .9rem; color: var(--muted); }}
.sanad-steps__dot {{ width: 26px; height: 26px; border-radius: 50%; display: inline-flex;
  align-items: center; justify-content: center; font-size: .8rem; font-weight: 700;
  background: var(--neutral-soft); color: var(--muted); border: 1px solid var(--border); }}
.sanad-steps__item--now {{ color: var(--g700); font-weight: 650; }}
.sanad-steps__item--now .sanad-steps__dot {{ background: var(--g600); color: #FFF; border-color: var(--g600); }}
.sanad-steps__item--done {{ color: var(--g600); }}
.sanad-steps__item--done .sanad-steps__dot {{ background: var(--g100); color: var(--g600); border-color: var(--g100); }}

/* ------------------------------------------------------------------ controls */
.stButton > button {{ border-radius: 999px; font-weight: 600; padding: .55rem 1.5rem;
  border: 1px solid var(--g600); transition: background .15s ease, border-color .15s ease; }}
.stButton > button[kind="primary"] {{ background: var(--g600); color: #FFF; }}
.stButton > button[kind="primary"]:hover {{ background: var(--g700); border-color: var(--g700); }}
.stButton > button[kind="secondary"] {{ background: var(--surface); color: var(--g700); border-color: var(--border); }}
.stButton > button[kind="secondary"]:hover {{ background: var(--g50); border-color: var(--g100); color: var(--g700); }}
.stButton > button[kind="tertiary"] {{ border-color: transparent; color: var(--g600); background: transparent; }}
[data-testid="stAppViewContainer"] input, [data-testid="stAppViewContainer"] textarea,
[data-testid="stAppViewContainer"] [data-baseweb="select"] > div {{
  border-radius: var(--r-sm) !important; border-color: var(--border) !important; background: var(--surface) !important;
}}
[data-testid="stAppViewContainer"] input:focus, [data-testid="stAppViewContainer"] textarea:focus {{
  border-color: var(--g600) !important; box-shadow: 0 0 0 2px rgba(1,78,43,.15) !important;
}}
[data-testid="stAppViewContainer"] label p {{ font-weight: 600; font-size: .9rem; color: var(--ink); }}
[data-baseweb="tag"] {{ background: var(--g600) !important; border-radius: 999px !important; }}

/* file upload */
[data-testid="stFileUploaderDropzone"] {{
  background: var(--surface); border: 1.5px dashed var(--gold400); border-radius: var(--r-lg); padding: 22px;
}}
[data-testid="stFileUploaderDropzone"]:hover {{ border-color: var(--g500); background: var(--g50); }}
[data-testid="stFileUploaderDropzoneInstructions"] {{ display: none; }}
[data-testid="stFileUploaderDropzone"] button {{ border-radius: 999px; border-color: var(--g600); color: var(--g700); }}
.sanad-drop {{ display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }}
.sanad-drop__icon {{ display: inline-flex; padding: 9px; border-radius: var(--r-md); background: var(--g100);
  color: var(--g600); }}
.sanad-drop__title {{ font-weight: 650; color: var(--ink); display: block; }}
.sanad-drop__hint {{ color: var(--muted); font-size: .86rem; }}
.sanad-file {{ display: flex; align-items: center; gap: 10px; background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--r-md); padding: 10px 14px; margin: 10px 0; color: var(--g700); }}
.sanad-file__name {{ font-weight: 600; }}
.sanad-file__size {{ color: var(--muted); font-size: .84rem; margin-inline-start: auto; }}

/* ------------------------------------------------------------------ surfaces */
[data-testid="stExpander"] {{ background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--r-md); box-shadow: none; overflow: hidden; }}
[data-testid="stExpander"] summary {{ color: var(--g700); font-weight: 600; }}
[data-testid="stExpander"] summary:hover {{ color: var(--gold600); }}
[data-testid="stDataFrame"] {{ border: 1px solid var(--border); border-radius: var(--r-md); overflow: hidden; }}

.sanad-metric {{ background: var(--surface); border: 1px solid var(--border); border-inline-start: 3px solid var(--gold500);
  border-radius: var(--r-md); padding: 14px 16px; height: 100%; }}
.sanad-metric__label {{ display: block; color: var(--muted); font-weight: 650; font-size: .72rem;
  letter-spacing: .07em; text-transform: uppercase; }}
.sanad-metric__value {{ display: block; color: var(--g700); font-size: 1.35rem; font-weight: 650; margin-top: 5px; }}
.sanad-metric__note {{ display: block; color: var(--muted); font-size: .8rem; margin-top: 4px; }}

.sanad-banner {{ display: block; padding: 13px 17px; border-radius: var(--r-md); margin: 4px 0 14px;
  border-inline-start: 4px solid var(--g600); background: var(--surface); border: 1px solid var(--border);
  border-inline-start-width: 4px; }}
.sanad-banner__title {{ font-weight: 700; color: var(--g700); margin-inline-end: 10px; }}
.sanad-banner__text {{ color: var(--ink); }}
.sanad-banner--good {{ border-inline-start-color: var(--g500); background: var(--g50); }}
.sanad-banner--caution {{ border-inline-start-color: var(--gold500); background: var(--gold100); border-color: #E7DBBE; }}
.sanad-banner--caution .sanad-banner__title {{ color: var(--gold700); }}
.sanad-banner--bad {{ border-inline-start-color: var(--danger); background: var(--danger-soft); border-color: #F0DAD6; }}
.sanad-banner--bad .sanad-banner__title {{ color: var(--danger); }}
.sanad-banner--neutral {{ border-inline-start-color: var(--neutral); }}

.sanad-pill {{ display: inline-flex; align-items: center; gap: 6px; padding: 3px 11px; border-radius: 999px;
  font-size: .78rem; font-weight: 650; white-space: nowrap; }}
.sanad-pill--good {{ background: var(--g100); color: var(--g700); }}
.sanad-pill--caution {{ background: var(--gold100); color: var(--gold700); }}
.sanad-pill--bad {{ background: var(--danger-soft); color: var(--danger); }}
.sanad-pill--neutral {{ background: var(--neutral-soft); color: var(--muted); }}

.sanad-finding {{ background: var(--surface); border: 1px solid var(--border); border-radius: var(--r-md);
  padding: 15px 18px; margin-bottom: 10px; }}
.sanad-finding__head {{ display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }}
.sanad-finding__title {{ font-weight: 650; color: var(--ink); font-size: .99rem; }}
.sanad-finding__articles {{ margin-inline-start: auto; color: var(--muted); font-size: .82rem; }}
.sanad-finding__text {{ margin: 9px 0 0; color: var(--muted); font-size: .92rem; }}
.sanad-quote {{ border-inline-start: 3px solid var(--g100); padding: 2px 0 2px 12px; margin: 10px 0 0;
  color: var(--ink); font-size: .9rem; }}
.sanad-quote--ar {{ direction: rtl; text-align: right; border-inline-start: none; border-inline-end: 3px solid var(--g100);
  padding: 2px 12px 2px 0; }}

.sanad-empty {{ text-align: center; padding: 46px 28px; background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--r-lg); }}
.sanad-empty__mark {{ height: 76px; width: auto; margin-bottom: 18px; }}
.sanad-empty__icon {{ display: inline-flex; padding: 13px; border-radius: 50%; background: var(--g100);
  margin-bottom: 14px; }}
.sanad-empty h3 {{ margin: 0 0 6px !important; }}
.sanad-empty p {{ margin: 0 auto; max-width: 56ch; color: var(--muted); }}

.sanad-progress {{ background: var(--surface); border: 1px solid var(--border); border-radius: var(--r-lg);
  padding: 24px 28px; }}
.sanad-progress__title {{ margin: 0 0 14px; font-weight: 650; color: var(--g700); }}
.sanad-progress ul {{ list-style: none; margin: 0; padding: 0; }}
.sanad-progress li {{ display: flex; gap: 11px; padding: 5px 0; color: var(--muted); font-size: .93rem; }}
.sanad-progress li.is-done {{ color: var(--g600); }}
.sanad-progress li span {{ color: var(--g500); font-weight: 700; }}

.sanad-source {{ display: flex; align-items: baseline; gap: 10px; padding: 9px 0; border-bottom: 1px solid var(--border); }}
.sanad-source__tier {{ color: var(--muted); font-size: .8rem; }}
.sanad-list {{ margin: 0; padding-inline-start: 18px; color: var(--muted); }}
.sanad-list li {{ margin-bottom: 7px; }}

.sanad-footer {{ display: flex; align-items: center; gap: 11px; margin-top: 40px; padding-top: 16px;
  border-top: 1px solid var(--border); }}
.sanad-footer img {{ height: 24px; width: auto; }}
.sanad-footer span {{ color: var(--muted); font-size: .8rem; }}

/* ------------------------------------------------------------------ small screens */
@media (max-width: 820px) {{
  [data-testid="stAppViewContainer"] .block-container {{ padding: 1.6rem 1.1rem 2.5rem; }}
  .sanad-hero {{ padding: 30px 12px 24px; }}
  .sanad-hero__title {{ font-size: 1.9rem; }}
  .sanad-steps {{ gap: 14px; }}
  .sanad-finding__articles {{ margin-inline-start: 0; }}
}}
</style>
"""

RTL_CSS = """
<style>
[data-testid="stAppViewContainer"], [data-testid="stSidebar"] { direction: rtl; }
[data-testid="stAppViewContainer"] .block-container, [data-testid="stSidebar"] > div { text-align: right; }
[data-testid="stSidebar"] .stButton > button { justify-content: flex-start; text-align: right; }
.sanad-metric, .sanad-banner, .sanad-finding, .sanad-file { text-align: right; }
[data-testid="stAppViewContainer"] input, [data-testid="stAppViewContainer"] textarea { text-align: right; }
.sanad-hero, .sanad-empty, .sanad-progress { text-align: center; }
.sanad-progress ul { text-align: right; }
</style>
"""


def apply(lang: str = "en") -> None:
    """Inject the stylesheet. Call once, right after the page config."""
    st.markdown(BASE_CSS, unsafe_allow_html=True)
    if lang == "ar":
        st.markdown(RTL_CSS, unsafe_allow_html=True)


def dropzone_label(title: str, hint: str) -> None:
    """Our own upload instructions; Streamlit's default English ones are hidden by the stylesheet."""
    st.markdown(f'<div class="sanad-drop"><span class="sanad-drop__icon">{icon("upload", 20, GREEN_600)}</span>'
                f'<span><span class="sanad-drop__title">{title}</span>'
                f'<span class="sanad-drop__hint">{hint}</span></span></div>', unsafe_allow_html=True)
