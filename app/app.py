import json
import html
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning


ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
DATA_PATH = ROOT_DIR / "data" / "all_tickets_processed_improved_v3.csv"
REPORT_PATH = MODELS_DIR / "model_report.json"

PROJECT_NAME = "Ticket Desk"
PROJECT_TAGLINE = "Support Intelligence Platform"
PROJECT_SUBTITLE = "Intelligent triage, SLA guidance, and support workflow automation."
PROJECT_DESCRIPTION = "A recruiter-ready support automation demo built with Python, Streamlit, and scikit-learn."

PAGES = [
    "Dashboard Overview",
    "Ticket Triage",
    "Model Performance",
    "Dataset Explorer",
    "About Project",
]

PAGE_LABELS = {
    "Dashboard Overview": "📊 Dashboard",
    "Ticket Triage": "🎫 Triage",
    "Model Performance": "📈 Performance",
    "Dataset Explorer": "🗂 Dataset",
    "About Project": "ℹ About",
}

SAMPLE_TICKETS = {
    "Production outage": (
        "Production database is unavailable for all users after the morning deployment. "
        "Customers cannot log in and the payment workflow is failing."
    ),
    "Access request": (
        "Please grant access to the analytics dashboard for a new marketing analyst. "
        "Manager approval has been attached."
    ),
    "Hardware issue": (
        "My laptop battery drains within 20 minutes and the keyboard randomly stops working during client calls."
    ),
    "Procurement": (
        "Need approval to purchase two monitors and a USB-C dock for the design team. Cost center and vendor quote are included."
    ),
    "Storage pressure": (
        "The shared drive is nearly full and backups are failing for the finance file share. We need urgent storage expansion."
    ),
    "Security escalation": (
        "A user reports a suspicious login notification and possible phishing email from an unknown sender."
    ),
}

TEXT_COLUMN_CANDIDATES = [
    "ticket_text",
    "text",
    "description",
    "summary",
    "issue",
    "problem",
    "request",
    "details",
]

CATEGORY_COLUMN_CANDIDATES = [
    "category",
    "issue_type",
    "ticket_category",
    "type",
    "label",
]

PRIORITY_COLUMN_CANDIDATES = [
    "priority",
    "urgency",
    "severity",
    "ticket_priority",
]

LOW_PRIORITY_KEYWORDS = ["information", "query", "general", "update", "change", "minor"]
MEDIUM_PRIORITY_KEYWORDS = ["slow", "delay", "intermittent", "error", "issue", "problem", "failed"]
HIGH_PRIORITY_KEYWORDS = ["outage", "urgent", "critical", "down", "failure", "blocked", "security"]

ROUTING_RULES = [
    ("access", "Identity & Access Management Team"),
    ("hardware", "Hardware Support Team"),
    ("hr support", "HR Operations Team"),
    ("human resources", "HR Operations Team"),
    ("administrative rights", "IT Admin Support Team"),
    ("admin rights", "IT Admin Support Team"),
    ("internal project", "Internal Tools Team"),
    ("project", "Internal Tools Team"),
    ("purchase", "Procurement Support Team"),
    ("procurement", "Procurement Support Team"),
    ("storage", "Infrastructure & Storage Team"),
    ("cloud", "Infrastructure & Storage Team"),
]

PRIORITY_SLA_RULES = {
    "high": {
        "label": "High",
        "guidance": "Immediate attention required. Suggested SLA: 2-4 hours.",
        "response_target": "2-4 hours",
        "resolution_target": "Same business day",
    },
    "medium": {
        "label": "Medium",
        "guidance": "Standard support queue. Suggested SLA: 24 hours.",
        "response_target": "24 hours",
        "resolution_target": "1 business day",
    },
    "low": {
        "label": "Low",
        "guidance": "Non-critical request. Suggested SLA: 2-3 business days.",
        "response_target": "2-3 business days",
        "resolution_target": "3 business days",
    },
}

PALETTE = {
    "bg": "#020617",
    "sidebar": "#061226",
    "card": "rgba(15, 23, 42, 0.92)",
    "card_elevated": "#111827",
    "text": "#F8FAFC",
    "muted": "#94A3B8",
    "border": "rgba(148, 163, 184, 0.18)",
    "blue": "#38BDF8",
    "cyan": "#22D3EE",
    "purple": "#8B5CF6",
    "green": "#22C55E",
    "amber": "#F59E0B",
    "red": "#EF4444",
}

px.defaults.template = "plotly_dark"
st.set_page_config(
    page_title=f"{PROJECT_NAME} — {PROJECT_TAGLINE}",
    page_icon="📩",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def normalize_text(value):
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def safe_read_csv(path):
    if not path.exists():
        return None
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error
    if last_error is not None:
        raise last_error
    return None


def resolve_column(columns, candidates):
    lookup = {str(column).lower(): column for column in columns}
    for candidate in candidates:
        match = lookup.get(candidate.lower())
        if match is not None:
            return match
    return None


def generate_priority_label(text):
    lowered = normalize_text(text).lower()
    if any(keyword in lowered for keyword in HIGH_PRIORITY_KEYWORDS):
        return "High"
    if any(keyword in lowered for keyword in MEDIUM_PRIORITY_KEYWORDS):
        return "Medium"
    if any(keyword in lowered for keyword in LOW_PRIORITY_KEYWORDS):
        return "Low"
    return "Low"


def get_sla_guidance(priority):
    key = normalize_text(priority).lower()
    return PRIORITY_SLA_RULES.get(key, PRIORITY_SLA_RULES["medium"])


def get_routing_team(category):
    normalized = normalize_text(category).lower()
    for keyword, team in ROUTING_RULES:
        if keyword in normalized:
            return team
    return "General IT Support Desk"


def get_urgency_message(priority):
    key = normalize_text(priority).lower()
    if key == "high":
        return "This ticket should be treated as urgent and monitored continuously until the service impact is contained."
    if key == "medium":
        return "This ticket should be handled in the standard support queue with normal operational monitoring."
    return "This ticket appears non-critical and can follow the normal service cadence unless new impact details emerge."


def summarize_ticket(text, limit=180):
    normalized = re.sub(r"\s+", " ", normalize_text(text)).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


@st.cache_resource(show_spinner=False)
def load_artifacts():
    artifacts = {"tfidf": None, "category_model": None, "priority_model": None}
    missing = []
    paths = {
        "tfidf": MODELS_DIR / "tfidf_vectorizer.pkl",
        "category_model": MODELS_DIR / "category_model.pkl",
        "priority_model": MODELS_DIR / "priority_model.pkl",
    }
    for key, path in paths.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", InconsistentVersionWarning)
                artifacts[key] = joblib.load(path)
        except Exception as exc:
            artifacts[key] = None
            missing.append(f"{path.name}: {exc}")
    return artifacts, missing


@st.cache_data(show_spinner=False)
def load_report():
    if not REPORT_PATH.exists():
        return {}
    try:
        return json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def load_dataset():
    # Primary: try the configured data path
    df = safe_read_csv(DATA_PATH)
    source = None
    if df is not None:
        source = DATA_PATH.name

    # Secondary: try to discover any CSV inside the data directory
    if df is None:
        data_dir = DATA_PATH.parent
        if data_dir.exists():
            csv_files = sorted([p for p in data_dir.iterdir() if p.suffix.lower() == ".csv"])
            if csv_files:
                df = safe_read_csv(csv_files[0])
                source = csv_files[0].name if df is not None else None

    # Tertiary: fall back to a small built-in sample to keep the UI usable
    if df is None:
        sample_rows = []
        for title, text in SAMPLE_TICKETS.items():
            sample_rows.append({"ticket_text": text, "category": "sample", "priority": "Low"})
        df = pd.DataFrame(sample_rows)
        source = "__sample_data__"

    # Normalize and detect columns
    text_column = None
    category_column = None
    priority_column = None

    # Special-case known dataset headers (common in provided CSVs)
    if df is not None:
        cols_lower = [str(c).lower() for c in df.columns]
        if "document" in cols_lower and text_column is None:
            text_column = next(c for c in df.columns if str(c).lower() == "document")
        if "topic_group" in cols_lower and category_column is None:
            category_column = next(c for c in df.columns if str(c).lower() == "topic_group")

    # Generic detection for other datasets
    if text_column is None:
        text_column = resolve_column(df.columns, TEXT_COLUMN_CANDIDATES)
    if category_column is None:
        category_column = resolve_column(df.columns, CATEGORY_COLUMN_CANDIDATES)
    if priority_column is None:
        priority_column = resolve_column(df.columns, PRIORITY_COLUMN_CANDIDATES)
    # Fallback: if no obvious text column detected, use the first non-numeric column
    if text_column is None:
        # Pick the first column that looks like text (object or pandas string dtype)
        from pandas.api.types import is_string_dtype

        for col in df.columns:
            try:
                if is_string_dtype(df[col]):
                    text_column = col
                    break
            except Exception:
                if df[col].dtype == object:
                    text_column = col
                    break
    # Fallback: if category not detected and there are multiple columns, use the second column
    if category_column is None and len(df.columns) >= 2:
        candidate = df.columns[1]
        if candidate != text_column:
            category_column = candidate
    if text_column is not None:
        df[text_column] = df[text_column].apply(normalize_text)
    if category_column is not None:
        df[category_column] = df[category_column].apply(normalize_text)
    if priority_column is not None:
        df[priority_column] = df[priority_column].apply(normalize_text)

    return {
        "df": df,
        "text_column": text_column,
        "category_column": category_column,
        "priority_column": priority_column,
        "source": source,
    }


def build_priority_series(df, text_column, priority_column=None):
    if text_column is None:
        return pd.Series(dtype=str)
    if priority_column is not None and priority_column in df.columns:
        series = df[priority_column].copy()
        generated = df[text_column].apply(generate_priority_label)
        series = series.where(series.astype(str).str.strip().ne(""), generated)
        return series.fillna(generated).astype(str)
    return df[text_column].apply(generate_priority_label).astype(str)


def prediction_payload(ticket_text, artifacts):
    vectorizer = artifacts.get("tfidf")
    category_model = artifacts.get("category_model")
    priority_model = artifacts.get("priority_model")
    if vectorizer is None or category_model is None or priority_model is None:
        raise RuntimeError("Model artifacts are not available.")

    transformed = vectorizer.transform([ticket_text])
    predicted_category = normalize_text(category_model.predict(transformed)[0])
    predicted_priority = normalize_text(priority_model.predict(transformed)[0])
    sla = get_sla_guidance(predicted_priority)
    team = get_routing_team(predicted_category)
    urgency = get_urgency_message(predicted_priority)
    summary = summarize_ticket(ticket_text, 220)
    response = (
        "Hello,\n\n"
        f"Thank you for contacting support. We have reviewed your request and classified it as {predicted_category}. "
        f"It has been routed to {team}. Current priority: {predicted_priority}. SLA guidance: {sla['guidance']} "
        "Our team will continue reviewing the issue and provide the next update within the expected response window.\n\n"
        f"Ticket summary: {summary}\n\n"
        "Best regards,\n"
        "Support Team"
    )
    return {
        "ticket_text": ticket_text,
        "predicted_category": predicted_category,
        "predicted_priority": predicted_priority,
        "sla": sla,
        "team": team,
        "urgency": urgency,
        "response": response,
        "summary": summary,
    }


# -----------------------------------------------------------------------------
# Styling helpers
# -----------------------------------------------------------------------------

def inject_css():
    st.markdown(
        f"""
        <style>
            :root {{
                --bg: {PALETTE['bg']};
                --sidebar: {PALETTE['sidebar']};
                --text: {PALETTE['text']};
                --muted: {PALETTE['muted']};
                --card: {PALETTE['card']};
                --card-elevated: {PALETTE['card_elevated']};
                --border: {PALETTE['border']};
                --blue: {PALETTE['blue']};
                --cyan: {PALETTE['cyan']};
                --purple: {PALETTE['purple']};
                --green: {PALETTE['green']};
                --amber: {PALETTE['amber']};
                --red: {PALETTE['red']};
                --radius: 18px;
                --shadow: 0 24px 60px rgba(2, 6, 23, 0.48);
            }}

            html, body, [class*="css"] {{
                font-family: Inter, "Segoe UI", Arial, sans-serif;
            }}

            html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] .main {{
                background:
                    radial-gradient(circle at top right, rgba(34,211,238,0.06), transparent 24%),
                    radial-gradient(circle at bottom left, rgba(139,92,246,0.05), transparent 25%),
                    var(--bg);
                color: var(--text);
            }}

            header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stDecoration"] {{
                display: none;
                visibility: hidden;
                height: 0;
            }}

            .block-container {{
                max-width: 100%;
                padding: 1rem 2rem 2rem;
            }}

            [data-testid="stSidebar"] {{
                background: linear-gradient(180deg, #020617 0%, var(--sidebar) 100%);
                border-right: 1px solid rgba(148,163,184,0.12);
                box-shadow: 24px 0 64px rgba(2, 6, 23, 0.45);
            }}
            [data-testid="stSidebar"] * {{ color: #E2E8F0 !important; }}
            [data-testid="stSidebar"] .sidebar-wrap {{ padding: 1.15rem 1rem 1rem; }}

            .sidebar-shell {{ display:flex; flex-direction:column; gap:0.85rem; }}
            .brand-block {{
                padding: 1rem;
                border: 1px solid rgba(148,163,184,0.12);
                border-radius: 18px;
                background: linear-gradient(180deg, rgba(15,23,42,0.92), rgba(2,6,23,0.94));
            }}
            .brand-name {{ font-size: 1.45rem; font-weight: 850; letter-spacing: -0.04em; line-height: 1.05; color: #fff; }}
            .brand-subtitle {{ margin-top: 0.2rem; font-size: 0.94rem; font-weight: 650; color: #BFDBFE; }}
            .brand-desc {{ margin-top: 0.7rem; font-size: 0.86rem; line-height: 1.55; color: #94A3B8; }}
            .sidebar-nav-title {{
                margin: 0.9rem 0 0.35rem;
                color: #94A3B8;
                text-transform: uppercase;
                letter-spacing: 0.12em;
                font-size: 0.72rem;
                font-weight: 850;
            }}
            .sidebar-note, .sidebar-footer {{
                margin-top: 1rem;
                padding-top: 0.95rem;
                border-top: 1px solid rgba(148,163,184,0.10);
                color: #94A3B8;
                font-size: 0.82rem;
                line-height: 1.5;
            }}

            .hero-card, .metric-card, .card, .chart-card, .result-card, .info-card, .empty-state, .email-card, .analysis-card, .report-card, .content-card, .card-shell, .business-card, .premium-card, .empty-card {{
                background: var(--card);
                border: 1px solid var(--border);
                border-radius: var(--radius);
                box-shadow: var(--shadow);
                backdrop-filter: blur(18px);
            }}

            .hero-card {{
                padding: 1.25rem 1.35rem;
                background:
                    radial-gradient(circle at top right, rgba(56,189,248,0.14), transparent 32%),
                    linear-gradient(180deg, rgba(15,23,42,0.96) 0%, rgba(17,24,39,0.96) 100%);
                position: relative;
                overflow: hidden;
            }}
            .hero-card::before, .metric-card::before, .business-card::before {{
                content: "";
                position: absolute;
                left: 0;
                top: 0;
                height: 3px;
                width: 100%;
                background: linear-gradient(90deg, rgba(56,189,248,0.95), rgba(34,211,238,0.95), rgba(139,92,246,0.85));
            }}
            .hero-grid {{ display:flex; gap:1rem; justify-content:space-between; align-items:stretch; }}
            .hero-copy {{ flex:1 1 auto; min-width:0; }}
            .hero-meta {{ min-width:260px; display:flex; flex-direction:column; gap:0.5rem; align-items:flex-end; justify-content:center; }}
            .hero-title {{ font-size:1.95rem; line-height:1.08; font-weight:850; letter-spacing:-0.05em; color:var(--text); margin:0; }}
            .hero-subtitle {{ margin-top:0.55rem; font-size:1rem; line-height:1.6; color:var(--muted); max-width:78ch; }}

            .badge, .pill {{ display:inline-flex; align-items:center; gap:0.35rem; border-radius:999px; padding:0.38rem 0.75rem; font-size:0.76rem; font-weight:800; line-height:1; border:1px solid transparent; white-space:nowrap; }}
            .badge-blue, .pill-blue {{ background: rgba(56,189,248,0.12); color:#BAE6FD; border-color: rgba(56,189,248,0.22); }}
            .badge-cyan, .pill-cyan {{ background: rgba(34,211,238,0.12); color:#CFFAFE; border-color: rgba(34,211,238,0.22); }}
            .badge-green, .pill-green {{ background: rgba(34,197,94,0.12); color:#DCFCE7; border-color: rgba(34,197,94,0.22); }}
            .badge-amber, .pill-amber {{ background: rgba(245,158,11,0.14); color:#FEF3C7; border-color: rgba(245,158,11,0.22); }}
            .badge-red, .pill-red {{ background: rgba(239,68,68,0.14); color:#FEE2E2; border-color: rgba(239,68,68,0.22); }}
            .badge-neutral, .pill-neutral {{ background: rgba(148,163,184,0.12); color:#E2E8F0; border-color: rgba(148,163,184,0.22); }}

            .section-title {{ font-size:1.35rem; line-height:1.18; font-weight:850; letter-spacing:-0.04em; color:var(--text); margin:0; }}
            .section-subtitle {{ margin-top:0.3rem; color:var(--muted); font-size:0.93rem; line-height:1.55; }}
            .section-wrap {{ margin-top:1rem; }}

            .metric-card {{ padding:0.9rem 0.95rem 0.85rem; min-height:112px; position:relative; overflow:hidden; background:linear-gradient(180deg, rgba(17,24,39,0.98), rgba(15,23,42,0.94)); }}
            .metric-label {{ margin:0 0 0.45rem; font-size:0.76rem; font-weight:850; text-transform:uppercase; letter-spacing:0.08em; color:var(--muted); }}
            .metric-value {{ font-size:1.65rem; line-height:1.02; font-weight:860; letter-spacing:-0.04em; color:var(--text); margin:0; }}
            .metric-caption {{ margin-top:0.4rem; font-size:0.82rem; line-height:1.45; color:var(--muted); }}

            .card, .chart-card, .info-card, .empty-state, .analysis-card, .report-card {{ padding:1rem 1.02rem; }}
            .card-title {{ margin:0; font-size:1rem; line-height:1.2; font-weight:850; color:var(--text); letter-spacing:-0.03em; }}
            .card-body {{ margin-top:0.38rem; color:var(--muted); font-size:0.94rem; line-height:1.6; }}
            .mini-card {{ border:1px solid var(--border); border-radius:14px; background:linear-gradient(180deg, rgba(17,24,39,0.95), rgba(15,23,42,0.92)); padding:0.95rem; }}
            .mini-copy {{ color:var(--text); font-size:0.94rem; line-height:1.55; }}
            .pill-row, .status-line {{ display:flex; flex-wrap:wrap; gap:0.45rem; }}

            .result-card {{ padding:0.92rem 0.95rem 0.88rem; min-height:104px; }}
            .result-label {{ font-size:0.76rem; text-transform:uppercase; letter-spacing:0.08em; font-weight:850; color:var(--muted); margin-bottom:0.35rem; }}
            .result-value {{ font-size:1.08rem; font-weight:850; line-height:1.2; color:var(--text); margin-bottom:0.35rem; overflow-wrap:anywhere; }}
            .result-caption {{ color:var(--muted); font-size:0.81rem; line-height:1.45; }}

            .empty-state, .empty-card {{ min-height:260px; display:flex; align-items:center; justify-content:center; text-align:center; padding:1.25rem; }}
            .empty-state-inner {{ max-width:360px; }}
            .empty-title {{ font-size:1.05rem; font-weight:850; color:var(--text); margin-top:0.75rem; }}
            .empty-subtitle {{ margin-top:0.35rem; color:var(--muted); font-size:0.92rem; line-height:1.55; }}

            .email-card {{ padding:1rem 1rem 0.95rem; }}
            .email-subject {{ font-size:0.84rem; font-weight:850; color:#CBD5E1; margin-bottom:0.55rem; }}
            .email-preview {{ border:1px solid var(--border); border-radius:14px; background:linear-gradient(180deg, rgba(2,6,23,0.92), rgba(15,23,42,0.94)); padding:0.95rem; color:#E2E8F0; font-size:0.92rem; line-height:1.68; white-space:pre-wrap; }}

            .section-divider {{ height:1px; background:linear-gradient(90deg, rgba(226,232,240,0), rgba(148,163,184,0.6), rgba(226,232,240,0)); margin:1rem 0; }}

            /* Primary CTA styling */
            .stButton > button {{ border-radius:14px; border:1px solid rgba(56,189,248,0.24); background:linear-gradient(135deg, #38BDF8 0%, #2563EB 55%, #8B5CF6 100%); color:white; font-weight:800; padding:0.72rem 1rem; box-shadow:0 14px 28px rgba(8,145,178,0.22); }}
            .stButton > button:hover {{ background:linear-gradient(135deg, #22D3EE 0%, #2563EB 55%, #7C3AED 100%); border-color:rgba(34,211,238,0.36); transform:translateY(-1px); }}
            .stButton > button[disabled], .stButton > button[aria-disabled="true"] {{ opacity:0.52; cursor:not-allowed; filter:grayscale(18%); box-shadow:none; }}

            /* Inputs and selects */
            .stTextArea textarea, .stTextInput input, .stSelectbox div[data-baseweb="select"] > div, .stMultiSelect div[data-baseweb="select"] > div, .stNumberInput input {{ border-radius:14px !important; border-color:rgba(148,163,184,0.20) !important; background:rgba(15,23,42,0.92) !important; color:var(--text) !important; box-shadow:none !important; }}
            .stTextArea textarea:focus, .stTextInput input:focus, .stSelectbox div[data-baseweb="select"] > div:focus-within, .stMultiSelect div[data-baseweb="select"] > div:focus-within {{ border-color:rgba(56,189,248,0.48) !important; box-shadow:0 0 0 1px rgba(56,189,248,0.16) !important; }}

            /* Dataframe/table polish for dark theme */
            div[data-testid="stDataFrame"], div[data-testid="stTable"] {{ border-radius:16px; overflow:hidden; border:1px solid var(--border); background:linear-gradient(180deg, rgba(17,24,39,0.96), rgba(15,23,42,0.96)); }}
            div[data-testid="stDataFrame"] * {{ color: var(--text) !important; }}
            div[data-testid="stDataFrame"] [role="gridcell"], div[data-testid="stDataFrame"] [role="columnheader"] {{ background:transparent !important; border-color:rgba(148,163,184,0.12) !important; }}
            div[data-testid="stDataFrame"] table {{ background:transparent !important; border-collapse:separate !important; border-spacing:0 8px !important; }}
            div[data-testid="stDataFrame"] thead th {{ background:linear-gradient(180deg, rgba(17,24,39,0.95), rgba(15,23,42,0.95)); color:var(--muted) !important; font-weight:800; border-bottom:1px solid rgba(148,163,184,0.06); padding:10px 12px; }}
            div[data-testid="stDataFrame"] tbody td {{ background:linear-gradient(180deg, rgba(2,6,23,0.28), rgba(2,6,23,0.12)); color:var(--text) !important; padding:12px; border-radius:10px; }}

            /* Tabs and expanders */
            .stTabs [data-baseweb="tab-list"] {{ gap:0.4rem; background:rgba(15,23,42,0.70); padding:0.35rem; border-radius:14px; border:1px solid rgba(148,163,184,0.14); }}
            .stTabs [data-baseweb="tab"] {{ border-radius:12px; color:var(--muted); background:transparent; padding:0.45rem 0.9rem; font-weight:700; }}
            .stTabs [aria-selected="true"] {{ color:var(--text) !important; background:linear-gradient(135deg, rgba(56,189,248,0.16), rgba(34,211,238,0.12)); border:1px solid rgba(56,189,248,0.24); }}

            .stExpander {{ border-radius:16px; border:1px solid rgba(148,163,184,0.16) !important; background:rgba(15,23,42,0.9); }}
            .stExpander summary {{ color:var(--text) !important; }}
            .stAlert {{ border-radius:16px; border:1px solid rgba(148,163,184,0.16); background:rgba(15,23,42,0.94); color:var(--text); }}

            .report-card, .analysis-card, .content-card, .card-shell, .business-card, .info-card, .chart-card, .premium-card {{ background:linear-gradient(180deg, rgba(17,24,39,0.96), rgba(15,23,42,0.94)); border:1px solid var(--border); box-shadow:var(--shadow); }}
            .business-card {{ position:relative; overflow:hidden; }}
            .business-card::before {{ content:""; position:absolute; inset:0 auto auto 0; width:100%; height:3px; background:linear-gradient(90deg, rgba(56,189,248,0.95), rgba(34,211,238,0.95), rgba(139,92,246,0.80)); }}
            .business-icon {{ display:inline-flex; align-items:center; justify-content:center; width:2rem; height:2rem; border-radius:999px; margin-bottom:0.65rem; background:rgba(56,189,248,0.14); color:#BAE6FD; font-weight:800; font-size:0.78rem; border:1px solid rgba(56,189,248,0.22); }}

            .analysis-header {{ display:flex; justify-content:space-between; align-items:flex-start; gap:1rem; }}
            .analysis-title {{ font-size:1.3rem; font-weight:850; letter-spacing:-0.04em; }}
            .analysis-subtitle {{ margin-top:0.25rem; color:var(--muted); font-size:0.92rem; }}

            .profile-avatar {{ width:86px; height:86px; border-radius:12px; background:linear-gradient(135deg, var(--blue), var(--cyan)); display:inline-block; flex:0 0 86px; box-shadow:0 14px 28px rgba(8,145,178,0.22); }}
            .profile-meta {{ font-size:0.95rem; }}
            .profile-meta .name {{ font-weight:800; font-size:1.05rem; color:var(--text); }}
            .profile-meta .title {{ color:var(--muted); margin-top:0.15rem; font-size:0.88rem; }}
            .profile-bio {{ margin-top:0.5rem; color:var(--muted); font-size:0.9rem; line-height:1.55; }}
            .profile-links a {{ text-decoration:none; }}

            .hint-text {{ color:var(--muted); font-size:0.84rem; line-height:1.45; }}
            .report-table-note {{ color:var(--muted); font-size:0.85rem; margin-top:0.15rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def priority_tone(priority):
    normalized = normalize_text(priority).lower()
    if normalized == "high":
        return "red"
    if normalized == "medium":
        return "amber"
    if normalized == "low":
        return "green"
    return "blue"


def pill(text, tone="blue"):
    tone_class = {
        "blue": "pill-blue",
        "cyan": "pill-cyan",
        "green": "pill-green",
        "amber": "pill-amber",
        "red": "pill-red",
        "neutral": "pill-neutral",
    }.get(str(tone).lower(), "pill-blue")
    return f'<span class="pill {tone_class}">{html.escape(str(text))}</span>'


def status_badge(text, tone="blue"):
    tone_class = {
        "blue": "badge-blue",
        "cyan": "badge-cyan",
        "green": "badge-green",
        "amber": "badge-amber",
        "red": "badge-red",
        "neutral": "badge-neutral",
    }.get(str(tone).lower(), "badge-blue")
    return f'<span class="badge {tone_class}">{html.escape(str(text))}</span>'


def style_plotly_dark(fig):
    if fig is None:
        return None
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["text"], size=12),
        title_font=dict(color=PALETTE["text"], size=18),
        margin=dict(l=10, r=10, t=50, b=10),
        height=390,
        showlegend=False,
        xaxis=dict(title="", gridcolor="rgba(148,163,184,0.15)", zeroline=False, tickfont=dict(color=PALETTE["muted"]), linecolor="rgba(148,163,184,0.16)"),
        yaxis=dict(title="", gridcolor="rgba(148,163,184,0.15)", zeroline=False, tickfont=dict(color=PALETTE["muted"]), linecolor="rgba(148,163,184,0.16)"),
    )
    return fig


def dark_dataframe(df: pd.DataFrame, height: int | None = None):
    """Render a pandas DataFrame with a compact dark theme using pandas Styler."""
    if df is None or getattr(df, "empty", True):
        st.info("No tabular data to display.")
        return
    try:
        styler = (
            df.style
            .set_table_styles([
                {"selector": "", "props": [("background-color", "transparent")]},
                {"selector": "th", "props": [("color", PALETTE["muted"]), ("font-weight", "700"), ("text-align", "left")]},
                {"selector": "td", "props": [("color", PALETTE["text"]), ("background-color", "rgba(2,6,23,0.16)"), ("padding", "6px 10px")]} ,
            ])
            .hide_index()
        )
        st.dataframe(styler, use_container_width=True, height=height)
    except Exception:
        # Fallback to default rendering when Styler is not supported
        st.dataframe(df, use_container_width=True, height=height)


def divider():
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


def section_header(title, subtitle):
    st.markdown(
        f"""
        <div class="section-wrap">
            <div class="section-title">{html.escape(str(title))}</div>
            <div class="section-subtitle">{html.escape(str(subtitle))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_hero(title, subtitle, badge=None):
    badge_html = f'<div style="margin-bottom:0.8rem;">{badge}</div>' if badge else ""
    st.markdown(
        f"""
        <div class="hero-card">
            {badge_html}
            <div class="hero-title">{html.escape(str(title))}</div>
            <div class="hero-subtitle">{html.escape(str(subtitle))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def glass_card(title, body):
    st.markdown(
        f"""
        <div class="card content-card">
            <div class="card-title">{html.escape(str(title))}</div>
            <div class="card-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def card(title, body):
    glass_card(title, body)


def metric_card(label, value, caption="", tone="blue"):
    accent = {
        "blue": PALETTE["blue"],
        "cyan": PALETTE["cyan"],
        "green": PALETTE["green"],
        "amber": PALETTE["amber"],
        "red": PALETTE["red"],
    }.get(str(tone).lower(), PALETTE["blue"])
    st.markdown(
        f"""
        <div class="metric-card" style="border-top: 3px solid {accent};">
            <div class="metric-label">{html.escape(str(label))}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-caption">{html.escape(str(caption))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def result_tile(label, value, caption="", tone="blue"):
    value_html = value if isinstance(value, str) and value.strip().startswith("<span") else html.escape(str(value))
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">{html.escape(str(label))}</div>
            <div class="result-value">{value_html}</div>
            <div class="result-caption">{html.escape(str(caption))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_state(title, subtitle):
    st.markdown(
        f"""
        <div class="empty-state empty-card">
            <div class="empty-state-inner">
                <div>{status_badge('Awaiting analysis', 'neutral')}</div>
                <div class="empty-title">{html.escape(str(title))}</div>
                <div class="empty-subtitle">{html.escape(str(subtitle))}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_bar_figure(series, title, top_n=None, horizontal=False):
    if series is None or getattr(series, "empty", True):
        return None
    counts = series.value_counts(dropna=False)
    if top_n is not None:
        counts = counts.head(top_n)
    chart_df = counts.reset_index()
    chart_df.columns = ["Label", "Count"]
    chart_df = chart_df.sort_values("Count", ascending=True if horizontal else False)
    fig = px.bar(chart_df, x="Count", y="Label", orientation="h" if horizontal else None)
    palette = [PALETTE["blue"], PALETTE["cyan"], PALETTE["purple"], PALETTE["green"], PALETTE["amber"]]
    colors = (palette * ((len(chart_df) // len(palette)) + 1))[: len(chart_df)]
    fig.update_traces(marker=dict(color=colors), marker_line_width=0, opacity=0.96)
    fig.update_layout(title=title)
    return style_plotly_dark(fig)


def chart_card(title, fig):
    st.markdown(
        f"""
        <div class="chart-card">
            <div class="card-title">{html.escape(str(title))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if fig is None:
        empty_state("No data available for this chart", "Adjust filters or add more data to populate this visualization.")
    else:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})


def report_table(report_dict):
    if not isinstance(report_dict, dict) or not report_dict:
        return pd.DataFrame()
    table = pd.DataFrame(report_dict).T
    preferred = ["precision", "recall", "f1-score", "support"]
    available = [column for column in preferred if column in table.columns]
    if available:
        table = table[available]
    for column in ["precision", "recall", "f1-score"]:
        if column in table.columns:
            table[column] = table[column].apply(lambda value: round(float(value) * 100, 2) if pd.notna(value) else value)
    if "support" in table.columns:
        table["support"] = table["support"].apply(lambda value: int(round(float(value))) if pd.notna(value) else value)
    return table


# -----------------------------------------------------------------------------
# Sidebar and page renderers
# -----------------------------------------------------------------------------

def build_sidebar(artifacts):
    with st.sidebar:
        st.markdown(
            f"""
            <div class="sidebar-shell">
                <div class="brand-block">
                    <div class="badge badge-cyan" style="margin-bottom:0.65rem;">NLP Automation System</div>
                    <div class="brand-name">{PROJECT_NAME}</div>
                    <div class="brand-subtitle">{PROJECT_TAGLINE}</div>
                    <div class="brand-desc">{PROJECT_DESCRIPTION}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-nav-title">Navigation</div>', unsafe_allow_html=True)
        selected = st.radio(
            "Navigation",
            PAGES,
            index=0,
            format_func=lambda page: PAGE_LABELS.get(page, page),
            label_visibility="collapsed",
        )

        st.markdown(
            f"<div style='margin-top:0.9rem;'>{status_badge('Models Ready', 'green') if all(artifacts.values()) else status_badge('Training Required', 'amber')}</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="sidebar-footer">
                Future Interns ML Task 2<br>
                Python · NLP · Streamlit
            </div>
            """,
            unsafe_allow_html=True,
        )
    return selected


def render_overview(report, dataset_info):
    df = dataset_info["df"] if dataset_info and dataset_info.get("df") is not None else None
    text_column = dataset_info.get("text_column") if dataset_info else None
    category_column = dataset_info.get("category_column") if dataset_info else None
    priority_column = dataset_info.get("priority_column") if dataset_info else None
    dataset_rows = len(df) if df is not None else None
    priority_series = build_priority_series(df, text_column, priority_column) if df is not None else None

    page_hero(
        "Support Intelligence Dashboard",
        "Monitor ticket distribution, model performance, and automated triage readiness from one workspace.",
        badge=status_badge("NLP Automation System", "cyan"),
    )

    st.markdown(
        f"""
        <div style="margin-top:0.85rem; display:flex; gap:0.45rem; flex-wrap:wrap;">
            {status_badge('Models Ready', 'green') if all(artifacts.values()) else status_badge('Training Required', 'amber')}
            {status_badge(f'{dataset_rows:,} Tickets' if dataset_rows is not None else 'Dataset Unavailable', 'blue')}
        </div>
        """,
        unsafe_allow_html=True,
    )

    divider()

    metrics = [
        ("Tickets Analyzed", f"{dataset_rows:,}" if dataset_rows is not None else "--", "Rows available for modeling and exploration", "blue"),
        ("Category Accuracy", f"{float(report.get('category_accuracy', 0)) * 100:.1f}%" if report else "--", "LinearSVC held-out score", "cyan"),
        ("Priority Accuracy", f"{float(report.get('priority_accuracy', 0)) * 100:.1f}%" if report else "--", "Logistic Regression held-out score", "green"),
        ("TF-IDF Features", f"{report.get('tfidf_feature_count', '--')}" if report else "--", "Vocabulary size used for ticket vectorization", "amber"),
    ]
    for column, spec in zip(st.columns(4, gap="small"), metrics):
        with column:
            metric_card(*spec)

    divider()

    section_header("Business Value", "A compact view of the product outcomes this platform is designed to improve.")
    for column, (title, body, icon) in zip(st.columns(3, gap="medium"), [
        ("Faster Triage", "Classify and prioritize support tickets in seconds so human agents can focus on resolution rather than manual sorting.", "01"),
        ("SLA Awareness", "Identify urgent requests early and surface clear response targets before support queues become overloaded.", "02"),
        ("Smarter Routing", "Route tickets to the correct support team using category-aware business logic and clear operational context.", "03"),
    ]):
        with column:
            st.markdown(
                f"""
                <div class="business-card card">
                    <div class="business-icon">{icon}</div>
                    <div class="card-title">{html.escape(title)}</div>
                    <div class="card-body">{html.escape(body)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    divider()
    chart_cols = st.columns(2, gap="large")
    with chart_cols[0]:
        fig = make_bar_figure(df[category_column] if df is not None and category_column is not None else None, "Category Distribution", top_n=12, horizontal=True)
        chart_card("Category Distribution", fig)
    with chart_cols[1]:
        fig = make_bar_figure(priority_series, "Priority Distribution", horizontal=False)
        chart_card("Priority Distribution", fig)


def render_ticket_triage(artifacts):
    page_hero(
        "Ticket Triage",
        "Paste a support request, run the analysis, and turn messy ticket text into a structured triage report.",
        badge=status_badge("Real-time NLP", "cyan"),
    )

    can_predict = all(artifacts.get(key) is not None for key in ["tfidf", "category_model", "priority_model"])
    if not can_predict:
        st.warning("Model artifacts are missing. Please run `python scripts/train_models.py` to regenerate the vectorizer and models.")

    if "ticket_input" not in st.session_state:
        st.session_state["ticket_input"] = next(iter(SAMPLE_TICKETS.values()))
    if "last_analysis" not in st.session_state:
        st.session_state["last_analysis"] = None

    left_col, right_col = st.columns([1.08, 1], gap="large")
    with left_col:
        glass_card("New Support Ticket", "Add a ticket description, use a sample for quick testing, and generate the triage output.")
        st.write("")
        sample_choice = st.selectbox("Example ticket", list(SAMPLE_TICKETS.keys()), index=0)
        if st.button("Load example ticket", use_container_width=True):
            st.session_state["ticket_input"] = SAMPLE_TICKETS[sample_choice]
            st.session_state["last_analysis"] = None
            st.rerun()

        ticket_text = st.text_area(
            "Ticket description",
            key="ticket_input",
            height=220,
            placeholder="Describe the support issue, incident, request, or service problem...",
        )
        analyze_clicked = st.button("Analyze Ticket", type="primary", use_container_width=True, disabled=not can_predict)
        st.caption("Tip: include the user impact, system affected, and urgency details.")

        if analyze_clicked:
            cleaned_text = normalize_text(ticket_text)
            if not cleaned_text:
                st.error("Please enter a support ticket before running the analysis.")
            else:
                try:
                    st.session_state["last_analysis"] = prediction_payload(cleaned_text, artifacts)
                except Exception as exc:
                    st.session_state["last_analysis"] = None
                    st.error(f"Ticket analysis failed: {exc}")

    with right_col:
        result = st.session_state.get("last_analysis")
        if not result:
            empty_state(
                "Triage report will appear here",
                "Run an analysis to view the predicted category, priority, SLA target, routing team, urgency note, and response draft.",
            )
        else:
            priority_value = result["predicted_priority"]
            priority_style = priority_tone(priority_value)
            st.markdown(
                f"""
                <div class="report-card">
                    <div class="analysis-header">
                        <div>
                            <div class="analysis-title">Triage Report</div>
                            <div class="analysis-subtitle">Completed analysis for the submitted support ticket.</div>
                        </div>
                        <div class="status-line">
                            {status_badge('Completed', 'green')}
                            {status_badge(priority_value, priority_style)}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            for column, (label, value, caption, tone) in zip(st.columns(4, gap="small"), [
                ("Category", result["predicted_category"], "Predicted issue type", "blue"),
                ("Priority", result["predicted_priority"], "Urgency classification", priority_style),
                ("SLA Target", result["sla"]["response_target"], "Suggested response window", "cyan"),
                ("Routing Team", result["team"], "Suggested support queue", "neutral"),
            ]):
                with column:
                    result_tile(label, value, caption, tone)

            divider()
            analysis_cols = st.columns([1, 1], gap="large")
            with analysis_cols[0]:
                glass_card("Urgency Explanation", result["urgency"])
                st.write("")
                glass_card(
                    "Routing Decision",
                    f"<div class='status-line'>{pill(result['predicted_category'], 'blue')}{pill(result['team'], 'neutral')}{pill(result['predicted_priority'], priority_style)}</div>",
                )
            with analysis_cols[1]:
                subject_line = f"Subject: {result['predicted_priority']} support ticket routed to {result['team']}"
                response_text = subject_line + "\n\n" + result["response"]
                st.markdown(
                    f"""
                    <div class="email-card">
                        <div class="card-title">First-Response Draft</div>
                        <div class="card-body">Email preview</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.code(response_text, language="text")
                try:
                    st.download_button("Download response", response_text, file_name="ai_response.txt", mime="text/plain")
                except Exception:
                    # graceful fallback if download button not available
                    st.text_area("Response (copy to clipboard)", value=response_text, height=140)


def render_model_performance():
    page_hero(
        "Model Performance Center",
        "Review the saved training report, evaluation metrics, and classification quality from the local ML pipeline.",
        badge=status_badge("Monitoring View", "cyan"),
    )
    if not report:
        st.warning("Model report not found. Run `python scripts/train_models.py` to generate `models/model_report.json`.")
        return

    for column, (label, value, caption, tone) in zip(st.columns(4, gap="small"), [
        ("Category Accuracy", f"{float(report.get('category_accuracy', 0)) * 100:.1f}%", "Held-out score for category prediction", "blue"),
        ("Priority Accuracy", f"{float(report.get('priority_accuracy', 0)) * 100:.1f}%", "Held-out score for urgency prediction", "cyan"),
        ("Feature Count", f"{report.get('tfidf_feature_count', '--')}", "Vocabulary size used by TF-IDF", "green"),
        ("Dataset Rows", f"{report.get('dataset_rows', '--')}", "Rows used in the local training run", "amber"),
    ]):
        with column:
            metric_card(label, value, caption, tone)

    divider()
    for column, (title, body) in zip(st.columns(2, gap="large"), [
        ("What category accuracy means", "This score shows how well the category classifier separates support ticket topics such as access, hardware, storage, and procurement on unseen data."),
        ("What priority accuracy means", "This score measures how well the urgency model identifies ticket priority so support leaders can handle SLA-sensitive requests sooner."),
    ]):
        with column:
            glass_card(title, body)

    divider()
    category_report = report.get("category_classification_report") or report.get("category_report") or {}
    priority_report = report.get("priority_classification_report") or report.get("priority_report") or {}
    tab_category, tab_priority, tab_raw = st.tabs(["Category Report", "Priority Report", "Raw Training Report"])
    with tab_category:
        df_report = report_table(category_report)
        if df_report.empty:
            st.info("Category classification report is not available.")
        else:
            dark_dataframe(df_report, height=360)
            st.caption("Precision, recall, F1-score, and support for each category.")
    with tab_priority:
        df_report = report_table(priority_report)
        if df_report.empty:
            st.info("Priority classification report is not available.")
        else:
            dark_dataframe(df_report, height=360)
            st.caption("Precision, recall, F1-score, and support for each priority class.")
    with tab_raw:
        with st.expander("Raw Training Report", expanded=False):
            st.json(report)


def render_dataset_explorer():
    page_hero(
        "Dataset Explorer",
        "Inspect the support ticket dataset, review sample rows, and explore category and priority distributions.",
        badge=status_badge("Data Operations", "blue"),
    )

    # Work with a local copy to avoid assigning to the global `dataset_info` variable
    ds = dataset_info

    # If no dataset was loaded, offer an upload fallback and keep a small sample available
    if not ds or ds.get("df") is None:
        st.warning("The dataset file was not found or could not be read. You can upload a CSV to explore your data.")
        uploaded = st.file_uploader("Upload dataset CSV", type=["csv"], help="Optional: upload a CSV with ticket text and optional category/priority columns.")
        if uploaded is not None:
            try:
                df_uploaded = pd.read_csv(uploaded)
                text_column = resolve_column(df_uploaded.columns, TEXT_COLUMN_CANDIDATES)
                category_column = resolve_column(df_uploaded.columns, CATEGORY_COLUMN_CANDIDATES)
                priority_column = resolve_column(df_uploaded.columns, PRIORITY_COLUMN_CANDIDATES)
                if text_column is not None:
                    df_uploaded[text_column] = df_uploaded[text_column].apply(normalize_text)
                if category_column is not None:
                    df_uploaded[category_column] = df_uploaded[category_column].apply(normalize_text)
                if priority_column is not None:
                    df_uploaded[priority_column] = df_uploaded[priority_column].apply(normalize_text)
                ds = {"df": df_uploaded, "text_column": text_column, "category_column": category_column, "priority_column": priority_column, "source": "uploaded"}
            except Exception as exc:
                st.error(f"Failed to read uploaded CSV: {exc}")
                return
        else:
            st.info("No dataset loaded — using a small sample to keep the explorer interactive.")
            if not ds:
                ds = load_dataset()

    df = ds["df"].copy()
    text_column = ds["text_column"]
    category_column = ds["category_column"]
    priority_column = ds["priority_column"]
    if text_column is None:
        st.error("No text column could be detected in the dataset.")
        return

    if "Priority" not in df.columns:
        df["Priority"] = build_priority_series(df, text_column, priority_column)
    else:
        df["Priority"] = df["Priority"].apply(normalize_text)

    for column, (label, value, caption, tone) in zip(st.columns(5, gap="small"), [
        ("Rows", f"{df.shape[0]:,}", "Total records", "blue"),
        ("Columns", f"{df.shape[1]:,}", "Detected fields", "cyan"),
        ("Text Column", text_column or "--", "Primary ticket text", "green"),
        ("Category Column", category_column or "--", "Issue label field", "amber"),
        ("Priority Source", normalize_text(priority_column) if priority_column else "generated", "Priority data source", "red"),
    ]):
        with column:
            metric_card(label, value, caption, tone)

    divider()
    section_header("Filters", "Use compact filters to narrow the dataset without cluttering the page.")
    filter_cols = st.columns([1, 1, 1.2], gap="medium")
    category_filter = []
    priority_filter = []
    search_term = ""

    with filter_cols[0]:
        if category_column is not None:
            category_options = sorted(df[category_column].dropna().astype(str).unique().tolist())
            category_filter = st.multiselect("Category", category_options, placeholder="Select one or more categories")
        else:
            st.info("No category column detected.")
    with filter_cols[1]:
        priority_options = sorted(df["Priority"].dropna().astype(str).unique().tolist())
        priority_filter = st.multiselect("Priority", priority_options, placeholder="Select one or more priorities")
    with filter_cols[2]:
        search_term = st.text_input("Search ticket text", placeholder="Type a keyword or phrase")

    filtered = df.copy()
    if category_column is not None and category_filter:
        filtered = filtered[filtered[category_column].isin(category_filter)]
    if priority_filter:
        filtered = filtered[filtered["Priority"].isin(priority_filter)]
    if search_term.strip():
        filtered = filtered[filtered[text_column].astype(str).str.contains(search_term.strip(), case=False, na=False)]

    for column, (label, value, caption, tone) in zip(st.columns(4, gap="small"), [
        ("Filtered Rows", f"{len(filtered):,}", "Matching current filters", "blue"),
        ("Visible Fields", f"{len(filtered.columns):,}", "Columns available in view", "cyan"),
        ("Category Groups", f"{filtered[category_column].nunique():,}" if category_column is not None else "--", "Unique categories in view", "green"),
        ("Priority Levels", f"{filtered['Priority'].nunique():,}", "Unique priorities in view", "amber"),
    ]):
        with column:
            metric_card(label, value, caption, tone)

    row_count = st.slider("Sample rows to display", min_value=5, max_value=min(25, max(len(filtered), 5)), value=min(10, max(len(filtered), 5)), step=1)
    section_header("Sample records", "A clean table view of the current subset of tickets.")
    dark_dataframe(filtered.head(row_count), height=320)

    divider()
    chart_cols = st.columns(2, gap="large")
    with chart_cols[0]:
        fig = make_bar_figure(filtered[category_column] if category_column is not None else None, "Category Distribution", top_n=12, horizontal=True)
        chart_card("Category Counts", fig)
    with chart_cols[1]:
        fig = make_bar_figure(filtered["Priority"], "Priority Distribution", horizontal=False)
        chart_card("Priority Counts", fig)


def render_about_project():
    page_hero("Ticket Desk", "A professional simulation of an enterprise support automation platform.", badge=status_badge("Product Case Study", "cyan"))
    divider()
    section_header("Problem Statement", "Support teams receive a constant stream of tickets that need to be categorized, prioritized, routed, and acknowledged quickly.")
    glass_card("Problem Statement", "Manual triage slows resolution, increases SLA risk, and makes support operations harder to scale. This project introduces a streamlined NLP workflow that gives support leaders a faster and more consistent operating model.")
    divider()
    section_header("Solution Overview", "The platform uses lightweight text classification plus business rules to deliver structured support guidance.")
    glass_card("Solution Overview", "Tickets are cleaned, vectorized with TF-IDF, classified into support categories, prioritized for SLA handling, routed to the appropriate team, and converted into a first-response draft for faster customer communication.")
    divider()
    section_header("ML Workflow", "A simple but realistic machine learning pipeline powers the support workflow.")
    for step, item in [
        ("01", "Clean and standardize support ticket text."),
        ("02", "Convert text into TF-IDF features."),
        ("03", "Train a LinearSVC model for category prediction."),
        ("04", "Train a Logistic Regression model for priority prediction."),
        ("05", "Use routing and SLA business rules for operational guidance."),
    ]:
        st.markdown(
            f"""
            <div class="card" style="margin-bottom:0.7rem; position:relative; overflow:hidden;">
                <div style="display:flex; gap:0.9rem; align-items:flex-start;">
                    <div class="badge badge-cyan" style="min-width:2.4rem; justify-content:center;">{step}</div>
                    <div>
                        <div class="card-title">{html.escape(item)}</div>
                        <div class="card-body">Each stage is intentionally lightweight so the project stays explainable and recruiter friendly.</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    divider()
    section_header("Tech Stack", "The implementation focuses on a practical Python and Streamlit stack.")
    stack_items = [
        ("Python", "blue"), ("Pandas", "neutral"), ("scikit-learn", "cyan"), ("TF-IDF", "blue"),
        ("LinearSVC", "green"), ("Logistic Regression", "amber"), ("Streamlit", "blue"), ("Plotly", "cyan"),
        ("Joblib", "neutral"), ("NLP", "green"),
    ]
    st.markdown("<div class='card'><div class='pill-row'>" + "".join(pill(name, tone) for name, tone in stack_items) + "</div></div>", unsafe_allow_html=True)
    divider()
    section_header("Business Impact", "What the system is designed to improve in a support environment.")
    for column, (title, body) in zip(st.columns(2, gap="large"), [
        ("Business Impact", "Faster ticket triage, better routing consistency, improved SLA tracking, and a more professional first-touch customer experience."),
        ("Internship Context", "This project began as Future Interns ML Task 2 and was upgraded into a recruiter-ready platform demo that showcases practical NLP, model training, and dashboard design."),
    ]):
        with column:
            glass_card(title, body)
    divider()
    section_header("Developer Profile", "A compact recruiter-facing profile block with direct links.")
    dev_skills = [
        ("Machine Learning", "green"),
        ("NLP", "cyan"),
        ("Data Analytics", "amber"),
        ("Streamlit", "blue"),
        ("scikit-learn", "cyan"),
        ("Full-Stack AI Applications", "neutral"),
        ("UI/UX", "amber"),
    ]
    profile_cols = st.columns([0.95, 1.4], gap="large")
    with profile_cols[0]:
        st.markdown(
            f"""
            <div class="card premium-card" style="height:100%; align-items:flex-start;">
                <div class="profile-avatar" aria-hidden="true"></div>
                    <div class="profile-meta">
                    <div class="name">A.J. Pardhiv</div>
                    <div class="title">AI & Data Science Student</div>
                    <div class="profile-bio">Google Certified Data Analyst building recruiter-ready Streamlit products with practical NLP, analytics, and interface polish.</div>
                    <div style="margin-top:0.65rem;">{status_badge('Google Certified Data Analyst', 'cyan')}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with profile_cols[1]:
        st.markdown("<div class='card'><div class='pill-row'>" + "".join(pill(name, tone) for name, tone in dev_skills) + "</div></div>", unsafe_allow_html=True)
        st.write("")
        button_cols = st.columns(2, gap="small")
        with button_cols[0]:
            st.markdown(
                "<a class='profile-link' href='https://github.com/itzPardhiv' target='_blank' rel='noopener' style='display:flex;justify-content:center;align-items:center;gap:0.5rem;padding:0.9rem 1rem;border-radius:14px;border:1px solid rgba(56,189,248,0.22);background:linear-gradient(135deg, rgba(56,189,248,0.16), rgba(34,211,238,0.10));color:#E0F2FE;text-decoration:none;font-weight:800;'>🐙 GitHub</a>",
                unsafe_allow_html=True,
            )
        with button_cols[1]:
            st.markdown(
                "<a class='profile-link' href='https://www.linkedin.com/in/aj-pardhiv-406a40333' target='_blank' rel='noopener' style='display:flex;justify-content:center;align-items:center;gap:0.5rem;padding:0.9rem 1rem;border-radius:14px;border:1px solid rgba(56,189,248,0.22);background:linear-gradient(135deg, rgba(139,92,246,0.16), rgba(56,189,248,0.10));color:#E0F2FE;text-decoration:none;font-weight:800;'>🔗 LinkedIn</a>",
                unsafe_allow_html=True,
            )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

artifacts, artifact_errors = load_artifacts()
report = load_report()
dataset_info = load_dataset()

if "last_analysis" not in st.session_state:
    st.session_state["last_analysis"] = None
if "ticket_input" not in st.session_state:
    st.session_state["ticket_input"] = next(iter(SAMPLE_TICKETS.values()))

inject_css()
selected_page = build_sidebar(artifacts)

if selected_page == "Dashboard Overview":
    render_overview(report, dataset_info)
elif selected_page == "Ticket Triage":
    render_ticket_triage(artifacts)
elif selected_page == "Model Performance":
    render_model_performance()
elif selected_page == "Dataset Explorer":
    render_dataset_explorer()
elif selected_page == "About Project":
    render_about_project()

if artifact_errors:
    with st.expander("Artifact loading notes", expanded=False):
        for error in artifact_errors:
            st.write(error)
