import json
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

PROJECT_NAME = "AI Ticket Desk"
PROJECT_TAGLINE = "Support Intelligence Platform"
PROJECT_SUBTITLE = "Enterprise NLP triage for support classification, priority prediction, SLA guidance, routing, and first-response drafting."

PAGES = [
    "Dashboard Overview",
    "Ticket Triage",
    "Model Performance",
    "Dataset Explorer",
    "About Project",
]

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

TEXT_COLUMN_CANDIDATES = ["Document", "Ticket", "Description", "Text", "Issue", "Summary"]
CATEGORY_COLUMN_CANDIDATES = ["Topic_group", "Category", "Label", "Department", "Class"]
PRIORITY_COLUMN_CANDIDATES = ["Priority"]

HIGH_PRIORITY_KEYWORDS = [
    "urgent",
    "critical",
    "down",
    "outage",
    "unable",
    "failed",
    "security",
    "production",
    "immediately",
]

MEDIUM_PRIORITY_KEYWORDS = [
    "issue",
    "error",
    "problem",
    "request",
    "access",
    "slow",
    "not working",
]

LOW_PRIORITY_KEYWORDS = [
    "information",
    "query",
    "general",
    "update",
    "change",
    "minor",
]

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

PLOTLY_THEME = "plotly_white"
PALETTE = ["#1f5eff", "#14b8a6", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4"]

st.set_page_config(
    page_title=f"{PROJECT_NAME} — {PROJECT_TAGLINE}",
    page_icon="📩",
    layout="wide",
    initial_sidebar_state="expanded",
)

px.defaults.template = PLOTLY_THEME

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #f8fafc 0%, #eef4ff 100%);
            color: #0f172a;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #111b33 100%);
        }
        [data-testid="stSidebar"] * {
            color: #e2e8f0 !important;
        }
        .hero-card {
            padding: 1.2rem 1.25rem;
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(31,94,255,0.08), rgba(20,184,166,0.06));
            border: 1px solid rgba(15,23,42,0.08);
            box-shadow: 0 12px 30px rgba(15,23,42,0.06);
        }
        .section-card {
            padding: 1rem 1.1rem;
            border-radius: 16px;
            background: rgba(255,255,255,0.75);
            border: 1px solid rgba(15,23,42,0.08);
            box-shadow: 0 10px 24px rgba(15,23,42,0.05);
        }
        .subtle {
            color: #475569;
        }
        .kpi-label {
            font-size: 0.82rem;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.2rem;
        }
        .kpi-value {
            font-size: 1.65rem;
            font-weight: 700;
            color: #0f172a;
            margin: 0;
        }
        .kpi-note {
            font-size: 0.85rem;
            color: #475569;
            margin-top: 0.2rem;
        }
        .stMetric {
            background: rgba(255,255,255,0.82);
            border: 1px solid rgba(15,23,42,0.08);
            padding: 0.75rem 0.9rem;
            border-radius: 14px;
            box-shadow: 0 8px 20px rgba(15,23,42,0.04);
        }
        footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


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

    encodings = ["utf-8-sig", "utf-8", "latin1"]
    last_error = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error

    if last_error is not None:
        raise last_error
    return None


def resolve_column(columns, candidates):
    column_lookup = {str(column).lower(): column for column in columns}
    for candidate in candidates:
        match = column_lookup.get(candidate.lower())
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
    if "misc" in normalized:
        return "General IT Support Desk"
    return "General IT Support Desk"


def get_urgency_message(priority):
    key = normalize_text(priority).lower()
    if key == "high":
        return (
            "This ticket should be treated as urgent and monitored continuously until the service impact is contained."
        )
    if key == "medium":
        return (
            "This ticket should be handled in the standard support queue with normal operational monitoring."
        )
    return (
        "This ticket appears non-critical and can follow the normal service cadence unless new impact details emerge."
    )


def generate_response_draft(ticket_text, category, priority, sla, team):
    summary = summarize_ticket(ticket_text, 220)
    return (
        "Hello,\n\n"
        f"Thank you for contacting the AI Ticket Desk. We have reviewed your request and classified it as {category}. "
        f"It has been routed to {team}. Current priority: {priority}. SLA guidance: {sla['guidance']} "
        f"Our team will continue reviewing the issue and provide the next update within the expected response window.\n\n"
        f"Ticket summary: {summary}\n\n"
        "Best regards,\n"
        "AI Ticket Desk Support Team"
    )


def summarize_ticket(text, limit=180):
    normalized = re.sub(r"\s+", " ", normalize_text(text)).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


@st.cache_resource(show_spinner=False)
def load_artifacts():
    artifacts = {
        "tfidf": None,
        "category_model": None,
        "priority_model": None,
    }
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
            missing.append(f"{path.name}: {exc}")
            artifacts[key] = None

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
    df = safe_read_csv(DATA_PATH)
    if df is None:
        return None

    text_column = resolve_column(df.columns, TEXT_COLUMN_CANDIDATES)
    category_column = resolve_column(df.columns, CATEGORY_COLUMN_CANDIDATES)
    priority_column = resolve_column(df.columns, PRIORITY_COLUMN_CANDIDATES)

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
    }


def classification_report_table(report_section):
    if not isinstance(report_section, dict) or not report_section:
        return pd.DataFrame()

    table = pd.DataFrame(report_section).T
    preferred_columns = ["precision", "recall", "f1-score", "support"]
    columns = [column for column in preferred_columns if column in table.columns]
    if columns:
        table = table[columns]

    for column in ["precision", "recall", "f1-score"]:
        if column in table.columns:
            table[column] = table[column].apply(lambda value: round(float(value) * 100, 2) if pd.notna(value) else value)
    if "support" in table.columns:
        table["support"] = table["support"].apply(lambda value: int(round(float(value))) if pd.notna(value) else value)

    return table


def metric_value(report, key, default="--"):
    value = report.get(key, default) if isinstance(report, dict) else default
    if value in [None, ""]:
        return default
    return value


def percent_text(value):
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return "--"


def load_sample_text(choice):
    return SAMPLE_TICKETS.get(choice, next(iter(SAMPLE_TICKETS.values())))


def build_priority_series(df, text_column, priority_column=None):
    if text_column is None:
        return pd.Series(dtype=str)

    if priority_column is not None and priority_column in df.columns:
        series = df[priority_column].copy()
        generated = df[text_column].apply(generate_priority_label)
        series = series.where(series.astype(str).str.strip().ne(""), generated)
        series = series.fillna(generated)
        return series.astype(str)

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
    response = generate_response_draft(ticket_text, predicted_category, predicted_priority, sla, team)

    return {
        "ticket_text": ticket_text,
        "predicted_category": predicted_category,
        "predicted_priority": predicted_priority,
        "sla": sla,
        "team": team,
        "urgency": urgency,
        "response": response,
    }


def render_page_header(title, subtitle):
    st.markdown(
        f"""
        <div class="hero-card">
            <div style="font-size:2rem; font-weight:800; color:#0f172a; line-height:1.15;">{title}</div>
            <div class="subtle" style="margin-top:0.35rem; font-size:1rem;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi(label, value, note):
    st.markdown(
        f"""
        <div class="section-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_distribution_chart(series, title, color_index=0):
    if series is None or series.empty:
        st.info(f"No data available for {title.lower()}.")
        return

    chart_df = series.value_counts().reset_index()
    chart_df.columns = ["Label", "Count"]
    fig = px.bar(
        chart_df,
        x="Label",
        y="Count",
        title=title,
        color="Label",
        color_discrete_sequence=PALETTE[color_index:] + PALETTE[:color_index],
    )
    fig.update_layout(showlegend=False, height=420, margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig, use_container_width=True)


def render_model_performance(report):
    report = report or {}
    category_report = report.get("category_classification_report") or report.get("category_report")
    priority_report = report.get("priority_classification_report") or report.get("priority_report")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Category Accuracy", percent_text(metric_value(report, "category_accuracy")))
    with c2:
        st.metric("Priority Accuracy", percent_text(metric_value(report, "priority_accuracy")))
    with c3:
        st.metric("TF-IDF Features", f"{metric_value(report, 'tfidf_feature_count')}")

    meta_cols = st.columns(4)
    with meta_cols[0]:
        st.metric("Dataset Rows", f"{metric_value(report, 'dataset_rows')}")
    with meta_cols[1]:
        st.metric("Text Column", metric_value(report, "text_column_used"))
    with meta_cols[2]:
        st.metric("Category Column", metric_value(report, "category_column_used"))
    with meta_cols[3]:
        st.metric("Priority Source", metric_value(report, "priority_source"))

    st.caption(f"Report generated: {metric_value(report, 'generated_timestamp')}")

    left, right = st.columns(2)
    with left:
        st.subheader("Category Classification Report")
        category_table = classification_report_table(category_report)
        if category_table.empty:
            st.info("Category classification report is not available.")
        else:
            st.dataframe(category_table, use_container_width=True)

    with right:
        st.subheader("Priority Classification Report")
        priority_table = classification_report_table(priority_report)
        if priority_table.empty:
            st.info("Priority classification report is not available.")
        else:
            st.dataframe(priority_table, use_container_width=True)


def render_dataset_explorer(dataset_info):
    if not dataset_info or dataset_info.get("df") is None:
        st.warning(
            "The dataset file was not found, but model-based ticket prediction can still work if the artifacts are available."
        )
        return

    df = dataset_info["df"].copy()
    text_column = dataset_info["text_column"]
    category_column = dataset_info["category_column"]
    priority_column = dataset_info["priority_column"]

    if text_column is None:
        st.error("No text column could be detected in the dataset.")
        return

    df["Priority"] = build_priority_series(df, text_column, priority_column)

    st.write(f"Dataset shape: **{df.shape[0]:,} rows x {df.shape[1]:,} columns**")
    st.write("Detected columns:")
    st.code(", ".join(df.columns.astype(str).tolist()), language="text")

    with st.expander("Filters", expanded=True):
        filter_cols = st.columns(3)
        category_filter = []
        priority_filter = []
        search_term = ""

        if category_column is not None:
            category_options = sorted(df[category_column].dropna().astype(str).unique().tolist())
            with filter_cols[0]:
                category_filter = st.multiselect("Category", category_options)
        else:
            with filter_cols[0]:
                st.info("No category column detected.")

        priority_options = sorted(df["Priority"].dropna().astype(str).unique().tolist())
        with filter_cols[1]:
            priority_filter = st.multiselect("Priority", priority_options)

        with filter_cols[2]:
            search_term = st.text_input("Search ticket text", placeholder="Type a keyword or phrase")

    filtered = df.copy()
    if category_column is not None and category_filter:
        filtered = filtered[filtered[category_column].isin(category_filter)]
    if priority_filter:
        filtered = filtered[filtered["Priority"].isin(priority_filter)]
    if search_term.strip():
        filtered = filtered[
            filtered[text_column].astype(str).str.contains(search_term.strip(), case=False, na=False)
        ]

    stats = st.columns(3)
    with stats[0]:
        st.metric("Filtered Rows", f"{len(filtered):,}")
    with stats[1]:
        st.metric("Columns", f"{df.shape[1]:,}")
    with stats[2]:
        st.metric("Visible Fields", f"{len(filtered.columns):,}")

    row_count = st.slider(
        "Sample rows to display",
        min_value=5,
        max_value=min(100, max(len(filtered), 5)),
        value=min(10, max(len(filtered), 5)),
        step=1,
    )
    st.dataframe(filtered.head(row_count), use_container_width=True)

    if category_column is not None:
        st.subheader("Category Counts")
        render_distribution_chart(filtered[category_column], "Category Distribution", color_index=0)

    st.subheader("Priority Counts")
    render_distribution_chart(filtered["Priority"], "Priority Distribution", color_index=2)


def render_about_page():
    st.markdown(
        """
        ### Problem Statement
        Support teams receive a constant stream of tickets that need to be categorized, prioritized, routed,
        and acknowledged quickly. Manual triage slows resolution, increases SLA risk, and makes support operations harder to scale.

        ### ML Workflow
        1. Clean and standardize support ticket text.
        2. Convert text into TF-IDF features.
        3. Train a LinearSVC model for category prediction.
        4. Train a Logistic Regression model for priority prediction.
        5. Use business rules to recommend SLA, routing team, and a first-response draft.

        ### Tech Stack
        - Python
        - Streamlit
        - pandas
        - scikit-learn
        - joblib
        - Plotly

        ### Business Impact
        - Faster ticket triage
        - Better routing consistency
        - Improved SLA tracking
        - Reduced manual effort for support analysts
        - More professional first-touch customer communication

        ### Internship Context
        This project began as Future Interns ML Internship Task 2 and was upgraded into a professional AI support platform for recruiter-ready presentation.
        """
    )


def render_overview(report, dataset_info):
    render_page_header(
        "AI Ticket Desk — Support Intelligence Platform",
        "Enterprise ticket intelligence for faster support triage, SLA guidance, and operational routing.",
    )

    if not report:
        st.warning(
            "Model report not found. Run `python scripts/train_models.py` to generate the trained artifacts and report file."
        )

    if dataset_info and dataset_info.get("df") is not None:
        df = dataset_info["df"]
        text_column = dataset_info["text_column"]
        category_column = dataset_info["category_column"]
        priority_column = dataset_info["priority_column"]
        priority_series = build_priority_series(df, text_column, priority_column)
    else:
        df = None
        text_column = None
        category_column = None
        priority_series = None

    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        render_kpi("Dataset Rows", f"{metric_value(report, 'dataset_rows') if report else (len(df) if df is not None else '--')}", "Rows used to train or inspect the dataset")
    with kpi_cols[1]:
        render_kpi("Category Accuracy", percent_text(metric_value(report, "category_accuracy")), "LinearSVC held-out test score")
    with kpi_cols[2]:
        render_kpi("Priority Accuracy", percent_text(metric_value(report, "priority_accuracy")), "Logistic Regression held-out test score")
    with kpi_cols[3]:
        render_kpi("TF-IDF Features", f"{metric_value(report, 'tfidf_feature_count')}", "Vocabulary size used for ticket vectorization")

    st.markdown(
        """
        <div class="section-card">
            <h4 style="margin-top:0; color:#0f172a;">Business Value</h4>
            <p class="subtle" style="margin-bottom:0;">
                AI Ticket Desk turns noisy support requests into structured operational decisions. It classifies tickets,
                predicts urgency, recommends SLA targets, and routes work to the correct team so support leaders can
                reduce manual triage and improve response consistency.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.subheader("Category Distribution")
        if df is not None and category_column is not None:
            render_distribution_chart(df[category_column], "Category Distribution", color_index=0)
        else:
            st.info("Dataset is not available for category analysis.")

    with chart_cols[1]:
        st.subheader("Priority Distribution")
        if priority_series is not None:
            render_distribution_chart(priority_series, "Priority Distribution", color_index=2)
        else:
            st.info("Dataset is not available for priority analysis.")


def render_ticket_triage(artifacts):
    render_page_header(
        "Ticket Triage",
        "Paste a support ticket, predict its category and priority, and generate the first response in seconds.",
    )

    can_predict = all(artifacts.get(key) is not None for key in ["tfidf", "category_model", "priority_model"])
    if not can_predict:
        st.warning(
            "Model artifacts are missing. Please run `python scripts/train_models.py` to regenerate the vectorizer and models."
        )

    if "ticket_input" not in st.session_state:
        st.session_state["ticket_input"] = next(iter(SAMPLE_TICKETS.values()))

    sample_cols = st.columns([2, 1])
    with sample_cols[0]:
        sample_choice = st.selectbox("Load a sample ticket", list(SAMPLE_TICKETS.keys()), index=0)
    with sample_cols[1]:
        st.write("")
        st.write("")
        if st.button("Load sample into editor", use_container_width=True):
            st.session_state["ticket_input"] = SAMPLE_TICKETS.get(sample_choice, st.session_state["ticket_input"])
            st.rerun()

    ticket_text = st.text_area(
        "Support ticket",
        key="ticket_input",
        height=230,
        placeholder="Paste a ticket about access, hardware, procurement, storage, or an internal project issue...",
    )

    analyze_clicked = st.button("Analyze Ticket", type="primary", use_container_width=True, disabled=not can_predict)

    if analyze_clicked:
        cleaned_text = normalize_text(ticket_text)
        if not cleaned_text:
            st.error("Please enter a support ticket before running the analysis.")
            return

        try:
            result = prediction_payload(cleaned_text, artifacts)
        except Exception as exc:
            st.error(f"Ticket analysis failed: {exc}")
            return

        top_cols = st.columns(4)
        with top_cols[0]:
            st.metric("Predicted Category", result["predicted_category"])
        with top_cols[1]:
            st.metric("Predicted Priority", result["predicted_priority"])
        with top_cols[2]:
            st.metric("Suggested SLA", result["sla"]["response_target"])
        with top_cols[3]:
            st.metric("Support Team", result["team"])

        insight_cols = st.columns([1.15, 0.85])
        with insight_cols[0]:
            st.markdown(
                f"""
                <div class="section-card">
                    <h4 style="margin-top:0; color:#0f172a;">Urgency Explanation</h4>
                    <p style="margin-bottom:0; color:#334155;">{result['urgency']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.subheader("First-Response Draft")
            st.text_area(
                "",
                value=result["response"],
                height=250,
                label_visibility="collapsed",
                disabled=True,
            )

        with insight_cols[1]:
            st.markdown(
                f"""
                <div class="section-card">
                    <h4 style="margin-top:0; color:#0f172a;">SLA Guidance</h4>
                    <p style="color:#334155; margin-bottom:0.4rem;">{result['sla']['guidance']}</p>
                    <p style="color:#475569; margin-bottom:0;">Response target: <strong>{result['sla']['response_target']}</strong><br/>
                    Resolution target: <strong>{result['sla']['resolution_target']}</strong></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class="section-card" style="margin-top:1rem;">
                    <h4 style="margin-top:0; color:#0f172a;">Routing Decision</h4>
                    <p style="color:#334155; margin-bottom:0.35rem;">Category: <strong>{result['predicted_category']}</strong></p>
                    <p style="color:#334155; margin-bottom:0.35rem;">Team: <strong>{result['team']}</strong></p>
                    <p style="color:#334155; margin-bottom:0;">Priority: <strong>{result['predicted_priority']}</strong></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.caption("Use the sample loader above or paste a custom ticket, then click Analyze Ticket.")


def render_model_page(report):
    render_page_header(
        "Model Performance",
        "Review the saved training results, classification quality, and the TF-IDF configuration used by the platform.",
    )

    if not report:
        st.warning(
            "Model report not found. Run `python scripts/train_models.py` to generate `models/model_report.json`."
        )
        return

    render_model_performance(report)


def render_dataset_page(dataset_info):
    render_page_header(
        "Dataset Explorer",
        "Inspect the support ticket dataset, view sample records, and review category or priority distributions.",
    )

    render_dataset_explorer(dataset_info)


def render_about_page_full():
    render_page_header(
        "About Project",
        "A recruiter-ready presentation of the Future Interns ML Task 2 project, upgraded into a professional AI support platform.",
    )
    render_about_page()


# App state
artifacts, artifact_errors = load_artifacts()
report = load_report()
dataset_info = load_dataset()

# Sidebar
with st.sidebar:
    st.markdown(
        f"""
        <div style="padding:1rem 0 0.4rem 0;">
            <div style="font-size:1.5rem; font-weight:800; color:#ffffff;">{PROJECT_NAME}</div>
            <div style="font-size:0.95rem; color:#cbd5e1; margin-top:0.25rem;">{PROJECT_TAGLINE}</div>
            <div style="font-size:0.82rem; color:#94a3b8; margin-top:0.5rem; line-height:1.5;">{PROJECT_SUBTITLE}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if all(artifacts.values()):
        st.success("Model artifacts ready")
    else:
        st.warning("Some model artifacts are missing")
    selected_page = st.radio("Navigation", PAGES, index=0, label_visibility="collapsed")
    st.divider()
    st.caption("If artifacts are missing, run: python scripts/train_models.py")


if artifact_errors:
    st.warning(
        "Some model files could not be loaded. Ticket prediction will be disabled until the artifacts are restored.\n\n"
        + "\n".join(f"- {error}" for error in artifact_errors)
    )

if selected_page == "Dashboard Overview":
    render_overview(report, dataset_info)
elif selected_page == "Ticket Triage":
    render_ticket_triage(artifacts)
elif selected_page == "Model Performance":
    render_model_page(report)
elif selected_page == "Dataset Explorer":
    render_dataset_page(dataset_info)
elif selected_page == "About Project":
    render_about_page_full()
