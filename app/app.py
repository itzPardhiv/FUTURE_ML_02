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
PROJECT_DESCRIPTION = (
    "Intelligent triage, SLA guidance, and support workflow automation."
)

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


# -----------------------------------------------------------------------------
# Data and model utilities
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
        return "This ticket should be treated as urgent and monitored continuously until the service impact is contained."
    if key == "medium":
        return "This ticket should be handled in the standard support queue with normal operational monitoring."
    return "This ticket appears non-critical and can follow the normal service cadence unless new impact details emerge."


def summarize_ticket(text, limit=180):
    normalized = re.sub(r"\s+", " ", normalize_text(text)).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


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


# -----------------------------------------------------------------------------
# UI helpers
# -----------------------------------------------------------------------------

TONE_MAP = {
    "blue": "tone-blue",
    "cyan": "tone-cyan",
    "green": "tone-green",
    "amber": "tone-amber",
    "red": "tone-red",
    "neutral": "tone-neutral",
}

def tone_class(tone):
    return TONE_MAP.get(str(tone).lower(), "tone-blue")


def pill(text, tone="blue"):
    return f'<span class="pill {tone_class(tone)}">{text}</span>'


def priority_tone(priority):
    key = normalize_text(priority).lower()
    if key == "high":
        return "red"
    if key == "medium":
        return "amber"
    if key == "low":
        return "green"
    return "blue"


def page_hero(title, subtitle, badge=None):
    badge_html = f'<div class="badge tone-blue">{badge}</div>' if badge else ""
    st.markdown(
        f"""
        <div class="hero-card hero-accent">
            <div class="hero-grid">
                <div>
                    {badge_html}
                    <div class="hero-title">{title}</div>
                    <div class="hero-subtitle">{subtitle}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label, value, caption="", tone="blue"):
    accent_class = tone_class(tone)
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def card(title, body):
    st.markdown(
        f"""
        <div class="content-card card-shell">
            <div class="card-title">{title}</div>
            <div class="card-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def result_tile(label, value, caption="", tone="blue"):
    value_html = value
    st.markdown(
        f"""
        <div class="result-tile">
            <div class="result-label">{label}</div>
            <div class="result-value">{value_html}</div>
            <div class="result-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chart_figure(series, title, color_index=0):
    if series is None or series.empty:
        return None

    chart_df = series.value_counts().reset_index()
    chart_df.columns = ["Label", "Count"]
    # Build an ordered color list from PALETTE (supports dict or list)
    if isinstance(PALETTE, dict):
        color_keys = ["blue", "cyan", "green", "amber", "red", "purple"]
        palette_list = [PALETTE[k] for k in color_keys if k in PALETTE]
    else:
        palette_list = list(PALETTE)

    if not palette_list:
        palette_list = ["#2563EB", "#0891B2", "#16A34A", "#F59E0B", "#DC2626"]

    color_seq = palette_list[color_index:] + palette_list[:color_index]

    fig = px.bar(
        chart_df,
        x="Label",
        y="Count",
        color="Label",
        title=title,
        color_discrete_sequence=color_seq,
    )

    # Use dark-friendly defaults for layout when the app is in dark theme
    fig.update_layout(
        showlegend=False,
        height=380,
        margin=dict(l=8, r=8, t=40, b=8),
        paper_bgcolor=PALETTE.get("card", "#071527") if isinstance(PALETTE, dict) else "#071527",
        plot_bgcolor=PALETTE.get("card", "#071527") if isinstance(PALETTE, dict) else "#071527",
        title_font=dict(size=18, color=PALETTE.get("text", "#E6EEF8") if isinstance(PALETTE, dict) else "#E6EEF8"),
        font=dict(color=PALETTE.get("muted", "#94A3B8") if isinstance(PALETTE, dict) else "#94A3B8", size=12),
        xaxis=dict(title="", tickfont=dict(color=PALETTE.get("muted", "#94A3B8") if isinstance(PALETTE, dict) else "#94A3B8"), gridcolor="rgba(255,255,255,0.03)", zeroline=False),
        yaxis=dict(title="", tickfont=dict(color=PALETTE.get("muted", "#94A3B8") if isinstance(PALETTE, dict) else "#94A3B8"), gridcolor="rgba(255,255,255,0.03)", zeroline=False),
    )
    fig.update_traces(marker_line_width=0, opacity=0.95)
    return fig


def chart_card(title, fig):
    st.markdown(
        f"""
        <div class="chart-card">
            <div class="card-title">{title}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if fig is None:
        st.info("No data available.")
    else:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})


def report_table(report_dict):
    if not isinstance(report_dict, dict) or not report_dict:
        return pd.DataFrame()

    table = pd.DataFrame(report_dict).T
    preferred_columns = ["precision", "recall", "f1-score", "support"]
    visible_columns = [column for column in preferred_columns if column in table.columns]
    if visible_columns:
        table = table[visible_columns]

    for column in ["precision", "recall", "f1-score"]:
        if column in table.columns:
            table[column] = table[column].apply(
                lambda value: round(float(value) * 100, 2) if pd.notna(value) else value
            )
    if "support" in table.columns:
        table["support"] = table["support"].apply(
            lambda value: int(round(float(value))) if pd.notna(value) else value
        )
    return table


def empty_state(title, subtitle):
    st.markdown(
        f"""
        <div class="empty-card">
            <div class="badge tone-neutral">AI Triage</div>
            <div class="empty-title">{title}</div>
            <div class="empty-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title, subtitle):
    st.markdown(
        f"""
        <div class="section-head">
            <div class="section-title">{title}</div>
            <div class="section-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def divider():
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


def safe_metric_text(value, fallback="Unavailable"):
    if value in [None, ""]:
        return fallback
    return str(value)


def metric_value(report, key, fallback="--"):
    if not isinstance(report, dict):
        return fallback
    value = report.get(key, fallback)
    if value in [None, ""]:
        return fallback
    return value


def format_percent(value):
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return "--"


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

def build_sidebar(artifacts):
    with st.sidebar:
        st.markdown(
            f"""
            <div class="sidebar-shell">
                <div class="brand-block">
                    <div class="brand-name">{PROJECT_NAME}</div>
                    <div class="brand-subtitle">{PROJECT_TAGLINE}</div>
                    <div style="margin-top:0.55rem;">{pill('NLP Automation', 'blue')}</div>
                    <div class="brand-desc">{PROJECT_DESCRIPTION}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-nav-title">Navigation</div>', unsafe_allow_html=True)
        selected_page = st.radio(
            "Navigation",
            PAGES,
            index=0,
            format_func=lambda page: PAGE_LABELS.get(page, page),
            label_visibility="collapsed",
        )

        if all(artifacts.values()):
            st.markdown(
                f"<div style='margin-top:0.9rem;'>{pill('Models Ready', 'green')}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='margin-top:0.9rem;'>{pill('Training Required', 'amber')}</div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div class="sidebar-note">
                Future Interns ML Task 2<br>
                Built with Python · scikit-learn · Streamlit
            </div>
            """,
            unsafe_allow_html=True,
        )

    return selected_page


# -----------------------------------------------------------------------------
# Page renderers
# -----------------------------------------------------------------------------

def render_overview(report, dataset_info):
    dataset_rows = metric_value(report, "dataset_rows") if report else None
    if dataset_info and dataset_info.get("df") is not None:
        df = dataset_info["df"]
        text_column = dataset_info["text_column"]
        category_column = dataset_info["category_column"]
        priority_column = dataset_info["priority_column"]
        priority_series = build_priority_series(df, text_column, priority_column)
        if dataset_rows in [None, "--"]:
            dataset_rows = len(df)
    else:
        df = None
        category_column = None
        priority_series = None

    page_hero(
        "Support Intelligence Dashboard",
        "Monitor ticket distribution, model performance, and AI triage readiness from one workspace.",
        badge=pill("NLP Automation System", "cyan"),
    )

    if report:
        dataset_badge = f"{dataset_rows:,} Tickets" if isinstance(dataset_rows, int) else safe_metric_text(dataset_rows, "Unavailable")
        st.markdown(
            f"""
            <div class="section-head" style="margin-top:0.8rem; margin-bottom:0.75rem;">
                <div class="status-line">
                    {pill('Dataset Ready', 'green') if df is not None else pill('Dataset Unavailable', 'amber')}
                    {pill('Models Ready', 'green') if report else pill('Training Required', 'amber')}
                    {pill(dataset_badge, 'neutral')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    divider()

    metrics = [
        (
            "Tickets Analyzed",
            f"{dataset_rows:,}" if isinstance(dataset_rows, int) else safe_metric_text(dataset_rows),
            "Rows available for training or review",
            "blue",
        ),
        (
            "Category Accuracy",
            format_percent(metric_value(report, "category_accuracy")),
            "LinearSVC held-out score",
            "cyan",
        ),
        (
            "Priority Accuracy",
            format_percent(metric_value(report, "priority_accuracy")),
            "Logistic Regression held-out score",
            "green",
        ),
        (
            "TF-IDF Features",
            safe_metric_text(metric_value(report, "tfidf_feature_count")),
            "Vocabulary used for vectorization",
            "amber",
        ),
    ]

    metric_cols = st.columns(4, gap="small")
    for column, (label, value, caption, tone) in zip(metric_cols, metrics):
        with column:
            metric_card(label, value, caption, tone)

    divider()

    business_cols = st.columns(3, gap="medium")
    business_cards = [
        (
            "Faster Triage",
            "Reduce manual handling time by surfacing the likely support category, urgency, and routing path instantly.",
            "01",
        ),
        (
            "SLA Awareness",
            "Highlight response targets early so support teams can prioritize time-sensitive tickets before they drift.",
            "02",
        ),
        (
            "Smarter Routing",
            "Direct tickets to the right support queue with a consistent business-rule layer on top of model predictions.",
            "03",
        ),
    ]
    for column, (title, body, icon) in zip(business_cols, business_cards):
        with column:
            st.markdown(
                f"""
                <div class="content-card business-card">
                    <div class="business-icon">{icon}</div>
                    <div class="card-title">{title}</div>
                    <div class="card-body">{body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    divider()

    chart_cols = st.columns(2, gap="large")
    with chart_cols[0]:
        fig = chart_figure(df[category_column], "Category Distribution", color_index=0) if df is not None and category_column is not None else None
        chart_card("Category Distribution", fig)
    with chart_cols[1]:
        fig = chart_figure(priority_series, "Priority Distribution", color_index=2) if priority_series is not None else None
        chart_card("Priority Distribution", fig)


def render_ticket_triage(artifacts):
    page_hero(
        "Ticket Triage",
        "Paste a support request, run the analysis, and turn messy ticket text into a structured AI triage report.",
        badge=pill("Real-time NLP", "cyan"),
    )

    can_predict = all(artifacts.get(key) is not None for key in ["tfidf", "category_model", "priority_model"])
    if not can_predict:
        st.warning("Model artifacts are missing. Please run `python scripts/train_models.py` to regenerate the vectorizer and models.")

    if "ticket_input" not in st.session_state:
        st.session_state["ticket_input"] = next(iter(SAMPLE_TICKETS.values()))
    if "last_analysis" not in st.session_state:
        st.session_state["last_analysis"] = None

    divider()

    left_col, right_col = st.columns([1.08, 1], gap="large")

    with left_col:
        st.markdown(
            """
            <div class="content-card card-shell">
                <div class="card-title">New Support Ticket</div>
                <div class="card-body">Add a ticket description, use a sample for quick testing, and generate the triage output.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")

        sample_choice = st.selectbox("Example tickets", list(SAMPLE_TICKETS.keys()), index=0)
        if st.button("Load example ticket", use_container_width=True):
            st.session_state["ticket_input"] = SAMPLE_TICKETS[sample_choice]
            st.session_state["last_analysis"] = None
            st.rerun()

        ticket_text = st.text_area(
            "Ticket description",
            key="ticket_input",
            height=210,
            placeholder="Describe the support issue, incident, request, or service problem...",
            label_visibility="visible",
        )

        analyze_clicked = st.button("Analyze Ticket", type="primary", use_container_width=True, disabled=not can_predict)
        st.caption("Tip: Keep the ticket concise but include the user impact, system affected, and urgency details.")

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
                "AI triage report will appear here",
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
                            <div class="analysis-title">AI Triage Report</div>
                            <div class="analysis-subtitle">Completed analysis for the submitted support ticket.</div>
                        </div>
                        <div class="status-line">
                            {pill('Completed', 'green')}
                            {pill(priority_value, priority_style)}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            result_cols = st.columns(2, gap="small")
            tiles = [
                ("Category", result["predicted_category"], "Predicted issue type", "blue"),
                ("Priority", result["predicted_priority"], "Urgency classification", priority_style),
                ("SLA Target", result["sla"]["response_target"], "Suggested response window", "cyan"),
                ("Routing Team", result["team"], "Suggested support queue", "neutral"),
            ]
            for index, (label, value, caption, tone) in enumerate(tiles):
                with result_cols[index % 2]:
                    result_tile(label, value, caption, tone)
                if index % 2 == 1 and index < len(tiles) - 1:
                    st.write("")

            divider()

            analysis_cols = st.columns([1, 1], gap="medium")
            with analysis_cols[0]:
                st.markdown(
                    f"""
                    <div class="info-card">
                        <div class="card-title">Urgency Explanation</div>
                        <div class="card-body">{result['urgency']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.write("")
                st.markdown(
                    f"""
                    <div class="info-card">
                        <div class="card-title">Routing Decision</div>
                        <div class="card-body">
                            <div class="status-line" style="margin-bottom:0.45rem;">{pill(result['predicted_category'], 'blue')}</div>
                            <div class="status-line" style="margin-bottom:0.45rem;">{pill(result['team'], 'neutral')}</div>
                            <div class="status-line">{pill(result['predicted_priority'], priority_style)}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with analysis_cols[1]:
                subject_line = f"Subject: {result['predicted_priority']} support ticket routed to {result['team']}"
                body_text = result["response"].split("\n\n")
                email_body = "\n\n".join(body_text)
                email_html = html.escape(subject_line + "\n\n" + email_body).replace("\n", "<br>")
                st.markdown(
                    f"""
                    <div class="email-card">
                        <div class="card-title">First-Response Draft</div>
                        <div class="card-body">Email preview</div>
                        <div class="email-preview">{email_html}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_model_page(report):
    page_hero(
        "Model Performance Center",
        "Track accuracy, dataset coverage, and the training report that powers the support automation workflow.",
        badge=pill("Monitoring", "cyan"),
    )

    if not report:
        st.warning("Model report not found. Run `python scripts/train_models.py` to generate `models/model_report.json`.")
        return

    divider()

    model_metrics = [
        ("Category Accuracy", format_percent(metric_value(report, "category_accuracy")), "Held-out score for ticket category prediction", "blue"),
        ("Priority Accuracy", format_percent(metric_value(report, "priority_accuracy")), "Held-out score for ticket urgency prediction", "cyan"),
        ("Feature Count", safe_metric_text(metric_value(report, "tfidf_feature_count")), "TF-IDF vocabulary size", "green"),
        ("Dataset Rows", safe_metric_text(metric_value(report, "dataset_rows")), "Rows used during training", "amber"),
    ]
    cols = st.columns(4, gap="small")
    for column, spec in zip(cols, model_metrics):
        with column:
            metric_card(*spec)

    divider()

    explanation_cols = st.columns(2, gap="medium")
    with explanation_cols[0]:
        card(
            "What category accuracy means",
            "This metric shows how often the model predicts the correct support category on unseen tickets. Higher values indicate better separation between issue types such as hardware, access, procurement, or storage.",
        )
    with explanation_cols[1]:
        card(
            "What priority accuracy means",
            "This metric measures whether the model correctly identifies ticket urgency so the support team can set the right SLA target and route time-sensitive requests faster.",
        )

    divider()

    category_report = report.get("category_classification_report") or report.get("category_report") or {}
    priority_report = report.get("priority_classification_report") or report.get("priority_report") or {}

    tab_category, tab_priority, tab_raw = st.tabs(["Category Report", "Priority Report", "Raw Training Report"])

    with tab_category:
        df_report = report_table(category_report)
        if df_report.empty:
            st.info("Category classification report is not available.")
        else:
            st.dataframe(df_report, use_container_width=True, height=360)
            st.caption("Precision, recall, F1-score, and support for each category.")

    with tab_priority:
        df_report = report_table(priority_report)
        if df_report.empty:
            st.info("Priority classification report is not available.")
        else:
            st.dataframe(df_report, use_container_width=True, height=360)
            st.caption("Precision, recall, F1-score, and support for each priority class.")

    with tab_raw:
        with st.expander("Raw Training Report", expanded=False):
            st.json(report)


def render_dataset_page(dataset_info):
    page_hero(
        "Dataset Explorer",
        "Explore the support ticket data with compact filters, summary metrics, and clean distribution charts.",
        badge=pill("Data Operations", "blue"),
    )

    if not dataset_info or dataset_info.get("df") is None:
        st.warning("The dataset file was not found, but model-based ticket prediction can still work if the artifacts are available.")
        return

    df = dataset_info["df"].copy()
    text_column = dataset_info["text_column"]
    category_column = dataset_info["category_column"]
    priority_column = dataset_info["priority_column"]

    if text_column is None:
        st.error("No text column could be detected in the dataset.")
        return

    if "Priority" not in df.columns:
        df["Priority"] = build_priority_series(df, text_column, priority_column)
    else:
        df["Priority"] = df["Priority"].apply(normalize_text)

    divider()

    summary_metrics = [
        ("Rows", f"{df.shape[0]:,}", "Total records in the dataset", "blue"),
        ("Columns", f"{df.shape[1]:,}", "Detected fields available", "cyan"),
        ("Text Column", safe_metric_text(text_column), "Primary ticket text field", "green"),
        ("Category Column", safe_metric_text(category_column), "Issue grouping field", "amber"),
        ("Priority Source", safe_metric_text(priority_column or "generated"), "Priority labels source", "red"),
    ]
    metric_cols = st.columns(5, gap="small")
    for column, spec in zip(metric_cols, summary_metrics):
        with column:
            metric_card(*spec)

    divider()

    section_header("Filters", "Narrow the dataset by category, priority, or text keyword without clutter.")
    filter_cols = st.columns([1, 1, 1.2], gap="medium")
    category_filter = []
    priority_filter = []
    search_term = ""

    if category_column is not None:
        category_options = sorted(df[category_column].dropna().astype(str).unique().tolist())
        with filter_cols[0]:
            category_filter = st.multiselect("Category", category_options, placeholder="Select categories")
    else:
        with filter_cols[0]:
            st.info("No category column detected.")

    priority_options = sorted(df["Priority"].dropna().astype(str).unique().tolist())
    with filter_cols[1]:
        priority_filter = st.multiselect("Priority", priority_options, placeholder="Select priorities")

    with filter_cols[2]:
        search_term = st.text_input("Search ticket text", placeholder="Type a keyword or phrase")

    filtered = df.copy()
    if category_column is not None and category_filter:
        filtered = filtered[filtered[category_column].isin(category_filter)]
    if priority_filter:
        filtered = filtered[filtered["Priority"].isin(priority_filter)]
    if search_term.strip():
        filtered = filtered[filtered[text_column].astype(str).str.contains(search_term.strip(), case=False, na=False)]

    divider()

    summary_cols = st.columns(4, gap="small")
    dataset_summary = [
        ("Filtered Rows", f"{len(filtered):,}", "Matching current filters", "blue"),
        ("Visible Fields", f"{len(filtered.columns):,}", "Columns available in view", "cyan"),
        ("Category Groups", f"{filtered[category_column].nunique():,}" if category_column is not None else "--", "Unique categories in view", "green"),
        ("Priority Levels", f"{filtered['Priority'].nunique():,}", "Unique priorities in view", "amber"),
    ]
    for column, spec in zip(summary_cols, dataset_summary):
        with column:
            metric_card(*spec)

    divider()

    row_count = st.slider(
        "Sample rows to display",
        min_value=5,
        max_value=min(25, max(len(filtered), 5)),
        value=min(10, max(len(filtered), 5)),
        step=1,
    )

    section_header("Sample records", "A clean table view of the current subset of tickets.")
    st.dataframe(filtered.head(row_count), use_container_width=True, height=320)

    divider()

    chart_cols = st.columns(2, gap="large")
    with chart_cols[0]:
        fig = chart_figure(filtered[category_column], "Category Distribution", color_index=0) if category_column is not None else None
        chart_card("Category Counts", fig)
    with chart_cols[1]:
        fig = chart_figure(filtered["Priority"], "Priority Distribution", color_index=2)
        chart_card("Priority Counts", fig)


def render_about_page():
    page_hero(
        PROJECT_NAME,
        "A professional simulation of an enterprise support automation platform.",
        badge=pill("Case Study", "cyan"),
    )

    divider()

    card(
        "Problem Statement",
        "Support teams receive a constant stream of unstructured tickets that must be categorized, prioritized, routed, and acknowledged quickly. Manual triage slows resolution, increases SLA risk, and makes support operations harder to scale.",
    )
    st.write("")

    card(
        "Solution Overview",
        "AI Ticket Desk uses a TF-IDF text pipeline plus lightweight classifiers to predict the most likely ticket category and urgency, then combines that output with support-routing rules and response drafting for a polished triage workflow.",
    )
    st.write("")

    card(
        "ML Workflow",
        "Clean the ticket text, vectorize it with TF-IDF, train a LinearSVC category model and a Logistic Regression priority model, evaluate the outputs, then load the saved artifacts inside the dashboard for interactive analysis.",
    )
    st.write("")

    card(
        "Tech Stack",
        "Python, Pandas, scikit-learn, TF-IDF, LinearSVC, Logistic Regression, Streamlit, Plotly, Joblib, NLP",
    )
    st.write("")

    card(
        "Business Impact",
        "This project demonstrates faster triage, better routing consistency, improved SLA awareness, and more professional first-touch support communication. It is positioned as a recruiter-ready product demo rather than a production deployment claim.",
    )
    st.write("")

    card(
        "Internship Context",
        "The project began as Future Interns ML Task 2 and was upgraded into a premium support-intelligence experience to show practical NLP, dashboard design, and product-thinking skills.",
    )

    st.write("")
    st.markdown(
        "<div class='content-card card-shell'><div class='card-title'>Tech badges</div><div class='status-line'>"
        + " ".join(
            [
                pill("Python", "blue"),
                pill("Pandas", "neutral"),
                pill("scikit-learn", "blue"),
                pill("TF-IDF", "cyan"),
                pill("LinearSVC", "neutral"),
                pill("Logistic Regression", "neutral"),
                pill("Streamlit", "blue"),
                pill("Plotly", "cyan"),
                pill("Joblib", "neutral"),
                pill("NLP", "green"),
            ]
        )
        + "</div></div>",
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Main app entrypoint is defined at the end of the file.
# -----------------------------------------------------------------------------
import html
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
PROJECT_SUBTITLE = (
    "Intelligent triage, SLA guidance, and support workflow automation."
)

PAGE_ORDER = [
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

MEDIUM_PRIORITY_KEYWORDS = ["issue", "error", "problem", "request", "access", "slow", "not working"]
LOW_PRIORITY_KEYWORDS = ["information", "query", "general", "update", "change", "minor"]

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
    "blue": "#2563EB",
    "cyan": "#0891B2",
    "green": "#16A34A",
    "amber": "#F59E0B",
    "red": "#DC2626",
    "text": "#0F172A",
    "muted": "#64748B",
    "border": "#E2E8F0",
    "bg": "#F6F8FC",
    "card": "#FFFFFF",
    "sidebar": "#07111F",
}

st.set_page_config(
    page_title=f"{PROJECT_NAME} — {PROJECT_TAGLINE}",
    page_icon="📩",
    layout="wide",
    initial_sidebar_state="expanded",
)

px.defaults.template = "plotly_dark"


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
                --border: {PALETTE['border']};
                --blue: {PALETTE['blue']};
                --cyan: {PALETTE['cyan']};
                --green: {PALETTE['green']};
                --amber: {PALETTE['amber']};
                --red: {PALETTE['red']};
                --radius: 16px;
                --shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
            }}

            html, body, [class*="css"] {{
                font-family: Inter, "Segoe UI", Arial, sans-serif;
            }}

            .stApp {{
                background: var(--bg);
                color: var(--text);
            }}

            header[data-testid="stHeader"], #MainMenu, footer {{
                visibility: hidden;
                height: 0;
            }}

            .block-container {{
                max-width: 1240px;
                padding-top: 1rem;
                padding-bottom: 2rem;
            }}

            [data-testid="stSidebar"] {{
                background: linear-gradient(180deg, #05111D 0%, var(--sidebar) 100%);
                border-right: 1px solid rgba(255,255,255,0.06);
            }}

            [data-testid="stSidebar"] * {{
                color: #E2E8F0 !important;
            }}

            [data-testid="stSidebar"] .sidebar-wrap {{
                padding: 1.1rem 1rem 1rem;
            }}

            .sidebar-title {{
                font-size: 1.45rem;
                font-weight: 850;
                letter-spacing: -0.04em;
                line-height: 1.05;
                margin: 0;
                color: #FFFFFF;
            }}

            .sidebar-subtitle {{
                margin-top: 0.22rem;
                font-size: 0.95rem;
                font-weight: 650;
                color: #BFDBFE;
            }}

            .sidebar-desc {{
                margin-top: 0.7rem;
                font-size: 0.86rem;
                line-height: 1.55;
                color: #94A3B8;
            }}

            .sidebar-footer {{
                margin-top: 1rem;
                padding-top: 0.9rem;
                border-top: 1px solid rgba(255,255,255,0.08);
                color: #94A3B8;
                font-size: 0.82rem;
                line-height: 1.5;
            }}

            .page-shell {{
                margin-top: 0.2rem;
            }}

            .hero-card,
            .metric-card,
            .card,
            .chart-card,
            .result-card,
            .info-card,
            .empty-state,
            .email-card,
            .analysis-card {{
                background: var(--card);
                border: 1px solid var(--border);
                border-radius: var(--radius);
                box-shadow: var(--shadow);
            }}

            .hero-card {{
                padding: 1.25rem 1.35rem;
                background: linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(248,251,255,1) 100%);
                position: relative;
                overflow: hidden;
            }}

            .hero-card::before {{
                content: "";
                position: absolute;
                inset: 0 auto auto 0;
                width: 100%;
                height: 3px;
                background: linear-gradient(90deg, var(--blue) 0%, var(--cyan) 100%);
            }}

            .hero-grid {{
                display: flex;
                gap: 1rem;
                justify-content: space-between;
                align-items: stretch;
            }}

            .hero-copy {{
                flex: 1 1 auto;
                min-width: 0;
            }}

            .hero-meta {{
                min-width: 260px;
                display: flex;
                flex-direction: column;
                gap: 0.5rem;
                align-items: flex-end;
                justify-content: center;
            }}

            .hero-title {{
                font-size: 2rem;
                line-height: 1.08;
                font-weight: 850;
                letter-spacing: -0.05em;
                color: var(--text);
                margin: 0;
            }}

            .hero-subtitle {{
                margin-top: 0.55rem;
                font-size: 1rem;
                line-height: 1.6;
                color: var(--muted);
                max-width: 78ch;
            }}

            .badge {{
                display: inline-flex;
                align-items: center;
                gap: 0.35rem;
                border-radius: 999px;
                padding: 0.38rem 0.75rem;
                font-size: 0.76rem;
                font-weight: 800;
                line-height: 1;
                border: 1px solid transparent;
                white-space: nowrap;
            }}

            .badge-blue {{ background: rgba(37,99,235,0.10); color: #1D4ED8; border-color: rgba(37,99,235,0.16); }}
            .badge-cyan {{ background: rgba(8,145,178,0.10); color: #0E7490; border-color: rgba(8,145,178,0.16); }}
            .badge-green {{ background: rgba(22,163,74,0.12); color: #15803D; border-color: rgba(22,163,74,0.18); }}
            .badge-amber {{ background: rgba(245,158,11,0.14); color: #B45309; border-color: rgba(245,158,11,0.18); }}
            .badge-red {{ background: rgba(220,38,38,0.12); color: #B91C1C; border-color: rgba(220,38,38,0.18); }}
            .badge-neutral {{ background: rgba(100,116,139,0.10); color: #475569; border-color: rgba(100,116,139,0.16); }}

            .section-title {{
                font-size: 1.35rem;
                line-height: 1.18;
                font-weight: 850;
                letter-spacing: -0.04em;
                color: var(--text);
                margin: 0;
            }}

            .section-subtitle {{
                margin-top: 0.3rem;
                color: var(--muted);
                font-size: 0.93rem;
                line-height: 1.55;
            }}

            .section-wrap {{
                margin-top: 1rem;
            }}

            .metric-card {{
                padding: 0.9rem 0.95rem 0.85rem;
                min-height: 112px;
                position: relative;
                overflow: hidden;
            }}

            .metric-card::before {{
                content: "";
                position: absolute;
                left: 0;
                top: 0;
                height: 3px;
                width: 100%;
                background: linear-gradient(90deg, rgba(37,99,235,0.9), rgba(8,145,178,0.9));
            }}

            .metric-label {{
                margin: 0 0 0.45rem;
                font-size: 0.76rem;
                font-weight: 850;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: var(--muted);
            }}

            .metric-value {{
                font-size: 1.65rem;
                line-height: 1.02;
                font-weight: 860;
                letter-spacing: -0.04em;
                color: var(--text);
                margin: 0;
            }}

            .metric-caption {{
                margin-top: 0.4rem;
                font-size: 0.82rem;
                line-height: 1.45;
                color: var(--muted);
            }}

            .card,
            .chart-card,
            .info-card,
            .empty-state,
            .analysis-card {{
                padding: 1rem 1.02rem;
            }}

            .card-title {{
                margin: 0;
                font-size: 1rem;
                line-height: 1.2;
                font-weight: 850;
                color: var(--text);
                letter-spacing: -0.03em;
            }}

            .card-body {{
                margin-top: 0.38rem;
                color: var(--muted);
                font-size: 0.94rem;
                line-height: 1.6;
            }}

            .info-grid {{
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.75rem;
            }}

            .mini-card {{
                border: 1px solid var(--border);
                border-radius: 14px;
                background: #FFFFFF;
                padding: 0.95rem;
            }}

            .mini-title {{
                font-size: 0.78rem;
                font-weight: 850;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: var(--muted);
                margin-bottom: 0.35rem;
            }}

            .mini-copy {{
                color: var(--text);
                font-size: 0.94rem;
                line-height: 1.55;
            }}

            .pill-row {{
                display: flex;
                flex-wrap: wrap;
                gap: 0.45rem;
            }}

            .pill {{
                display: inline-flex;
                align-items: center;
                gap: 0.3rem;
                border-radius: 999px;
                padding: 0.36rem 0.7rem;
                font-size: 0.78rem;
                font-weight: 750;
                line-height: 1;
                border: 1px solid transparent;
                white-space: nowrap;
            }}

            .pill-blue {{ background: rgba(37,99,235,0.10); color: #1D4ED8; border-color: rgba(37,99,235,0.16); }}
            .pill-cyan {{ background: rgba(8,145,178,0.10); color: #0E7490; border-color: rgba(8,145,178,0.16); }}
            .pill-green {{ background: rgba(22,163,74,0.12); color: #15803D; border-color: rgba(22,163,74,0.16); }}
            .pill-amber {{ background: rgba(245,158,11,0.14); color: #B45309; border-color: rgba(245,158,11,0.18); }}
            .pill-red {{ background: rgba(220,38,38,0.12); color: #B91C1C; border-color: rgba(220,38,38,0.18); }}
            .pill-neutral {{ background: rgba(100,116,139,0.10); color: #475569; border-color: rgba(100,116,139,0.16); }}

            .result-card {{
                padding: 0.92rem 0.95rem 0.88rem;
                min-height: 104px;
            }}

            .result-label {{
                font-size: 0.76rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-weight: 850;
                color: var(--muted);
                margin-bottom: 0.35rem;
            }}

            .result-value {{
                font-size: 1.08rem;
                font-weight: 850;
                line-height: 1.2;
                color: var(--text);
                margin-bottom: 0.35rem;
                overflow-wrap: anywhere;
            }}

            .result-caption {{
                color: var(--muted);
                font-size: 0.81rem;
                line-height: 1.45;
            }}

            .empty-state {{
                min-height: 260px;
                display: flex;
                align-items: center;
                justify-content: center;
                text-align: center;
                padding: 1.25rem;
            }}

            .empty-state-inner {{
                max-width: 360px;
            }}

            .empty-title {{
                font-size: 1.05rem;
                font-weight: 850;
                color: var(--text);
                margin-top: 0.75rem;
            }}

            .empty-subtitle {{
                margin-top: 0.35rem;
                color: var(--muted);
                font-size: 0.92rem;
                line-height: 1.55;
            }}

            .email-card {{
                padding: 1rem 1rem 0.95rem;
            }}

            .email-subject {{
                font-size: 0.84rem;
                font-weight: 850;
                color: #334155;
                margin-bottom: 0.55rem;
            }}

            .email-body {{
                border: 1px solid var(--border);
                border-radius: 14px;
                background: linear-gradient(180deg, #FFFFFF 0%, #FBFDFF 100%);
                padding: 0.95rem 0.95rem 0.95rem;
                color: #243244;
                font-size: 0.92rem;
                line-height: 1.68;
                white-space: pre-wrap;
            }}

            .hint-text {{
                color: var(--muted);
                font-size: 0.84rem;
                line-height: 1.45;
            }}

            .section-divider {{
                height: 1px;
                background: linear-gradient(90deg, rgba(226,232,240,0), rgba(226,232,240,1), rgba(226,232,240,0));
                margin: 1rem 0;
            }}

            .stButton > button {{
                border-radius: 14px;
                border: 1px solid rgba(37,99,235,0.22);
                background: linear-gradient(180deg, #2563EB 0%, #1D4ED8 100%);
                color: white;
                font-weight: 800;
                padding: 0.72rem 1rem;
                box-shadow: 0 8px 16px rgba(37,99,235,0.16);
            }}

            .stButton > button:hover {{
                background: linear-gradient(180deg, #1D4ED8 0%, #1E40AF 100%);
                border-color: rgba(29,78,216,0.35);
            }}

            .stTextArea textarea,
            .stTextInput input,
            .stSelectbox div[data-baseweb="select"] > div,
            .stMultiSelect div[data-baseweb="select"] > div {{
                border-radius: 14px !important;
                border-color: #D7DFEA !important;
                box-shadow: none !important;
            }}

            .stDataFrame, .stTable {{
                border-radius: 14px;
                overflow: hidden;
            }}

            .report-table-note {{
                color: var(--muted);
                font-size: 0.85rem;
                margin-top: 0.15rem;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


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
    summary = summarize_ticket(ticket_text, 220)
    response = (
        "Hello,\n\n"
        f"Thank you for contacting the AI Ticket Desk. We have reviewed your request and classified it as {predicted_category}. "
        f"It has been routed to {team}. Current priority: {predicted_priority}. SLA guidance: {sla['guidance']} "
        "Our team will continue reviewing the issue and provide the next update within the expected response window.\n\n"
        f"Ticket summary: {summary}\n\n"
        "Best regards,\n"
        "AI Ticket Desk Support Team"
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
    safe_text = html.escape(str(text))
    tone_class = {
        "blue": "pill-blue",
        "cyan": "pill-cyan",
        "green": "pill-green",
        "amber": "pill-amber",
        "red": "pill-red",
        "neutral": "pill-neutral",
    }.get(str(tone).lower(), "pill-blue")
    return f'<span class="pill {tone_class}">{safe_text}</span>'


def status_badge(text, tone="blue"):
    safe_text = html.escape(str(text))
    tone_class = {
        "blue": "badge-blue",
        "cyan": "badge-cyan",
        "green": "badge-green",
        "amber": "badge-amber",
        "red": "badge-red",
        "neutral": "badge-neutral",
    }.get(str(tone).lower(), "badge-blue")
    return f'<span class="badge {tone_class}">{safe_text}</span>'


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
    badge_html = f'<div style="margin-bottom:0.8rem;">{pill(badge, "blue")}</div>' if badge else ""
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


def metric_card(label, value, caption="", tone="blue"):
    tone_colors = {
        "blue": PALETTE["blue"],
        "cyan": PALETTE["cyan"],
        "green": PALETTE["green"],
        "amber": PALETTE["amber"],
        "red": PALETTE["red"],
    }
    accent = tone_colors.get(str(tone).lower(), PALETTE["blue"])
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


def card(title, body):
    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">{html.escape(str(title))}</div>
            <div class="card-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def result_tile(label, value, caption="", tone="blue"):
    badge_html = value if isinstance(value, str) and value.strip().startswith("<span") else html.escape(str(value))
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">{html.escape(str(label))}</div>
            <div class="result-value">{badge_html}</div>
            <div class="result-caption">{html.escape(str(caption))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_state(title, subtitle):
    st.markdown(
        f"""
        <div class="empty-state">
            <div class="empty-state-inner">
                <div>{status_badge('Awaiting analysis', 'neutral')}</div>
                <div class="empty-title">{html.escape(str(title))}</div>
                <div class="empty-subtitle">{html.escape(str(subtitle))}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_bar_figure(series, title, color, top_n=None, horizontal=False):
    if series is None or series.empty:
        return None

    counts = series.value_counts(dropna=False)
    if top_n is not None:
        counts = counts.head(top_n)
    chart_df = counts.reset_index()
    chart_df.columns = ["Label", "Count"]
    chart_df = chart_df.sort_values("Count", ascending=True if horizontal else False)

    if horizontal:
        fig = px.bar(chart_df, x="Count", y="Label", orientation="h")
    else:
        fig = px.bar(chart_df, x="Label", y="Count")

    fig.update_traces(marker_color=color, marker_line_width=0)
    fig.update_layout(
        title=title,
        showlegend=False,
        height=390,
        margin=dict(l=10, r=10, t=55, b=10),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        title_font=dict(size=18, color=PALETTE["text"]),
        xaxis=dict(title="", tickfont=dict(color=PALETTE["muted"]), gridcolor="#EEF2F7"),
        yaxis=dict(title="", tickfont=dict(color=PALETTE["muted"]), gridcolor="#EEF2F7"),
    )
    return fig


def chart_card(title, fig):
    st.markdown(
        f"""
        <div class="chart-card">
            <div class="card-title">{html.escape(str(title))}</div>
            <div class="card-body">Operational distribution view for the active dataset or report slice.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if fig is None:
        st.info("No data available for this chart.")
    else:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})


def report_table(report_dict):
    if not isinstance(report_dict, dict) or not report_dict:
        return pd.DataFrame()

    table = pd.DataFrame(report_dict).T
    preferred_columns = ["precision", "recall", "f1-score", "support"]
    available = [column for column in preferred_columns if column in table.columns]
    if available:
        table = table[available]

    for column in ["precision", "recall", "f1-score"]:
        if column in table.columns:
            table[column] = table[column].apply(lambda value: round(float(value) * 100, 2) if pd.notna(value) else value)
    if "support" in table.columns:
        table["support"] = table["support"].apply(lambda value: int(round(float(value))) if pd.notna(value) else value)

    return table


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


@st.cache_data(show_spinner=False)
def load_report():
    if not REPORT_PATH.exists():
        return {}
    try:
        return json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


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


def render_overview():
    dataset_rows = metric_value = None
    if dataset_info and dataset_info.get("df") is not None:
        df = dataset_info["df"]
        text_column = dataset_info["text_column"]
        category_column = dataset_info["category_column"]
        priority_column = dataset_info["priority_column"]
        priority_series = build_priority_series(df, text_column, priority_column)
        dataset_rows = len(df)
    else:
        df = None
        text_column = None
        category_column = None
        priority_series = None
        dataset_rows = None

    status_text = "Models Ready" if all(artifacts.values()) else "Training Required"
    dataset_text = f"{dataset_rows:,} Tickets" if dataset_rows is not None else "Dataset Unavailable"

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-grid">
                <div class="hero-copy">
                    <div class="hero-title">Support Intelligence Dashboard</div>
                    <div class="hero-subtitle">Monitor ticket distribution, model performance, and AI triage readiness from one workspace.</div>
                </div>
                <div class="hero-meta">
                    <div>{status_badge(status_text, 'green' if all(artifacts.values()) else 'amber')}</div>
                    <div>{status_badge(dataset_text, 'blue')}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    divider()

    metric_specs = [
        ("Tickets Analyzed", f"{dataset_rows:,}" if dataset_rows is not None else "--", "Rows available for modeling and exploration", "blue"),
        ("Category Accuracy", f"{float(report.get('category_accuracy', 0)) * 100:.1f}%" if report else "--", "LinearSVC held-out score", "cyan"),
        ("Priority Accuracy", f"{float(report.get('priority_accuracy', 0)) * 100:.1f}%" if report else "--", "Logistic Regression held-out score", "green"),
        ("TF-IDF Features", f"{report.get('tfidf_feature_count', '--')}", "Vocabulary size used for ticket vectorization", "amber"),
    ]
    cols = st.columns(4, gap="small")
    for col, (label, value, caption, tone) in zip(cols, metric_specs):
        with col:
            metric_card(label, value, caption, tone=tone)

    divider()

    section_header(
        "Business Value",
        "A compact view of the product outcomes this platform is designed to improve.",
    )
    biz_cols = st.columns(3, gap="medium")
    biz_cards = [
        ("Faster Triage", "Classify and prioritize support tickets in seconds so human agents can focus on resolution rather than manual sorting."),
        ("SLA Awareness", "Identify urgent requests early and surface clear response targets before support queues become overloaded."),
        ("Smarter Routing", "Route tickets to the correct support team using category-aware business logic and clear operational context."),
    ]
    for col, (title, body) in zip(biz_cols, biz_cards):
        with col:
            card(title, body)

    divider()

    chart_cols = st.columns(2, gap="large")
    with chart_cols[0]:
        fig = make_bar_figure(
            df[category_column] if df is not None and category_column is not None else None,
            "Category Distribution",
            PALETTE["blue"],
            top_n=12,
            horizontal=True,
        )
        chart_card("Category Distribution", fig)

    with chart_cols[1]:
        fig = make_bar_figure(priority_series, "Priority Distribution", PALETTE["cyan"], horizontal=False)
        chart_card("Priority Distribution", fig)


def render_ticket_triage():
    page_hero(
        "Ticket Triage",
        "Paste a support ticket, predict its category and priority, and generate the first response in seconds.",
        badge="AI Analysis Workflow",
    )

    can_predict = all(artifacts.get(key) is not None for key in ["tfidf", "category_model", "priority_model"])
    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.markdown(
            """
            <div class="card">
                <div class="card-title">New Support Ticket</div>
                <div class="card-body">Choose a sample or paste a new request to analyze support urgency, routing, and response guidance.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        sample_choice = st.selectbox("Example ticket", list(SAMPLE_TICKETS.keys()), index=0)
        if st.button("Load example", use_container_width=True):
            st.session_state["ticket_input"] = SAMPLE_TICKETS.get(sample_choice, st.session_state["ticket_input"])
            st.session_state["triage_result"] = None
            st.rerun()

        ticket_text = st.text_area(
            "Support ticket",
            value=st.session_state["ticket_input"],
            height=220,
            placeholder="Paste a ticket about access, hardware, procurement, storage, or an internal project issue...",
        )
        st.session_state["ticket_input"] = ticket_text

        if st.button("Analyze Ticket", type="primary", use_container_width=True, disabled=not can_predict):
            cleaned_text = normalize_text(ticket_text)
            if not cleaned_text:
                st.error("Please enter a support ticket before running the analysis.")
            else:
                try:
                    st.session_state["triage_result"] = prediction_payload(cleaned_text, artifacts)
                except Exception as exc:
                    st.error(f"Ticket analysis failed: {exc}")

        st.markdown(
            '<div class="hint-text">Tip: use the sample dropdown to test the workflow quickly, then replace it with a custom support request.</div>',
            unsafe_allow_html=True,
        )

    with right:
        result = st.session_state.get("triage_result")
        if not result:
            empty_state(
                "AI triage report will appear here",
                "Once a ticket is analyzed, this panel becomes a compact AI report with category, priority, SLA, routing, and response guidance.",
            )
        else:
            st.markdown(
                f"""
                <div class="analysis-card">
                    <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;">
                        <div>
                            <div class="badge badge-cyan" style="margin-bottom:0.6rem;">Completed</div>
                            <div class="section-title" style="font-size:1.3rem;">AI Triage Report</div>
                            <div class="section-subtitle">A structured summary of the model output and the recommended support workflow.</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")

            tile_cols = st.columns(2, gap="small")
            with tile_cols[0]:
                result_tile("Category", result["predicted_category"], "Predicted issue type", tone="blue")
            with tile_cols[1]:
                result_tile("Priority", pill(result["predicted_priority"], priority_tone(result["predicted_priority"])), "Urgency level", tone=priority_tone(result["predicted_priority"]))
            tile_cols = st.columns(2, gap="small")
            with tile_cols[0]:
                result_tile("SLA Target", result["sla"]["response_target"], "Suggested response window", tone="cyan")
            with tile_cols[1]:
                result_tile("Routing Team", result["team"], "Suggested operating queue", tone="blue")

            st.write("")
            left_detail, right_detail = st.columns([1.05, 0.95], gap="large")
            with left_detail:
                card("Urgency Explanation", result["urgency"])
                st.write("")
                card(
                    "Routing Decision",
                    (
                        f"<div class='pill-row'>"
                        f"{pill(result['predicted_category'], 'blue')}"
                        f"{pill(result['team'], 'neutral')}"
                        f"{pill(result['predicted_priority'], priority_tone(result['predicted_priority']))}"
                        f"</div>"
                    ),
                )
            with right_detail:
                st.markdown(
                    f"""
                    <div class="email-card">
                        <div class="card-title">First-Response Draft</div>
                        <div class="card-body">Email preview</div>
                        <div class="email-subject">Subject: Update on your support request</div>
                        <div class="email-body">{html.escape(result['response']).replace(chr(10), '<br>')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_model_performance():
    page_hero(
        "Model Performance Center",
        "Review the saved training report, evaluation metrics, and classification quality from the local ML pipeline.",
        badge="Monitoring View",
    )

    if not report:
        st.warning("Model report not found. Run `python scripts/train_models.py` to generate `models/model_report.json`.")
        return

    cols = st.columns(4, gap="small")
    metrics = [
        ("Category Accuracy", f"{float(report.get('category_accuracy', 0)) * 100:.1f}%", "Held-out score for category prediction", "blue"),
        ("Priority Accuracy", f"{float(report.get('priority_accuracy', 0)) * 100:.1f}%", "Held-out score for urgency prediction", "cyan"),
        ("Feature Count", f"{report.get('tfidf_feature_count', '--')}", "Vocabulary size used by TF-IDF", "green"),
        ("Dataset Rows", f"{report.get('dataset_rows', '--')}", "Rows used in the local training run", "amber"),
    ]
    for col, (label, value, caption, tone) in zip(cols, metrics):
        with col:
            metric_card(label, value, caption, tone=tone)

    divider()

    exp_cols = st.columns(2, gap="large")
    with exp_cols[0]:
        card(
            "What category accuracy means",
            "This score shows how well the category classifier separates support ticket topics such as access, hardware, storage, and procurement on unseen data.",
        )
    with exp_cols[1]:
        card(
            "What priority accuracy means",
            "This score measures how well the urgency model identifies ticket priority so support leaders can handle SLA-sensitive requests sooner.",
        )

    divider()

    category_report = report.get("category_classification_report") or report.get("category_report")
    priority_report = report.get("priority_classification_report") or report.get("priority_report")

    tabs = st.tabs(["Category Report", "Priority Report", "Raw Training Report"])
    with tabs[0]:
        section_header("Category Classification Report", "Precision, recall, F1-score, and support for each category in the held-out evaluation.")
        table = report_table(category_report)
        if table.empty:
            st.info("Category classification report is not available.")
        else:
            st.dataframe(table, use_container_width=True, height=360)
    with tabs[1]:
        section_header("Priority Classification Report", "Performance metrics for the urgency model used by the AI triage workflow.")
        table = report_table(priority_report)
        if table.empty:
            st.info("Priority classification report is not available.")
        else:
            st.dataframe(table, use_container_width=True, height=360)
    with tabs[2]:
        st.markdown('<div class="card-title" style="margin-bottom:0.4rem;">Raw Training Report</div>', unsafe_allow_html=True)
        with st.expander("Raw Training Report", expanded=False):
            st.json(report)


def render_dataset_explorer():
    page_hero(
        "Dataset Explorer",
        "Inspect the support ticket dataset, review sample rows, and explore category and priority distributions.",
        badge="Data Operations",
    )

    if not dataset_info or dataset_info.get("df") is None:
        st.warning("The dataset file was not found, but model-based ticket prediction can still work if the artifacts are available.")
        return

    df = dataset_info["df"].copy()
    text_column = dataset_info["text_column"]
    category_column = dataset_info["category_column"]
    priority_column = dataset_info["priority_column"]

    if text_column is None:
        st.error("No text column could be detected in the dataset.")
        return

    if "Priority" not in df.columns:
        df["Priority"] = build_priority_series(df, text_column, priority_column)
    else:
        df["Priority"] = df["Priority"].apply(normalize_text)

    summary_cols = st.columns(5, gap="small")
    summary_specs = [
        ("Rows", f"{df.shape[0]:,}", "Total records", "blue"),
        ("Columns", f"{df.shape[1]:,}", "Detected fields", "cyan"),
        ("Text Column", text_column or "--", "Primary ticket text", "green"),
        ("Category Column", category_column or "--", "Issue label field", "amber"),
        ("Priority Source", normalize_text(priority_column) if priority_column else "generated", "Priority data source", "red"),
    ]
    for col, (label, value, caption, tone) in zip(summary_cols, summary_specs):
        with col:
            metric_card(label, value, caption, tone=tone)

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

    stat_cols = st.columns(4, gap="small")
    stat_specs = [
        ("Filtered Rows", f"{len(filtered):,}", "Matching current filters", "blue"),
        ("Columns", f"{df.shape[1]:,}", "Dataset schema size", "cyan"),
        ("Visible Fields", f"{len(filtered.columns):,}", "Fields shown in the table", "green"),
        ("Sample Count", "Live", "Adjust below with the slider", "amber"),
    ]
    for col, (label, value, caption, tone) in zip(stat_cols, stat_specs):
        with col:
            metric_card(label, value, caption, tone=tone)

    row_count = st.slider(
        "Sample rows to display",
        min_value=5,
        max_value=min(50, max(len(filtered), 5)),
        value=min(10, max(len(filtered), 5)),
        step=1,
    )

    section_header("Sample records", "Review representative ticket rows with the current filters applied.")
    st.dataframe(filtered.head(row_count), use_container_width=True, height=340)

    divider()

    chart_cols = st.columns(2, gap="large")
    with chart_cols[0]:
        fig = make_bar_figure(
            filtered[category_column] if category_column is not None else None,
            "Category Distribution",
            PALETTE["blue"],
            top_n=12,
            horizontal=True,
        )
        chart_card("Category Counts", fig)
    with chart_cols[1]:
        fig = make_bar_figure(filtered["Priority"], "Priority Distribution", PALETTE["cyan"], horizontal=False)
        chart_card("Priority Counts", fig)


def render_about_project():
    page_hero(
        "AI Ticket Desk",
        "A professional simulation of an enterprise support automation platform.",
        badge="Product Case Study",
    )

    divider()
    section_header("Problem Statement", "Support teams receive a constant stream of tickets that need to be categorized, prioritized, routed, and acknowledged quickly.")
    card(
        "Problem Statement",
        "Manual triage slows resolution, increases SLA risk, and makes support operations harder to scale. AI Ticket Desk introduces a streamlined NLP workflow that gives support leaders a faster and more consistent operating model.",
    )

    divider()
    section_header("Solution Overview", "The platform uses lightweight text classification plus business rules to deliver structured support guidance.")
    card(
        "Solution Overview",
        "Tickets are cleaned, vectorized with TF-IDF, classified into support categories, prioritized for SLA handling, routed to the appropriate team, and converted into a first-response draft for faster customer communication.",
    )

    divider()
    section_header("ML Workflow", "A simple but realistic machine learning pipeline powers the support workflow.")
    workflow_items = [
        "Clean and standardize support ticket text.",
        "Convert text into TF-IDF features.",
        "Train a LinearSVC model for category prediction.",
        "Train a Logistic Regression model for priority prediction.",
        "Use routing and SLA business rules for operational guidance.",
    ]
    for item in workflow_items:
        st.markdown(
            f"<div class='mini-card' style='margin-bottom:0.65rem;'><div class='mini-copy'>{html.escape(item)}</div></div>",
            unsafe_allow_html=True,
        )

    divider()
    section_header("Tech Stack", "The implementation focuses on a practical Python and Streamlit stack.")
    stack_items = [
        ("Python", "blue"),
        ("Pandas", "neutral"),
        ("scikit-learn", "cyan"),
        ("TF-IDF", "blue"),
        ("LinearSVC", "green"),
        ("Logistic Regression", "amber"),
        ("Streamlit", "blue"),
        ("Plotly", "cyan"),
        ("Joblib", "neutral"),
        ("NLP", "green"),
    ]
    st.markdown(
        "<div class='card'><div class='pill-row'>"
        + "".join(pill(name, tone) for name, tone in stack_items)
        + "</div></div>",
        unsafe_allow_html=True,
    )

    divider()
    section_header("Business Impact", "What the system is designed to improve in a support environment.")
    impact_cols = st.columns(2, gap="large")
    with impact_cols[0]:
        card(
            "Business Impact",
            "Faster ticket triage, better routing consistency, improved SLA tracking, and a more professional first-touch customer experience.",
        )
    with impact_cols[1]:
        card(
            "Internship Context",
            "This project began as Future Interns ML Internship Task 2 and was upgraded into a recruiter-ready platform demo that showcases practical NLP, model training, and dashboard design.",
        )


def render_page(page_name):
    if page_name == "Dashboard Overview":
        render_overview()
    elif page_name == "Ticket Triage":
        render_ticket_triage()
    elif page_name == "Model Performance":
        render_model_performance()
    elif page_name == "Dataset Explorer":
        render_dataset_explorer()
    elif page_name == "About Project":
        render_about_project()


artifacts, artifact_errors = load_artifacts()
report = load_report()
dataset_info = load_dataset()


if "triage_result" not in st.session_state:
    st.session_state["triage_result"] = None
if "ticket_input" not in st.session_state:
    st.session_state["ticket_input"] = next(iter(SAMPLE_TICKETS.values()))


with st.sidebar:
    st.markdown(
        f"""
        <div class="sidebar-wrap">
            <div class="badge badge-cyan" style="margin-bottom:0.9rem;">NLP Automation</div>
            <div class="sidebar-title">{PROJECT_NAME}</div>
            <div class="sidebar-subtitle">{PROJECT_TAGLINE}</div>
            <div class="sidebar-desc">{PROJECT_SUBTITLE}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if all(artifacts.values()):
        st.markdown(f"<div style='padding:0 1rem 0.75rem 1rem;'>{status_badge('Models ready', 'green')}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='padding:0 1rem 0.75rem 1rem;'>{status_badge('Training required', 'amber')}</div>", unsafe_allow_html=True)

    st.markdown("<div style='padding:0 1rem 0.45rem 1rem; color:#94A3B8; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.08em; font-weight:800;'>Navigation</div>", unsafe_allow_html=True)
    sidebar_choice = st.radio(
        "Navigation",
        list(PAGE_LABELS.keys()),
        format_func=lambda key: PAGE_LABELS[key],
        label_visibility="collapsed",
        index=0,
    )
    st.markdown(
        """
        <div class="sidebar-footer">
            <div>Future Interns ML Task 2</div>
            <div>Built with Python · scikit-learn · Streamlit</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if artifact_errors:
    st.warning(
        "Some model files could not be loaded. Ticket prediction will be disabled until the artifacts are restored.\n\n"
        + "\n".join(f"- {error}" for error in artifact_errors)
    )


render_page(sidebar_choice)
