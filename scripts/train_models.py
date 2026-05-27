import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "all_tickets_processed_improved_v3.csv"
MODELS_DIR = ROOT_DIR / "models"
REPORT_PATH = MODELS_DIR / "model_report.json"

TEXT_COLUMN_CANDIDATES = ["Document", "Ticket", "Description", "Text", "Issue", "Summary"]
CATEGORY_COLUMN_CANDIDATES = ["Topic_group", "Category", "Label", "Department", "Class"]

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


def load_dataset(path):
    """Load the CSV with a small encoding fallback chain."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    encodings_to_try = ["utf-8-sig", "utf-8", "latin1"]
    last_error = None

    for encoding in encodings_to_try:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error

    raise UnicodeDecodeError(
        last_error.encoding if last_error else "utf-8",
        last_error.object if last_error else b"",
        last_error.start if last_error else 0,
        last_error.end if last_error else 0,
        "Unable to decode the dataset with common encodings.",
    )


def resolve_column(columns, preferred_names):
    """Return the first matching column name using case-insensitive matching."""
    column_lookup = {column.lower(): column for column in columns}

    for candidate in preferred_names:
        match = column_lookup.get(candidate.lower())
        if match is not None:
            return match

    return None


def normalize_label(value):
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def generate_priority_label(text):
    lowered = str(text).lower()

    if any(keyword in lowered for keyword in HIGH_PRIORITY_KEYWORDS):
        return "High"
    if any(keyword in lowered for keyword in MEDIUM_PRIORITY_KEYWORDS):
        return "Medium"
    if any(keyword in lowered for keyword in LOW_PRIORITY_KEYWORDS):
        return "Low"

    return "Low"


def safe_train_test_split(frame, label_column, test_size=0.2, random_state=42):
    """Split with stratification when the class distribution allows it."""
    label_counts = frame[label_column].value_counts()
    can_stratify = len(label_counts) > 1 and label_counts.min() >= 2

    if can_stratify:
        return train_test_split(
            frame,
            test_size=test_size,
            random_state=random_state,
            stratify=frame[label_column],
        )

    return train_test_split(
        frame,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )


def build_model_report(
    dataset_rows,
    text_column,
    category_column,
    priority_source,
    tfidf,
    category_accuracy,
    priority_accuracy,
    category_report,
    priority_report,
):
    return {
        "generated_timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset_rows": int(dataset_rows),
        "text_column_used": text_column,
        "category_column_used": category_column,
        "priority_source": priority_source,
        "tfidf_feature_count": int(len(tfidf.vocabulary_)),
        "category_accuracy": float(category_accuracy),
        "priority_accuracy": float(priority_accuracy),
        "category_classification_report": category_report,
        "priority_classification_report": priority_report,
    }


def main():
    print("AI Ticket Desk - Support Intelligence Platform")
    print("=" * 52)
    print("Loading dataset...")

    df = load_dataset(DATA_PATH)
    print(f"Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]:,} columns")

    text_column = resolve_column(df.columns, TEXT_COLUMN_CANDIDATES)
    category_column = resolve_column(df.columns, CATEGORY_COLUMN_CANDIDATES)
    priority_column = resolve_column(df.columns, ["Priority"])

    if text_column is None:
        available_columns = ", ".join(df.columns.astype(str).tolist())
        raise ValueError(
            "No text column found. Expected one of: "
            f"{', '.join(TEXT_COLUMN_CANDIDATES)}. Available columns: {available_columns}"
        )

    if category_column is None:
        available_columns = ", ".join(df.columns.astype(str).tolist())
        raise ValueError(
            "No category column found. Expected one of: "
            f"{', '.join(CATEGORY_COLUMN_CANDIDATES)}. Available columns: {available_columns}"
        )

    working_df = df[[text_column, category_column]].copy()
    working_df[text_column] = working_df[text_column].astype(str).str.strip()
    working_df[category_column] = working_df[category_column].map(normalize_label)
    working_df = working_df.dropna(subset=[text_column, category_column]).copy()
    working_df = working_df[working_df[text_column] != ""].copy()

    if working_df.empty:
        raise ValueError("No usable rows remain after validating text and category columns.")

    if priority_column is not None:
        working_df[priority_column] = df.loc[working_df.index, priority_column].map(normalize_label)
        working_df["Priority"] = working_df[priority_column].fillna(
            working_df[text_column].map(generate_priority_label)
        )
        priority_source = "existing"
    else:
        working_df["Priority"] = working_df[text_column].map(generate_priority_label)
        priority_source = "generated"

    working_df = working_df.rename(columns={text_column: "__text__", category_column: "__category__"})
    working_df["__category__"] = working_df["__category__"].astype(str)
    working_df["Priority"] = working_df["Priority"].astype(str)

    print(f"Text column used: {text_column}")
    print(f"Category column used: {category_column}")
    print(f"Priority source: {priority_source}")
    print(f"Usable rows after cleaning: {len(working_df):,}")
    print("Preparing train/test splits...")

    category_train_df, category_test_df = safe_train_test_split(working_df, "__category__")
    priority_train_df, priority_test_df = safe_train_test_split(working_df, "Priority")

    tfidf = TfidfVectorizer(
        max_features=10000,
        stop_words="english",
        min_df=2,
        sublinear_tf=True,
    )

    print("Training TF-IDF vectorizer...")
    category_x_train = tfidf.fit_transform(category_train_df["__text__"])
    category_x_test = tfidf.transform(category_test_df["__text__"])

    print("Training category model (LinearSVC)...")
    category_model = LinearSVC(class_weight="balanced", random_state=42, dual=False)
    category_model.fit(category_x_train, category_train_df["__category__"])
    category_predictions = category_model.predict(category_x_test)

    category_accuracy = accuracy_score(category_test_df["__category__"], category_predictions)
    category_report = classification_report(
        category_test_df["__category__"],
        category_predictions,
        output_dict=True,
        zero_division=0,
    )

    print("Training priority model (LogisticRegression)...")
    priority_x_train = tfidf.transform(priority_train_df["__text__"])
    priority_x_test = tfidf.transform(priority_test_df["__text__"])

    priority_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    priority_model.fit(priority_x_train, priority_train_df["Priority"])
    priority_predictions = priority_model.predict(priority_x_test)

    priority_accuracy = accuracy_score(priority_test_df["Priority"], priority_predictions)
    priority_report = classification_report(
        priority_test_df["Priority"],
        priority_predictions,
        output_dict=True,
        zero_division=0,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Saving artifacts...")
    joblib.dump(tfidf, MODELS_DIR / "tfidf_vectorizer.pkl")
    joblib.dump(category_model, MODELS_DIR / "category_model.pkl")
    joblib.dump(priority_model, MODELS_DIR / "priority_model.pkl")

    report = build_model_report(
        dataset_rows=len(working_df),
        text_column=text_column,
        category_column=category_column,
        priority_source=priority_source,
        tfidf=tfidf,
        category_accuracy=category_accuracy,
        priority_accuracy=priority_accuracy,
        category_report=category_report,
        priority_report=priority_report,
    )

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nTraining complete.")
    print(f"Category accuracy: {category_accuracy:.4f}")
    print(f"Priority accuracy: {priority_accuracy:.4f}")
    print("Artifacts saved:")
    print(f"- {MODELS_DIR / 'tfidf_vectorizer.pkl'}")
    print(f"- {MODELS_DIR / 'category_model.pkl'}")
    print(f"- {MODELS_DIR / 'priority_model.pkl'}")
    print(f"- {REPORT_PATH}")


if __name__ == "__main__":
    main()
