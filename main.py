import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent

IMPORTANT_FOLDERS = [
    "data",
    "models",
    "app",
    "scripts",
    "notebooks",
    "visuals",
]

IMPORTANT_FILES = [
    "data/all_tickets_processed_improved_v3.csv",
    "models/tfidf_vectorizer.pkl",
    "models/category_model.pkl",
    "models/priority_model.pkl",
    "models/model_report.json",
    "app/app.py",
    "scripts/train_models.py",
    "README.md",
    "requirements.txt",
]


def format_status(exists):
    return "[OK]" if exists else "[MISSING]"


def print_banner():
    print("=" * 72)
    print("Ticket Desk — Support Intelligence Platform")
    print("Project Launcher and Health Checker")
    print("=" * 72)


def check_folders():
    print("\nFolders")
    print("-" * 72)
    for folder in IMPORTANT_FOLDERS:
        folder_path = ROOT_DIR / folder
        print(f"{format_status(folder_path.exists())} {folder}/")


def check_files():
    print("\nFiles")
    print("-" * 72)
    for file_name in IMPORTANT_FILES:
        file_path = ROOT_DIR / file_name
        print(f"{format_status(file_path.exists())} {file_name}")


def print_model_report():
    report_path = ROOT_DIR / "models" / "model_report.json"
    if not report_path.exists():
        print("\nModel Report")
        print("-" * 72)
        print("[MISSING] models/model_report.json")
        return

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print("\nModel Report")
        print("-" * 72)
        print(f"[ERROR] Could not read model_report.json: {exc}")
        return

    def as_percent(value):
        try:
            return f"{float(value) * 100:.1f}%"
        except Exception:
            return "--"

    print("\nModel Report")
    print("-" * 72)
    print(f"Dataset rows       : {report.get('dataset_rows', '--')}")
    print(f"Category accuracy   : {as_percent(report.get('category_accuracy'))}")
    print(f"Priority accuracy   : {as_percent(report.get('priority_accuracy'))}")
    print(f"TF-IDF feature count: {report.get('tfidf_feature_count', '--')}")


def print_next_steps():
    models_ready = all((ROOT_DIR / path).exists() for path in [
        "models/tfidf_vectorizer.pkl",
        "models/category_model.pkl",
        "models/priority_model.pkl",
    ])
    app_ready = (ROOT_DIR / "app" / "app.py").exists()

    print("\nSuggested Commands")
    print("-" * 72)
    if not models_ready:
        print("To train or rebuild the models:")
        print("  python scripts/train_models.py")

    if app_ready and models_ready:
        print("To launch the dashboard:")
        print("  streamlit run app/app.py")

    if not models_ready and app_ready:
        print("The dashboard file exists, but model artifacts are missing.")
        print("Run the training command above before launching Streamlit.")


def main():
    print_banner()
    check_folders()
    check_files()
    print_model_report()
    print_next_steps()

    print("\nStatus complete.")
    print("Built for Ticket Desk — Support Intelligence Platform")


if __name__ == "__main__":
    main()
