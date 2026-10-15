
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# All paths are relative to the project directory.
DATA_PATH = Path("data/raw/SMSSpamCollection")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

RANDOM_STATE = 42


def load_data():
    """Load the tab-separated SMS dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Download it and place it in data/raw/."
        )

    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8",
        on_bad_lines="error",
    )

    df = df.dropna(subset=["label", "message"])
    df["label"] = df["label"].str.strip().str.lower()
    df["message"] = df["message"].astype(str).str.strip()

    df = df[df["label"].isin(["ham", "spam"])]
    df = df[df["message"] != ""]
    df = df.drop_duplicates(subset=["label", "message"])

    if df["label"].nunique() != 2:
        raise ValueError("Expected both ham and spam labels.")

    return df


def build_model():
    """Create a text-processing and classification pipeline."""
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    print("\nDataset overview")
    print("-" * 40)
    print(f"Total usable messages: {len(df)}")
    print("\nClass distribution:")
    print(df["label"].value_counts())

    X = df["message"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = build_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, pos_label="spam", zero_division=0
    )
    recall = recall_score(
        y_test, predictions, pos_label="spam", zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, pos_label="spam", zero_division=0
    )

    print("\nEvaluation on held-out test data")
    print("-" * 40)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Spam precision: {precision:.4f}")
    print(f"Spam recall:    {recall:.4f}")
    print(f"Spam F1-score:  {f1:.4f}")
    print("\nClassification report:")
    report = classification_report(
        y_test, predictions, zero_division=0
    )
    print(report)

    # Save the entire pipeline so inference uses the same TF-IDF setup.
    model_path = MODEL_DIR / "spam_classifier.joblib"
    joblib.dump(model, model_path)

    # Save actual results for the project documentation.
    report_path = OUTPUT_DIR / "classification_report.txt"
    with report_path.open("w", encoding="utf-8") as file:
        file.write(f"Test samples: {len(y_test)}\n")
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Spam precision: {precision:.4f}\n")
        file.write(f"Spam recall: {recall:.4f}\n")
        file.write(f"Spam F1-score: {f1:.4f}\n\n")
        file.write(report)

    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=["ham", "spam"],
        display_labels=["Not Spam", "Spam"],
        cmap="Blues",
        values_format="d",
    )
    display.ax_.set_title("SMS Spam Detection — Logistic Regression")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "confusion_matrix.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    print(f"\nSaved model: {model_path}")
    print(f"Saved report: {report_path}")
    print(f"Saved confusion matrix: {OUTPUT_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
