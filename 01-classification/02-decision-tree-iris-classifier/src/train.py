from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"

MODEL_PATH = MODEL_DIR / "iris_decision_tree.joblib"
REPORT_PATH = OUTPUT_DIR / "classification_report.txt"
CONFUSION_PATH = OUTPUT_DIR / "confusion_matrix.png"
TREE_PATH = OUTPUT_DIR / "decision_tree.png"

RANDOM_STATE = 42


def main():
    # Create output folders
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the built-in Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    print("Decision Tree - Iris Flower Classifier")
    print("-" * 45)
    print(f"Total samples: {len(X)}")
    print(f"Features: {list(iris.feature_names)}")
    print(f"Classes: {list(iris.target_names)}")

    # Split the data: 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Create and train the Decision Tree
    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=4,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)

    # Evaluate on the held-out test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=iris.target_names,
        zero_division=0,
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Test accuracy: {accuracy:.2%}")
    print("\nClassification report:")
    print(report)

    # Save the trained model and required metadata together
    joblib.dump(
        {
            "model": model,
            "feature_names": list(iris.feature_names),
            "target_names": list(iris.target_names),
        },
        MODEL_PATH,
    )

    # Save the report
    REPORT_PATH.write_text(
        f"Test accuracy: {accuracy:.4f}\n\n{report}",
        encoding="utf-8",
    )

    # Save confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=iris.target_names,
        cmap="Blues",
    )
    plt.title("Iris Classifier - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_PATH, dpi=150)
    plt.close()

    # Save a visualization of the trained tree
    plt.figure(figsize=(18, 10))
    plot_tree(
        model,
        feature_names=iris.feature_names,
        class_names=list(iris.target_names),
        filled=True,
        rounded=True,
        fontsize=9,
    )
    plt.title("Decision Tree - Iris Flower Classification")
    plt.tight_layout()
    plt.savefig(TREE_PATH, dpi=150)
    plt.close()

    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Report saved: {REPORT_PATH}")
    print(f"Confusion matrix saved: {CONFUSION_PATH}")
    print(f"Decision tree saved: {TREE_PATH}")


if __name__ == "__main__":
    main()