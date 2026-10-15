from pathlib import Path

import joblib


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "models" / "iris_decision_tree.joblib"


def load_model():
    """Load the saved model and its metadata."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run python src/train.py first."
        )

    return joblib.load(MODEL_PATH)


def predict_species(measurements):
    """Predict the Iris species from four numerical measurements."""
    if len(measurements) != 4:
        raise ValueError("Exactly four measurements are required.")

    try:
        values = [float(value) for value in measurements]
    except (TypeError, ValueError) as exc:
        raise ValueError("All measurements must be valid numbers.") from exc

    if any(not 0 < value <= 10 for value in values):
        raise ValueError(
            "Each measurement must be greater than 0 and at most 10 cm."
        )

    bundle = load_model()
    model = bundle["model"]
    target_names = bundle["target_names"]

    prediction = int(model.predict([values])[0])
    probabilities = model.predict_proba([values])[0]

    return {
        "species": target_names[prediction],
        "confidence": float(probabilities[prediction]),
        "probabilities": {
            target_names[int(class_id)]: float(probability)
            for class_id, probability in zip(model.classes_, probabilities)
        },
    }


def main():
    print("Iris Flower Species Predictor")
    print("Enter measurements in centimeters.")
    print("Type 'exit' to quit.\n")

    feature_labels = [
        "Sepal length",
        "Sepal width",
        "Petal length",
        "Petal width",
    ]

    while True:
        measurements = []

        for label in feature_labels:
            value = input(f"{label} (cm): ").strip()

            if value.lower() == "exit":
                print("Goodbye!")
                return

            measurements.append(value)

        try:
            result = predict_species(measurements)

            print(f"\nPredicted species: {result['species']}")
            print(
                "Model confidence: "
                f"{result['confidence']:.2%}"
            )
            print("Class probabilities:")

            for species, probability in result["probabilities"].items():
                print(f"  {species}: {probability:.2%}")

        except (ValueError, FileNotFoundError) as error:
            print(f"Error: {error}")

        print()


if __name__ == "__main__":
    main()