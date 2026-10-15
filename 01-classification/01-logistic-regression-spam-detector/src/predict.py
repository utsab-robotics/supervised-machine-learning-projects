
from pathlib import Path

import joblib

MODEL_PATH = Path("models/spam_classifier.joblib")


def predict_message(message: str) -> dict:
    """Predict whether a single SMS is spam or legitimate."""
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Please provide a non-empty message.")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run python src/train.py first."
        )

    model = joblib.load(MODEL_PATH)
    prediction = model.predict([message])[0]
    probabilities = model.predict_proba([message])[0]

    classes = list(model.classes_)
    spam_probability = float(probabilities[classes.index("spam")])

    return {
        "message": message,
        "prediction": "SPAM" if prediction == "spam" else "NOT SPAM",
        "spam_probability": spam_probability,
    }


if __name__ == "__main__":
    print("SMS Spam Detector")
    print("Type 'exit' to quit.\n")

    while True:
        message = input("Enter an SMS message: ").strip()

        if message.lower() == "exit":
            break

        try:
            result = predict_message(message)
            print(f"Prediction: {result['prediction']}")
            print(
                "Estimated spam probability: "
                f"{result['spam_probability']:.2%}\n"
            )
        except (ValueError, FileNotFoundError) as error:
            print(f"Error: {error}\n")
