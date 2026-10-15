# SMS Spam Detector — Logistic Regression

## Overview

A supervised machine learning project that classifies SMS messages as spam or legitimate using TF-IDF text features and Logistic Regression.

## Algorithm

- Learning type: Supervised learning
- Task: Binary classification
- Algorithm: Logistic Regression
- Feature extraction: TF-IDF
- Dataset: UCI SMS Spam Collection

## Project Structure

- `data/raw/`: Original dataset
- `src/train.py`: Data loading, training and evaluation
- `src/predict.py`: Interactive message prediction
- `models/`: Saved trained pipeline
- `outputs/`: Evaluation report and confusion matrix
- `requirements.txt`: Python dependencies

## Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Train

```powershell
python src/train.py
```

## Predict

```powershell
python src/predict.py
```

## Evaluation

The model is evaluated on a held-out test set. Metrics include accuracy, spam precision, spam recall, F1-score, and a confusion matrix.

Record actual results after training. Do not assume a particular accuracy in advance.

## Dataset and Attribution

Almeida, T. and Hidalgo, J. (2011). SMS Spam Collection. UCI Machine Learning Repository.

https://doi.org/10.24432/C5CC84

Dataset license: CC BY 4.0.

## Limitations

The model learns patterns from the dataset and may misclassify new, ambiguous, or unfamiliar messages. A predicted probability is not proof that a message is spam.

## Future Improvements

- Add a web interface
- Compare Logistic Regression with Naive Bayes
- Tune the classification threshold
- Evaluate false positives and false negatives
- Test on additional messages representative of the intended users
