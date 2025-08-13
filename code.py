"""
investment_readiness_model.py
Predicts startup investment readiness using financial, strategic, and perceptual features.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib

# Optional: Transformers for Novelty & Sentiment
from sentence_transformers import SentenceTransformer
from transformers import pipeline as hf_pipeline


# ----------------------------
# CONFIGURATION
# ----------------------------
USE_TRANSFORMERS = False  # Set to True for real SBERT & FinBERT usage
RANDOM_STATE = 42
PROBA_THRESHOLD = 0.58  # Tuned to achieve ~0.77 F1 in synthetic baseline

FEATURE_COLUMNS = [
    "Total_Funding_USD_M",
    "Burn_Rate_USD_M",
    "Runway_Months",
    "Revenue_USD_M",
    "Revenue_Growth_Rate",
    "Net_Profit_Margin",
    "CAC",
    "LTV",
    "LTV_CAC_Ratio",
    "ARR_USD_M",
    "Churn_Rate",
    "Novelty_Score",
    "Sustainability_Alignment",
    "Sentiment_Score",
    "Social_Buzz_Score"
]


# ----------------------------
# FEATURE ENGINEERING FUNCTIONS
# ----------------------------
def compute_novelty(text: str, market_texts: list[str]) -> float:
    """
    Compute novelty score using SBERT embeddings + cosine similarity.
    Returns 1 - max cosine similarity with existing market ideas.
    """
    if not USE_TRANSFORMERS:
        # Deterministic placeholder for offline mode
        return (abs(hash(text)) % 1000) / 1000.0

    model = SentenceTransformer("all-MiniLM-L6-v2")
    startup_vec = model.encode([text])[0]
    market_vecs = model.encode(market_texts)

    # Cosine similarities
    sims = np.dot(market_vecs, startup_vec) / (
        np.linalg.norm(market_vecs, axis=1) * np.linalg.norm(startup_vec)
    )
    return float(1 - np.max(sims))


def compute_sentiment(text: str) -> float:
    """
    Compute sentiment score using FinBERT sentiment pipeline.
    Returns score in range [-1, 1].
    """
    if not USE_TRANSFORMERS:
        # Placeholder: map hash to [-1, 1]
        return ((abs(hash(text)) % 2000) / 1000.0) - 1.0

    sentiment_model = hf_pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")
    result = sentiment_model(text)[0]
    label = result["label"]
    score = result["score"]

    if label == "Positive":
        return score
    elif label == "Negative":
        return -score
    else:  # Neutral
        return 0.0


# ----------------------------
# MODEL TRAINING
# ----------------------------
def train_model(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(C=2.0, kernel="rbf", probability=True, random_state=RANDOM_STATE))
    ])

    pipeline.fit(X_train, y_train)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= PROBA_THRESHOLD).astype(int)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return pipeline


# ----------------------------
# MAIN EXECUTION
# ----------------------------
if __name__ == "__main__":
    # Example: Load synthetic dataset (replace with your real data CSV)
    df = pd.read_csv("data/synthetic_startups.csv")

    # Train the model
    model = train_model(df)

    # Save the trained model
    joblib.dump(model, "artifacts/investment_readiness_svm.joblib")
    print("Model saved to artifacts/investment_readiness_svm.joblib")

    # Example inference
    sample = df.iloc[[0]][FEATURE_COLUMNS]
    readiness_proba = model.predict_proba(sample)[:, 1][0]
    readiness_pred = int(readiness_proba >= PROBA_THRESHOLD)
    print(f"\nSample prediction: Proba={readiness_proba:.3f}, Ready={readiness_pred}")
