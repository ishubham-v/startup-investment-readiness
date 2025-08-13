"""
fine_tune_features.py
Performs feature selection and hyperparameter tuning for the Investment Readiness SVM model.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib

RANDOM_STATE = 42
PROBA_THRESHOLD = 0.58

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


def fine_tune_model(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    # Step 1: Base SVC model
    svc = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)

    # Step 2: Feature selection using RFECV (Recursive Feature Elimination with Cross-Validation)
    selector = RFECV(
        estimator=svc,
        step=1,
        cv=5,
        scoring="f1",
        min_features_to_select=5,
        n_jobs=-1
    )
    selector = selector.fit(X_train, y_train)

    selected_features = [f for f, keep in zip(FEATURE_COLUMNS, selector.support_) if keep]
    print("\nSelected Features:", selected_features)

    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]

    # Step 3: Hyperparameter tuning with GridSearchCV
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))
    ])

    param_grid = {
        "clf__C": [0.5, 1, 2, 5],
        "clf__gamma": ["scale", 0.1, 0.01, 0.001]
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train_sel, y_train)
    print("\nBest Parameters:", grid_search.best_params_)

    # Step 4: Evaluate tuned model
    y_proba = grid_search.predict_proba(X_test_sel)[:, 1]
    y_pred = (y_proba >= PROBA_THRESHOLD).astype(int)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Step 5: Save tuned model and selected features
    joblib.dump({
        "model": grid_search.best_estimator_,
        "features": selected_features
    }, "artifacts/fine_tuned_investment_readiness_model.joblib")
    print("\nFine-tuned model saved to artifacts/fine_tuned_investment_readiness_model.joblib")


if __name__ == "__main__":
    # Load synthetic dataset (replace with real dataset for production)
    df = pd.read_csv("data/synthetic_startups.csv")
    fine_tune_model(df)
