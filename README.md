# startup-investment-readiness
Startup investment readiness scoring with SBERT novelty &amp; sentiment, via RBF-SVM
Machine Learning–Driven Startup Investment Readiness Scoring
Overcoming Subjectivity Bias with Novelty & Sentiment-Enhanced Feature Modelling (Jan 2025 – May 2025)

This project implements an end-to-end machine learning pipeline to assess startup investment readiness, integrating:

Financial features: Funding, burn rate, runway, revenue growth, margins, LTV/CAC, ARR, churn rate.

Strategic features:

Idea Novelty via SBERT embeddings and cosine similarity against market vectors.

Sustainability Alignment with UN SDGs.

Perceptual features:

FinBERT-based sentiment analysis of news coverage.

Social buzz scores from online presence metrics.

A Support Vector Machine (RBF kernel) is trained on these 15 engineered features, with VC-risk-aligned sigmoid thresholding to optimize both F1-score and decision quality.

✨ Highlights
SBERT-powered novelty modelling to measure originality in startup concepts.

FinBERT sentiment scoring to quantify market and media perception.

Synthetic dataset benchmark achieving ~80% accuracy and 0.77 F1-score.

Bias mitigation by shifting from purely subjective evaluation to feature-driven modelling.

Modular feature engineering pipeline — drop in real data to replace synthetic baseline.

