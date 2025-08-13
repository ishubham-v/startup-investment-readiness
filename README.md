Alright — let’s break down **the mathematics** behind your **Machine Learning–Driven Startup Investment Readiness** project so you can clearly explain it in a thesis, README, or interview.

---

## **1️⃣ Problem Formulation**

We want to predict **investment readiness** $y \in \{0,1\}$ of a startup based on a set of **15 engineered features** $x \in \mathbb{R}^{15}$ from three domains:

$$
x = [x_\text{financial}, x_\text{strategic}, x_\text{perceptual}]
$$

The goal is to model:

$$
\hat{y} = f(x) = 
\begin{cases}
1, & \text{if startup is investment ready} \\
0, & \text{otherwise}
\end{cases}
$$

---

## **2️⃣ Feature Engineering Mathematics**

### **(a) Financial Features**

These are mostly **normalized numerical KPIs** like funding, runway, LTV/CAC ratio, etc.
Some derived metrics:

1. **Runway (months)**:

$$
\text{Runway} = \frac{\text{Total Funding}}{\text{Monthly Burn Rate}}
$$

2. **LTV/CAC Ratio**:

$$
\text{LTV\_CAC} = \frac{\text{LTV}}{\text{CAC}}
$$

3. **ARR (Annual Recurring Revenue)**:

$$
\text{ARR} = \text{Monthly Revenue} \times 12
$$

---

### **(b) Strategic Features**

#### **Idea Novelty Score (SBERT + Cosine Similarity)**

Given:

* $e_s$ = SBERT embedding of startup description
* $e_m^{(i)}$ = embedding of the $i$-th market idea

**Cosine Similarity**:

$$
\cos(e_s, e_m^{(i)}) = \frac{e_s \cdot e_m^{(i)}}{\| e_s \| \, \| e_m^{(i)} \|}
$$

Novelty score is:

$$
\text{Novelty} = 1 - \max_i \cos(e_s, e_m^{(i)})
$$

This ensures that **lower similarity to existing ideas → higher novelty**.

---

#### **Sustainability Alignment**

$$
\text{Sustainability\_Alignment} =
\begin{cases}
1, & \text{if startup aligns with at least one UN SDG} \\
0, & \text{otherwise}
\end{cases}
$$

---

### **(c) Perceptual Features**

#### **Sentiment Score (FinBERT)**

FinBERT outputs probabilities for three classes:

$$
[p_\text{pos}, p_\text{neg}, p_\text{neu}]
$$

We map these to a scalar sentiment score:

$$
\text{Sentiment} = p_\text{pos} - p_\text{neg}
$$

Result is in range $[-1, 1]$.

#### **Social Buzz Score**

If we have engagement counts (likes, shares, mentions), we normalize:

$$
\text{Buzz Score} = \frac{\text{Current Engagement Count}}{\text{Max Engagement in Dataset}}
$$

This ensures values are in range $[0,1]$.

---

## **3️⃣ Model Mathematics (SVM with RBF Kernel)**

We use a **Support Vector Classifier** with **Radial Basis Function (RBF)** kernel:

### **(a) RBF Kernel**

$$
K(x_i, x_j) = \exp\left(-\gamma \| x_i - x_j \|^2 \right)
$$

* $\gamma$ controls the influence of a single sample (higher = more complex boundary).

---

### **(b) SVM Decision Function**

The classifier finds a decision boundary:

$$
f(x) = \sum_{i=1}^N \alpha_i y_i K(x_i, x) + b
$$

Where:

* $\alpha_i$ = learned support vector coefficients
* $b$ = bias term
* $y_i \in \{-1, 1\}$ = class labels

---

### **(c) Probability Calibration**

We get decision scores $f(x)$ and convert to probabilities $p$ using **Platt scaling**:

$$
p(y=1|x) = \frac{1}{1 + \exp(A f(x) + B)}
$$

where $A$ and $B$ are fitted from validation data.

---

### **(d) VC Risk-Aligned Thresholding**

Instead of default $0.5$, we use $t = 0.58$ to reflect a VC's preference for reducing false positives:

$$
\hat{y} =
\begin{cases}
1, & p(y=1|x) \geq 0.58 \\
0, & \text{otherwise}
\end{cases}
$$

---

## **4️⃣ Model Fine-Tuning Mathematics**

We fine-tune:

1. **Features** using **Recursive Feature Elimination with Cross Validation (RFECV)** — eliminates least important features iteratively.
2. **Hyperparameters** $C$ and $\gamma$ using **GridSearchCV**:

   * $C$ = regularization (larger = less regularization, more complex model)
   * $\gamma$ = kernel spread parameter

Grid search:

$$
(C, \gamma) \in \{0.5, 1, 2, 5\} \times \{\text{scale}, 0.1, 0.01, 0.001\}
$$

---

## **5️⃣ Evaluation Metrics**

* **Accuracy**:

$$
\text{Accuracy} = \frac{\text{TP + TN}}{\text{TP + TN + FP + FN}}
$$

* **F1-score** (harmonic mean of precision and recall):

$$
F1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

* **Precision**:

$$
\text{Precision} = \frac{\text{TP}}{\text{TP + FP}}
$$

* **Recall**:

$$
\text{Recall} = \frac{\text{TP}}{\text{TP + FN}}
$$

---

## **6️⃣ End-to-End Pipeline**

1. **Input**: Startup data (financial, strategic, perceptual features).
2. **Feature Engineering**:

   * Calculate novelty (SBERT cosine similarity).
   * Calculate sentiment (FinBERT probabilities).
   * Normalize numerical KPIs.
3. **Model**: RBF SVM with calibrated probabilities.
4. **Decision**: Apply VC threshold (0.58) for classification.
5. **Output**:

   * Probability of investment readiness.
   * Binary decision (Ready / Not Ready).
