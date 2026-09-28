# Avenix — Lead Scoring & Propensity-to-Convert ML

An end-to-end Machine Learning project for predicting the probability that a lead will convert into a customer.

Avenix transforms historical lead and marketing data into an explainable **Lead Scoring System** that can estimate conversion probability for new leads and classify them into actionable scoring groups.

---

## 🎯 Project Goal

The main goal of Avenix is to build a complete and reproducible ML pipeline:

**Raw Data → Data Understanding → EDA → Feature Engineering → ML Models → Evaluation → Explainability → Lead Scoring**

For each new lead, the final system should:

1. Process the lead's information.
2. Predict the probability of conversion.
3. Convert the probability into a lead score/category.
4. Explain which features influenced the prediction.

Example:

```text
New Lead
   ↓
Data Processing
   ↓
Feature Engineering
   ↓
ML Model
   ↓
Conversion Probability
   ↓
Lead Score
   ↓
High / Medium / Low
   ↓
SHAP Explanation
```

---

## 📊 Dataset

The project uses the **X Education Lead Scoring dataset** containing historical information about leads, their interactions with the website, marketing channels, activities, and other lead attributes.

The dataset contains features such as:

* Lead Origin
* Lead Source
* Total Visits
* Total Time Spent on Website
* Page Views Per Visit
* Last Activity
* Country
* Specialization
* Current Occupation
* Lead Quality
* City
* Lead Profile
* Last Notable Activity
* and other lead-related attributes

The target variable is:

```text
Converted
```

where:

```text
0 → Lead did not convert
1 → Lead converted
```

The original dataset archive is included in:

```text
data/raw/Leads X Education.csv.zip
```

> Dataset usage and redistribution should follow the terms and license of the original dataset source.

---

# 🧠 Machine Learning Problem

This project is formulated as a **binary classification problem**.

Given information about a lead:

```text
X = lead characteristics and interactions
```

we want to estimate:

```text
P(Converted = 1 | X)
```

The resulting probability can then be used for lead scoring and prioritization.

---

# 🏗️ Project Pipeline

## Phase 1 — Project Setup

* Initialize Git repository
* Create project structure
* Configure Python environment
* Install dependencies
* Add dataset
* Prepare reproducible development environment

---

## Phase 2 — Data Understanding & EDA

We investigate the dataset before training any model.

### Tasks

* Load and inspect the dataset
* Understand columns and data types
* Analyze missing values
* Detect duplicated records
* Investigate outliers
* Analyze target distribution
* Explore categorical variables
* Explore numerical variables
* Study relationships between features and conversion
* Identify possible data leakage

### Important

Some variables may contain information that is only available **after or very close to the conversion event**.

Examples that require careful investigation include:

* `Last Activity`
* `Lead Quality`
* `Lead Profile`
* `Last Notable Activity`

We must determine whether each feature would realistically be available at the moment a lead needs to be scored.

The goal is to avoid **data leakage** and build a model that can actually be used in a real-world scenario.

---

# ⚙️ Phase 3 — Feature Engineering

Raw data cannot always be used directly by ML models.

We will prepare the features through:

* Missing-value handling
* Categorical encoding
* Numerical preprocessing
* Feature transformation
* Feature selection
* New feature creation
* Train / validation / test splitting

Possible derived features may include:

* Website engagement indicators
* Visit intensity
* Time spent per visit
* Activity frequency
* Interaction-based features

Feature engineering decisions will be documented rather than added blindly.

---

# 🤖 Phase 4 — Model Development

We will establish a simple baseline before testing more advanced models.

### Baseline

* Logistic Regression

### Candidate models

* Logistic Regression
* Random Forest
* XGBoost
* LightGBM

The models will be compared using the same validation methodology.

### Hyperparameter Optimization

For promising models, we may use:

* Cross-validation
* Grid Search
* Randomized Search
* Optuna

The final model will be selected based on the project's evaluation criteria rather than simply choosing the most complex model.

---

# 📈 Phase 5 — Model Evaluation

Accuracy alone is not enough for a lead-scoring problem, especially when the target classes may be imbalanced.

We will evaluate models using:

### Classification Metrics

* ROC-AUC
* PR-AUC
* Precision
* Recall
* F1-score
* Confusion Matrix

### Business-oriented analysis

* Probability calibration
* Threshold analysis
* Lift
* Gain
* Lead ranking

The objective is to understand not only whether the model predicts correctly, but also whether its predictions are useful for prioritizing leads.

---

# 🔍 Phase 6 — Model Explainability

Avenix is designed to be explainable.

We will use **SHAP (SHapley Additive exPlanations)** to understand model predictions.

SHAP will help answer questions such as:

> Why did the model assign this lead a high conversion probability?

For example:

```text
Lead
 ↓
Conversion Probability: 0.82
 ↓
Important factors:
 + High website engagement
 + High time spent on website
 + Relevant lead source
 - Low recent activity
```

We will analyze both:

### Global Explainability

Which features are generally most important?

### Local Explainability

Why did the model make a prediction for a specific lead?

---

# 🎯 Phase 7 — Final Lead Scoring System

The final model will be saved and used to score new leads.

Example:

```text
Input Lead
    ↓
Preprocessing
    ↓
Feature Engineering
    ↓
Trained Model
    ↓
Conversion Probability
    ↓
Lead Score
```

Example output:

```text
Lead ID: 10234

Conversion Probability: 0.87
Lead Score: High
```

A possible scoring scheme:

```text
High    → high conversion probability
Medium  → moderate conversion probability
Low     → low conversion probability
```

The exact thresholds will be determined through validation and threshold analysis rather than arbitrarily.

---

# 🧪 Reproducibility

The project is structured so that another developer can clone the repository and reproduce the workflow.

```bash
git clone <repository-url>
cd lead-scoring-ml
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 📁 Project Structure

```text
lead-scoring-ml/
│
├── data/
│   ├── raw/
│   │   └── Leads X Education.csv.zip
│   │
│   └── processed/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_baseline_model.ipynb
│   ├── 05_model_training.ipynb
│   ├── 06_model_evaluation.ipynb
│   └── 07_shap_explainability.ipynb
│
├── src/
│   ├── data_processing.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│
├── reports/
│   ├── figures/
│   └── results/
│
├── predict.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🛠️ Tech Stack

### Programming

* Python

### Data Analysis

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* XGBoost
* LightGBM

### Explainability

* SHAP

### Model Persistence

* Joblib

### Development

* Jupyter Notebook
* Git
* GitHub

---

# 👥 Team Workflow

Avenix is developed as a collaborative project.

The work is divided into several areas:

### Data Science / ML

Responsible for:

* Data analysis
* Feature engineering
* Model development
* Evaluation
* Explainability

### Engineering

Responsible for:

* Project structure
* Reusable Python modules
* Training pipeline
* Prediction pipeline
* Model saving/loading

### Documentation

Responsible for:

* README
* Experiment documentation
* Results
* Visualizations
* Final presentation

Although responsibilities are divided, every team member should understand the complete ML pipeline.

---

# 📚 Project Philosophy

Avenix is not intended to be a copy-paste ML project.

For every major step we ask:

### What?

What are we doing?

### Why?

Why is this method appropriate for the problem?

### How?

How does it work technically?

### Result?

What should we expect from the result?

This approach helps us understand the complete process rather than simply producing a model.

---

# 🚀 Expected Final Result

By the end of the project, Avenix should provide:

* A clean and reproducible data pipeline
* Exploratory data analysis
* Leakage-aware feature engineering
* Multiple ML models
* Model comparison
* Cross-validation
* Hyperparameter optimization
* Robust evaluation
* Probability calibration
* Lead scoring
* SHAP-based explanations
* Saved production-ready model artifacts
* A prediction pipeline for new leads

---

# 🔮 Future Improvements

Possible future extensions include:

* Web-based lead scoring dashboard
* REST API for predictions
* Real-time lead scoring
* CRM integration
* Automated model retraining
* Monitoring model performance
* Data drift detection
* Experiment tracking
* Docker deployment
* Cloud deployment

---

# 👨‍💻 Project Status

**Status:** 🚧 In Development

Current stage:

```text
Project Setup
     ↓
Dataset Preparation
     ↓
Data Understanding
     ↓
EDA
     ↓
Feature Engineering
     ↓
Model Development
     ↓
Evaluation
     ↓
Explainability
     ↓
Final Lead Scoring System
```

---

## 📌 Avenix

**Avenix — Lead Scoring & Propensity-to-Convert ML**

An end-to-end Machine Learning project focused on turning historical lead data into actionable and explainable conversion predictions.
