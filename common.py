"""
common.py  -  code that several steps reuse (data loading, the train/test split,
the preprocessing Pipeline, the 7 models, the metrics and the money calculation).
"""
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier

from config import CLEAN_DIR, RANDOM_STATE, LOSS_EGP, OFFER_EGP

# Columns of customers.csv (customer_id and the targets are NOT features)
NUM_COLS = ["age", "tenure_months", "data_usage_gb", "voice_minutes", "sms_count",
            "num_services", "support_calls", "late_payments", "monthly_bill_egp"]
CAT_COLS = ["gender", "governorate", "contract_type", "plan", "payment_method",
            "device_type", "has_home_internet"]


def load_clean_customers():
    """customers_clean.csv written by step 01 (missing values are still there on purpose:
    they are filled inside the Pipeline, using the training rows only)."""
    return pd.read_csv(CLEAN_DIR / "customers_clean.csv")


def churn_split():
    """Part 3.1: X / y and the 80/20 stratified split. The test set stays locked until step 05."""
    cust = load_clean_customers()
    y = (cust["churn"] == "Yes").astype(int)
    X = cust.drop(columns=["customer_id", "churn"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    return cust, X_train, X_test, y_train, y_test


def preprocessor(scale, num_cols=None, cat_cols=None):
    """Impute + (optionally) scale numbers, impute + one-hot encode text.
    Used inside a Pipeline, so everything is learned from the training rows only."""
    num_cols = NUM_COLS if num_cols is None else num_cols
    cat_cols = CAT_COLS if cat_cols is None else cat_cols
    num_steps = [("imp", SimpleImputer(strategy="median"))] + ([("sc", StandardScaler())] if scale else [])
    return ColumnTransformer([
        ("num", Pipeline(num_steps), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
    ], sparse_threshold=0)


def build_models():
    """The 7 models of Part 3 (default settings). Scaling only where the algorithm needs it."""
    return {
        "Logistic Regression": Pipeline([("prep", preprocessor(True)),  ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))]),
        "Decision Tree":       Pipeline([("prep", preprocessor(False)), ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))]),
        "SVM":                 Pipeline([("prep", preprocessor(True)),  ("clf", SVC(random_state=RANDOM_STATE))]),
        "KNN":                 Pipeline([("prep", preprocessor(True)),  ("clf", KNeighborsClassifier())]),
        "Naive Bayes":         Pipeline([("prep", preprocessor(False)), ("clf", GaussianNB())]),
        "Random Forest":       Pipeline([("prep", preprocessor(False)), ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))]),
        "XGBoost":             Pipeline([("prep", preprocessor(False)), ("clf", XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=1))]),
    }


def stratified_cv():
    """5-fold stratified cross-validation, the same folds everywhere."""
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def clf_scores(y_true, y_hat):
    """Accuracy + precision / recall / F1 of the churn class."""
    return {"Accuracy": accuracy_score(y_true, y_hat),
            "Precision": precision_score(y_true, y_hat, zero_division=0),
            "Recall": recall_score(y_true, y_hat, zero_division=0),
            "F1": f1_score(y_true, y_hat, zero_division=0)}


def money(y_true, y_hat, success=1.0):
    """Part 3.5 money impact.
    Cost with model    = 200 x (TP + FP) + 1,500 x FN   (TP: offer sent, customer stays if the offer works)
    Cost with no model = 1,500 x (TP + FN)               (every leaver is lost)
    success = share of offers that really keep the customer (1.0 = the brief's assumption)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    cost_model = OFFER_EGP * (tp + fp) + LOSS_EGP * (fn + (1 - success) * tp)
    cost_none = LOSS_EGP * (tp + fn)
    return {"TP": tp, "FP": fp, "FN": fn, "Cost with model": cost_model,
            "Cost with no model": cost_none, "Money saved": cost_none - cost_model}
