"""
STEP 4  -  PART 4 (4.1, 4.2, 4.3, 4.4, 4.6): Fair comparison and tuning.

Everything here uses the TRAINING set only. At the end we choose the final model and save
the choice; the test set is opened ONE time, in step 05.

Run:  python 04_cv_and_tuning.py          (needs step 01; takes about 1 minute)
"""
import json
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from common import build_models, churn_split, money, preprocessor, stratified_cv
from config import MODEL_DIR, RANDOM_STATE, header, save_fig, show

cust, X_train, X_test, y_train, y_test = churn_split()
skf = stratified_cv()

# ---------------------------------------------------------------------------
header("4.1  5-fold Stratified Cross Validation for all 7 models (mean and std of F1)")
rows, folds = {}, {}
for name, model in build_models().items():
    r = cross_validate(model, X_train, y_train, cv=skf, scoring=["f1", "recall", "precision"], n_jobs=1)
    folds[name] = r["test_f1"]
    rows[name] = {"F1 mean": r["test_f1"].mean(), "F1 std": r["test_f1"].std(),
                  "F1 min": r["test_f1"].min(), "F1 max": r["test_f1"].max(),
                  "Recall mean": r["test_recall"].mean(), "Precision mean": r["test_precision"].mean()}
cv_table = pd.DataFrame(rows).T.sort_values("F1 mean", ascending=False)
show(cv_table.round(3), "cross-validation results", save_as="04_cv_results")
print("Most stable (lowest F1 std):", cv_table["F1 std"].idxmin(), "| Highest F1:", cv_table["F1 mean"].idxmax())

fig, ax = plt.subplots(figsize=(9, 4.5))
order = cv_table.index.tolist()
ax.boxplot([folds[n] for n in order], tick_labels=order, showmeans=True)
ax.set(title="F1 across the 5 CV folds (default settings)", ylabel="F1 (churn class)")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
save_fig("4_1_cv_boxplot")

# ---------------------------------------------------------------------------
header("4.2  GridSearchCV on a Decision Tree (4 parameters, scored with F1, cv=5)")
dt_pipe = Pipeline([("prep", preprocessor(False)), ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))])
dt_space = {
    "clf__criterion": ["gini", "entropy"],
    "clf__max_depth": [3, 4, 5, 6, 8, 10, None],
    "clf__min_samples_leaf": [1, 5, 10, 20, 40],
    "clf__min_samples_split": [2, 10, 30],
}
n_grid = int(np.prod([len(v) for v in dt_space.values()]))
t0 = time.perf_counter()
grid = GridSearchCV(dt_pipe, dt_space, scoring="f1", cv=skf, n_jobs=2).fit(X_train, y_train)
t_grid = time.perf_counter() - t0
print(f"{n_grid} combinations x 5 folds = {n_grid * 5} fits in {t_grid:.1f}s")
print("Best parameters:", {k.replace("clf__", ""): v for k, v in grid.best_params_.items()})
print("Best CV F1: %.3f   (default Decision Tree: %.3f)" % (grid.best_score_, cv_table.loc["Decision Tree", "F1 mean"]))

# ---------------------------------------------------------------------------
header("4.3  RandomizedSearchCV on XGBoost (8 parameters, n_iter = 40), timed")
xgb_pipe = Pipeline([("prep", preprocessor(False)),
                     ("clf", XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=1))])
xgb_space = {
    "clf__n_estimators": randint(100, 500),
    "clf__max_depth": randint(2, 8),
    "clf__learning_rate": loguniform(0.01, 0.3),
    "clf__subsample": uniform(0.6, 0.4),
    "clf__colsample_bytree": uniform(0.5, 0.5),
    "clf__min_child_weight": randint(1, 10),
    "clf__gamma": uniform(0, 5),
    "clf__reg_lambda": uniform(0.5, 5),
}
N_ITER = 40
t0 = time.perf_counter()
rand_xgb = RandomizedSearchCV(xgb_pipe, xgb_space, n_iter=N_ITER, scoring="f1", cv=skf,
                              random_state=RANDOM_STATE, n_jobs=2).fit(X_train, y_train)
t_rand_xgb = time.perf_counter() - t0
print(f"{N_ITER} combinations x 5 folds = {N_ITER * 5} fits in {t_rand_xgb:.1f}s")
print("Best parameters:", {k.replace("clf__", ""): (round(v, 3) if isinstance(v, float) else v) for k, v in rand_xgb.best_params_.items()})
print("Best CV F1: %.3f   (default XGBoost: %.3f)" % (rand_xgb.best_score_, cv_table.loc["XGBoost", "F1 mean"]))

# ---------------------------------------------------------------------------
header("4.4  Grid vs Randomized: time, score, number of combinations")
# For a fair comparison, Randomized search also runs on the SAME Decision Tree space as the grid.
t0 = time.perf_counter()
rand_dt = RandomizedSearchCV(dt_pipe, dt_space, n_iter=30, scoring="f1", cv=skf,
                             random_state=RANDOM_STATE, n_jobs=2).fit(X_train, y_train)
t_rand_dt = time.perf_counter() - t0
comparison = pd.DataFrame([
    ["GridSearchCV", "Decision Tree", n_grid, n_grid * 5, t_grid, grid.best_score_],
    ["RandomizedSearchCV (same space)", "Decision Tree", 30, 150, t_rand_dt, rand_dt.best_score_],
    ["RandomizedSearchCV (continuous space)", "XGBoost", N_ITER, N_ITER * 5, t_rand_xgb, rand_xgb.best_score_],
], columns=["Method", "Model", "Combinations tried", "Model fits", "Time (s)", "Best CV F1"]).set_index("Method")
show(comparison.round(3), "grid vs randomized", save_as="04_grid_vs_random")
print("""
When to use Grid      : a small discrete space, you want to be exhaustive / reproducible, or to fine-tune around a good region.
When to use Randomized: many parameters or continuous ranges, a limited time budget, the first exploration.""")

# ---------------------------------------------------------------------------
header("Tuning the front-runner (Logistic Regression) so the final comparison is fair")
lr_pipe = Pipeline([("prep", preprocessor(True)),
                    ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))])
lr_space = {"clf__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30], "clf__class_weight": [None, "balanced"]}
grid_lr = GridSearchCV(lr_pipe, lr_space, scoring="f1", cv=skf, n_jobs=2).fit(X_train, y_train)
print("Best parameters:", {k.replace("clf__", ""): v for k, v in grid_lr.best_params_.items()})
print("Best CV F1: %.3f   (default Logistic Regression: %.3f)" % (grid_lr.best_score_, cv_table.loc["Logistic Regression", "F1 mean"]))

candidates = {
    "Logistic Regression (default)": cv_table.loc["Logistic Regression", "F1 mean"],
    "Logistic Regression (tuned)": grid_lr.best_score_,
    "XGBoost (default)": cv_table.loc["XGBoost", "F1 mean"],
    "XGBoost (tuned)": rand_xgb.best_score_,
    "Decision Tree (tuned)": grid.best_score_,
    "Random Forest (default)": cv_table.loc["Random Forest", "F1 mean"],
}
cand_table = pd.Series(candidates).sort_values(ascending=False).rename("CV F1").to_frame()
show(cand_table.round(3), "candidates for the final model (cross-validation F1 only)", save_as="04_final_candidates")

# ---------------------------------------------------------------------------
header("4.6  Bonus: class_weight and the decision threshold (still on training data only)")
FINAL_NAME = cand_table.index[0]
searches = {"Logistic Regression (tuned)": grid_lr, "XGBoost (tuned)": rand_xgb, "Decision Tree (tuned)": grid}
final_model = (searches[FINAL_NAME].best_estimator_ if FINAL_NAME in searches
               else build_models()[FINAL_NAME.split(" (")[0]])
print("Final model, chosen on cross-validation only:", FINAL_NAME)

oof_proba = cross_val_predict(final_model, X_train, y_train, cv=skf, method="predict_proba")[:, 1]
rows = []
for th in np.round(np.arange(0.10, 0.76, 0.05), 2):
    hat = (oof_proba >= th).astype(int)
    rows.append({"threshold": th, "Precision": precision_score(y_train, hat), "Recall": recall_score(y_train, hat),
                 "F1": f1_score(y_train, hat), "Customers flagged %": hat.mean() * 100,
                 "Money saved (offer works 100%)": money(y_train, hat, 1.0)["Money saved"],
                 "Money saved (offer works 50%)": money(y_train, hat, 0.5)["Money saved"]})
th_table = pd.DataFrame(rows).set_index("threshold")
show(th_table.round(3), "what happens when the threshold moves (out-of-fold, 2,400 training customers)", save_as="04_thresholds")

TH_F1 = float(th_table["F1"].idxmax())
TH_MONEY = float(th_table["Money saved (offer works 100%)"].idxmax())
print(f"F1-best threshold: {TH_F1} | money-best threshold if every offer works: {TH_MONEY}")
thresholds = {"0.50 (default)": 0.5}
if abs(TH_F1 - 0.5) > 1e-9:
    thresholds[f"{TH_F1:.2f} (best F1 in CV)"] = TH_F1
thresholds[f"{TH_MONEY:.2f} (max money in CV if every offer works)"] = TH_MONEY

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
th_table[["Precision", "Recall", "F1"]].plot(ax=axes[0], marker="o")
axes[0].axvline(0.5, color="grey", ls="--", lw=1)
axes[0].set(title="Precision / Recall / F1 vs threshold", ylabel="Score")
th_table[["Money saved (offer works 100%)", "Money saved (offer works 50%)"]].plot(ax=axes[1], marker="o")
axes[1].set(title="Money saved vs threshold (2,400 training customers)", ylabel="EGP saved vs doing nothing")
plt.tight_layout()
save_fig("4_6_threshold")

# ---------------------------------------------------------------------------
# Save the choices. Step 05 will open the test set exactly one time with them.
joblib.dump(final_model, MODEL_DIR / "final_model.joblib")
choices = {
    "final_name": FINAL_NAME,
    "cv_f1_of_final": float(candidates[FINAL_NAME]),
    "thresholds": thresholds,
    "logistic_regression_best_params": {k.replace("clf__", ""): v for k, v in grid_lr.best_params_.items()},
}
(MODEL_DIR / "final_choices.json").write_text(json.dumps(choices, indent=2))
print(f"\nSaved the final model and the thresholds in {MODEL_DIR}")
print("Next: python 05_final_test.py")
