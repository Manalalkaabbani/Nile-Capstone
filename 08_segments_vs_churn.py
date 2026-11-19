"""
STEP 8  -  PART 7: Put the two worlds together (Supervised + Unsupervised).

7.1  Add the segment from step 06 to customers.csv and show the churn rate per segment.
7.2  On those customers, add the segment as a feature to the churn model. Does the score improve?

Only TRAINING customers (the step-05 split) are used; the test set is not touched again.

Run:  python 08_segments_vs_churn.py          (needs steps 04 and 06)
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from common import CAT_COLS, NUM_COLS, churn_split, preprocessor
from config import CLEAN_DIR, MODEL_DIR, RANDOM_STATE, header, save_fig, show

cust, X_train, X_test, y_train, y_test = churn_split()
segments = pd.read_csv(CLEAN_DIR / "usage_segments.csv")
usage = pd.read_csv(CLEAN_DIR / "usage_clean.csv")

# ---------------------------------------------------------------------------
header("7.1  Churn rate per segment")
cust["segment"] = cust["customer_id"].map(dict(zip(segments["customer_id"], segments["segment"])))   # NaN after the first 2,000
with_seg = cust.dropna(subset=["segment"]).copy()
with_seg["churn_flag"] = (with_seg["churn"] == "Yes").astype(int)
print(f"customers with a segment: {len(with_seg):,} of {len(cust):,}")


def wilson(k, n, z=1.96):
    """95% confidence interval of a rate (small groups need it)."""
    p = k / n; d = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return centre - half, centre + half


seg = with_seg.groupby("segment")["churn_flag"].agg(customers="size", churned="sum", churn_rate="mean")
seg[["CI low", "CI high"]] = [wilson(k, n) for k, n in zip(seg["churned"], seg["customers"])]
seg = seg.sort_values("churn_rate", ascending=False)
overall = with_seg["churn_flag"].mean()
show((seg * [1, 1, 100, 100, 100]).round(1), "churn rate per segment (%)", save_as="08_churn_per_segment")
chi2, p_value, _, _ = chi2_contingency(pd.crosstab(with_seg["segment"], with_seg["churn_flag"]))
print(f"overall churn rate here: {overall:.1%} | chi-square test 'segment and churn are independent': p = {p_value:.2f}")
print("(p > 0.05 = no proof that the segments differ in churn)")

fig, ax = plt.subplots(figsize=(8, 4.3))
y_pos = np.arange(len(seg))
ax.barh(y_pos, seg["churn_rate"] * 100,
        xerr=[(seg["churn_rate"] - seg["CI low"]) * 100, (seg["CI high"] - seg["churn_rate"]) * 100], capsize=4)
ax.axvline(overall * 100, color="crimson", ls="--", label=f"Overall {overall:.1%}")
ax.set_yticks(y_pos); ax.set_yticklabels([f"{s} (n={n})" for s, n in zip(seg.index, seg["customers"])]); ax.invert_yaxis()
ax.set(title="Churn rate per usage segment (bars = 95% confidence interval)", xlabel="Churn rate (%)")
ax.legend()
plt.tight_layout()
save_fig("7_1_churn_per_segment")

# ---------------------------------------------------------------------------
header("7.2  Does the segment help the churn model?  (honest answer; 'no' is fine)")
best = json.loads((MODEL_DIR / "final_choices.json").read_text())["logistic_regression_best_params"]
print("same tuned Logistic Regression as step 04:", best)


def make_model(num_cols, cat_cols):
    return Pipeline([("prep", preprocessor(True, num_cols, cat_cols)),
                     ("clf", LogisticRegression(C=best["C"], class_weight=best["class_weight"], max_iter=2000, random_state=RANDOM_STATE))])


idx = X_train.index[cust.loc[X_train.index, "segment"].notna()]           # training customers that have a segment
usage_cols = usage.drop(columns="customer_id").columns.tolist()
usage_by_id = usage.set_index("customer_id")[usage_cols]

XA = X_train.loc[idx].copy()                                              # A: original features
XB = XA.assign(segment=cust.loc[idx, "segment"].values)                   # B: + segment
XC = XA.join(usage_by_id.reindex(cust.loc[idx, "customer_id"]).set_index(idx))   # C: + 12 raw usage columns
yy = y_train.loc[idx]
print(f"rows used: {len(idx)} | churn rate {yy.mean():.3f}")

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)
variants = {"A: original features": (XA, NUM_COLS, CAT_COLS),
            "B: + segment": (XB, NUM_COLS, CAT_COLS + ["segment"]),
            "C: + 12 raw usage columns": (XC, NUM_COLS + usage_cols, CAT_COLS)}
results, f1_folds = {}, {}
for name, (X_v, num_cols, cat_cols) in variants.items():
    r = cross_validate(make_model(num_cols, cat_cols), X_v, yy, cv=cv,
                       scoring=["f1", "roc_auc", "recall", "precision"], n_jobs=1)
    f1_folds[name] = r["test_f1"]
    results[name] = {"F1": r["test_f1"].mean(), "F1 std": r["test_f1"].std(), "ROC-AUC": r["test_roc_auc"].mean(),
                     "Recall": r["test_recall"].mean(), "Precision": r["test_precision"].mean()}
show(pd.DataFrame(results).T.round(4), "5 x 5 repeated cross-validation, training customers only", save_as="08_segment_as_feature")

for name in ["B: + segment", "C: + 12 raw usage columns"]:
    diff = f1_folds[name] - f1_folds["A: original features"]
    print(f"{name} minus A: mean F1 change {diff.mean():+.4f}; better in {(diff > 0).sum()} of {len(diff)} folds; Wilcoxon p = {wilcoxon(diff).pvalue:.2f}")

print("""
How to read it: if F1 does not go up, say so honestly and explain why:
  (1) churn is almost the same in every segment, so the segment holds no new information about leaving;
  (2) contract, tenure, late payments and bill already explain most of the churn;
  (3) with ~1,600 customers, extra columns mostly add noise.
Segments are valuable for choosing OFFERS, not for predicting churn.
(Part 8, the CEO report, is a document: see Manal_ML_Capstone_Report.pdf)""")
