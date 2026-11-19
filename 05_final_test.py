"""
STEP 5  -  PART 4.5: Choose one final model and evaluate it on the untouched TEST set - ONE time.

The model and the thresholds were fixed in step 04 using training data only.
Here predict_proba is called on the test set ONE time; all numbers below come from that single call.
If you change something after seeing these numbers, the test set is no longer a test.

Run:  python 05_final_test.py          (needs step 04)
"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from common import churn_split, clf_scores, money
from config import MODEL_DIR, header, save_fig, show

cust, X_train, X_test, y_train, y_test = churn_split()
choices = json.loads((MODEL_DIR / "final_choices.json").read_text())
final_model = joblib.load(MODEL_DIR / "final_model.joblib")
name = choices["final_name"]
print("Final model:", name)

final_model.fit(X_train, y_train)                         # train on the full training set
test_proba = final_model.predict_proba(X_test)[:, 1]      # <- the ONLY look at the test set

rows, matrices = {}, {}
for label, th in choices["thresholds"].items():
    hat = (test_proba >= th).astype(int)
    rows["threshold " + label] = clf_scores(y_test, hat)
    matrices["threshold " + label] = confusion_matrix(y_test, hat)
rows["Cross-validation F1 (mean of 5 folds)"] = {"Accuracy": np.nan, "Precision": np.nan, "Recall": np.nan, "F1": choices["cv_f1_of_final"]}
rows["Baseline: nobody churns"] = clf_scores(y_test, np.zeros(len(y_test), dtype=int))

header("4.5  TEST-SET RESULT (used once)")
show(pd.DataFrame(rows).T.round(3), "all four metrics, compared with the cross-validation score and the baseline", save_as="05_final_test_result")

fig, axes = plt.subplots(1, len(matrices), figsize=(5.5 * len(matrices), 4.3))
for ax, (label, cm) in zip(np.atleast_1d(axes), matrices.items()):
    ConfusionMatrixDisplay(cm, display_labels=["Stayed", "Churned"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name}\n{label} (test set)", fontsize=10)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
save_fig("4_5_test_confusion_matrix")

header("Money impact on the test set")
for label, th in choices["thresholds"].items():
    hat = (test_proba >= th).astype(int)
    for success in (1.0, 0.5):
        m = money(y_test, hat, success)
        print(f"threshold {label} | offer works {int(success * 100)}%: flagged {hat.sum()} of {len(hat)}, "
              f"cost {int(m['Cost with model']):,} vs {int(m['Cost with no model']):,} doing nothing -> saved {int(m['Money saved']):,} EGP")

print("""
Questions to answer in your report:
  * Are the test F1 and the cross-validation F1 close?  If not, what could be wrong? (leakage, a small test set, bad luck)
  * Did we beat the 74% 'nobody churns' baseline AND find the leavers (recall)?
Next: python 06_clustering_pca.py""")
