"""
STEP 3  -  PART 3: Who will leave? (Classification). Target: churn

* The test set is NOT touched here (it is opened once, in step 05).
* To judge the 7 models fairly we use 5-fold OUT-OF-FOLD predictions on the TRAINING set:
  every training customer is predicted by a model that never saw that customer.

Run:  python 03_churn_models.py          (needs step 01)
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import cross_val_predict, train_test_split

from common import CAT_COLS, build_models, churn_split, clf_scores, money, stratified_cv
from config import LOSS_EGP, OFFER_EGP, RANDOM_STATE, header, save_fig, show

# ---------------------------------------------------------------------------
header("3.1  Prepare the data")
cust, X_train, X_test, y_train, y_test = churn_split()
print("train:", X_train.shape, "| test:", X_test.shape, "(locked until step 05)")
print("churn rate  train: %.3f | test: %.3f" % (y_train.mean(), y_test.mean()))
skf = stratified_cv()

# ---------------------------------------------------------------------------
header("3.2  Train the 7 models with default settings (out-of-fold predictions)")
oof = {name: cross_val_predict(model, X_train, y_train, cv=skf) for name, model in build_models().items()}
print("done:", list(oof))

# ---------------------------------------------------------------------------
header("3.3  Accuracy, Precision, Recall, F1 (churn class), sorted by F1")
rows = {name: clf_scores(y_train, p) for name, p in oof.items()}
rows["Baseline: nobody churns"] = clf_scores(y_train, np.zeros(len(y_train), dtype=int))
scores = pd.DataFrame(rows).T.sort_values("F1", ascending=False).round(3)
show(scores, "all models vs the 'nobody churns' baseline", save_as="03_model_scores")
ranking = scores.drop(index="Baseline: nobody churns").index.tolist()

# ---------------------------------------------------------------------------
header("3.4  Confusion matrices of the best 2 models")
best2 = ranking[:2]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, name in zip(axes, best2):
    ConfusionMatrixDisplay(confusion_matrix(y_train, oof[name]), display_labels=["Stayed", "Churned"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name}\n(out-of-fold, training set)")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
save_fig("3_4_confusion_matrices")
for name in best2:
    tn, fp, fn, tp = confusion_matrix(y_train, oof[name]).ravel()
    print(f"{name}: TN={tn} FP={fp} FN={fn} TP={tp}")
print("False Negative = a leaver we missed (costs 1,500 EGP).  False Positive = a wasted offer (costs 200 EGP).")

# ---------------------------------------------------------------------------
header(f"3.5  Business cost (loss = {LOSS_EGP} EGP, offer = {OFFER_EGP} EGP)")
cost_all = pd.DataFrame({n: money(y_train, oof[n]) for n in oof}).T.sort_values("Money saved", ascending=False)
as_int = lambda d: d.astype({"Cost with model": int, "Cost with no model": int, "Money saved": int})
show(as_int(cost_all), "offer works 100% of the time (the brief's assumption), 2,400 training customers", save_as="03_money_impact")
top3 = ranking[:3]
cost_half = pd.DataFrame({n: money(y_train, oof[n], success=0.5) for n in top3}).T
show(as_int(cost_half), "offer works for only 50% of the true leavers (top-3 models)")

n_tr, n_leave = len(y_train), int(y_train.sum())
best_money = cost_all.index[0]
strategies = pd.DataFrame({
    "Do nothing": [LOSS_EGP * n_leave] * 2,
    "Offer to everyone": [OFFER_EGP * n_tr + LOSS_EGP * (1 - s) * n_leave for s in (1.0, 0.5)],
    f"Use the model ({best_money})": [money(y_train, oof[best_money], s)["Cost with model"] for s in (1.0, 0.5)],
}, index=["Offer works 100%", "Offer works 50%"]).astype(int)
show(strategies, "three strategies side by side (EGP, lower is better)")
print(f"\nBest-F1 model: {ranking[0]}  |  best-money model: {best_money}  -> same model? {ranking[0] == best_money}")

# ---------------------------------------------------------------------------
header("3.6  Feature importance: Random Forest vs XGBoost")


def grouped_importance(pipe):
    """Add the one-hot columns of each text column back together."""
    names = pipe.named_steps["prep"].get_feature_names_out()
    out = {}
    for n, v in zip(names, pipe.named_steps["clf"].feature_importances_):
        n = n.split("__", 1)[1]
        key = next((c for c in CAT_COLS if n.startswith(c + "_")), n)
        out[key] = out.get(key, 0) + v
    return pd.Series(out)


models = build_models()
rf = models["Random Forest"].fit(X_train, y_train)
xgb = models["XGBoost"].fit(X_train, y_train)
imp = pd.DataFrame({"Random Forest": grouped_importance(rf), "XGBoost": grouped_importance(xgb)})
imp = imp.div(imp.sum()).sort_values("Random Forest", ascending=False)
show((imp * 100).round(1).head(10), "importance (%)", save_as="03_feature_importance")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, col in zip(axes, imp.columns):
    s = imp[col].sort_values().tail(10)
    ax.barh(s.index, s.values * 100)
    ax.set(title=f"{col}: top 10 features", xlabel="Importance (%)")
plt.tight_layout()
save_fig("3_6_feature_importance")
print("Top-5 RF :", imp["Random Forest"].nlargest(5).index.tolist())
print("Top-5 XGB:", imp["XGBoost"].nlargest(5).index.tolist())
print("Do they agree? overlap in top 5:", len(set(imp["Random Forest"].nlargest(5).index) & set(imp["XGBoost"].nlargest(5).index)),
      "of 5 | rank correlation: %.2f" % spearmanr(imp["Random Forest"], imp["XGBoost"])[0])

# Tie-breaker: permutation importance (computed inside the TRAINING set only)
Xa, Xv, ya, yv = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, random_state=RANDOM_STATE)
perm = {}
for name in ["Random Forest", "XGBoost"]:
    m = build_models()[name].fit(Xa, ya)
    r = permutation_importance(m, Xv, yv, scoring="f1", n_repeats=15, random_state=RANDOM_STATE, n_jobs=1)
    perm[name] = pd.Series(r.importances_mean, index=Xv.columns)
perm = pd.DataFrame(perm)
perm["Average"] = perm.mean(axis=1)
perm = perm.sort_values("Average", ascending=False)
show(perm.head(10).round(3), "permutation importance (drop in F1 when a column is shuffled)", save_as="03_permutation_importance")

# Direction of the main drivers, to write the 'top 5 reasons' in simple words
d = cust.assign(churn_flag=(cust["churn"] == "Yes").astype(int))
print("\nChurn rate by contract :", (d.groupby("contract_type").churn_flag.mean() * 100).round(1).to_dict())
print("Churn rate tenure <=6 vs >24 months:", round(d[d.tenure_months <= 6].churn_flag.mean() * 100, 1), "vs", round(d[d.tenure_months > 24].churn_flag.mean() * 100, 1))
print("Churn rate by late payments 0/1/2/3+:", [round(d[d.late_payments == k].churn_flag.mean() * 100, 1) for k in (0, 1, 2)] + [round(d[d.late_payments >= 3].churn_flag.mean() * 100, 1)])
print("Churn rate by plan     :", (d.groupby("plan").churn_flag.mean() * 100).round(1).to_dict())
print("Churn rate by support calls 0-1 / 2-3 / 4+:", [round(d[d.support_calls.between(a, b)].churn_flag.mean() * 100, 1) for a, b in ((0, 1), (2, 3), (4, 9))])
print("\nNext: python 04_cv_and_tuning.py")
