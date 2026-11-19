"""
STEP 2  -  PART 2: Predict the monthly bill (Regression).  Target: monthly_bill_egp

customer_id (an ID) and churn (the other target) are NOT used as features.

Run:  python 02_regression.py          (needs step 01)
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import load_clean_customers
from config import DATA_CAP_GB, RANDOM_STATE, header, save_fig, show

cust = load_clean_customers()
y = cust["monthly_bill_egp"]
X = cust.drop(columns=["customer_id", "monthly_bill_egp", "churn"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)   # 80/20
print("train:", X_train.shape, "| test:", X_test.shape)

# ---------------------------------------------------------------------------
header("2.1  One variable: data_usage_gb only")
simple = Pipeline([("imp", SimpleImputer(strategy="median")), ("lr", LinearRegression())])
simple.fit(X_train[["data_usage_gb"]], y_train)
slope, intercept = simple.named_steps["lr"].coef_[0], simple.named_steps["lr"].intercept_
print(f"bill = {intercept:.1f} + {slope:.2f} x data_usage_gb")
print(f"=> every extra GB goes with about {slope:.1f} EGP more on the monthly bill (careful: this mixes the effect of the plan).")

grid = pd.DataFrame({"data_usage_gb": np.linspace(0, DATA_CAP_GB, 100)})
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(X_train["data_usage_gb"], y_train, s=10, alpha=0.35, label="Customers (training set)")
ax.plot(grid["data_usage_gb"], simple.predict(grid), color="crimson", lw=2.5, label="Linear fit")
ax.set(title="Monthly bill vs data usage (one-variable model)", xlabel="Data usage (GB per month)", ylabel="Monthly bill (EGP)")
ax.legend()
save_fig("2_1_simple_regression")

# ---------------------------------------------------------------------------
header("2.2  Multiple variables (text columns are encoded)")
num_feats = X.select_dtypes("number").columns.tolist()
cat_feats = X.select_dtypes(exclude="number").columns.tolist()
print("numeric:", num_feats, "\ncategorical:", cat_feats)


def make_model(scale):
    num_steps = [("imp", SimpleImputer(strategy="median"))] + ([("sc", StandardScaler())] if scale else [])
    prep = ColumnTransformer([
        ("num", Pipeline(num_steps), num_feats),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", drop="first"))]), cat_feats)])
    return Pipeline([("prep", prep), ("lr", LinearRegression())])


multi = make_model(scale=True).fit(X_train, y_train)       # for comparing the size of the effects
multi_raw = make_model(scale=False).fit(X_train, y_train)  # same predictions, coefficients in real units

# ---------------------------------------------------------------------------
header("2.3  R2, RMSE and MAE on the test set (both models in one table)")


def metrics(name, pred):
    return {"Model": name, "R2": r2_score(y_test, pred),
            "RMSE (EGP)": np.sqrt(mean_squared_error(y_test, pred)), "MAE (EGP)": mean_absolute_error(y_test, pred)}


table = pd.DataFrame([metrics("Simple: data_usage_gb only", simple.predict(X_test[["data_usage_gb"]])),
                      metrics("Multiple: all features", multi.predict(X_test))]).set_index("Model").round(3)
show(table, "bill models", save_as="02_bill_model_metrics")
print("Average bill in the test set (EGP):", round(y_test.mean(), 1))

print("""
2.4  For a non-technical manager (write it in your own words):
  R2   = the share of the differences between customers' bills that the model explains.
  RMSE = a typical size of our mistake in EGP; big misses count extra.
  MAE  = the plain average size of our mistakes in EGP. MAE is better when a few very large bills
         should not dominate the score (some bills reach 1,600 EGP while the median is 371 EGP).""")

# ---------------------------------------------------------------------------
header("2.5  Coefficients and residuals")
names = [n.split("__", 1)[1] for n in multi.named_steps["prep"].get_feature_names_out()]
coef = pd.DataFrame({"feature": names,
                     "effect per +1 SD (EGP)": multi.named_steps["lr"].coef_,
                     "effect per +1 unit (EGP)": multi_raw.named_steps["lr"].coef_})
coef = coef.reindex(coef["effect per +1 SD (EGP)"].abs().sort_values(ascending=False).index).reset_index(drop=True)
show(coef.head(12).round(2), "top coefficients (the first 3 = the features that move the bill most)", save_as="02_coefficients")

fig, ax = plt.subplots(figsize=(8, 5))
top = coef.head(10).iloc[::-1]
ax.barh(top["feature"], top["effect per +1 SD (EGP)"],
        color=["seagreen" if v > 0 else "indianred" for v in top["effect per +1 SD (EGP)"]])
ax.set(title="Top 10 coefficients (standardised): effect on the bill", xlabel="EGP change in bill per +1 SD of the feature")
plt.tight_layout()
save_fig("2_5_coefficients")

pred = multi.predict(X_test)
resid = y_test - pred
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].scatter(pred, resid, s=10, alpha=0.4)
axes[0].axhline(0, color="crimson")
axes[0].set(title="Residuals vs predicted bill", xlabel="Predicted bill (EGP)", ylabel="Residual = actual - predicted (EGP)")
sns.histplot(resid, bins=40, kde=True, ax=axes[1])
axes[1].set(title="Distribution of residuals", xlabel="Residual (EGP)")
plt.tight_layout()
save_fig("2_5_residuals")
print("Residual spread (std) by predicted-bill quartile:",
      pd.Series(resid.values).groupby(pd.qcut(pred, 4)).std().round(1).to_dict())
print("Do you see a pattern? Look for a curve (bad fit) or a funnel (errors grow with the bill).")
print("\nNext: python 03_churn_models.py")
