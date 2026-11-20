"""
STEP 1  -  PART 1 (all of it): EDA = Explore, audit and clean the data.   ONE file, four sections:

    Section A  Load the three files                    (1.1)
    Section B  Audit: find the problems                (1.1, 1.2)
    Section C  Clean the data and save it              (1.3)
    Section D  Explore: charts and the churn column    (1.4, 1.5)

Run:  python 01_eda.py          (about 15 seconds; needs only the 3 CSV files)
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import CLEAN_DIR, DATA_CAP_GB, RAW_DIR, header, save_fig, show

# ============================================================================
# SECTION A  -  LOAD THE THREE FILES                                    (1.1)
# ============================================================================
raw_customers = pd.read_csv(RAW_DIR / "customers.csv")
raw_usage = pd.read_csv(RAW_DIR / "usage_behavior.csv")
raw_transactions = pd.read_csv(RAW_DIR / "service_transactions.csv")

for name, df in [("customers.csv", raw_customers), ("usage_behavior.csv", raw_usage),
                 ("service_transactions.csv", raw_transactions)]:
    header(f"A. {name}: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print("-- data types:", df.dtypes.astype(str).to_dict())
    missing = df.isna().sum()
    print("-- missing values per column:", missing[missing > 0].to_dict() or "none")
    print("-- exact duplicate rows:", int(df.duplicated().sum()))
    show(df.describe(include="all").T.round(2), "basic statistics")
# Write 3-5 lines on what you see (Part 1.1).

# ============================================================================
# SECTION B  -  AUDIT: FIND THE PROBLEMS (change nothing yet)           (1.2)
# ============================================================================
header("B. Hint from the brief: value_counts() of every text column")
for col in raw_customers.select_dtypes(exclude="number").columns.drop("customer_id"):
    print(f"\n{col}: {raw_customers[col].value_counts(dropna=False).to_dict()}")

header("B. Hint from the brief: the min / max of every number column")
show(raw_customers.select_dtypes("number").agg(["min", "max"]).T, "customers.csv")
show(raw_usage.drop(columns="customer_id").agg(["min", "max"]).T, "usage_behavior.csv")

CANONICAL_CONTRACTS = {"Monthly", "1-Year", "2-Year"}
basket_size = raw_transactions.groupby("transaction_id").size()
audit = pd.DataFrame([
    ["Duplicate rows (customer repeated)", "customers / all columns", int(raw_customers.duplicated().sum()), "Drop exact duplicates, keep first"],
    ["Missing values", "customers / age", int(raw_customers.age.isna().sum()), "Median (inside the Pipeline, train only)"],
    ["Missing values", "customers / data_usage_gb", int(raw_customers.data_usage_gb.isna().sum()), "Median (inside the Pipeline, train only)"],
    ["Missing values", "customers / voice_minutes", int(raw_customers.voice_minutes.isna().sum()), "Median (inside the Pipeline, train only)"],
    ["Missing values", "customers / payment_method", int(raw_customers.payment_method.isna().sum()), "Mode (inside the Pipeline, train only)"],
    ["Missing values", "usage_behavior / social_media_hours_week", int(raw_usage.social_media_hours_week.isna().sum()), "Median (no split needed: unsupervised)"],
    ["Text written in different ways (trailing space)", "customers / gender", int((raw_customers.gender != raw_customers.gender.str.strip()).sum()), "strip() whitespace"],
    ["Text written in different ways (capital letters)", "customers / contract_type", int((~raw_customers.contract_type.isin(CANONICAL_CONTRACTS)).sum()), "Standardise to Monthly / 1-Year / 2-Year"],
    [f"Extreme / impossible values (> {DATA_CAP_GB} GB)", "customers / data_usage_gb", int((raw_customers.data_usage_gb > DATA_CAP_GB).sum()), f"Cap at {DATA_CAP_GB} GB"],
    ["Extreme values (> 30 calls per day)", "usage_behavior / calls_per_day", int((raw_usage.calls_per_day > 30).sum()), "Keep and flag. Look at them again in DBSCAN (step 06)"],
    ["Zero data use (voice-only customers?)", "usage_behavior / avg_daily_data_mb", int((raw_usage.avg_daily_data_mb == 0).sum()), "Keep: a real behaviour, not an error"],
    ["One-item baskets (cannot make a rule)", "service_transactions / transaction_id", int((basket_size == 1).sum()), "Keep: they still count in the support"],
], columns=["Problem found", "File / column", "Rows", "How it is fixed"])
show(audit, "B. DATA AUDIT TABLE (Part 1.2)", save_as="01_data_audit")

# ============================================================================
# SECTION C  -  CLEAN THE DATA                                          (1.3)
# ============================================================================
# A different rule for each type of problem (drop / fix text / cap / median / mode).
#
# DATA-LEAKAGE RULE: anything that LEARNS from the data (median, mode, scaler, encoder) must be fitted
# on the training set only. So here we do only fixes that learn nothing: duplicates, text, a fixed cap.
# The missing values of customers.csv stay as they are and are filled later INSIDE the Pipeline
# (steps 02 - 05). usage_behavior.csv has no train/test split, so its few gaps are filled once here.
header("C. Cleaning customers.csv")
customers = raw_customers.copy()
customers["gender"] = customers["gender"].str.strip()                                  # 'Male ' -> 'Male'
customers["contract_type"] = customers["contract_type"].str.strip().str.title()        # 'monthly' -> 'Monthly'
before = len(customers)
customers = customers.drop_duplicates().reset_index(drop=True)                         # drop duplicates
print(f"duplicates dropped: {before - len(customers)} -> {len(customers):,} customers, unique ids: {customers.customer_id.is_unique}")
n_capped = int((customers.data_usage_gb > DATA_CAP_GB).sum())
customers["data_usage_gb"] = customers["data_usage_gb"].clip(upper=DATA_CAP_GB)        # cap (a fixed rule)
print(f"data_usage_gb capped at {DATA_CAP_GB} GB: {n_capped} rows changed, max now {customers.data_usage_gb.max()}")
print("gender values       :", sorted(customers.gender.unique()))
print("contract_type values:", sorted(customers.contract_type.unique()))
print("missing values left (filled later inside the Pipeline):",
      customers.isna().sum()[customers.isna().sum() > 0].to_dict())

header("C. Cleaning usage_behavior.csv")
usage = raw_usage.copy()
median_social = usage["social_media_hours_week"].median()
usage["social_media_hours_week"] = usage["social_media_hours_week"].fillna(median_social)
print(f"missing social_media_hours_week filled with the median ({median_social}); missing now: {int(usage.isna().sum().sum())}")
print("calls_per_day > 30 and zero-data customers are KEPT: real behaviour (DBSCAN will show them).")

header("C. service_transactions.csv")
transactions = raw_transactions.copy()
print("no missing values, no duplicates: kept as it is (one-item baskets stay: they count in the support).")

customers.to_csv(CLEAN_DIR / "customers_clean.csv", index=False)
usage.to_csv(CLEAN_DIR / "usage_clean.csv", index=False)
transactions.to_csv(CLEAN_DIR / "transactions_clean.csv", index=False)
print(f"\nSaved the 3 cleaned files in {CLEAN_DIR}")
print("""
Why each rule (write one sentence each in your report):
  duplicates -> drop      : the same customer counted twice gets double weight and can sit in train AND test.
  gender text -> strip    : 'Male ' and 'Male' are the same category.
  contract -> title case  : the same contract written in two ways.
  age, data_usage_gb, voice_minutes (missing) -> median, in the Pipeline : not pulled by heavy users; learned from train only.
  payment_method (missing) -> mode, in the Pipeline : a category has no median.
  data_usage_gb up to 1,288 GB -> cap : keeps the customer but stops giant values from dragging the regression line.
  social_media_hours_week (missing) -> median : tiny gap, no train/test split in this file.""")

# ============================================================================
# SECTION D  -  EXPLORE: 6 CHARTS + THE CHURN COLUMN                   (1.4, 1.5)
# ============================================================================
# For the charts only: a filled-in copy. (The models fill gaps inside the Pipeline instead.)
eda = customers.copy()
for col in ["age", "data_usage_gb", "voice_minutes"]:
    eda[col] = eda[col].fillna(eda[col].median())
eda["payment_method"] = eda["payment_method"].fillna(eda["payment_method"].mode()[0])
eda["churn_flag"] = (eda["churn"] == "Yes").astype(int)
eda["tenure_group"] = pd.cut(eda["tenure_months"], bins=[0, 6, 12, 24, 48, 72],
                             labels=["0-6", "7-12", "13-24", "25-48", "49-72"])
# Under every chart, write ONE business sentence in your report.

header("D. Chart 1: churn rate by contract type (required)")
fig, ax = plt.subplots(figsize=(6, 4))
r = eda.groupby("contract_type")["churn_flag"].mean().reindex(["Monthly", "1-Year", "2-Year"]) * 100
sns.barplot(x=r.index, y=r.values, ax=ax)
for i, v in enumerate(r.values):
    ax.text(i, v + 0.5, f"{v:.1f}%", ha="center")
ax.set(title="Churn rate by contract type", xlabel="Contract type", ylabel="Churn rate (%)")
save_fig("1_4_churn_by_contract"); print(r.round(1).to_dict())

header("D. Chart 2: churn rate by tenure group (required)")
fig, ax = plt.subplots(figsize=(6, 4))
r = eda.groupby("tenure_group", observed=True)["churn_flag"].mean() * 100
sns.barplot(x=r.index.astype(str), y=r.values, ax=ax)
for i, v in enumerate(r.values):
    ax.text(i, v + 0.5, f"{v:.1f}%", ha="center")
ax.set(title="Churn rate by tenure group", xlabel="Tenure (months)", ylabel="Churn rate (%)")
save_fig("1_4_churn_by_tenure"); print(r.round(1).to_dict())

header("D. Chart 3: correlation heatmap of the numeric columns (required)")
num_cols = eda.select_dtypes("number").columns.drop("churn_flag").tolist() + ["churn_flag"]
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(eda[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 8})
ax.set_title("Correlation heatmap of numeric columns (churn_flag: 1 = left)")
save_fig("1_4_correlation_heatmap")
print(eda[num_cols].corr()["churn_flag"].drop("churn_flag").sort_values().round(2).to_dict())

header("D. Chart 4: churn rate by support calls and by late payments")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, col, title in zip(axes, ["support_calls", "late_payments"], ["Support calls", "Late payments"]):
    r = eda.groupby(col)["churn_flag"].agg(["mean", "size"])
    r = r[r["size"] >= 30]                          # hide groups that are too small to trust
    sns.barplot(x=r.index, y=r["mean"] * 100, ax=ax)
    ax.set(title=f"Churn rate by {title.lower()}", xlabel=title, ylabel="Churn rate (%)")
plt.tight_layout()
save_fig("1_4_churn_support_late")
print(eda.groupby("support_calls")["churn_flag"].mean().round(3).to_dict())
print(eda.groupby("late_payments")["churn_flag"].mean().round(3).to_dict())

header("D. Chart 5: monthly bill by plan, and churn rate by plan")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.boxplot(data=eda, x="plan", y="monthly_bill_egp", order=["Basic", "Standard", "Premium"], ax=axes[0])
axes[0].set(title="Monthly bill by plan", xlabel="Plan", ylabel="Monthly bill (EGP)")
r = eda.groupby("plan")["churn_flag"].mean().reindex(["Basic", "Standard", "Premium"]) * 100
sns.barplot(x=r.index, y=r.values, ax=axes[1])
axes[1].set(title="Churn rate by plan", xlabel="Plan", ylabel="Churn rate (%)")
plt.tight_layout()
save_fig("1_4_bill_and_churn_by_plan")
print(eda.groupby("plan")["monthly_bill_egp"].median().round(0).to_dict(), r.round(1).to_dict())

header("D. 1.5  The churn column")
churn_rate = (customers["churn"] == "Yes").mean()
print(f"Customers who left: {churn_rate:.1%}  ({(customers.churn == 'Yes').sum():,} of {len(customers):,})")
print(f"A model that always says 'nobody churns' gets {1 - churn_rate:.1%} accuracy and finds ZERO leavers.")
print("=> accuracy is dangerous; judge models with precision / recall / F1 of the churn class.")
print("\nNext: python 02_regression.py")
