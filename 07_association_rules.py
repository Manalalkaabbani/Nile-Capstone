"""
STEP 7  -  PART 6: Which services go together? (Association Rules / Apriori).  File: service_transactions.csv

Run:  python 07_association_rules.py          (needs step 01)
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules

from config import CLEAN_DIR, TABLE_DIR, header, save_fig, show

tx = pd.read_csv(CLEAN_DIR / "transactions_clean.csv")

# ---------------------------------------------------------------------------
header("6.1  Reshape into a basket table (one row per transaction, one True/False column per service)")
basket = pd.crosstab(tx["transaction_id"], tx["service"]).astype(bool)     # counts -> True / False (mlxtend wants bool)
print("basket table:", basket.shape, "| dtypes:", set(basket.dtypes.astype(str)))
size = basket.sum(axis=1)
print("services per basket: mean %.2f, max %d, one-item baskets: %d" % (size.mean(), size.max(), (size == 1).sum()))
show(basket.head(3), "first 3 baskets")

# ---------------------------------------------------------------------------
header("6.2  The 5 most popular services (support)")
support = basket.mean().sort_values(ascending=False)
top5 = support.head(5)
fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(x=top5.values * 100, y=top5.index, ax=ax, color=sns.color_palette("deep")[0])
for i, v in enumerate(top5.values):
    ax.text(v * 100 + 0.5, i, f"{v:.1%}", va="center")
ax.set(title="Top 5 services by support (share of baskets that contain them)", xlabel="Support (% of 4,500 baskets)", ylabel="")
plt.tight_layout()
save_fig("6_2_top5_services")
print((support * 100).round(1).to_dict())

# ---------------------------------------------------------------------------
header("6.3  Apriori (min_support = 0.03), rules (min_confidence = 0.5), sorted by lift")
frequent = apriori(basket, min_support=0.03, use_colnames=True)
rules = association_rules(frequent, metric="confidence", min_threshold=0.5)
rules["if customer has"] = rules["antecedents"].map(lambda s: " + ".join(sorted(s)))
rules["then likely to buy"] = rules["consequents"].map(lambda s: " + ".join(sorted(s)))
rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
print(f"{len(frequent)} frequent itemsets -> {len(rules)} rules")
cols = ["if customer has", "then likely to buy", "support", "confidence", "lift"]
show(rules[cols].head(20).round(3), "top 20 rules by lift")
rules[cols].round(4).to_csv(TABLE_DIR / "07_all_rules.csv", index=False)

# ---------------------------------------------------------------------------
header("6.4  Support, Confidence and Lift on ONE real rule: Intl_Calling -> Roaming_Pack")
n = len(basket)
p_a, p_c = basket["Intl_Calling"].mean(), basket["Roaming_Pack"].mean()
p_both = (basket["Intl_Calling"] & basket["Roaming_Pack"]).mean()
print(f"Support    = P(both)            = {p_both:.3f}   ({int(p_both * n)} of {n} baskets)")
print(f"Confidence = P(both) / P(Intl)  = {p_both:.3f} / {p_a:.3f} = {p_both / p_a:.3f}")
print(f"Lift       = Confidence / P(Roaming) = {p_both / p_a:.3f} / {p_c:.3f} = {p_both / p_a / p_c:.2f}")

one_to_one = (rules["antecedents"].map(len) == 1) & (rules["consequents"].map(len) == 1)
show(rules[(rules["lift"] < 1.25) & one_to_one].sort_values("confidence", ascending=False)[cols].head(5).round(3),
     "high confidence but lift near 1 = NOT interesting (the second service is just popular)")
show(rules[(rules["then likely to buy"] == "Home_Internet") & (rules["antecedents"].map(len) == 1)].sort_values("confidence", ascending=False)[cols].head(3).round(3),
     "trap: a router upgrade needs home internet, so this is a dependency, not a cross-sell")

# ---------------------------------------------------------------------------
header("6.5  The 3 best (distinct) rules and who should get the offer")


def pick(a, c):
    return rules[(rules["if customer has"] == a) & (rules["then likely to buy"] == c)].iloc[0]


best3 = pd.DataFrame([pick("Intl_Calling", "Roaming_Pack"), pick("Device_Installment", "Antivirus"),
                      pick("Cloud_Storage + Home_Internet", "Streaming_Bundle")])[cols]
show(best3.round(3), "best 3 rules", save_as="07_best3_rules")
reach = {
    "Intl_Calling -> Roaming_Pack": int((basket["Intl_Calling"] & ~basket["Roaming_Pack"]).sum()),
    "Device_Installment -> Antivirus": int((basket["Device_Installment"] & ~basket["Antivirus"]).sum()),
    "Cloud_Storage + Home_Internet -> Streaming_Bundle": int((basket["Cloud_Storage"] & basket["Home_Internet"] & ~basket["Streaming_Bundle"]).sum()),
}
print("\nBaskets that have the 'if' part but not the 'then' service yet (the offer audience):", reach)
print("""
Offers (write your own wording):
  1. Intl_Calling -> Roaming_Pack            : 'Traveller pack': 20% off Roaming with International Calling. Send to: Intl Calling buyers without Roaming; the Global Talkers segment.
  2. Device_Installment -> Antivirus         : 'Protected phone': 3 months free Antivirus with every phone instalment plan. Send to: instalment customers without Antivirus (show at checkout).
  3. Cloud_Storage + Home_Internet -> Streaming_Bundle : 'Home Entertainment': Streaming Bundle at 25% off for 3 months. Send to: home-internet + cloud-storage customers without Streaming.
Next: python 08_segments_vs_churn.py""")
