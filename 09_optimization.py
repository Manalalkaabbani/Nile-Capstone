"""
STEP 9  -  OPERATIONS RESEARCH: who should get the 200 EGP offer so that the company saves the MOST money?

Idea: every customer has a probability p of leaving. An offer costs 200 EGP and, with success rate s,
saves 1,500 EGP. So the expected net value of contacting one customer is   s * p * 1500 - 200.
This is an optimisation problem:   maximise  sum( x_i * value_i )   with  x_i = 1 (offer) or 0 (no offer),
and optionally a budget / call-capacity limit.

What this step does (all decisions use TRAINING data only; the test set only reports the result):
  9.1  Calibrate the model's probabilities (the 'balanced' model over-states risk, and money needs honest probabilities)
  9.2  The optimal rule without a limit:  contact a customer if p >= 200 / (1500 * s)
  9.3  Check it with a real integer-programming solver (scipy milp) under a budget
  9.4  The efficient frontier: how much is saved for each extra EGP of campaign budget
  9.5  How the best plan changes with the offer success rate (sensitivity)
  9.6  Compare the policies on the test customers

Run:  python 09_optimization.py          (needs step 04)
"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from common import churn_split, stratified_cv
from config import LOSS_EGP, MODEL_DIR, OFFER_EGP, RANDOM_STATE, header, save_fig, show

cust, X_train, X_test, y_train, y_test = churn_split()
final_model = joblib.load(MODEL_DIR / "final_model.joblib")
y_te = y_test.to_numpy()

# ---------------------------------------------------------------- 9.1 calibration
header("9.1  Honest probabilities (calibration)")
oof = cross_val_predict(final_model, X_train, y_train, cv=stratified_cv(), method="predict_proba")[:, 1]
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
calib = LogisticRegression(C=1e6).fit(logit(oof).reshape(-1, 1), y_train)      # Platt scaling, learned on training only
cal = lambda p: calib.predict_proba(logit(p).reshape(-1, 1))[:, 1]

final_model.fit(X_train, y_train)
p_train = cal(oof)
p_test = cal(final_model.predict_proba(X_test)[:, 1])
print(f"average predicted churn  raw model: {oof.mean():.3f} -> calibrated: {p_train.mean():.3f} | real churn rate: {y_train.mean():.3f}")
print(f"test: average calibrated probability {p_test.mean():.3f} | real churn rate {y_te.mean():.3f}")
print("Why: the tuned model uses class_weight='balanced', which pushes probabilities up. For money we need real ones.")

# ---------------------------------------------------------------- 9.2 the optimal rule
header("9.2  The optimal rule when there is no limit")
print("Expected net value of one offer = s x p x 1,500 - 200   ->   contact the customer when it is above 0, i.e.")
print("    p  >=  200 / (1,500 x s)\n")
rows = []
for s in (1.0, 0.8, 0.6, 0.5, 0.4, 0.3):
    cut = OFFER_EGP / (LOSS_EGP * s)
    rows.append({"offer success rate": f"{int(s * 100)}%", "contact if risk >=": round(cut, 3)})
show(pd.DataFrame(rows).set_index("offer success rate"), "the optimal risk cut-off for each success rate", save_as="09_optimal_cutoffs")
print("The model's own default (0.50) ignores this: it is the right cut-off only if a wasted offer costs as much as a lost customer.")

# ---------------------------------------------------------------- 9.3 integer programming
header("9.3  Check with a real solver: maximise expected saving under a budget (integer programming)")
def solve_budget(p, s, budget):
    """max sum(x_i * (s*p_i*1500 - 200))  s.t.  200 * sum(x_i) <= budget,  x_i in {0,1}."""
    value = s * p * LOSS_EGP - OFFER_EGP
    res = milp(c=-value, constraints=LinearConstraint(np.full((1, len(p)), OFFER_EGP), ub=budget),
               integrality=np.ones(len(p)), bounds=Bounds(0, 1))
    x = np.round(res.x).astype(int)
    return x, float(value @ x)

def greedy(p, s, budget):
    """Same problem, solved by sorting: offer to the riskiest customers first, only while the value is positive."""
    value = s * p * LOSS_EGP - OFFER_EGP
    k = int(budget // OFFER_EGP)
    order = np.argsort(-value)[:k]
    order = order[value[order] > 0]
    return order, float(value[order].sum())

s_demo, budget_demo = 1.0, 40_000
x, best = solve_budget(p_train, s_demo, budget_demo)
order, best_greedy = greedy(p_train, s_demo, budget_demo)
print(f"2,400 training customers, budget {budget_demo:,} EGP (= {budget_demo // OFFER_EGP} offers), success rate 100%:")
print(f"  solver : contacts {x.sum()} customers, expected net saving {best:,.0f} EGP")
print(f"  sorting: contacts {len(order)} customers, expected net saving {best_greedy:,.0f} EGP  -> same answer: {abs(best - best_greedy) < 1e-6}")
print("Because every offer costs the same, the optimum is simply 'riskiest first'. The solver confirms it.")

# ---------------------------------------------------------------- 9.4 efficient frontier
header("9.4  The efficient frontier: saving vs campaign budget (training customers, scaled to 600)")
scale = 600 / len(p_train)
budgets = np.arange(0, 240_001, 4_000)      # budget for the 2,400 training customers
frontier = {}
for s in (1.0, 0.5):
    frontier[s] = [greedy(p_train, s, b)[1] * scale for b in budgets]
fig, ax = plt.subplots(figsize=(8, 4.8))
for s, vals in frontier.items():
    ax.plot(budgets * scale, vals, label=f"offer works {int(s * 100)}%")
ax.set_xlabel("Campaign budget (EGP) per 600 customers"); ax.set_ylabel("Expected net saving (EGP)")
ax.set_title("Extra budget stops paying off once the next customer is not worth 200 EGP", fontsize=11); ax.legend()
plt.tight_layout(); save_fig("9_4_budget_frontier")
for b in (10_000, 20_000, 30_000, 40_000, 55_000, 60_000):
    i = int(np.argmin(np.abs(budgets * scale - b)))
    print(f"  budget {b:>6,} EGP -> saving {frontier[1.0][i]:>9,.0f} (works 100%) | {frontier[0.5][i]:>9,.0f} (works 50%)")

# ---------------------------------------------------------------- 9.5 sensitivity
header("9.5  Sensitivity: the best plan for each offer success rate (training customers, scaled to 600)")
rows = []
for s in (1.0, 0.8, 0.6, 0.5, 0.4, 0.3):
    value = s * p_train * LOSS_EGP - OFFER_EGP
    pick = value > 0
    rows.append({"offer success rate": f"{int(s * 100)}%", "contact if risk >=": round(OFFER_EGP / (LOSS_EGP * s), 3),
                 "customers contacted %": round(100 * pick.mean(), 1),
                 "expected net saving (EGP per 600)": round(value[pick].sum() * scale),
                 "campaign cost (EGP per 600)": round(OFFER_EGP * pick.sum() * scale)})
sens = pd.DataFrame(rows).set_index("offer success rate")
show(sens, "optimal plan vs success rate", save_as="09_sensitivity")

# ---------------------------------------------------------------- 9.6 policies on the test set
header("9.6  Compare the policies on the 600 test customers (real outcomes)")
def realised(pick, s):
    """Real net saving vs doing nothing: every offered leaver is kept with probability s; every offer costs 200."""
    return float((s * y_te[pick] * LOSS_EGP).sum() - OFFER_EGP * pick.sum())

policies = {
    "Do nothing": np.zeros(len(y_te), dtype=bool),
    "Offer to everyone": np.ones(len(y_te), dtype=bool),
    "Model, default cut-off 0.50 (raw probability)": final_model.predict_proba(X_test)[:, 1] >= 0.5,
}
for s in (1.0, 0.5):
    policies[f"OPTIMAL for success {int(s * 100)}% (risk >= {OFFER_EGP / (LOSS_EGP * s):.2f})"] = p_test >= OFFER_EGP / (LOSS_EGP * s)
table = pd.DataFrame({name: {"customers contacted": int(m.sum()), "leavers reached": int(y_te[m].sum()),
                             "net saving if offers work 100% (EGP)": round(realised(m, 1.0)),
                             "net saving if offers work 50% (EGP)": round(realised(m, 0.5))}
                      for name, m in policies.items()}).T
show(table, "policies compared on the test set", save_as="09_policy_comparison")
print("Read it: the optimal plan contacts MORE customers than the default model, because a 200 EGP offer is cheap next to a 1,500 EGP loss.")
print("If the real success rate turns out lower, raise the cut-off: the table in 9.2 tells you by how much.")

(MODEL_DIR / "optimal_policy.json").write_text(json.dumps({
    "cutoff_if_offer_always_works": OFFER_EGP / LOSS_EGP, "cutoff_if_half_work": OFFER_EGP / (LOSS_EGP * 0.5)}, indent=2))
print("\nSaved the optimal cut-offs in", MODEL_DIR)
print("(Part 8 report: add one paragraph from this step.)")
