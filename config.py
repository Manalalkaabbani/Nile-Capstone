"""
config.py  -  settings shared by every step.

>>> THE ONLY THING YOU MAY NEED TO CHANGE IS RAW_DIR (the folder with your 3 CSV files) <<<
"""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")                      # charts are saved as PNG files (no pop-up windows)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ----------------------------------------------------------------------------
# 1) WHERE ARE THE FILES?
# ----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent

# Put customers.csv, usage_behavior.csv and service_transactions.csv in data/raw/
# or write your own folder here, for example:
#   RAW_DIR = Path(r"C:\Users\Manal\Documents\ML_project_data")
RAW_DIR = PROJECT_DIR / "data" / "raw"

CLEAN_DIR = PROJECT_DIR / "data" / "clean"      # step 01 writes the cleaned files here
FIG_DIR = PROJECT_DIR / "outputs" / "figures"   # every chart (PNG)
TABLE_DIR = PROJECT_DIR / "outputs" / "tables"  # every result table (CSV)
MODEL_DIR = PROJECT_DIR / "outputs" / "models"  # saved model + choices made in step 04

for folder in (CLEAN_DIR, FIG_DIR, TABLE_DIR, MODEL_DIR):
    folder.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# 2) SETTINGS
# ----------------------------------------------------------------------------
RANDOM_STATE = 42          # the brief asks for random_state=42 everywhere
DATA_CAP_GB = 150          # a customer cannot really use more than this (see step 01)
LOSS_EGP = 1500            # cost of losing one customer (from the brief)
OFFER_EGP = 200            # cost of one retention offer (from the brief)

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"figure.dpi": 100, "axes.titleweight": "bold", "axes.titlesize": 12})
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)


# ----------------------------------------------------------------------------
# 3) SMALL HELPERS
# ----------------------------------------------------------------------------
def save_fig(name):
    """Save the current chart to outputs/figures/<name>.png and close it."""
    plt.gcf().savefig(FIG_DIR / f"{name}.png", dpi=130, bbox_inches="tight")
    plt.close("all")
    print(f"   [chart saved] outputs/figures/{name}.png")


def show(df, title=None, save_as=None):
    """Print a table (and optionally save it to outputs/tables/<save_as>.csv)."""
    if title:
        print(f"\n--- {title}")
    print(df.to_string() if hasattr(df, "to_string") else df)
    if save_as:
        df.to_csv(TABLE_DIR / f"{save_as}.csv")


def header(text):
    print("\n" + "=" * 90 + f"\n{text}\n" + "=" * 90)
