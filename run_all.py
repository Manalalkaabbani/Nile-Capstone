"""
run_all.py  -  runs the 8 steps in order (about 2 minutes).  You can also run each step by hand.

    python run_all.py
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = [
    "01_eda.py", "02_regression.py", "03_churn_models.py", "04_cv_and_tuning.py",
    "05_final_test.py", "06_clustering_pca.py", "07_association_rules.py", "08_segments_vs_churn.py",
]

for step in STEPS:
    print(f"\n>>>>>>>>>> {step}", flush=True)
    t0 = time.perf_counter()
    result = subprocess.run([sys.executable, step], cwd=HERE)
    if result.returncode != 0:
        sys.exit(f"\nSTOPPED: {step} failed. Fix it, then run it again.")
    print(f"<<<<<<<<<< {step} finished in {time.perf_counter() - t0:.0f}s", flush=True)

print("\nAll steps finished. Charts: outputs/figures | tables: outputs/tables")
