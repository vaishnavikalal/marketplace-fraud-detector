"""
=============================================================
 AMAZON MARKETPLACE FRAUD DETECTION
 Full Pipeline Runner — runs all 5 steps end to end

 Usage:  python run_all.py
 Time:   ~2–4 minutes on a standard laptop

 Steps:
   1. Generate realistic Amazon-style dataset (500 products, 50K reviews)
   2. Engineer 6 fraud detection features per product
   3. Train & evaluate Random Forest classifier
   4. Produce 12 EDA + seller leaderboard charts
   5. Generate executive summary + final report
=============================================================
"""

import subprocess, sys, time, os

STEPS = [
    ("step1_generate_data.py",    "Generating dataset (500 products, 50K reviews)"),
    ("step2_feature_engineering.py", "Engineering 6 fraud signals per product"),
    ("step3_model.py",            "Training Random Forest classifier"),
    ("step4_eda_charts.py",       "Producing EDA & seller leaderboard charts"),
    ("step5_insights.py",         "Building executive summary & final report"),
]

def run_step(script, description):
    print(f"\n{'─'*62}")
    print(f"  STEP {STEPS.index((script, description))+1}/5 — {description}")
    print(f"{'─'*62}")
    t0     = time.time()
    result = subprocess.run([sys.executable, script], capture_output=False, text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n❌  {script} failed. Check error above.")
        sys.exit(1)
    print(f"\n   ⏱  Completed in {elapsed:.1f}s")


if __name__ == "__main__":
    print("=" * 62)
    print("  AMAZON MARKETPLACE FRAUD DETECTION PIPELINE")
    print("  Replicating Amazon's Seller Integrity Methodology")
    print("=" * 62)

    total_start = time.time()
    for script, desc in STEPS:
        run_step(script, desc)

    total = time.time() - total_start
    print(f"\n{'='*62}")
    print(f"  ✅  PIPELINE COMPLETE  —  {total:.0f}s total")
    print(f"{'='*62}")
    print("\n  📁  Output files:")
    output_files = [
        "outputs/eda_charts.png         — 9-panel EDA analysis",
        "outputs/model_analysis.png     — ML model evaluation",
        "outputs/seller_leaderboard.png — Flagged seller dashboard",
        "outputs/executive_summary.png  — Stakeholder presentation",
        "outputs/final_report.txt       — Business narrative",
        "outputs/model_report.txt       — Model metrics",
        "data/product_features.csv      — Engineered feature matrix",
        "data/predictions.csv           — Per-product fraud scores",
    ]
    for f in output_files:
        print(f"    {f}")

    print("\n  GitHub README, resume bullets, and LinkedIn post")
    print("  are in README.md\n")
