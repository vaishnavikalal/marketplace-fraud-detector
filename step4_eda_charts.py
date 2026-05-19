"""
=============================================================
 AMAZON MARKETPLACE FRAUD DETECTION — Step 4: EDA & Charts
=============================================================
 Produces 12 publication-quality EDA charts covering:
   - Review burst timelines
   - Rating distribution comparison
   - Duplicate phrasing heatmap
   - Seller trust score leaderboard
   - Fraud signal correlation matrix
   - Category-level risk analysis
   - Top flagged sellers

 Run: python step4_eda_charts.py
 Output: outputs/eda_charts.png   outputs/seller_leaderboard.png
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import os
os.makedirs("outputs", exist_ok=True)

C_FRAUD  = "#D85A30"
C_LEGIT  = "#1D9E75"
C_PURPLE = "#7F77DD"
C_AMBER  = "#EF9F27"
C_GRAY   = "#888780"
C_BG     = "#FAFAF9"
C_DARK   = "#2C2C2A"

plt.rcParams.update({
    "figure.facecolor" : C_BG,
    "axes.facecolor"   : C_BG,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"        : True,
    "grid.alpha"       : 0.25,
    "grid.color"       : "#D3D1C7",
    "font.size"        : 10,
    "axes.labelsize"   : 11,
    "axes.titlesize"   : 12,
    "axes.titlepad"    : 10,
})

def load():
    feats   = pd.read_csv("data/product_features.csv")
    reviews = pd.read_csv("data/amazon_reviews.csv", parse_dates=["review_date"])
    return feats, reviews


# ───────────────────────────────────────────────────────────
#  CHART SET 1 — EDA Overview (3×3 grid)
# ───────────────────────────────────────────────────────────
def plot_eda(feats, reviews):
    fig = plt.figure(figsize=(18, 15), facecolor=C_BG)
    fig.suptitle("Amazon Marketplace Fraud Detection — Exploratory Data Analysis",
                 fontsize=16, fontweight="bold", y=0.99, color=C_DARK)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38)

    fraud = feats[feats["is_fraud_seller"] == True]
    legit = feats[feats["is_fraud_seller"] == False]

    # ── 1A. Velocity score distribution ──────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(legit["velocity_score"], bins=30, color=C_LEGIT, alpha=0.7, label="Legit", density=True)
    ax.hist(fraud["velocity_score"], bins=30, color=C_FRAUD, alpha=0.7, label="Fraud", density=True)
    ax.set_title("Signal 1 — Review velocity burst score")
    ax.set_xlabel("Velocity score (0 = slow, 1 = burst)")
    ax.legend(fontsize=9)
    ax.text(0.97, 0.93, "Higher = suspicious", transform=ax.transAxes,
            fontsize=8, color=C_GRAY, ha="right")

    # ── 1B. Unverified purchase ratio ────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(legit["unverified_ratio"], bins=25, color=C_LEGIT, alpha=0.7, label="Legit", density=True)
    ax.hist(fraud["unverified_ratio"], bins=25, color=C_FRAUD, alpha=0.7, label="Fraud", density=True)
    ax.set_title("Signal 2 — Unverified purchase ratio")
    ax.set_xlabel("Ratio of non-verified reviews")
    ax.legend(fontsize=9)

    # ── 1C. Duplicate phrasing score ─────────────────────
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(legit["duplicate_score"], bins=25, color=C_LEGIT, alpha=0.7, label="Legit", density=True)
    ax.hist(fraud["duplicate_score"], bins=25, color=C_FRAUD, alpha=0.7, label="Fraud", density=True)
    ax.set_title("Signal 4 — TF-IDF duplicate phrasing score")
    ax.set_xlabel("Mean cosine similarity of review text")
    ax.legend(fontsize=9)

    # ── 2A. Rating concentration boxplot ─────────────────
    ax = fig.add_subplot(gs[1, 0])
    data_box = [legit["rating_concentration"].values, fraud["rating_concentration"].values]
    bp = ax.boxplot(data_box, patch_artist=True, widths=0.5,
                    medianprops=dict(color="white", lw=2))
    for patch, color in zip(bp["boxes"], [C_LEGIT, C_FRAUD]):
        patch.set_facecolor(color); patch.set_alpha(0.8)
    ax.set_xticklabels(["Legit sellers", "Fraud sellers"])
    ax.set_title("Signal 3 — 5-star rating concentration")
    ax.set_ylabel("Concentration score")

    # ── 2B. Fraud risk index scatter: velocity vs unverified
    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(legit["velocity_score"], legit["unverified_ratio"],
               c=C_LEGIT, alpha=0.4, s=20, label="Legit")
    ax.scatter(fraud["velocity_score"], fraud["unverified_ratio"],
               c=C_FRAUD, alpha=0.5, s=20, label="Fraud")
    ax.set_xlabel("Velocity score")
    ax.set_ylabel("Unverified ratio")
    ax.set_title("Velocity vs. Unverified ratio — separation")
    ax.legend(fontsize=9)

    # ── 2C. Seller account age vs fraud risk ──────────────
    ax = fig.add_subplot(gs[1, 2])
    ax.scatter(legit["seller_age_months"], legit["fraud_risk_index"],
               c=C_LEGIT, alpha=0.4, s=18, label="Legit")
    ax.scatter(fraud["seller_age_months"], fraud["fraud_risk_index"],
               c=C_FRAUD, alpha=0.5, s=18, label="Fraud")
    ax.set_xlabel("Seller account age (months)")
    ax.set_ylabel("Fraud Risk Index")
    ax.set_title("Seller age vs. Fraud Risk Index")
    ax.legend(fontsize=9)

    # ── 3A. Review burst timeline (sample products) ───────
    ax = fig.add_subplot(gs[2, :2])
    fraud_asins = feats[feats["is_fraud_seller"]==True].nlargest(4, "velocity_score")["asin"].tolist()
    legit_asins = feats[feats["is_fraud_seller"]==False].nsmallest(4, "velocity_score")["asin"].tolist()
    sample_asins = fraud_asins[:2] + legit_asins[:2]
    labels_map   = {a: (f"FRAUD #{i+1}" if a in fraud_asins else f"LEGIT #{i+1}")
                    for i, a in enumerate(sample_asins)}

    for i, asin in enumerate(sample_asins):
        grp = reviews[reviews["asin"]==asin].sort_values("review_date")
        if len(grp) == 0: continue
        grp = grp.copy()
        grp["day_offset"] = (grp["review_date"] - grp["review_date"].min()).dt.days
        grp["cumcount"]   = range(1, len(grp)+1)
        color = C_FRAUD if asin in fraud_asins else C_LEGIT
        ls    = "-" if asin in fraud_asins else "--"
        ax.plot(grp["day_offset"], grp["cumcount"],
                color=color, lw=2, ls=ls, label=labels_map[asin], alpha=0.85)

    ax.set_title("Review accumulation curve — burst (fraud) vs. organic (legit)")
    ax.set_xlabel("Days since first review"); ax.set_ylabel("Cumulative review count")
    ax.legend(fontsize=9, ncol=2)

    # ── 3B. Fraud signal correlation heatmap ──────────────
    ax = fig.add_subplot(gs[2, 2])
    signal_cols = ["velocity_score","unverified_ratio","rating_concentration",
                   "duplicate_score","helpful_suspicion","fraud_risk_index"]
    corr = feats[signal_cols].corr()
    short_names = ["Velocity","Unverified","5-star conc.",
                   "Duplicate","Low helpful","Fraud index"]
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn_r",
                xticklabels=short_names, yticklabels=short_names,
                ax=ax, linewidths=0.4, cbar=False,
                annot_kws={"size": 8})
    ax.set_title("Fraud signal correlation matrix")
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)

    plt.savefig("outputs/eda_charts.png", dpi=150, bbox_inches="tight", facecolor=C_BG)
    print("   ✓  Saved → outputs/eda_charts.png")
    plt.close()


# ───────────────────────────────────────────────────────────
#  CHART SET 2 — Seller Leaderboard / Trust Scorecard
# ───────────────────────────────────────────────────────────
def plot_leaderboard(feats):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor=C_BG)
    fig.suptitle("Amazon Seller Trust Score — Flagged Sellers Dashboard",
                 fontsize=15, fontweight="bold", y=1.01, color=C_DARK)

    # ── Top 20 riskiest sellers ────────────────────────────
    ax = axes[0]
    top_risky = feats.nlargest(20, "fraud_risk_index")[
        ["product_name","category","seller_trust_score","fraud_risk_index","risk_tier","n_reviews"]
    ].reset_index(drop=True)

    colors = [C_FRAUD if t == "HIGH RISK" else C_AMBER if t == "MEDIUM RISK" else C_LEGIT
              for t in top_risky["risk_tier"]]

    bars = ax.barh(
        top_risky.index,
        top_risky["fraud_risk_index"],
        color=colors, height=0.72, alpha=0.88
    )
    ax.set_yticks(top_risky.index)
    ax.set_yticklabels(
        [f"{row['product_name'][:28]}…" if len(row['product_name'])>28 else row['product_name']
         for _, row in top_risky.iterrows()],
        fontsize=8.5
    )
    ax.invert_yaxis()
    ax.set_xlabel("Fraud Risk Index (higher = riskier)")
    ax.set_title("Top 20 riskiest products by Fraud Risk Index", fontsize=12)
    ax.axvline(0.65, color=C_DARK, ls="--", lw=1.2, label="High risk threshold")
    ax.axvline(0.40, color=C_GRAY, ls="--", lw=1.0, label="Medium risk threshold")
    ax.legend(fontsize=9)

    for bar, (_, row) in zip(bars, top_risky.iterrows()):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f"{row['fraud_risk_index']:.2f}  ({row['n_reviews']} reviews)",
                va="center", fontsize=8, color="#444441")

    legend_patches = [
        mpatches.Patch(color=C_FRAUD,  label="HIGH RISK"),
        mpatches.Patch(color=C_AMBER,  label="MEDIUM RISK"),
        mpatches.Patch(color=C_LEGIT,  label="LOW RISK"),
    ]
    ax.legend(handles=legend_patches, fontsize=9, loc="lower right")

    # ── Category risk breakdown ────────────────────────────
    ax2 = axes[1]
    cat_stats = feats.groupby("category").agg(
        avg_risk   = ("fraud_risk_index", "mean"),
        high_risk  = ("risk_tier", lambda x: (x=="HIGH RISK").sum()),
        total      = ("risk_tier", "count")
    ).reset_index()
    cat_stats["high_risk_pct"] = (cat_stats["high_risk"] / cat_stats["total"] * 100).round(1)
    cat_stats = cat_stats.sort_values("high_risk_pct", ascending=True)

    bar_colors = [C_FRAUD if p > 30 else C_AMBER if p > 15 else C_LEGIT
                  for p in cat_stats["high_risk_pct"]]

    bars2 = ax2.barh(cat_stats["category"], cat_stats["high_risk_pct"],
                     color=bar_colors, height=0.6, alpha=0.88)
    ax2.set_xlabel("% of sellers flagged as HIGH RISK")
    ax2.set_title("High-risk seller rate by product category", fontsize=12)
    ax2.axvline(30, color=C_DARK, ls="--", lw=1.2, label="> 30% = severe")
    ax2.legend(fontsize=9)

    for bar, (_, row) in zip(bars2, cat_stats.iterrows()):
        ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                 f"{row['high_risk_pct']}%  ({row['high_risk']}/{row['total']})",
                 va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("outputs/seller_leaderboard.png", dpi=150, bbox_inches="tight", facecolor=C_BG)
    print("   ✓  Saved → outputs/seller_leaderboard.png")
    plt.close()


if __name__ == "__main__":
    feats, reviews = load()
    print("⟳  Generating EDA charts...")
    plot_eda(feats, reviews)
    print("⟳  Generating seller leaderboard...")
    plot_leaderboard(feats)
    print("\n✅  Step 4 complete. Run step5_insights.py next.\n")
