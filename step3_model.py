"""
=============================================================
 AMAZON MARKETPLACE FRAUD DETECTION — Step 3: ML Model
=============================================================
 Trains a Random Forest classifier on the 6 engineered features.
 Evaluates precision, recall, F1, and ROC-AUC.
 Saves model + predictions for dashboard use.

 Run: python step3_model.py
 Output: data/predictions.csv  |  outputs/model_report.txt
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os, warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection   import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing     import StandardScaler
from sklearn.metrics           import (classification_report, confusion_matrix,
                                       roc_auc_score, roc_curve,
                                       precision_recall_curve, average_precision_score)

FEATURES_PATH = "data/product_features.csv"
PRED_PATH     = "data/predictions.csv"
REPORT_PATH   = "outputs/model_report.txt"
os.makedirs("outputs", exist_ok=True)

FEATURE_COLS = [
    "velocity_score", "unverified_ratio", "rating_concentration",
    "duplicate_score", "helpful_suspicion", "reviewer_overlap_score",
    "n_reviews", "avg_rating", "seller_age_months"
]

# ── Palette ────────────────────────────────────────────────
C_FRAUD  = "#D85A30"   # coral — fraud / risk
C_LEGIT  = "#1D9E75"   # teal  — legit / safe
C_PURPLE = "#7F77DD"
C_GRAY   = "#888780"
BG       = "#FAFAF9"

plt.rcParams.update({
    "figure.facecolor" : BG,
    "axes.facecolor"   : BG,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
    "grid.color"       : "#D3D1C7",
    "font.size"        : 11,
})


def load_features():
    df = pd.read_csv(FEATURES_PATH)
    df = df.dropna(subset=FEATURE_COLS + ["is_fraud_seller"])
    print(f"✓  Loaded {len(df)} products for modelling")
    return df


def train_model(df):
    X = df[FEATURE_COLS]
    y = df["is_fraud_seller"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"   Fraud rate — train: {y_train.mean():.1%} | test: {y_test.mean():.1%}")

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",   # handles imbalance
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # Cross-validated AUC
    cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs  = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    print(f"   5-fold CV AUC: {aucs.mean():.3f} ± {aucs.std():.3f}")

    return clf, X_train, X_test, y_train, y_test


def evaluate(clf, X_test, y_test, df, feature_cols):
    y_pred      = clf.predict(X_test)
    y_prob      = clf.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred,
                                    target_names=["Legit", "Fraud"])
    auc    = roc_auc_score(y_test, y_prob)
    ap     = average_precision_score(y_test, y_prob)

    print(f"\n── Classification Report ─────────────────────────────")
    print(report)
    print(f"   ROC-AUC  : {auc:.4f}")
    print(f"   Avg Prec : {ap:.4f}")

    # Save report
    with open(REPORT_PATH, "w") as f:
        f.write("AMAZON FRAUD DETECTION — MODEL REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(report + "\n")
        f.write(f"ROC-AUC          : {auc:.4f}\n")
        f.write(f"Avg Precision    : {ap:.4f}\n")

    return y_pred, y_prob, auc, ap


def plot_all(clf, X_test, y_test, y_pred, y_prob, auc, ap, feature_cols):
    fig = plt.figure(figsize=(18, 14), facecolor=BG)
    fig.suptitle("Amazon Marketplace Fraud Detection — Model Analysis",
                 fontsize=16, fontweight="bold", y=0.98, color="#2C2C2A")

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

    # ── 1. Confusion Matrix ────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    cm  = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="RdYlGn",
                xticklabels=["Legit","Fraud"],
                yticklabels=["Legit","Fraud"],
                ax=ax1, linewidths=0.5, cbar=False,
                annot_kws={"size": 13, "weight": "bold"})
    ax1.set_title("Confusion matrix", fontsize=12, pad=10)
    ax1.set_xlabel("Predicted"); ax1.set_ylabel("Actual")

    # ── 2. ROC Curve ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax2.plot(fpr, tpr, color=C_PURPLE, lw=2, label=f"AUC = {auc:.3f}")
    ax2.plot([0,1],[0,1], "--", color=C_GRAY, lw=1, label="Random")
    ax2.fill_between(fpr, tpr, alpha=0.08, color=C_PURPLE)
    ax2.set_title("ROC curve", fontsize=12, pad=10)
    ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("True Positive Rate")
    ax2.legend(fontsize=10); ax2.set_xlim(0,1); ax2.set_ylim(0,1.02)

    # ── 3. Precision-Recall Curve ─────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ax3.plot(rec, prec, color=C_FRAUD, lw=2, label=f"AP = {ap:.3f}")
    ax3.fill_between(rec, prec, alpha=0.08, color=C_FRAUD)
    ax3.set_title("Precision-Recall curve", fontsize=12, pad=10)
    ax3.set_xlabel("Recall"); ax3.set_ylabel("Precision")
    ax3.legend(fontsize=10); ax3.set_xlim(0,1); ax3.set_ylim(0,1.02)

    # ── 4. Feature Importance ─────────────────────────────
    ax4 = fig.add_subplot(gs[1, :2])
    importances = clf.feature_importances_
    fi_df = pd.DataFrame({"feature": feature_cols, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=True)

    NICE_NAMES = {
        "velocity_score"        : "Review velocity burst",
        "unverified_ratio"      : "Unverified purchase ratio",
        "rating_concentration"  : "5-star rating concentration",
        "duplicate_score"       : "Duplicate phrasing (TF-IDF)",
        "helpful_suspicion"     : "Low helpful-vote signal",
        "reviewer_overlap_score": "Cross-product reviewer overlap",
        "n_reviews"             : "Total review count",
        "avg_rating"            : "Average star rating",
        "seller_age_months"     : "Seller account age",
    }
    fi_df["label"] = fi_df["feature"].map(NICE_NAMES)

    colors = [C_FRAUD if f in [
        "velocity_score","unverified_ratio","rating_concentration",
        "duplicate_score","helpful_suspicion","reviewer_overlap_score"
    ] else C_GRAY for f in fi_df["feature"]]

    bars = ax4.barh(fi_df["label"], fi_df["importance"], color=colors, height=0.65)
    ax4.set_title("Feature importance — fraud signals (orange) vs. baseline features (gray)",
                  fontsize=12, pad=10)
    ax4.set_xlabel("Importance score")
    for bar, val in zip(bars, fi_df["importance"]):
        ax4.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center", fontsize=9, color="#444441")

    # ── 5. Fraud Risk Distribution ────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    df_all = pd.read_csv("data/product_features.csv")
    fraud = df_all[df_all["is_fraud_seller"]==True]["fraud_risk_index"]
    legit = df_all[df_all["is_fraud_seller"]==False]["fraud_risk_index"]
    ax5.hist(legit, bins=30, color=C_LEGIT, alpha=0.65, label="Legit", density=True)
    ax5.hist(fraud, bins=30, color=C_FRAUD, alpha=0.65, label="Fraud", density=True)
    ax5.axvline(0.40, color=C_GRAY, ls="--", lw=1.2, label="Medium threshold")
    ax5.axvline(0.65, color="#2C2C2A", ls="--", lw=1.2, label="High threshold")
    ax5.set_title("Fraud Risk Index distribution", fontsize=12, pad=10)
    ax5.set_xlabel("Fraud Risk Index"); ax5.set_ylabel("Density")
    ax5.legend(fontsize=9)

    # ── 6. Trust Score by Category ───────────────────────
    ax6 = fig.add_subplot(gs[2, :])
    cat_trust = (
        df_all.groupby(["category","is_fraud_seller"])["seller_trust_score"]
        .mean().reset_index()
    )
    cat_fraud = cat_trust[cat_trust["is_fraud_seller"]==True].sort_values("seller_trust_score")
    cat_legit = cat_trust[cat_trust["is_fraud_seller"]==False]
    cat_legit = cat_legit.set_index("category").reindex(cat_fraud["category"]).reset_index()

    x      = np.arange(len(cat_fraud))
    width  = 0.38
    ax6.bar(x - width/2, cat_fraud["seller_trust_score"], width, color=C_FRAUD, alpha=0.85, label="Fraud sellers")
    ax6.bar(x + width/2, cat_legit["seller_trust_score"], width, color=C_LEGIT, alpha=0.85, label="Legit sellers")
    ax6.set_xticks(x); ax6.set_xticklabels(cat_fraud["category"], rotation=35, ha="right", fontsize=9)
    ax6.set_title("Average Seller Trust Score by product category", fontsize=12, pad=10)
    ax6.set_ylabel("Trust Score (0–100)")
    ax6.legend(fontsize=10)
    ax6.set_ylim(0, 105)

    plt.savefig("outputs/model_analysis.png", dpi=150, bbox_inches="tight",
                facecolor=BG)
    print("   ✓  Saved → outputs/model_analysis.png")
    plt.close()


def save_predictions(clf, df, feature_cols):
    df = df.copy()
    df["fraud_probability"] = clf.predict_proba(df[feature_cols])[:, 1].round(4)
    df["model_prediction"]  = (df["fraud_probability"] >= 0.5).map({True:"FRAUD", False:"LEGIT"})
    df["correct"]           = df["model_prediction"].map({"FRAUD": True, "LEGIT": False}) == df["is_fraud_seller"]
    df.to_csv(PRED_PATH, index=False)
    print(f"   ✓  Predictions saved → {PRED_PATH}")


if __name__ == "__main__":
    df             = load_features()
    clf, X_tr, X_te, y_tr, y_te = train_model(df)
    y_pred, y_prob, auc, ap     = evaluate(clf, X_te, y_te, df, FEATURE_COLS)
    plot_all(clf, X_te, y_te, y_pred, y_prob, auc, ap, FEATURE_COLS)
    save_predictions(clf, df, FEATURE_COLS)

    print(f"\n── Final model metrics ───────────────────────────────")
    print(f"   ROC-AUC          : {auc:.4f}")
    print(f"   Avg Precision    : {ap:.4f}")
    print(f"   Report saved     : {REPORT_PATH}")
    print(f"\n✅  Step 3 complete. Run step4_dashboard.py next.\n")
