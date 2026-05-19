"""
=============================================================
 AMAZON MARKETPLACE FRAUD DETECTION — Step 2: Feature Engineering
=============================================================
 Builds 6 original fraud-detection signals per product:

   1. review_velocity_score   — burst of reviews in short window
   2. unverified_ratio        — % of reviews not "verified purchase"
   3. rating_concentration    — how skewed ratings are toward 5★
   4. duplicate_phrasing_score— TF-IDF cosine similarity among reviews
   5. helpful_vote_ratio      — real reviews get more helpful votes
   6. reviewer_overlap_score  — same reviewers across multiple products

 Run: python step2_feature_engineering.py
 Output: data/product_features.csv
=============================================================
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

REVIEWS_PATH  = "data/amazon_reviews.csv"
PRODUCTS_PATH = "data/products.csv"
OUTPUT_PATH   = "data/product_features.csv"


def load_data():
    print("⟳  Loading data...")
    reviews  = pd.read_csv(REVIEWS_PATH, parse_dates=["review_date"])
    products = pd.read_csv(PRODUCTS_PATH)
    print(f"   ✓  {len(reviews):,} reviews | {len(products)} products")
    return reviews, products


# ── Feature 1: Review velocity score ──────────────────────
# High score = many reviews arrived in a tight time window (burst signal)
def compute_velocity_score(grp):
    if len(grp) < 3:
        return 0.0
    dates = grp["review_date"].sort_values()
    # Reviews per day in first 30-day window
    window_end   = dates.iloc[0] + pd.Timedelta(days=30)
    burst_count  = (dates <= window_end).sum()
    total        = len(dates)
    score        = burst_count / total
    # Amplify if the burst is very tight (< 7 days)
    tight_window = dates.iloc[0] + pd.Timedelta(days=7)
    tight_count  = (dates <= tight_window).sum()
    if tight_count >= 5:
        score = min(1.0, score * 1.4)
    return round(score, 4)


# ── Feature 2: Unverified purchase ratio ──────────────────
def compute_unverified_ratio(grp):
    if len(grp) == 0:
        return 0.0
    ratio = (~grp["verified_purchase"]).sum() / len(grp)
    return round(ratio, 4)


# ── Feature 3: Rating concentration (5-star skew) ─────────
# Gini-like score: 1.0 = all reviews are 5-star (suspicious)
def compute_rating_concentration(grp):
    if len(grp) == 0:
        return 0.0
    counts = grp["rating"].value_counts(normalize=True)
    five_star_pct = counts.get(5, 0.0)
    # Also flag if avg > 4.7 with >20 reviews
    avg    = grp["rating"].mean()
    n      = len(grp)
    score  = five_star_pct
    if avg > 4.7 and n > 20:
        score = min(1.0, score * 1.2)
    return round(score, 4)


# ── Feature 4: Duplicate phrasing score (TF-IDF cosine) ───
# Computes average pairwise cosine similarity of review texts
# High similarity = copy-paste / template reviews
def compute_duplicate_score(grp):
    texts = grp["review_text"].dropna().tolist()
    if len(texts) < 3:
        return 0.0
    # Sample up to 50 reviews for speed
    sample = texts[:50]
    try:
        vec    = TfidfVectorizer(min_df=1, stop_words="english")
        tfidf  = vec.fit_transform(sample)
        sim    = cosine_similarity(tfidf)
        # Average of upper triangle (exclude self-similarity diagonal)
        n = sim.shape[0]
        upper = sim[np.triu_indices(n, k=1)]
        return round(float(upper.mean()), 4)
    except Exception:
        return 0.0


# ── Feature 5: Helpful vote ratio ─────────────────────────
# Fake reviews rarely get helpful votes; low ratio is suspicious
def compute_helpful_ratio(grp):
    if len(grp) == 0:
        return 0.0
    avg_helpful = grp["helpful_votes"].mean()
    # Invert: low helpful → high suspicion score
    # Normalise to 0-1 (cap at 10 votes avg = fully legit)
    suspicion = max(0.0, 1.0 - (avg_helpful / 10.0))
    return round(suspicion, 4)


# ── Feature 6: Reviewer overlap score ─────────────────────
# Fraudsters use the same reviewer accounts across products
def compute_reviewer_overlap(reviews_df, products_df):
    print("   ⟳  Computing reviewer overlap (cross-product)...")
    # For each reviewer: how many distinct ASINs did they review?
    reviewer_asin = (
        reviews_df.groupby("reviewer_id")["asin"]
        .nunique()
        .reset_index()
        .rename(columns={"asin": "n_products_reviewed"})
    )
    # Merge back to get per-review reviewer breadth
    merged = reviews_df.merge(reviewer_asin, on="reviewer_id", how="left")
    # High overlap = reviewer leaves reviews on many products quickly
    # Score per ASIN = mean reviewer breadth (normalised)
    max_breadth = merged["n_products_reviewed"].max()
    merged["reviewer_breadth_norm"] = merged["n_products_reviewed"] / max_breadth

    overlap = (
        merged.groupby("asin")["reviewer_breadth_norm"]
        .mean()
        .reset_index()
        .rename(columns={"reviewer_breadth_norm": "reviewer_overlap_score"})
    )
    overlap["reviewer_overlap_score"] = overlap["reviewer_overlap_score"].round(4)
    return overlap


# ── Composite Fraud Trust Index ────────────────────────────
# Weights tuned to match importance of each signal
WEIGHTS = {
    "velocity_score"        : 0.25,
    "unverified_ratio"      : 0.20,
    "rating_concentration"  : 0.20,
    "duplicate_score"       : 0.20,
    "helpful_suspicion"     : 0.10,
    "reviewer_overlap_score": 0.05,
}

def compute_fraud_index(row):
    score = sum(row[feat] * w for feat, w in WEIGHTS.items())
    return round(min(1.0, score), 4)


# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    reviews, products = load_data()

    features = []
    total = len(products)
    print(f"⟳  Engineering features for {total} products...")

    for idx, (_, prod) in enumerate(products.iterrows()):
        asin = prod["asin"]
        grp  = reviews[reviews["asin"] == asin]

        features.append({
            "asin"                : asin,
            "n_reviews"           : len(grp),
            "avg_rating"          : round(grp["rating"].mean(), 2) if len(grp) else 0,
            "velocity_score"      : compute_velocity_score(grp),
            "unverified_ratio"    : compute_unverified_ratio(grp),
            "rating_concentration": compute_rating_concentration(grp),
            "duplicate_score"     : compute_duplicate_score(grp),
            "helpful_suspicion"   : compute_helpful_ratio(grp),
        })

        if (idx + 1) % 100 == 0:
            print(f"   {idx+1}/{total} products processed...")

    features_df = pd.DataFrame(features)

    # Reviewer overlap (cross-product computation)
    overlap_df  = compute_reviewer_overlap(reviews, products)
    features_df = features_df.merge(overlap_df, on="asin", how="left")
    features_df["reviewer_overlap_score"] = features_df["reviewer_overlap_score"].fillna(0)

    # Composite fraud index
    features_df["fraud_risk_index"] = features_df.apply(compute_fraud_index, axis=1)

    # Seller trust score (inverted, 0-100)
    features_df["seller_trust_score"] = ((1 - features_df["fraud_risk_index"]) * 100).round(1)

    # Risk tier classification
    def assign_tier(score):
        if score >= 0.65: return "HIGH RISK"
        if score >= 0.40: return "MEDIUM RISK"
        return "LOW RISK"
    features_df["risk_tier"] = features_df["fraud_risk_index"].apply(assign_tier)

    # Merge ground truth for validation
    features_df = features_df.merge(
        products[["asin","product_name","category","brand","price","is_fraud_seller","seller_age_months","seller_id"]],
        on="asin", how="left"
    )

    features_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n── Feature summary ──────────────────────────────────")
    print(features_df[["velocity_score","unverified_ratio","rating_concentration",
                        "duplicate_score","helpful_suspicion","fraud_risk_index"]].describe().round(3).to_string())
    print(f"\n── Risk tier distribution ───────────────────────────")
    print(features_df["risk_tier"].value_counts().to_string())
    print(f"\n✅  Features saved → {OUTPUT_PATH}")
    print("    Run step3_model.py next.\n")
