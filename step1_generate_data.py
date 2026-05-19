"""
=============================================================
 AMAZON MARKETPLACE FRAUD DETECTION — Step 1: Data Generation
=============================================================
 Generates a realistic Amazon-style review dataset with
 embedded fraud patterns (burst reviews, duplicate phrasing,
 fake verified purchases, rating manipulation).

 Run: python step1_generate_data.py
 Output: data/amazon_reviews.csv  +  data/products.csv
=============================================================
"""

import pandas as pd
import numpy as np
import random
import json
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

# ── Config ────────────────────────────────────────────────
N_PRODUCTS   = 500
N_REVIEWS    = 50_000
FRAUD_RATE   = 0.28          # 28% of sellers are fraudulent
OUTPUT_DIR   = "data"

# ── Realistic review text templates ───────────────────────
LEGIT_REVIEWS = [
    "Works exactly as described. Happy with the purchase overall.",
    "Good quality for the price. Shipping was fast.",
    "I've been using this for 3 months and it still works great.",
    "Not perfect but does the job. Would recommend with caveats.",
    "Exceeded my expectations! Really impressed.",
    "Decent product. Instructions could be clearer.",
    "Returned it — didn't fit my needs. Not the product's fault.",
    "Great build quality. Feels premium compared to competitors.",
    "Bought as a gift, recipient loved it.",
    "Had a small defect but customer service resolved it quickly.",
    "Does exactly what I needed. No complaints.",
    "A bit pricey but quality justifies it.",
    "Took a while to ship but product is solid.",
    "Perfect for everyday use. Very satisfied.",
    "Minor issue after a week but still giving 4 stars overall.",
    "Looked cheaper in photos. Pleasantly surprised in person.",
    "My third purchase from this brand — consistent quality.",
    "Great value. Would buy again.",
    "Instructions are in three languages, a bit confusing.",
    "Does what it says. Can't ask for more at this price point.",
]

FAKE_REVIEW_TEMPLATES = [
    "Amazing product! Best I've ever used! Highly recommend to everyone!!!",
    "WOW! This product changed my life! 5 stars without hesitation!",
    "Absolutely love this! Perfect in every way! Must buy!!!",
    "Best purchase ever made! Incredible quality! Will buy again!",
    "Outstanding product! Exceeded all expectations! 5 stars!",
    "Fantastic! Just fantastic! Everyone should own this product!",
    "Perfect product! Perfect seller! Perfect experience! Love it!",
    "Amazing amazing amazing! Can't believe how good this is!!!",
    "This is the best product in this category! No doubt! Buy now!",
    "Superb quality! Fast delivery! Excellent in every single way!",
]

CATEGORIES = [
    "Electronics", "Home & Kitchen", "Sports & Outdoors",
    "Beauty & Personal Care", "Toys & Games", "Office Products",
    "Health & Household", "Clothing", "Automotive", "Books"
]

BRANDS_LEGIT = [
    "SoundWave", "HomeEssentials", "ProFit", "PureGlow", "KidZone",
    "OfficeMax", "WellLife", "StyleCo", "AutoPro", "PageTurner"
]

BRANDS_FRAUD = [
    "BestDealz", "TopQualityShop", "AmazingFinds", "SuperValue",
    "MegaSavings", "PremiumChoice", "EliteGoods", "UltraDeals"
]


# ── Generate Products ──────────────────────────────────────
def generate_products():
    products = []
    for i in range(N_PRODUCTS):
        is_fraud = random.random() < FRAUD_RATE
        category = random.choice(CATEGORIES)
        brand    = random.choice(BRANDS_FRAUD if is_fraud else BRANDS_LEGIT)

        base_price  = round(random.uniform(8, 350), 2)
        seller_age  = random.randint(1, 60) if not is_fraud else random.randint(1, 12)

        products.append({
            "asin"            : f"B{str(i).zfill(9)}",
            "product_name"    : f"{brand} {category} Product {i+1}",
            "category"        : category,
            "brand"           : brand,
            "price"           : base_price,
            "is_fraud_seller" : is_fraud,
            "seller_age_months": seller_age,
            "seller_id"       : f"SELLER_{str(i % 150).zfill(4)}",
        })
    return pd.DataFrame(products)


# ── Generate Reviews ───────────────────────────────────────
def generate_reviews(products_df):
    reviews = []
    base_date = datetime(2022, 1, 1)

    for _, prod in products_df.iterrows():
        is_fraud = prod["is_fraud_seller"]

        # Fraud sellers: burst of reviews in first 2 weeks
        if is_fraud:
            n_reviews   = random.randint(40, 120)
            burst_window = 14                  # days
        else:
            n_reviews   = random.randint(5, 60)
            burst_window = 180

        product_start = base_date + timedelta(days=random.randint(0, 600))

        for j in range(n_reviews):
            # Date: fraud = clustered tightly, legit = spread out
            if is_fraud:
                review_date = product_start + timedelta(
                    days=random.randint(0, burst_window),
                    hours=random.randint(0, 23)
                )
            else:
                review_date = product_start + timedelta(
                    days=random.randint(0, 365),
                    hours=random.randint(0, 23)
                )

            # Rating manipulation
            if is_fraud:
                # Mostly 5-stars with occasional 4
                rating = random.choices([5, 4, 3], weights=[80, 15, 5])[0]
                review_text = random.choice(FAKE_REVIEW_TEMPLATES)
                verified    = random.choices([True, False], weights=[30, 70])[0]
                helpful     = random.randint(0, 2)
            else:
                rating      = random.choices([1,2,3,4,5], weights=[8,7,15,35,35])[0]
                review_text = random.choice(LEGIT_REVIEWS)
                # Sometimes paraphrase slightly
                if random.random() < 0.1:
                    review_text = review_text + " " + random.choice(["Great!", "Solid.", "Recommended."])
                verified    = random.choices([True, False], weights=[75, 25])[0]
                helpful     = random.randint(0, 45)

            reviewer_id = (
                f"RFAKE_{random.randint(1,500):04d}" if is_fraud
                else f"RREAL_{random.randint(1,8000):05d}"
            )

            reviews.append({
                "review_id"        : f"R{len(reviews):07d}",
                "asin"             : prod["asin"],
                "reviewer_id"      : reviewer_id,
                "rating"           : rating,
                "review_text"      : review_text,
                "verified_purchase": verified,
                "helpful_votes"    : helpful,
                "review_date"      : review_date,
                "category"         : prod["category"],
            })

    df = pd.DataFrame(reviews)
    df["review_date"] = pd.to_datetime(df["review_date"])
    return df


# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("⟳  Generating product catalog...")
    products = generate_products()
    products.to_csv(f"{OUTPUT_DIR}/products.csv", index=False)
    print(f"   ✓  {len(products)} products saved → data/products.csv")

    print("⟳  Generating review dataset...")
    reviews = generate_reviews(products)
    reviews.to_csv(f"{OUTPUT_DIR}/amazon_reviews.csv", index=False)
    print(f"   ✓  {len(reviews):,} reviews saved → data/amazon_reviews.csv")

    fraud_sellers = products["is_fraud_seller"].sum()
    print(f"\n── Dataset summary ──────────────────────────────────")
    print(f"   Products      : {len(products)}")
    print(f"   Reviews       : {len(reviews):,}")
    print(f"   Fraud sellers : {fraud_sellers} ({fraud_sellers/len(products)*100:.1f}%)")
    print(f"   Legit sellers : {len(products)-fraud_sellers}")
    print(f"─────────────────────────────────────────────────────")
    print("\n✅  Step 1 complete. Run step2_feature_engineering.py next.\n")
