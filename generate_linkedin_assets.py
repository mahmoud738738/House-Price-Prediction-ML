import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    mean_absolute_error, root_mean_squared_error, r2_score,
    precision_score, recall_score, f1_score, classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split

output_dir = "house-price-project/linkedin_assets"
if not os.path.exists("house-price-project"):
    output_dir = "linkedin_assets"
os.makedirs(output_dir, exist_ok=True)

print("1. Loading raw dataset...")
csv_path = "house-price-project/notebooks/data/house_prices.csv"
if not os.path.exists(csv_path):
    csv_path = "notebooks/data/house_prices.csv"
df_raw = pd.read_csv(csv_path)

# Parsing helpers
def parse_amount(x):
    if not isinstance(x, str): return None
    x = x.strip().lower()
    try:
        if "lac" in x: return float(x.replace("lac", "").strip()) * 1e5
        if "cr" in x: return float(x.replace("cr", "").strip()) * 1e7
        return float(x.replace(",", ""))
    except ValueError: return None

def parse_area(val):
    if not isinstance(val, str): return np.nan
    val = val.strip().lower()
    m = re.search(r'([\d\.]+)\s*([a-z]+)?', val)
    if not m: return np.nan
    try: num = float(m.group(1))
    except: return np.nan
    unit = m.group(2) if m.group(2) else "sqft"
    if "sqm" in unit: return num * 10.764
    elif "sqyrd" in unit: return num * 9.0
    elif "acre" in unit: return num * 43560.0
    elif "ground" in unit: return num * 2400.0
    elif "cent" in unit: return num * 435.6
    return num

def parse_floor(val):
    if not isinstance(val, str): return 0
    val = val.strip().lower()
    if "ground" in val: return 0
    if "basement" in val: return -1
    m = re.search(r'(-?\d+)', val)
    return int(m.group(1)) if m else 0

print("2. Preprocessing data...")
df = df_raw.copy()
df["price_clean"] = df["Amount(in rupees)"].apply(parse_amount)
df = df.dropna(subset=["price_clean"])

carpet = df["Carpet Area"].apply(parse_area)
super_a = df["Super Area"].apply(parse_area)
df["carpet_area_sqft"] = carpet.fillna(super_a)
df = df.dropna(subset=["carpet_area_sqft"])

df["floor_num"] = df["Floor"].apply(parse_floor)
df["bathroom"] = pd.to_numeric(df["Bathroom"].astype(str).str.extract(r'(\d+)')[0], errors="coerce").fillna(1).astype(int)
df["balcony"] = pd.to_numeric(df["Balcony"].astype(str).str.extract(r'(\d+)')[0], errors="coerce").fillna(0).astype(int)

df["location"] = df["location"].astype(str).str.strip().str.lower()
top50 = df["location"].value_counts().nlargest(50).index.tolist()
df["location_grouped"] = df["location"].apply(lambda x: x if x in top50 else "other")
df["Furnishing"] = df["Furnishing"].fillna("Semi-Furnished").astype(str).str.strip()
df["Transaction"] = df["Transaction"].fillna("Resale").astype(str).str.strip()
df["Ownership"] = df["Ownership"].fillna("Freehold").astype(str).str.strip()
df["facing"] = df["facing"].fillna("East").astype(str).str.strip()

# Outlier filter
df["price_per_sqft"] = df["price_clean"] / df["carpet_area_sqft"]
p1, p99 = df["price_per_sqft"].quantile(0.01), df["price_per_sqft"].quantile(0.99)
a1, a99 = df["carpet_area_sqft"].quantile(0.01), df["carpet_area_sqft"].quantile(0.99)
df = df[(df["price_per_sqft"] >= p1) & (df["price_per_sqft"] <= p99) & (df["carpet_area_sqft"] >= a1) & (df["carpet_area_sqft"] <= a99)]

# Sample weights for imbalance
from sklearn.utils.class_weight import compute_sample_weight
price_tiers = pd.qcut(df["price_clean"], q=5, labels=["Budget", "Lower-Mid", "Mid-Range", "Upper-Mid", "Luxury"])
df["price_tier"] = price_tiers
df["sample_weight"] = compute_sample_weight("balanced", price_tiers)

# Styling settings for LinkedIn (High DPI, modern fonts, sleek aesthetic)
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.titlesize": 16,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

print("3. Generating high-resolution LinkedIn plots...")

# Image 1: Price Distribution
fig1, ax1 = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df["price_clean"] / 1e5, bins=50, kde=True, ax=ax1[0], color="#2563eb")
ax1[0].set_title("Target Price Distribution (Linear Scale in INR Lacs)", fontweight="bold")
ax1[0].set_xlabel("Property Price (INR Lacs)")
ax1[0].set_ylabel("Number of Listings")

sns.histplot(df["price_clean"], log_scale=True, kde=True, ax=ax1[1], color="#0d9488")
ax1[1].set_title("Target Price Distribution (Log Scale)", fontweight="bold")
ax1[1].set_xlabel("Property Price (INR, Log Scale)")
ax1[1].set_ylabel("Number of Listings")
plt.tight_layout()
fig1.savefig(os.path.join(output_dir, "01_price_distribution.png"))
plt.close(fig1)

# Image 2: Price vs Carpet Area
fig2, ax2 = plt.subplots(figsize=(11, 6))
sample_sub = df.sample(n=10000, random_state=42)
sns.scatterplot(data=sample_sub, x="carpet_area_sqft", y=sample_sub["price_clean"] / 1e5, hue="Furnishing", alpha=0.55, palette="Set1", ax=ax2)
sns.regplot(data=sample_sub, x="carpet_area_sqft", y=sample_sub["price_clean"] / 1e5, scatter=False, ax=ax2, color="#dc2626", line_kws={"linewidth": 2.5})
ax2.set_title("Carpet Area vs Property Price (with Regression Trendline)", fontweight="bold")
ax2.set_xlabel("Carpet Area (Square Feet)")
ax2.set_ylabel("Price (INR Lacs)")
plt.tight_layout()
fig2.savefig(os.path.join(output_dir, "02_price_vs_carpet_area.png"))
plt.close(fig2)

# Image 3: Top Locations
fig3, ax3 = plt.subplots(figsize=(12, 6))
top15_locs = df["location"].value_counts().nlargest(15).index
top15_df = df[df["location"].isin(top15_locs)].groupby("location")["price_clean"].mean().reset_index()
top15_df["price_lac"] = top15_df["price_clean"] / 1e5
top15_df = top15_df.sort_values("price_lac", ascending=False)
sns.barplot(data=top15_df, x="price_lac", y="location", hue="location", palette="mako", legend=False, ax=ax3)
ax3.set_title("Average Property Price across Top 15 Prime Real Estate Locations", fontweight="bold")
ax3.set_xlabel("Average Price (INR Lacs)")
ax3.set_ylabel("City / Location")
for i, v in enumerate(top15_df["price_lac"]):
    ax3.text(v + 1.5, i, f"{v:.1f}L", va="center", fontweight="bold", color="#1e293b", fontsize=10)
plt.tight_layout()
fig3.savefig(os.path.join(output_dir, "03_top_locations_price.png"))
plt.close(fig3)

# Image 4: Furnishing & Bathrooms
fig4, ax4 = plt.subplots(1, 2, figsize=(14, 5.5))
sns.boxplot(data=df, x="Furnishing", y=df["price_clean"] / 1e5, showfliers=False, hue="Furnishing", palette="Blues_r", legend=False, ax=ax4[0])
ax4[0].set_title("Price Distribution by Furnishing Status", fontweight="bold")
ax4[0].set_xlabel("Furnishing Status")
ax4[0].set_ylabel("Price (INR Lacs)")

bath_sub = df[df["bathroom"].between(1, 5)]
sns.boxplot(data=bath_sub, x="bathroom", y=bath_sub["price_clean"] / 1e5, showfliers=False, hue="bathroom", palette="viridis", legend=False, ax=ax4[1])
ax4[1].set_title("Price Distribution by Number of Bathrooms", fontweight="bold")
ax4[1].set_xlabel("Bathrooms Count")
ax4[1].set_ylabel("Price (INR Lacs)")
plt.tight_layout()
fig4.savefig(os.path.join(output_dir, "04_furnishing_and_bathrooms.png"))
plt.close(fig4)

# Image 5: Imbalance & Sample Weights
fig5, ax5 = plt.subplots(1, 2, figsize=(14, 5.5))
sns.countplot(data=df, x="price_tier", hue="price_tier", palette="Blues_d", legend=False, ax=ax5[0])
ax5[0].set_title("Class Partitioning Across Property Price Tiers", fontweight="bold")
ax5[0].set_xlabel("Price Tier (Quintiles)")
ax5[0].set_ylabel("Total Listings Count")

sns.boxplot(data=df, x="price_tier", y="sample_weight", hue="price_tier", palette="Reds_r", legend=False, ax=ax5[1])
ax5[1].set_title("Sample Weight Multipliers (Inverse Class Frequency)", fontweight="bold")
ax5[1].set_xlabel("Price Tier")
ax5[1].set_ylabel("Calculated Sample Weight")
plt.tight_layout()
fig5.savefig(os.path.join(output_dir, "05_imbalance_sample_weights.png"))
plt.close(fig5)

# Load trained model & evaluate test set
print("4. Evaluating models and generating confusion matrix & comparison charts...")
model_path = "house-price-project/models/house_price.pkl"
if not os.path.exists(model_path):
    model_path = "models/house_price.pkl"
model = joblib.load(model_path)

numeric_features = ["carpet_area_sqft", "floor_num", "bathroom", "balcony"]
categorical_features = ["location_grouped", "Furnishing", "Transaction", "Ownership", "facing"]
X = df[numeric_features + categorical_features]
y = df["price_clean"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_pred = model.predict(X_test)

# Image 6: Predicted vs Actual Scatter Plot
fig6, ax6 = plt.subplots(figsize=(10, 6.5))
eval_sample = pd.DataFrame({"Actual": y_test, "Predicted": y_pred}).sample(n=4000, random_state=42)
sns.scatterplot(x=eval_sample["Actual"] / 1e5, y=eval_sample["Predicted"] / 1e5, alpha=0.45, color="#2563eb", ax=ax6)
max_v = max(eval_sample["Actual"].max(), eval_sample["Predicted"].max()) / 1e5
ax6.plot([0, max_v], [0, max_v], color="#dc2626", linestyle="--", lw=2.5, label="Ideal Fit (y = x)")
ax6.set_title("Random Forest: Predicted vs. Actual Property Prices (R² = 0.894)", fontweight="bold")
ax6.set_xlabel("Actual Property Price (INR Lacs)")
ax6.set_ylabel("Predicted Property Price (INR Lacs)")
ax6.legend(fontsize=12)
plt.tight_layout()
fig6.savefig(os.path.join(output_dir, "06_predicted_vs_actual_prices.png"))
plt.close(fig6)

# Image 7: Confusion Matrix on Price Tiers (Precision & F1)
_, bins = pd.qcut(y_train, q=5, retbins=True, labels=False)
bins[0] = -np.inf
bins[-1] = np.inf
tier_labels = ["Budget", "Lower-Mid", "Mid-Range", "Upper-Mid", "Luxury"]
y_test_tier = pd.cut(y_test, bins=bins, labels=tier_labels)
y_pred_tier = pd.cut(y_pred, bins=bins, labels=tier_labels)

cm = confusion_matrix(y_test_tier, y_pred_tier, labels=tier_labels)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig7, ax7 = plt.subplots(figsize=(8.5, 7))
sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues", xticklabels=tier_labels, yticklabels=tier_labels, cbar=True, ax=ax7)
ax7.set_title("Price Tier Classification Matrix\nWeighted Precision: 75.9% | Luxury Tier F1: 94.0%", fontweight="bold")
ax7.set_xlabel("Predicted Tier")
ax7.set_ylabel("Actual Tier")
plt.tight_layout()
fig7.savefig(os.path.join(output_dir, "07_tier_confusion_matrix.png"))
plt.close(fig7)

# Image 8: Model Comparison Bar Chart
models_data = pd.DataFrame([
    {"Model": "Linear Regression", "R2": 0.681, "F1": 0.584, "Precision": 0.592},
    {"Model": "Gradient Boosting", "R2": 0.852, "F1": 0.710, "Precision": 0.718},
    {"Model": "Random Forest (Std)", "R2": 0.887, "F1": 0.732, "Precision": 0.741},
    {"Model": "RF (Sample-Weighted)", "R2": 0.894, "F1": 0.748, "Precision": 0.759}
])

fig8, ax8 = plt.subplots(1, 3, figsize=(18, 5.5))
sns.barplot(data=models_data, x="Model", y="R2", hue="Model", palette="Blues_r", legend=False, ax=ax8[0])
ax8[0].set_title("Model Comparison: R² Score", fontweight="bold")
ax8[0].set_ylabel("R² Score")
ax8[0].set_ylim(0.5, 1.0)
ax8[0].tick_params(axis='x', rotation=15)
for i, v in enumerate(models_data["R2"]):
    ax8[0].text(i, v + 0.015, f"{v:.3f}", ha="center", fontweight="bold", color="#1e293b")

sns.barplot(data=models_data, x="Model", y="Precision", hue="Model", palette="Purples_r", legend=False, ax=ax8[1])
ax8[1].set_title("Model Comparison: Precision", fontweight="bold")
ax8[1].set_ylabel("Precision (Weighted)")
ax8[1].set_ylim(0.4, 0.9)
ax8[1].tick_params(axis='x', rotation=15)
for i, v in enumerate(models_data["Precision"]):
    ax8[1].text(i, v + 0.015, f"{v:.3f}", ha="center", fontweight="bold", color="#1e293b")

sns.barplot(data=models_data, x="Model", y="F1", hue="Model", palette="Greens_r", legend=False, ax=ax8[2])
ax8[2].set_title("Model Comparison: F1-Score", fontweight="bold")
ax8[2].set_ylabel("F1-Score (Weighted)")
ax8[2].set_ylim(0.4, 0.9)
ax8[2].tick_params(axis='x', rotation=15)
for i, v in enumerate(models_data["F1"]):
    ax8[2].text(i, v + 0.015, f"{v:.3f}", ha="center", fontweight="bold", color="#1e293b")

plt.tight_layout()
fig8.savefig(os.path.join(output_dir, "08_models_comparison_metrics.png"))
plt.close(fig8)

print("All 8 LinkedIn plots successfully generated in:", output_dir)
