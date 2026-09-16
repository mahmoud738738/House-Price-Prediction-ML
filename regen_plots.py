import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import re
from sklearn.utils.class_weight import compute_sample_weight

output_dir = "figure"
os.makedirs(output_dir, exist_ok=True)

df_raw = pd.read_csv("notebooks/data/house_prices.csv")

def parse_amount(val):
    if not isinstance(val, str): return np.nan
    val = val.strip().lower()
    if val == "" or val == "price on request": return np.nan
    m = re.search(r'([\d\.]+)\s*(lac|cr)?', val)
    if not m: return np.nan
    try: num = float(m.group(1))
    except: return np.nan
    unit = m.group(2)
    if unit == "lac": return num * 100000.0
    elif unit == "cr": return num * 10000000.0
    return num

def parse_area(val):
    if not isinstance(val, str): return np.nan
    val = val.strip().lower()
    m = re.search(r'([\d\.]+)\s*([a-z]+)?', val)
    if not m: return np.nan
    try: num = float(m.group(1))
    except ValueError: return np.nan
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

df = df_raw.copy()
df = df.drop(columns=["Index"], errors="ignore")
df = df.drop_duplicates().reset_index(drop=True)

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

df["price_per_sqft"] = df["price_clean"] / df["carpet_area_sqft"]
p1, p99 = df["price_per_sqft"].quantile(0.01), df["price_per_sqft"].quantile(0.99)
a1, a99 = df["carpet_area_sqft"].quantile(0.01), df["carpet_area_sqft"].quantile(0.99)
df = df[(df["price_per_sqft"] >= p1) & (df["price_per_sqft"] <= p99) & (df["carpet_area_sqft"] >= a1) & (df["carpet_area_sqft"] <= a99)]

bins = [0, 4_500_000, 7_500_000, 12_500_000, 25_000_000, float("inf")]
labels = ["Budget (Q1)", "Lower-Mid (Q2)", "Mid-Range (Q3)", "Upper-Mid (Q4)", "Luxury (Q5)"]
df["price_tier"] = pd.cut(df["price_clean"], bins=bins, labels=labels, include_lowest=True)
df["sample_weight"] = compute_sample_weight("balanced", df["price_tier"])

sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#f8fafc", "figure.facecolor": "#ffffff"})
fig5, ax5 = plt.subplots(1, 2, figsize=(14, 5.5))
sns.countplot(data=df, x="price_tier", hue="price_tier", palette="Blues_d", legend=False, ax=ax5[0])
ax5[0].set_title("Class Partitioning Across Property Price Tiers", fontweight="bold")
ax5[0].set_xlabel("Price Tier (Fixed Custom)")
ax5[0].set_ylabel("Total Listings Count")
ax5[0].tick_params(axis='x', rotation=15)

sns.boxplot(data=df, x="price_tier", y="sample_weight", hue="price_tier", palette="Reds_r", legend=False, ax=ax5[1])
ax5[1].set_title("Sample Weight Multipliers (Inverse Class Frequency)", fontweight="bold")
ax5[1].set_xlabel("Price Tier")
ax5[1].set_ylabel("Calculated Sample Weight")
ax5[1].tick_params(axis='x', rotation=15)
plt.tight_layout()
fig5.savefig(os.path.join(output_dir, "05_imbalance_sample_weights.png"), dpi=300)
plt.close(fig5)

model = joblib.load("models/house_price.pkl")

numeric_features = ["carpet_area_sqft", "floor_num", "bathroom", "balcony"]
categorical_features = ["location_grouped", "Furnishing", "Transaction", "Ownership", "facing"]
X = df[numeric_features + categorical_features]
y = df["price_clean"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_pred = model.predict(X_test)

tier_labels = ["Budget", "Lower-Mid", "Mid-Range", "Upper-Mid", "Luxury"]
y_test_tier = pd.cut(y_test, bins=bins, labels=tier_labels, include_lowest=True)
y_pred_tier = pd.cut(y_pred, bins=bins, labels=tier_labels, include_lowest=True)

cm = confusion_matrix(y_test_tier, y_pred_tier, labels=tier_labels)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig7, ax7 = plt.subplots(figsize=(8.5, 7))
sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues", xticklabels=tier_labels, yticklabels=tier_labels, cbar=True, ax=ax7)
ax7.set_title("Price Tier Classification Matrix", fontweight="bold")
ax7.set_xlabel("Predicted Tier")
ax7.set_ylabel("Actual Tier")
plt.tight_layout()
fig7.savefig(os.path.join(output_dir, "07_tier_confusion_matrix.png"), dpi=300)
plt.close(fig7)
print("Finished generating plots.")
