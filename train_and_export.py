import os
import sys
import json
import base64
import io
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.utils.class_weight import compute_sample_weight

print("Starting complete ML workflow...")

# 1. Load Data
data_path = r"notebooks/data/house_prices.csv"
print(f"Loading data from {data_path}...")
df_raw = pd.read_csv(data_path)
print(f"Raw shape: {df_raw.shape}")

# Preprocessing functions
def parse_amount(x):
    if not isinstance(x, str):
        return None
    x = x.strip().lower()
    try:
        if "lac" in x:
            return float(x.replace("lac", "").strip()) * 1e5
        if "cr" in x:
            return float(x.replace("cr", "").strip()) * 1e7
        return float(x.replace(",", ""))
    except ValueError:
        return None

def parse_area(val):
    if not isinstance(val, str):
        return np.nan
    val = val.strip().lower()
    m = re.search(r'([\d\.]+)\s*([a-z]+)?', val)
    if not m:
        return np.nan
    try:
        num = float(m.group(1))
    except ValueError:
        return np.nan
    unit = m.group(2) if m.group(2) else "sqft"
    if "sqm" in unit:
        return num * 10.764
    elif "sqyrd" in unit:
        return num * 9.0
    elif "acre" in unit:
        return num * 43560.0
    elif "ground" in unit:
        return num * 2400.0
    elif "cent" in unit:
        return num * 435.6
    return num

def parse_floor(val):
    if not isinstance(val, str):
        return 0
    val = val.strip().lower()
    if "ground" in val:
        return 0
    if "basement" in val:
        return -1
    m = re.search(r'(-?\d+)', val)
    return int(m.group(1)) if m else 0

df = df_raw.copy()
df = df.drop(columns=["Index"], errors="ignore")
dups_count = df.duplicated().sum()
df = df.drop_duplicates().reset_index(drop=True)
print(f"Removed {dups_count:,} duplicate rows. Remaining unique listings: {df.shape[0]:,}")

# Cleaning
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

# Outliers filter
df["price_per_sqft"] = df["price_clean"] / df["carpet_area_sqft"]
p1, p99 = df["price_per_sqft"].quantile(0.01), df["price_per_sqft"].quantile(0.99)
a1, a99 = df["carpet_area_sqft"].quantile(0.01), df["carpet_area_sqft"].quantile(0.99)
df = df[(df["price_per_sqft"] >= p1) & (df["price_per_sqft"] <= p99) & (df["carpet_area_sqft"] >= a1) & (df["carpet_area_sqft"] <= a99)]

print(f"Cleaned dataset shape: {df.shape}")

# Feature columns
numeric_features = ["carpet_area_sqft", "floor_num", "bathroom", "balcony"]
categorical_features = ["location_grouped", "Furnishing", "Transaction", "Ownership", "facing"]

X = df[numeric_features + categorical_features]
y = df["price_clean"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Imbalance weighting (Data-driven thresholds)
bins = [0, 4_500_000, 7_500_000, 12_500_000, 25_000_000, float("inf")]
labels = ["Budget (Q1)", "Lower-Mid (Q2)", "Mid-Range (Q3)", "Upper-Mid (Q4)", "Luxury (Q5)"]
price_bins_train = pd.cut(y_train, bins=bins, labels=labels, include_lowest=True)
sample_weights_train = compute_sample_weight('balanced', price_bins_train)

# Preprocessor
preprocessor = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_features),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
])

# Define models
models = {
    "LinearRegression": Pipeline([
        ("prep", preprocessor),
        ("reg", LinearRegression())
    ]),
    "RandomForest": Pipeline([
        ("prep", preprocessor),
        ("reg", RandomForestRegressor(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1))
    ]),
    "RandomForest_Weighted": Pipeline([
        ("prep", preprocessor),
        ("reg", RandomForestRegressor(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1))
    ]),
    "GradientBoosting": Pipeline([
        ("prep", preprocessor),
        ("reg", GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42))
    ])
}

results = {}
preds = {}

print("Training LinearRegression...")
models["LinearRegression"].fit(X_train, y_train)
preds["LinearRegression"] = models["LinearRegression"].predict(X_test)
results["LinearRegression"] = {
    "MAE": float(mean_absolute_error(y_test, preds["LinearRegression"])),
    "RMSE": float(root_mean_squared_error(y_test, preds["LinearRegression"])),
    "R2": float(r2_score(y_test, preds["LinearRegression"]))
}

print("Training RandomForest (Standard)...")
models["RandomForest"].fit(X_train, y_train)
preds["RandomForest"] = models["RandomForest"].predict(X_test)
results["RandomForest"] = {
    "MAE": float(mean_absolute_error(y_test, preds["RandomForest"])),
    "RMSE": float(root_mean_squared_error(y_test, preds["RandomForest"])),
    "R2": float(r2_score(y_test, preds["RandomForest"]))
}

print("Training RandomForest (Minority Sample-Weighted)...")
models["RandomForest_Weighted"].fit(X_train, y_train, reg__sample_weight=sample_weights_train)
preds["RandomForest_Weighted"] = models["RandomForest_Weighted"].predict(X_test)
results["RandomForest_Weighted"] = {
    "MAE": float(mean_absolute_error(y_test, preds["RandomForest_Weighted"])),
    "RMSE": float(root_mean_squared_error(y_test, preds["RandomForest_Weighted"])),
    "R2": float(r2_score(y_test, preds["RandomForest_Weighted"]))
}

print("Training GradientBoosting...")
models["GradientBoosting"].fit(X_train, y_train)
preds["GradientBoosting"] = models["GradientBoosting"].predict(X_test)
results["GradientBoosting"] = {
    "MAE": float(mean_absolute_error(y_test, preds["GradientBoosting"])),
    "RMSE": float(root_mean_squared_error(y_test, preds["GradientBoosting"])),
    "R2": float(r2_score(y_test, preds["GradientBoosting"]))
}

print("Running 5-fold CV on RandomForest...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(models["RandomForest"], X_train.iloc[:20000], y_train.iloc[:20000], cv=kf, scoring='r2', n_jobs=-1)
cv_mean = float(cv_scores.mean())
cv_std = float(cv_scores.std())

print("Evaluation Results:")
for name, metric in results.items():
    print(f"{name:25s} | MAE: {metric['MAE']:,.2f} | RMSE: {metric['RMSE']:,.2f} | R2: {metric['R2']:.4f}")
print(f"5-Fold CV R2 on RandomForest: {cv_mean:.4f} (+/- {cv_std:.4f})")

# Determine winner
best_name = max(results, key=lambda k: results[k]["R2"])
best_model = models[best_name]
print(f"Winner: {best_name}")

# Export model & locations
os.makedirs("models", exist_ok=True)
os.makedirs("backend/models", exist_ok=True)
joblib.dump(best_model, "models/house_price.pkl", compress=3)
joblib.dump(best_model, "backend/models/house_price.pkl", compress=3)

locations_list = sorted(df["location_grouped"].unique().tolist())
with open("locations.json", "w") as f:
    json.dump(locations_list, f, indent=2)
with open("backend/locations.json", "w") as f:
    json.dump(locations_list, f, indent=2)

# Save metrics summary
summary = {
    "results": results,
    "cv_mean_r2": cv_mean,
    "cv_std_r2": cv_std,
    "winner": best_name,
    "sample_test_prediction": float(best_model.predict(X_test.iloc[[0]])[0]),
    "sample_actual": float(y_test.iloc[0])
}
with open("metrics_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("Exported house_price.pkl, locations.json, and metrics_summary.json successfully!")
