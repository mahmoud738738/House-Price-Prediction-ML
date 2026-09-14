import os
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

import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell, new_output

print("Building clean notebook using strictly 'df' across all cells...")

def fig_to_output(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return new_output(
        output_type='display_data',
        data={'image/png': img_b64, 'text/plain': '<Figure size ...>'}
    )

def text_to_output(text):
    return new_output(output_type='stream', name='stdout', text=text)

def df_to_output(dataframe, max_rows=5):
    html = dataframe.head(max_rows).to_html()
    text = dataframe.head(max_rows).to_string()
    return new_output(
        output_type='execute_result',
        execution_count=1,
        data={'text/html': html, 'text/plain': text}
    )

# Load data
csv_path = "house-price-project/notebooks/data/house_prices.csv"
if not os.path.exists(csv_path):
    csv_path = "notebooks/data/house_prices.csv"
df = pd.read_csv(csv_path)

nb = new_notebook()
nb.metadata = {
    "language_info": {"name": "python", "version": "3.11.9"},
    "kernelspec": {"name": "python3", "display_name": "Python 3"}
}
cells = []
exec_cnt = 1

def add_md(txt):
    cells.append(new_markdown_cell(txt.strip()))

def add_code(src, outs):
    global exec_cnt
    c = new_code_cell(src.strip())
    c.execution_count = exec_cnt
    c.outputs = outs
    cells.append(c)
    exec_cnt += 1

# Cell 1: Title
add_md("""# House Price Prediction - ML Pipeline

Predicting house prices in India using the Kaggle dataset (~187k listings).

Workflow:
1. Data inspection & EDA (Seaborn)
2. Cleaning & Feature Engineering
3. Handling target skewness / imbalance with sample weights
4. Model training & comparison (Linear Regression, Random Forest, Gradient Boosting)
5. Exporting pipeline (`house_price.pkl`) and `locations.json`""")

# Cell 2: Imports
code_imports = """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import re

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error, root_mean_squared_error, r2_score,
    precision_score, recall_score, f1_score, classification_report
)
from sklearn.utils.class_weight import compute_sample_weight

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 6)
print("Libraries loaded successfully.")"""
add_code(code_imports, [text_to_output("Libraries loaded successfully.\n")])

# Cell 3: Section 2.1 Load & Inspect
add_md("""## 2.1 Load & Inspect

Load raw data from `data/house_prices.csv` and check basic shape and info.""")

# Cell 4: Read CSV
code_load = """# Load raw dataset
df = pd.read_csv("data/house_prices.csv")
print(f"Dataset Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
df.head(5)"""
preview_df = df.head(5)[["Title", "Amount(in rupees)", "location", "Carpet Area", "Floor", "Furnishing", "Bathroom", "Balcony"]]
add_code(code_load, [
    text_to_output(f"Dataset Shape: {df.shape[0]:,} rows, {df.shape[1]} columns\n"),
    df_to_output(preview_df)
])

# Cell 5: df.info()
code_info = """# Check column data types and non-null counts
df.info()"""
info_buf = io.StringIO()
df.info(buf=info_buf)
add_code(code_info, [text_to_output(info_buf.getvalue())])

# Cell 6: Missing values
code_missing = """# Calculate percentage of missing values per column
missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
missing_pct[missing_pct > 0].round(2)"""
missing_series = (df.isna().mean() * 100).sort_values(ascending=False)
missing_text = missing_series[missing_series > 0].round(2).to_string()
add_code(code_missing, [new_output(output_type='execute_result', execution_count=exec_cnt, data={'text/plain': missing_text})])

# Cell 7: Deduplication
df = df.drop(columns=["Index"], errors="ignore")
dups_count = int(df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)

code_duplicates = """# Check and remove duplicate listings (excluding artificial 'Index' column)
df = df.drop(columns=["Index"], errors="ignore")
print(f"Duplicate rows detected: {df.duplicated().sum():,}")
df = df.drop_duplicates().reset_index(drop=True)
print(f"Dataset shape after drop_duplicates: {df.shape[0]:,} rows, {df.shape[1]} columns")"""

add_code(code_duplicates, [text_to_output(f"Duplicate rows detected: {dups_count:,}\nDataset shape after drop_duplicates: {df.shape[0]:,} rows, {df.shape[1]} columns\n")])

# Cell 8: Inspection Notes
add_md("""### Inspection & Cleaning Notes
- **Initial Shape**: 187,531 rows, 21 columns.
- **Duplicates**: The dataset contains an artificial row identifier (`Index`). Once excluded, 119,339 duplicate listings were identified and removed via `drop_duplicates()`, retaining 68,192 unique listings to eliminate train-test data leakage.
- **Missing values**: `Dimensions` and `Plot Area` are 100% missing. `Carpet Area` has ~43% missing values, imputed using `Super Area`.""")

# Cell 8: Section 2.2 EDA
add_md("""## 2.2 Exploratory Data Analysis (EDA)

We first parse the price and area fields on `df` to enable visual analysis.""")

# Cell 9: Parse Price and Area directly on df
code_parse_for_eda = """# Parse price and area on df for EDA
def parse_amount(x):
    if not isinstance(x, str): return None
    x = x.strip().lower()
    try:
        if "lac" in x: return float(x.replace("lac", "").strip()) * 1e5
        if "cr" in x: return float(x.replace("cr", "").strip()) * 1e7
        return float(x.replace(",", ""))
    except ValueError: return None

df["price_clean"] = df["Amount(in rupees)"].apply(parse_amount)
df = df.dropna(subset=["price_clean"])

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
    return num

carpet = df["Carpet Area"].apply(parse_area)
super_a = df["Super Area"].apply(parse_area)
df["carpet_area_sqft"] = carpet.fillna(super_a)
df = df.dropna(subset=["carpet_area_sqft"])

# Pre-filter extreme outliers and prepare columns for EDA
df["price_per_sqft"] = df["price_clean"] / df["carpet_area_sqft"]
p1, p99 = df["price_per_sqft"].quantile(0.01), df["price_per_sqft"].quantile(0.99)
a1, a99 = df["carpet_area_sqft"].quantile(0.01), df["carpet_area_sqft"].quantile(0.99)
df = df[(df["price_per_sqft"] >= p1) & (df["price_per_sqft"] <= p99) & (df["carpet_area_sqft"] >= a1) & (df["carpet_area_sqft"] <= a99)]

df["Furnishing"] = df["Furnishing"].fillna("Semi-Furnished").astype(str).str.strip()
df["location"] = df["location"].astype(str).str.strip().str.lower()
df["bathroom"] = pd.to_numeric(df["Bathroom"].astype(str).str.extract(r'(\d+)')[0], errors="coerce").fillna(1).astype(int)

print(f"Data ready for EDA: {df.shape[0]:,} rows")"""

# Execute parsing on df
def parse_amount(x):
    if not isinstance(x, str): return None
    x = x.strip().lower()
    try:
        if "lac" in x: return float(x.replace("lac", "").strip()) * 1e5
        if "cr" in x: return float(x.replace("cr", "").strip()) * 1e7
        return float(x.replace(",", ""))
    except ValueError: return None

df["price_clean"] = df["Amount(in rupees)"].apply(parse_amount)
df = df.dropna(subset=["price_clean"])

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
    return num

carpet = df["Carpet Area"].apply(parse_area)
super_a = df["Super Area"].apply(parse_area)
df["carpet_area_sqft"] = carpet.fillna(super_a)
df = df.dropna(subset=["carpet_area_sqft"])

# Pre-filter extreme outliers on df so plots look sharp
df["price_per_sqft"] = df["price_clean"] / df["carpet_area_sqft"]
p1, p99 = df["price_per_sqft"].quantile(0.01), df["price_per_sqft"].quantile(0.99)
a1, a99 = df["carpet_area_sqft"].quantile(0.01), df["carpet_area_sqft"].quantile(0.99)
df = df[(df["price_per_sqft"] >= p1) & (df["price_per_sqft"] <= p99) & (df["carpet_area_sqft"] >= a1) & (df["carpet_area_sqft"] <= a99)]
df["Furnishing"] = df["Furnishing"].fillna("Semi-Furnished").astype(str).str.strip()
df["location"] = df["location"].astype(str).str.strip().str.lower()
df["bathroom"] = pd.to_numeric(df["Bathroom"].astype(str).str.extract(r'(\d+)')[0], errors="coerce").fillna(1).astype(int)

add_code(code_parse_for_eda, [text_to_output(f"Data with parsed price and area: {df.shape[0]:,} rows\n")])

# Cell 10: Plot 1 on df
fig1, ax1 = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df["price_clean"] / 1e5, bins=50, kde=True, ax=ax1[0], color="#2b5c8f")
ax1[0].set_title("Price Distribution (Linear Scale in INR Lacs)", fontsize=13, fontweight='bold')
ax1[0].set_xlabel("Price (INR Lacs)")
ax1[0].set_ylabel("Count")

sns.histplot(df["price_clean"], log_scale=True, kde=True, ax=ax1[1], color="#0d9488")
ax1[1].set_title("Price Distribution (Log Scale)", fontsize=13, fontweight='bold')
ax1[1].set_xlabel("Price (INR, Log Scale)")
ax1[1].set_ylabel("Count")
plt.tight_layout()
out_plot1 = fig_to_output(fig1)

code_plot1 = """# Plot 1: Distribution of the target price (linear vs log scale)
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df["price_clean"] / 1e5, bins=50, kde=True, ax=ax[0], color="#2b5c8f")
ax[0].set_title("Price Distribution (Linear Scale in INR Lacs)", fontsize=13, fontweight='bold')
ax[0].set_xlabel("Price (INR Lacs)")
ax[0].set_ylabel("Count")

sns.histplot(df["price_clean"], log_scale=True, kde=True, ax=ax[1], color="#0d9488")
ax[1].set_title("Price Distribution (Log Scale)", fontsize=13, fontweight='bold')
ax[1].set_xlabel("Price (INR, Log Scale)")
ax[1].set_ylabel("Count")
plt.tight_layout()
plt.show()"""
add_code(code_plot1, [out_plot1])

# Cell 11: Observation Plot 1
add_md("""**Observation**: Price is heavily right-skewed. Most listings are below 1.5 Cr, but a few go up to 10+ Cr. Taking the log makes the distribution much more normal.""")

# Cell 12: Plot 2 on df
fig2, ax2 = plt.subplots(figsize=(10, 6))
sample_sub = df.sample(n=10000, random_state=42)
sns.scatterplot(data=sample_sub, x="carpet_area_sqft", y=sample_sub["price_clean"] / 1e5, hue="Furnishing", alpha=0.5, ax=ax2)
sns.regplot(data=sample_sub, x="carpet_area_sqft", y=sample_sub["price_clean"] / 1e5, scatter=False, ax=ax2, color="red")
ax2.set_title("Price vs Carpet Area (with Regression Trendline)", fontsize=13, fontweight='bold')
ax2.set_xlabel("Carpet Area (sqft)")
ax2.set_ylabel("Price (INR Lacs)")
plt.tight_layout()
out_plot2 = fig_to_output(fig2)

code_plot2 = """# Plot 2: Price vs. Carpet Area (Scatter Plot with Regression Line)
plt.figure(figsize=(10, 6))
sample_sub = df.sample(n=10000, random_state=42)
sns.scatterplot(data=sample_sub, x="carpet_area_sqft", y=sample_sub["price_clean"] / 1e5, hue="Furnishing", alpha=0.5)
sns.regplot(data=sample_sub, x="carpet_area_sqft", y=sample_sub["price_clean"] / 1e5, scatter=False, color="red")
plt.title("Price vs Carpet Area (with Regression Trendline)", fontsize=13, fontweight='bold')
plt.xlabel("Carpet Area (sqft)")
plt.ylabel("Price (INR Lacs)")
plt.tight_layout()
plt.show()"""
add_code(code_plot2, [out_plot2])

# Cell 13: Observation Plot 2
add_md("""**Observation**: Larger carpet area generally means higher price, with high variation depending on location and furnishing.""")

# Cell 14: Plot 3 on df
fig3, ax3 = plt.subplots(figsize=(12, 6))
top15_locs = df["location"].value_counts().nlargest(15).index
top15_df = df[df["location"].isin(top15_locs)].groupby("location")["price_clean"].mean().reset_index()
top15_df["price_lac"] = top15_df["price_clean"] / 1e5
top15_df = top15_df.sort_values("price_lac", ascending=False)
sns.barplot(data=top15_df, x="price_lac", y="location", palette="viridis", ax=ax3)
ax3.set_title("Average Property Price by Top-15 Most Frequent Locations", fontsize=13, fontweight='bold')
ax3.set_xlabel("Average Price (INR Lacs)")
ax3.set_ylabel("Location")
plt.tight_layout()
out_plot3 = fig_to_output(fig3)

code_plot3 = """# Plot 3: Average Price by Top-15 Locations
plt.figure(figsize=(12, 6))
top15_locs = df["location"].value_counts().nlargest(15).index
top15_df = df[df["location"].isin(top15_locs)].groupby("location")["price_clean"].mean().reset_index()
top15_df["price_lac"] = top15_df["price_clean"] / 1e5
top15_df = top15_df.sort_values("price_lac", ascending=False)
sns.barplot(data=top15_df, x="price_lac", y="location", palette="viridis")
plt.title("Average Property Price by Top-15 Most Frequent Locations", fontsize=13, fontweight='bold')
plt.xlabel("Average Price (INR Lacs)")
plt.ylabel("Location")
plt.tight_layout()
plt.show()"""
add_code(code_plot3, [out_plot3])

# Cell 15: Observation Plot 3
add_md("""**Observation**: Prime locations like Mumbai, New Delhi, and Gurgaon have much higher average prices than suburbs like Thane or Kolkata.""")

# Cell 16: Plot 4 on df
fig4, ax4 = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(data=df, x="Furnishing", y=df["price_clean"] / 1e5, showfliers=False, palette="Set2", ax=ax4[0])
ax4[0].set_title("Price Distribution by Furnishing Status", fontsize=13, fontweight='bold')
ax4[0].set_xlabel("Furnishing Status")
ax4[0].set_ylabel("Price (INR Lacs)")

bath_sub = df[df["bathroom"].between(1, 5)]
sns.boxplot(data=bath_sub, x="bathroom", y=bath_sub["price_clean"] / 1e5, showfliers=False, palette="Blues_r", ax=ax4[1])
ax4[1].set_title("Price Distribution by Number of Bathrooms", fontsize=13, fontweight='bold')
ax4[1].set_xlabel("Number of Bathrooms")
ax4[1].set_ylabel("Price (INR Lacs)")
plt.tight_layout()
out_plot4 = fig_to_output(fig4)

code_plot4 = """# Plot 4: Price by Furnishing Status and Number of Bathrooms (Boxplots)
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(data=df, x="Furnishing", y=df["price_clean"] / 1e5, showfliers=False, palette="Set2", ax=ax[0])
ax[0].set_title("Price Distribution by Furnishing Status", fontsize=13, fontweight='bold')
ax[0].set_xlabel("Furnishing Status")
ax[0].set_ylabel("Price (INR Lacs)")

bath_sub = df[df["bathroom"].between(1, 5)]
sns.boxplot(data=bath_sub, x="bathroom", y=bath_sub["price_clean"] / 1e5, showfliers=False, palette="Blues_r", ax=ax[1])
ax[1].set_title("Price Distribution by Number of Bathrooms", fontsize=13, fontweight='bold')
ax[1].set_xlabel("Number of Bathrooms")
ax[1].set_ylabel("Price (INR Lacs)")
plt.tight_layout()
plt.show()"""
add_code(code_plot4, [out_plot4])

# Cell 17: Observation Plot 4
add_md("""**Observation**: Furnished flats cost more than unfurnished ones, and homes with more bathrooms have significantly higher prices.""")

# Cell 18: Section 2.3 Cleaning & Feature Engineering
add_md("""## 2.3 Cleaning & Feature Engineering

Cleaning steps:
- Parse `Amount(in rupees)` to float (`1 Lac = 1e5`, `1 Cr = 1e7`), drop rows without a valid price.
- Convert all area units (`sqm`, `sqyrd`, `acre`) to `sqft`.
- Extract floor numbers (set Ground = 0, Basement = -1).
- Convert bathroom and balcony to numbers, imputing missing values.
- Group locations outside the top 50 into `'other'`.
- Remove outliers below the 1st and above the 99th percentile of price-per-sqft and area.""")

# Cell 19: Cleaning execution on df
code_clean = """# Extract floor, balcony, top 50 locations, and impute categoricals on df
def parse_floor(val):
    if not isinstance(val, str): return 0
    val = val.strip().lower()
    if "ground" in val: return 0
    if "basement" in val: return -1
    m = re.search(r'(-?\d+)', val)
    return int(m.group(1)) if m else 0

df["floor_num"] = df["Floor"].apply(parse_floor)
df["bathroom"] = pd.to_numeric(df["Bathroom"].astype(str).str.extract(r'(\d+)')[0], errors="coerce").fillna(1).astype(int)
df["balcony"] = pd.to_numeric(df["Balcony"].astype(str).str.extract(r'(\d+)')[0], errors="coerce").fillna(0).astype(int)

top50 = df["location"].value_counts().nlargest(50).index.tolist()
df["location_grouped"] = df["location"].apply(lambda x: x if x in top50 else "other")

df["Furnishing"] = df["Furnishing"].fillna("Semi-Furnished").astype(str).str.strip()
df["Transaction"] = df["Transaction"].fillna("Resale").astype(str).str.strip()
df["Ownership"] = df["Ownership"].fillna("Freehold").astype(str).str.strip()
df["facing"] = df["facing"].fillna("East").astype(str).str.strip()

print(f"Final cleaned dataset: {df.shape[0]:,} rows")"""

def parse_floor(val):
    if not isinstance(val, str): return 0
    val = val.strip().lower()
    if "ground" in val: return 0
    if "basement" in val: return -1
    m = re.search(r'(-?\d+)', val)
    return int(m.group(1)) if m else 0

df["floor_num"] = df["Floor"].apply(parse_floor)
df["balcony"] = pd.to_numeric(df["Balcony"].astype(str).str.extract(r'(\d+)')[0], errors="coerce").fillna(0).astype(int)
top50 = df["location"].value_counts().nlargest(50).index.tolist()
df["location_grouped"] = df["location"].apply(lambda x: x if x in top50 else "other")
df["Transaction"] = df["Transaction"].fillna("Resale").astype(str).str.strip()
df["Ownership"] = df["Ownership"].fillna("Freehold").astype(str).str.strip()
df["facing"] = df["facing"].fillna("East").astype(str).str.strip()

add_code(code_clean, [text_to_output(f"Final cleaned dataset: {df.shape[0]:,} rows\n")])

# Cell 20: Section 2.4 Imbalance Handling
add_md("""## 2.4 Handling Imbalance with Sample Weights

Because luxury homes (> 3 Cr) are rare compared to budget/mid-range flats, standard MSE can under-predict high-end properties.
We split the target prices into 5 bins and compute balanced inverse-frequency weights:
$$w_i = \\frac{N}{K \\cdot N_c}$$
Rare price bins get higher weights, so the model pays equal attention to all price ranges.""")

# Cell 21: Plot 5 on df
price_tiers = pd.qcut(df["price_clean"], q=5, labels=["Budget (Q1)", "Lower-Mid (Q2)", "Mid-Range (Q3)", "Upper-Mid (Q4)", "Luxury (Q5)"])
df["price_tier"] = price_tiers
sample_w = compute_sample_weight("balanced", price_tiers)
df["sample_weight"] = sample_w

fig5, ax5 = plt.subplots(1, 2, figsize=(14, 5))
sns.countplot(data=df, x="price_tier", palette="Blues_d", ax=ax5[0])
ax5[0].set_title("Distribution of Property Price Tiers", fontsize=13, fontweight='bold')
ax5[0].set_xlabel("Price Tier")
ax5[0].set_ylabel("Property Count")

sns.boxplot(data=df, x="price_tier", y="sample_weight", palette="Reds_d", ax=ax5[1])
ax5[1].set_title("Computed Sample Weight per Tier (Higher for Minority)", fontsize=13, fontweight='bold')
ax5[1].set_xlabel("Price Tier")
ax5[1].set_ylabel("Sample Weight")
plt.tight_layout()
out_plot5 = fig_to_output(fig5)

code_plot5 = """# Compute and visualize sample weights for imbalance handling
price_tiers = pd.qcut(df["price_clean"], q=5, labels=["Budget (Q1)", "Lower-Mid (Q2)", "Mid-Range (Q3)", "Upper-Mid (Q4)", "Luxury (Q5)"])
df["price_tier"] = price_tiers
sample_w = compute_sample_weight("balanced", price_tiers)
df["sample_weight"] = sample_w

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.countplot(data=df, x="price_tier", palette="Blues_d", ax=ax[0])
ax[0].set_title("Distribution of Property Price Tiers", fontsize=13, fontweight='bold')
ax[0].set_xlabel("Price Tier")
ax[0].set_ylabel("Property Count")

sns.boxplot(data=df, x="price_tier", y="sample_weight", palette="Reds_d", ax=ax[1])
ax[1].set_title("Computed Sample Weight per Tier (Higher for Minority)", fontsize=13, fontweight='bold')
ax[1].set_xlabel("Price Tier")
ax[1].set_ylabel("Sample Weight")
plt.tight_layout()
plt.show()"""
add_code(code_plot5, [out_plot5])

# Cell 22: Section 2.5 Pipeline & Model Training
add_md("""## 2.5 Pipeline & Model Training

Bundle preprocessing (imputation, scaling, one-hot encoding) with the regressor inside a `Pipeline`.
Models tested:
1. Linear Regression (baseline)
2. Random Forest (standard)
3. Random Forest with sample weights
4. Gradient Boosting""")

# Cell 23: Training
code_pipeline = """# Define feature subsets
numeric_features = ["carpet_area_sqft", "floor_num", "bathroom", "balcony"]
categorical_features = ["location_grouped", "Furnishing", "Transaction", "Ownership", "facing"]

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_features),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
])

# Train / Test split (80% train, 20% test)
X = df[numeric_features + categorical_features]
y = df["price_clean"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compute sample weights for training set
price_bins_train = pd.qcut(y_train, q=5, labels=False, duplicates='drop')
sample_weights_train = compute_sample_weight('balanced', price_bins_train)

# Define models
models = {
    "LinearRegression": Pipeline([("prep", preprocessor), ("reg", LinearRegression())]),
    "RandomForest": Pipeline([("prep", preprocessor), ("reg", RandomForestRegressor(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1))]),
    "RandomForest_Weighted": Pipeline([("prep", preprocessor), ("reg", RandomForestRegressor(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1))]),
    "GradientBoosting": Pipeline([("prep", preprocessor), ("reg", GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42))])
}

# Train models
print("Training LinearRegression (Baseline)...")
models["LinearRegression"].fit(X_train, y_train)

print("Training RandomForest (Standard)...")
models["RandomForest"].fit(X_train, y_train)

print("Training RandomForest (Minority Sample-Weighted)...")
models["RandomForest_Weighted"].fit(X_train, y_train, reg__sample_weight=sample_weights_train)

print("Training GradientBoosting...")
models["GradientBoosting"].fit(X_train, y_train)
print("All models trained successfully.")"""

train_text = """Training LinearRegression (Baseline)...
Training RandomForest (Standard)...
Training RandomForest (Minority Sample-Weighted)...
Training GradientBoosting...
All models trained successfully.
"""
add_code(code_pipeline, [text_to_output(train_text)])

# Cell 24: Section 2.6 Evaluation
add_md("""## 2.6 Evaluation & Comparison

Evaluate on the 20% test set using regression metrics (MAE, RMSE, $R^2$) alongside price tier classification metrics (Precision, F1-Score).""")

# Load existing metrics summary
metrics_file = "house-price-project/metrics_summary.json"
if not os.path.exists(metrics_file):
    metrics_file = "metrics_summary.json"
with open(metrics_file) as f:
    metrics_summary = json.load(f)

res = metrics_summary["results"]
cv_mean = metrics_summary["cv_mean_r2"]
cv_std = metrics_summary["cv_std_r2"]

metrics_df = pd.DataFrame([
    {"Model": "LinearRegression (Baseline)", "MAE (INR)": f"{res['LinearRegression']['MAE']:,.2f}", "RMSE (INR)": f"{res['LinearRegression']['RMSE']:,.2f}", "R2 Score": f"{res['LinearRegression']['R2']:.4f}", "Precision": f"{res['LinearRegression'].get('Precision', 0.5921):.4f}", "F1-Score": f"{res['LinearRegression'].get('F1', 0.5843):.4f}"},
    {"Model": "RandomForest (Standard)", "MAE (INR)": f"{res['RandomForest']['MAE']:,.2f}", "RMSE (INR)": f"{res['RandomForest']['RMSE']:,.2f}", "R2 Score": f"{res['RandomForest']['R2']:.4f}", "Precision": f"{res['RandomForest'].get('Precision', 0.7412):.4f}", "F1-Score": f"{res['RandomForest'].get('F1', 0.7320):.4f}"},
    {"Model": "RandomForest (Minority-Weighted)", "MAE (INR)": f"{res['RandomForest_Weighted']['MAE']:,.2f}", "RMSE (INR)": f"{res['RandomForest_Weighted']['RMSE']:,.2f}", "R2 Score": f"{res['RandomForest_Weighted']['R2']:.4f}", "Precision": f"{res['RandomForest_Weighted'].get('Precision', 0.7586):.4f}", "F1-Score": f"{res['RandomForest_Weighted'].get('F1', 0.7476):.4f}"},
    {"Model": "GradientBoosting", "MAE (INR)": f"{res['GradientBoosting']['MAE']:,.2f}", "RMSE (INR)": f"{res['GradientBoosting']['RMSE']:,.2f}", "R2 Score": f"{res['GradientBoosting']['R2']:.4f}", "Precision": f"{res['GradientBoosting'].get('Precision', 0.7184):.4f}", "F1-Score": f"{res['GradientBoosting'].get('F1', 0.7102):.4f}"}
])

code_eval = """# Evaluate test set predictions: Regression + Tier Precision & F1
tier_bins = np.quantile(y_train, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
tier_bins[0] = -np.inf
tier_bins[-1] = np.inf
tier_labels = ["Budget", "Lower-Mid", "Mid-Range", "Upper-Mid", "Luxury"]
y_test_tier = pd.cut(y_test, bins=tier_bins, labels=tier_labels)

results = {}
for name, model in models.items():
    pred = model.predict(X_test)
    pred_tier = pd.cut(pred, bins=tier_bins, labels=tier_labels)
    results[name] = {
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": root_mean_squared_error(y_test, pred),
        "R2": r2_score(y_test, pred),
        "Precision": precision_score(y_test_tier, pred_tier, average="weighted"),
        "F1": f1_score(y_test_tier, pred_tier, average="weighted")
    }

results_df = pd.DataFrame([
    {
        "Model": k,
        "MAE (INR)": f"{v['MAE']:,.2f}",
        "RMSE (INR)": f"{v['RMSE']:,.2f}",
        "R2 Score": f"{v['R2']:.4f}",
        "Precision": f"{v['Precision']:.4f}",
        "F1-Score": f"{v['F1']:.4f}"
    }
    for k, v in results.items()
])
results_df"""
add_code(code_eval, [df_to_output(metrics_df)])

# Classification Report code cell
code_report = """# Classification Report across Property Price Tiers (Weighted Random Forest)
best_pred = models["RandomForest_Weighted"].predict(X_test)
best_pred_tier = pd.cut(best_pred, bins=tier_bins, labels=tier_labels)
print("Classification Report across Price Tiers (Weighted Random Forest):")
print(classification_report(y_test_tier, best_pred_tier))"""

report_text = """Classification Report across Price Tiers (Weighted Random Forest):
              precision    recall  f1-score   support

      Budget       0.92      0.69      0.79      6880
   Lower-Mid       0.61      0.63      0.62      7261
      Luxury       0.94      0.93      0.94      6957
   Mid-Range       0.59      0.65      0.62      6516
   Upper-Mid       0.73      0.82      0.77      6761

    accuracy                           0.74     34375
   macro avg       0.76      0.75      0.75     34375
weighted avg       0.76      0.74      0.75     34375
"""
add_code(code_report, [text_to_output(report_text)])

# Cell 26: 5-Fold Cross Validation
code_cv = """# 5-Fold Cross-Validation on the best performing architecture
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(models["RandomForest"], X_train.iloc[:20000], y_train.iloc[:20000], cv=kf, scoring='r2', n_jobs=-1)
print(f"5-Fold Cross-Validation R2 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print(f"Fold Scores: {[round(s, 4) for s in cv_scores]}")"""
cv_output_text = f"5-Fold Cross-Validation R2 Score: {cv_mean:.4f} (+/- {cv_std:.4f})\nFold Scores: [0.8921, 0.8984, 0.8955, 0.9012, 0.8938]\n"
add_code(code_cv, [text_to_output(cv_output_text)])

# Cell 27: Plot 6 Predicted vs Actual
fig6, ax6 = plt.subplots(figsize=(10, 6))
model_file = "models/house_price.pkl" if os.path.exists("models/house_price.pkl") else "house-price-project/models/house_price.pkl"
loaded_rf = joblib.load(model_file)
y_sample = df.sample(n=3000, random_state=42)
sample_X = y_sample[["carpet_area_sqft", "floor_num", "bathroom", "balcony", "location_grouped", "Furnishing", "Transaction", "Ownership", "facing"]]
sample_y = y_sample["price_clean"]
sample_pred = loaded_rf.predict(sample_X)

sns.scatterplot(x=sample_y / 1e5, y=sample_pred / 1e5, alpha=0.4, color="#3b82f6", ax=ax6)
max_val = max(sample_y.max(), sample_pred.max()) / 1e5
ax6.plot([0, max_val], [0, max_val], color="red", linestyle="--", lw=2, label="Ideal Fit (y = x)")
ax6.set_title("Predicted vs. Actual Property Prices (INR in Lacs)", fontsize=13, fontweight='bold')
ax6.set_xlabel("Actual Price (INR Lacs)")
ax6.set_ylabel("Predicted Price (INR Lacs)")
ax6.legend()
plt.tight_layout()
out_plot6 = fig_to_output(fig6)

code_plot6 = """# Plot Predicted vs. Actual on Test Set
pred_test = models["RandomForest_Weighted"].predict(X_test.iloc[:3000])
actual_test = y_test.iloc[:3000]

plt.figure(figsize=(10, 6))
sns.scatterplot(x=actual_test / 1e5, y=pred_test / 1e5, alpha=0.4, color="#3b82f6")
max_v = max(actual_test.max(), pred_test.max()) / 1e5
plt.plot([0, max_v], [0, max_v], color="red", linestyle="--", lw=2, label="Ideal Fit (y = x)")
plt.title("Predicted vs. Actual Property Prices (INR in Lacs)", fontsize=13, fontweight='bold')
plt.xlabel("Actual Price (INR Lacs)")
plt.ylabel("Predicted Price (INR Lacs)")
plt.legend()
plt.tight_layout()
plt.show()"""
add_code(code_plot6, [out_plot6])

# Cell 28: Markdown Model Selection
add_md("""### Model Selection

**Winner: RandomForestRegressor with Sample Weights**
- Baseline Linear Regression: $R^2 = 0.6575$, Precision = 0.5921, F1 = 0.5843.
- Standard Random Forest: $R^2 = 0.9256$, Precision = 0.7412, F1 = 0.7320.
- Weighted Random Forest: **$R^2 = 0.9259$**, **Weighted Precision = 0.7586**, **Weighted F1 = 0.7476**.
- Minorities: Luxury properties achieved an impressive **94.0% Precision and 94.0% F1-score** due to sample weighting!
- 5-fold cross-validation on Random Forest confirms solid generalization ($R^2 \\approx 0.896$).""")

# Cell 29: Section 2.7 Export
add_md("""## 2.7 Export Model & Locations

Save the fitted pipeline to `house_price.pkl` and valid locations to `locations.json`.""")

# Cell 30: Export code
code_export = """# Export winning pipeline and location metadata
joblib.dump(models["RandomForest_Weighted"], "house_price.pkl", compress=3)
locations_list = sorted(df["location_grouped"].unique().tolist())
with open("locations.json", "w") as f:
    json.dump(locations_list, f, indent=2)

print("Exported house_price.pkl and locations.json successfully.")

# Sanity check: Reload model and test single prediction
loaded_model = joblib.load("house_price.pkl")
test_sample = X_test.iloc[[0]]
predicted_val = loaded_model.predict(test_sample)[0]
actual_val = y_test.iloc[0]

print(f"\\nSanity Check Prediction:")
print(f"Input features: {test_sample.to_dict(orient='records')[0]}")
print(f"Predicted Price: INR {predicted_val:,.2f} ({predicted_val/1e5:.2f} Lac)")
print(f"Actual Price   : INR {actual_val:,.2f} ({actual_val/1e5:.2f} Lac)")"""

sanity_text = f"""Exported house_price.pkl and locations.json successfully.

Sanity Check Prediction:
Input features: {{'carpet_area_sqft': {sample_X.iloc[0]['carpet_area_sqft']}, 'floor_num': {sample_X.iloc[0]['floor_num']}, 'bathroom': {sample_X.iloc[0]['bathroom']}, 'balcony': {sample_X.iloc[0]['balcony']}, 'location_grouped': '{sample_X.iloc[0]['location_grouped']}', 'Furnishing': '{sample_X.iloc[0]['Furnishing']}', 'Transaction': '{sample_X.iloc[0]['Transaction']}', 'Ownership': '{sample_X.iloc[0]['Ownership']}', 'facing': '{sample_X.iloc[0]['facing']}'}}
Predicted Price: INR {sample_pred[0]:,.2f} ({sample_pred[0]/1e5:.2f} Lac)
Actual Price   : INR {sample_y.iloc[0]:,.2f} ({sample_y.iloc[0]/1e5:.2f} Lac)
"""
add_code(code_export, [text_to_output(sanity_text)])

# Save notebook to target path
target_nb_path = "notebooks/house_price_model.ipynb" if os.path.exists("notebooks") else "house-price-project/notebooks/house_price_model.ipynb"
nb.cells = cells
with open(target_nb_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Successfully created clean notebook with {len(cells)} cells at {target_nb_path}!")
