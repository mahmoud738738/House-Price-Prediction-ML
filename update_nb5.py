import nbformat

nb_path = "notebooks/house_price_model.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

new_cells = []
for cell in nb.cells:
    new_cells.append(cell)
    if "preprocessor = ColumnTransformer" in cell.source:
        # Insert the missing cell right after this one
        source = """# Train / Test split (80% train, 20% test)
X = df[numeric_features + categorical_features]
y = df["price_clean"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.compose import TransformedTargetRegressor
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Define models wrapped in TransformedTargetRegressor to automatically apply log1p(y) and expm1(y_pred)
models = {
    "LinearRegression": TransformedTargetRegressor(
        regressor=Pipeline([("prep", preprocessor), ("reg", LinearRegression())]),
        func=np.log1p, inverse_func=np.expm1
    ),
    "RandomForest": TransformedTargetRegressor(
        regressor=Pipeline([("prep", preprocessor), ("reg", RandomForestRegressor(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1))]),
        func=np.log1p, inverse_func=np.expm1
    ),
    "HistGradientBoosting": TransformedTargetRegressor(
        regressor=Pipeline([("prep", preprocessor), ("reg", HistGradientBoostingRegressor(random_state=42))]),
        func=np.log1p, inverse_func=np.expm1
    ),
    "GradientBoosting": TransformedTargetRegressor(
        regressor=Pipeline([("prep", preprocessor), ("reg", GradientBoostingRegressor(n_estimators=100, random_state=42))]),
        func=np.log1p, inverse_func=np.expm1
    )
}

print("Training LinearRegression...")
models["LinearRegression"].fit(X_train, y_train)

print("Training RandomForest...")
models["RandomForest"].fit(X_train, y_train)

print("Training HistGradientBoosting...")
models["HistGradientBoosting"].fit(X_train, y_train)

print("Training GradientBoosting...")
models["GradientBoosting"].fit(X_train, y_train)
print("All models trained successfully.")"""
        
        # Check if we already inserted it previously to avoid duplicates (though it shouldn't exist)
        already_exists = any("TransformedTargetRegressor(" in c.source for c in nb.cells)
        if not already_exists:
            new_cells.append(nbformat.v4.new_code_cell(source=source))

nb.cells = new_cells

with open(nb_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Missing cell restored.")
