import nbformat
import sys

nb_path = "notebooks/house_price_model.ipynb"
try:
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
except Exception:
    sys.exit(1)

new_cells = []
skip = False
for cell in nb.cells:
    if "Let's dynamically calculate the quantiles" in cell.source:
        continue
    
    if "Compute and visualize sample weights" in cell.source:
        cell.source = """# Since we are using Log Transformation, we don't need manual Sample Weights anymore!
# We will use TransformedTargetRegressor with np.log1p during training.
print("Switched from Sample Weights to Log Transformation (TransformedTargetRegressor) for better stability on right-skewed property prices.")"""
    
    if "price_bins_train" in cell.source and "compute_sample_weight" in cell.source:
        continue
    
    if "from sklearn.ensemble import HistGradientBoostingRegressor" not in cell.source and "import numpy as np" in cell.source:
        pass # we just add it to the model def cell

    if "models = {" in cell.source and "LinearRegression" in cell.source:
        cell.source = """from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.compose import TransformedTargetRegressor
import numpy as np

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

    if "RandomForest_Weighted" in cell.source:
        cell.source = cell.source.replace("RandomForest_Weighted", "HistGradientBoosting")

    new_cells.append(cell)

nb.cells = new_cells
with open(nb_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Notebook modified for log transformation.")
