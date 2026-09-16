import re

with open("train_and_export.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Remove the sample weights section
content = re.sub(
    r'# Calculate and print data-driven quantiles.*?sample_weights_train = compute_sample_weight\(\'balanced\', price_bins_train\)',
    '',
    content,
    flags=re.DOTALL
)

# 2. Add imports
content = content.replace(
    'from sklearn.linear_model import LinearRegression',
    'from sklearn.linear_model import LinearRegression\nfrom sklearn.ensemble import HistGradientBoostingRegressor\nfrom sklearn.compose import TransformedTargetRegressor\nimport numpy as np'
)

# 3. Update model definitions
models_def = """# Define models with Log Transformation (TransformedTargetRegressor)
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
}"""
content = re.sub(
    r'# Define models\nmodels = \{.*?\n\}',
    models_def,
    content,
    flags=re.DOTALL
)

# 4. Remove the weighted training block
content = re.sub(
    r'print\("Training RandomForest \(Minority Sample-Weighted\)\.\.\."\)\nmodels\["RandomForest_Weighted"\].fit\(X_train, y_train, reg__sample_weight=sample_weights_train\)\npreds\["RandomForest_Weighted"\] = models\["RandomForest_Weighted"\].predict\(X_test\)\nresults\["RandomForest_Weighted"\] = \{\n    "MAE": float\(mean_absolute_error\(y_test, preds\["RandomForest_Weighted"\]\)\),\n    "RMSE": float\(root_mean_squared_error\(y_test, preds\["RandomForest_Weighted"\]\)\),\n    "R2": float\(r2_score\(y_test, preds\["RandomForest_Weighted"\]\)\)\n\}',
    'print("Training HistGradientBoosting...")\nmodels["HistGradientBoosting"].fit(X_train, y_train)\npreds["HistGradientBoosting"] = models["HistGradientBoosting"].predict(X_test)\nresults["HistGradientBoosting"] = {\n    "MAE": float(mean_absolute_error(y_test, preds["HistGradientBoosting"])),\n    "RMSE": float(root_mean_squared_error(y_test, preds["HistGradientBoosting"])),\n    "R2": float(r2_score(y_test, preds["HistGradientBoosting"]))\n}',
    content
)

with open("train_and_export.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Updated train_and_export.py")
