import nbformat

nb_path = "notebooks/house_price_model.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == "code":
        # 1. Fix sparse=False for OneHotEncoder
        if "OneHotEncoder(handle_unknown=\"ignore\")" in cell.source:
            cell.source = cell.source.replace("OneHotEncoder(handle_unknown=\"ignore\")", "OneHotEncoder(handle_unknown=\"ignore\", sparse_output=False)")
            
        # 2. Add train_test_split before the models if it's missing
        if "models = {" in cell.source and "LinearRegression" in cell.source:
            if "train_test_split" not in cell.source:
                cell.source = """# Train / Test split (80% train, 20% test)
X = df[numeric_features + categorical_features]
y = df["price_clean"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

""" + cell.source

with open(nb_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

# Also fix train_and_export.py
with open("train_and_export.py", "r", encoding="utf-8") as f:
    tr = f.read()

tr = tr.replace("OneHotEncoder(handle_unknown=\"ignore\")", "OneHotEncoder(handle_unknown=\"ignore\", sparse_output=False)")

with open("train_and_export.py", "w", encoding="utf-8") as f:
    f.write(tr)

print("Fixed sparse array and train_test_split.")
