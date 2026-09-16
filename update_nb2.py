import nbformat

nb_path = "notebooks/house_price_model.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == "code":
        if "bins = [0, 5_000_000, 10_000_000, 20_000_000, 30_000_000, float('inf')]" in cell.source or 'bins = [0, 5_000_000, 10_000_000, 20_000_000, 30_000_000, float("inf")]' in cell.source:
            cell.source = cell.source.replace("5_000_000", "4_500_000")
            cell.source = cell.source.replace("10_000_000", "7_500_000")
            cell.source = cell.source.replace("20_000_000", "12_500_000")
            cell.source = cell.source.replace("30_000_000", "25_000_000")

with open(nb_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Notebook updated successfully.")
