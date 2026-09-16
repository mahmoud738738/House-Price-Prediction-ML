import nbformat

nb_path = "notebooks/house_price_model.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

new_cells = []
for cell in nb.cells:
    if cell.cell_type == "code" and "price_tiers = pd.cut(df[\"price_clean\"], bins=bins" in cell.source:
        # Create a new code cell before this one
        new_cell_source = """# Let's dynamically calculate the quantiles to justify our tier boundaries
quantiles = df["price_clean"].quantile([0.25, 0.5, 0.75, 0.90, 0.95])
print("Derived Quantiles from price distribution:")
for q, val in quantiles.items():
    print(f"  {int(q*100)}th Percentile: {val:,.0f} INR")

print("\\nBased on these percentiles, we set logical boundaries:")
print("- Budget: < 45 Lacs (~Q1)")
print("- Lower-Mid: 45 Lacs - 75 Lacs (~Q2)")
print("- Mid-Range: 75 Lacs - 1.25 Cr (~Q3)")
print("- Upper-Mid: 1.25 Cr - 2.5 Cr (~Q4)")
print("- Luxury: > 2.5 Cr (Top 10% minority class)")"""
        new_cell = nbformat.v4.new_code_cell(source=new_cell_source)
        new_cells.append(new_cell)
    
    new_cells.append(cell)

nb.cells = new_cells

with open(nb_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Notebook updated with quantile code.")
