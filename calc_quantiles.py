import pandas as pd
import re

def parse_amount(val):
    if not isinstance(val, str): return None
    val = val.strip().lower()
    m = re.search(r'([\d\.]+)\s*(lac|cr)?', val)
    if not m: return None
    try: num = float(m.group(1))
    except: return None
    unit = m.group(2)
    if unit == "lac": return num * 100000.0
    elif unit == "cr": return num * 10000000.0
    return num

df = pd.read_csv('notebooks/data/house_prices.csv')
df = df.drop_duplicates(subset=[c for c in df.columns if c != 'Index'])
df['p'] = df['Amount(in rupees)'].apply(parse_amount)
df = df.dropna(subset=['p'])

print("Percentiles for price_clean (in INR):")
print(df['p'].describe(percentiles=[0.25, 0.5, 0.75, 0.85, 0.90, 0.95, 0.98, 0.99]).apply(lambda x: f"{x:,.0f}"))
