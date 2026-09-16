import pandas as pd
import numpy as np
df = pd.read_csv('notebooks/data/house_prices.csv')

import re
def parse_amount(val):
    if not isinstance(val, str): return None
    m = re.search(r'([\d\.]+)\s*(lac|cr)?', str(val).strip().lower())
    if not m: return None
    try: num = float(m.group(1))
    except: return None
    unit = m.group(2)
    if unit == "lac": return num * 100000.0
    elif unit == "cr": return num * 10000000.0
    return num
    
df['p'] = df['Amount(in rupees)'].apply(parse_amount)
df = df.dropna(subset=['p'])

bins = [0, 4_500_000, 7_500_000, 12_500_000, 25_000_000, float("inf")]
labels = ["Budget", "Lower-Mid", "Mid-Range", "Upper-Mid", "Luxury"]
tier = pd.cut(df['p'], bins=bins, labels=labels)
print(tier.value_counts(normalize=True) * 100)
