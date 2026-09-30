import pandas as pd

# --- 1. Load your data ---
df = pd.read_parquet("gafanha.parquet")


# --- 2. Expected EU units ---
expected_units = {
    "Benzene":       "µg/m³",
    "CO":            "mg/m³",
    "NO":            "µg/m³",
    "NO2":           "µg/m³",
    "NOx":           "µg/m³",
    "O3":            "µg/m³",
    "PM10":          "µg/m³",
    "PM25":          "µg/m³",
    "SO2":           "µg/m³",
    "Precipitation": "mm",
    "Radiation":     "W/m²",
    "Temperature":   "°C",
    "WindDirection": "deg",
    "WindSpeed":     "m/s",
}

# --- 3. Plausible value ranges ---
plausible_ranges = {
    "CO":      (0, 50),
    "NO2":     (0, 500),
    "O3":      (0, 400),
    "PM10":    (0, 1000),
    "PM25":    (0, 500),
    "SO2":     (0, 500),
    "Benzene": (0, 50),
}

# --- 4. Check unit mismatches ---
def check_unit(row):
    expected = expected_units.get(row["collection"])
    if expected is None:
        return "unknown_parameter"
    return "ok" if row["unit"] == expected else "MISMATCH"

df["unit_check"] = df.apply(check_unit, axis=1)
mismatches = df[df["unit_check"] == "MISMATCH"].copy()

# --- 5. Check out-of-range values ---
range_flags = []
for param, (lo, hi) in plausible_ranges.items():
    sub = df[df["collection"] == param]
    bad = sub[(sub["value"] < lo) | (sub["value"] > hi)]
    range_flags.append(bad)
range_issues = pd.concat(range_flags) if range_flags else pd.DataFrame()

# --- 6. Print results ---
pd.set_option("display.max_rows", None)
pd.set_option("display.width", None)

if not mismatches.empty:
    print(f"\n❌ {len(mismatches)} rows with WRONG UNIT LABEL:\n")
    mismatches["expected_unit"] = mismatches["collection"].map(expected_units)
    print(mismatches[["station", "timestamp", "collection", "value", "unit", "expected_unit"]])
else:
    print("\n✅ No unit label mismatches found.")

if not range_issues.empty:
    print(f"\n⚠️ {len(range_issues)} rows with OUT-OF-RANGE VALUES:\n")
    print(range_issues[["station", "timestamp", "collection", "value", "unit"]])
else:
    print("\n✅ No out-of-range values found.")

# --- 7. Save flagged rows to CSV for review ---
mismatches.to_csv("unit_mismatches.csv", index=True)
range_issues.to_csv("range_issues.csv", index=True)
print("\nSaved 'unit_mismatches.csv' and 'range_issues.csv' for review.")
