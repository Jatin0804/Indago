import os
import pandas as pd

ALIAS_FILE = "country_aliases.xlsx"
ENRICHED_DIR = "enriched_LocSheets"
ORIGINAL_DIR = "." 
RESULT_DIR = "result"

# Ensure result folder exists
os.makedirs(RESULT_DIR, exist_ok=True)

# Load alias sheet and normalize
alias_df = pd.read_excel(ALIAS_FILE, header=None)
alias_df_cleaned = alias_df.apply(lambda col: col.map(lambda x: str(x).strip().lower() if pd.notna(x) else x))

# Get user input
user_input = input("Enter a location name, alias, or abbreviation: ").strip().lower()

# Match alias
matched_country = None
for idx, row in alias_df_cleaned.iterrows():
    if user_input in row.values:
        matched_country = alias_df.iloc[idx, 0]  # Preserve original case
        print(f"✅ Match found: '{user_input}' belongs to → {matched_country}")
        break

if not matched_country:
    print(f"❌ No match found for: '{user_input}'")
    exit()

# Process all enriched files
for filename in os.listdir(ENRICHED_DIR):
    if filename.endswith(".xlsx"):
        enriched_path = os.path.join(ENRICHED_DIR, filename)
        print(f"\n📂 Processing: {filename}")

        enriched_df = pd.read_excel(enriched_path)

        # Identify the country column
        country_col = next((col for col in enriched_df.columns if col.strip().lower() == 'country'), None)
        if not country_col:
            print("❌ No 'country' column found. Skipping file.")
            continue

        country_col_values = enriched_df[country_col].astype(str).str.strip().str.lower()
        matching_indices = enriched_df[country_col_values == matched_country.lower()].index.tolist()

        if not matching_indices:
            print(f"❌ No matches found for '{matched_country}' in {filename}")
            continue

        # Infer original file name (e.g., enriched_c1_locations.xlsx → c1.xlsx)
        original_base = filename.replace("enriched_", "").replace("_locations", "")
        original_filename = f"{original_base}"
        original_path = os.path.join(ORIGINAL_DIR, original_filename)

        if not os.path.exists(original_path):
            print(f"⚠️ Original file '{original_filename}' not found. Skipping.")
            continue

        # Load and extract matching rows
        original_df = pd.read_excel(original_path)
        matched_rows = original_df.loc[matching_indices]

        result_file = f"matched_from_{original_base}"
        result_path = os.path.join(RESULT_DIR, result_file)
        matched_rows.to_excel(result_path, index=False)

        print(f"✅ Saved {len(matched_rows)} matched rows to {result_path}")
