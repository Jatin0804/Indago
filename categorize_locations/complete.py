import os
import asyncio
from openai import AsyncOpenAI
import pandas as pd
import dotenv
from openai import OpenAI
from opencage.geocoder import OpenCageGeocode
from tqdm import tqdm
import re
import time
import tkinter as tk
from tkinter import filedialog

dotenv.load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAPI"))
client2 = OpenAI(api_key=os.getenv("OPENAPI"))
OPENCAGE_KEY = os.getenv("OPENCAGE_API")

STAGE1_INPUT = "Input_files"
STAGE1_OUTPUT = "Stage1_LocSheets"
os.makedirs(STAGE1_OUTPUT, exist_ok=True)

STAGE2_INPUT = 'Stage1_LocSheets'
STAGE2_OUTPUT = 'Stage2_enriched_LocSheets'
os.makedirs(STAGE2_OUTPUT, exist_ok=True)

STAGE3_INPUT = 'Stage2_enriched_LocSheets'
STAGE3_OUTPUT = 'Stage3_Results'
os.makedirs(STAGE3_OUTPUT, exist_ok=True)

FINAL_OUTPUT = 'Final_Results'
os.makedirs(FINAL_OUTPUT, exist_ok=True)

ALIAS_FILE = "Dataset/country_aliases.xlsx"
ORIGINAL_DIR = "."

# Define standard business fields for mapping
STANDARD_BUSINESS_FIELDS = [
    "Company Name",
    "Business Description",
    "Website",
    "Revenue",
    "Number of Employees",
    "Company Phone Number",
    "Parent Company",
    "Active Investors",
    "Contact First Name",
    "Contact Last Name",
    "Contact Title",
    "Contact Email",
    "Geography"
]

# Stage 1: Identify location-related columns in Excel files
async def analyze_column_with_gpt(column_name, sample_data):
    prompt = f"""
    You are an AI assistant. I will provide you with a column name and some sample data from an Excel sheet. 
    Your task is to determine if the column is related to location information (e.g., city, state, country, address, etc.).

    Column Name: {column_name}
    Sample Data: {sample_data}

    Respond with "Yes" if the column is related to location, otherwise respond with "No".
    """
    try:
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return column_name, response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Error analyzing column '{column_name}': {e}")
        return column_name, "Error"

# Stage 1: Convert CSV to Excel if needed
def convert_csv_to_excel(csv_path):
    """
    Converts a CSV file to an Excel (.xlsx) file in the same directory.
    Returns the path to the new Excel file.
    """
    df = pd.read_csv(csv_path)
    excel_path = csv_path.rsplit('.', 1)[0] + '.xlsx'
    df.to_excel(excel_path, index=False, engine='openpyxl')
    print(f"🔄 Converted CSV to Excel: {excel_path}")
    return excel_path


# Stage 1: Process each Excel file to identify location-related columns
async def process_file(filepath):
    """
    Process a single file (CSV or Excel), identify location-related columns using GPT, and save as Excel.
    If it's a CSV, convert it to Excel first.
    """
    print(f"\n📂 Processing file: {filepath}")
    try:
        ext = os.path.splitext(filepath)[1].lower()

        # Convert CSV to Excel first
        if ext == ".csv":
            filepath = convert_csv_to_excel(filepath)  # replaces with new .xlsx path

        df = pd.read_excel(filepath)  # now guaranteed to be Excel

        if df.empty:
            print("⚠️ Skipping empty file.")
            return

        tasks = []
        for column in df.columns:
            sample_data = df[column].dropna().head(5).tolist()
            if not sample_data:
                continue
            tasks.append(analyze_column_with_gpt(column, sample_data))

        results = await asyncio.gather(*tasks)
        location_cols = [col for col, result in results if result.lower() == "yes"]

        if location_cols:
            filtered_df = df[location_cols]
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            output_path = os.path.join(STAGE1_OUTPUT, f"{base_name}_locations.xlsx")
            filtered_df.to_excel(output_path, index=False, engine='openpyxl')
            print(f"✅ Saved to: {output_path}")
        else:
            print("ℹ️ No location-related columns found.")

    except Exception as e:
        print(f"❌ Error processing file '{filepath}': {e}")


# Stage 2: Get GPT ranking of columns based on their usefulness for location
def rank_columns_with_gpt(columns):
    prompt = (
        f"Rank these columns by how useful they are for determining a geographic location."
        f"keep in mind the usefulness of locations atrributes in the following order: city is the most useful, state is next, then country is useful, then complete address is useful, and then any municipality or local address or street etc\n"
        f"(city, state, country): {', '.join(columns)}.\n"
        f"Return a comma-separated list in order of usefulness."
    )
    response = client2.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    ranked = response.choices[0].message.content
    ranked_columns = [col.strip() for col in ranked.split(',') if col.strip() in columns]
    print(f"Ranked columns: {ranked_columns}")
    return ranked_columns

# Stage 2: Call OpenCage API to geocode location
def get_geocoded_info(geocoder, query):
    try:
        result = geocoder.geocode(query)
        if result and len(result) > 0:
            components = result[0]['components']
            return {
                'City': components.get('city') or components.get('town') or components.get('village'),
                'State': components.get('state'),
                'Country': components.get('country')
            }
    except Exception:
        pass
    return {'City': None, 'State': None, 'Country': None}

# Stage 2: Process each Excel file - MODIFIED to send columns one by one
def process_file_2(filename):
    print(f"📂 Processing: {filename}")
    df = pd.read_excel(os.path.join(STAGE2_INPUT, filename))
    if df.empty:
        print(f"⚠️ Skipping empty file: {filename}")
        return

    ranked_columns = rank_columns_with_gpt(list(df.columns))
    if not ranked_columns:
        print(f"⚠️ No ranked columns found for {filename}")
        if len(df.columns) == 1:
            ranked_columns = list(df.columns)  # fallback to that one column
            print(f"🛟 Falling back to single column: {ranked_columns}")
        else:
            return

    geocoder = OpenCageGeocode(OPENCAGE_KEY)

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Geocoding"):
        enriched = {'City': None, 'State': None, 'Country': None}
        
        # Try each column one by one, in order of ranking
        for col in ranked_columns:
            if pd.notna(row.get(col)) and str(row[col]).strip():
                query = str(row[col]).strip()
                print(f"🔍 Trying column: {col} with value: {query}")
                
                enriched = get_geocoded_info(geocoder, query)
                
                # If we got a result, stop trying more columns
                if any([enriched['City'], enriched['State'], enriched['Country']]):
                    print(f"✅ Found result using column: {col}")
                    break
                else:
                    print(f"❌ No geocode result for: {query} using column: {col}")
        
        results.append(enriched)

    enriched_df = pd.DataFrame(results)
    output_path = os.path.join(STAGE2_OUTPUT, f"enriched_{filename}")
    enriched_df.to_excel(output_path, index=False)
    print(f"✅ Saved enriched file: {output_path}")



# STAGE 3: Load alias sheet and normalize
def find_country_aliases(user_input):
    alias_df = pd.read_excel(ALIAS_FILE, header=None)
    alias_df_cleaned = alias_df.apply(lambda col: col.map(lambda x: str(x).strip().lower() if pd.notna(x) else x))

    # Match alias
    matched_country = None
    for idx, row in alias_df_cleaned.iterrows():
        if user_input in row.values:
            matched_country = alias_df.iloc[idx, 0]  # Preserve original case
            print(f"✅ Match found: '{user_input}' belongs to → {matched_country}")
            return matched_country
        
    
    if not matched_country:
        print(f"❌ No match found for: '{user_input} as a country.'")
        return None


def find_matches():
    while True:
        user_input = input("Enter a country name or alias (or 'exit' to quit): ").strip()
        if user_input.lower() == 'exit':
            print("Exiting...")
            break

        stage3_time = time.time()
        matched_country = find_country_aliases(user_input)
        if not matched_country:
            matched_country = user_input

        # Create a country-specific folder
        safe_country_name = re.sub(r'[^\w\-_. ]', '_', matched_country.strip())
        COUNTRY_FOLDER = os.path.join(STAGE3_OUTPUT, safe_country_name)
        os.makedirs(COUNTRY_FOLDER, exist_ok=True)
        print(f"📁 Created folder for country: {COUNTRY_FOLDER}")

        all_matched_rows = []

        for filename in os.listdir(STAGE3_INPUT):
            if filename.endswith(".xlsx"):
                enriched_path = os.path.join(STAGE2_OUTPUT, filename)
                print(f"\n📂 Processing: {filename}")
                enriched_df = pd.read_excel(enriched_path)

                # Normalize column names
                columns_lower = {col.lower(): col for col in enriched_df.columns}
                country_col = columns_lower.get("country")
                state_col = columns_lower.get("state")
                city_col = columns_lower.get("city")

                if not any([country_col, state_col, city_col]):
                    print("❌ No 'country', 'state', or 'city' column found. Skipping.")
                    continue

                matching_indices = []

                # Try matching by Country
                if country_col:
                    country_vals = enriched_df[country_col].astype(str).str.strip().str.lower()
                    matching_indices = enriched_df[country_vals == matched_country.lower()].index.tolist()

                # Try matching by State if no country match
                if not matching_indices and state_col:
                    state_vals = enriched_df[state_col].astype(str).str.strip().str.lower()
                    matching_indices = enriched_df[state_vals == matched_country.lower()].index.tolist()

                # Try matching by City if no state match
                if not matching_indices and city_col:
                    city_vals = enriched_df[city_col].astype(str).str.strip().str.lower()
                    matching_indices = enriched_df[city_vals == matched_country.lower()].index.tolist()

                if not matching_indices:
                    print(f"❌ No match found for '{matched_country}' in {filename}")
                    continue

                # Infer original file name
                original_base = filename.replace("enriched_", "").replace("_locations", "")
                original_filename = f"{original_base}"
                original_path = os.path.join(STAGE1_INPUT, original_filename)

                if not os.path.exists(original_path):
                    print(f"⚠️ Original file '{original_filename}' not found. Skipping.")
                    continue

                original_df = pd.read_excel(original_path)
                matched_rows = original_df.loc[matching_indices]

                all_matched_rows.append(matched_rows)

                # Store in the country-specific subfolder
                result_file = f"matched_from_{original_base}"
                result_path = os.path.join(COUNTRY_FOLDER, result_file)
                matched_rows.to_excel(result_path, index=False)

                print(f"✅ Saved {len(matched_rows)} matched rows to {result_path}")

        combine_results_for_country(COUNTRY_FOLDER)
        stage3_end_time = time.time()
        print(f"🕒 Stage 3 time taken for {matched_country}: {stage3_end_time - stage3_time:.2f} seconds")
        

def combine_results_for_country(COUNTRY_FOLDER):
    """
    Combines all Excel files from the country-specific subfolder in Stage3_Results
    into a single Excel file, handling different columns across sheets.
    """
    all_results = []
    files = [
        os.path.join(COUNTRY_FOLDER, fname)
        for fname in os.listdir(COUNTRY_FOLDER)
        if fname.lower().endswith((".xlsx", ".csv"))
    ]


    for file_path in files:
        try:
            print(f"🔍 Processing: {file_path}")
            if file_path.endswith(".csv"):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)

            standardized_df, _ = map_to_standard_fields(df)
            standardized_df.insert(0, "Source File", os.path.basename(file_path))  
            all_results.append(standardized_df)

        except Exception as e:
            print(f"⚠️ Error processing {file_path}: {str(e)}")

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        output_path = os.path.join(FINAL_OUTPUT, "standardized_output.xlsx")
        final_df.to_excel(output_path, index=False)
        print(f"✅ Standardized data saved to: {output_path}")
    else:
        print("❌ No data processed.")


def map_to_standard_fields(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    column_list = "\n".join([f"- {col}" for col in df.columns])
    mapping_prompt = f"""
    Map the following columns to standard business fields:
    {column_list}
    Standard fields to map to:
    {', '.join(STANDARD_BUSINESS_FIELDS)}

    Important mapping rules:
    1. For Company Name:
       - First priority: Exact match to "Company Name" or "Companies"
       - Only if no exact match: Consider "Company Former Name", "Company Also Known As", "Company Legal Name"
       - Never use "Company ID" for company name
    2. For revenue, look for columns containing revenue or financial metrics or similar terms
       - Revenue, Annual Revenue, Total Revenue, Rev etc.
    3. For Geography, map ALL columns containing location information:
       - HQ Location, HQ City, HQ State, HQ Country
       - Office Location, Address, Region
       - Any column with City, State, Country in name
    4. For contact information, look for columns with contact details (name, email, phone, etc.)
    5. For phone numbers, look for columns containing "phone", "contact", "telephone", or similar terms
    6. Map only employee count to Number of Employees, not followers of social media.
    7. Always return mappings in format: Column Name -> Standard Field Name
    """

    for attempt in range(3):
        try:
            response = client2.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that maps business report columns to standard fields. For company name, prioritize exact matches and avoid using IDs."},
                    {"role": "user", "content": mapping_prompt}
                ],
                temperature=0.1
            )
            break
        except (client2.error.RateLimitError, client2.error.Timeout) as e:
            print(f"⏳ GPT retry {attempt + 1}/3 due to: {e}")
            time.sleep(2 ** attempt)
    else:
        print("❌ GPT API failed after 3 retries.")
        return pd.DataFrame(), {}

    mapping_text = response.choices[0].message.content.strip()

    # Save raw GPT mapping output
    with open(f"{FINAL_OUTPUT}/raw_column_mappings.txt", "a", encoding="utf-8") as f:
        f.write("=== Mapping Output ===\n")
        f.write(mapping_text + "\n\n")

    mappings = {}
    for line in mapping_text.split('\n'):
        if '->' in line:
            col, field = line.split('->')
            col = col.strip().lstrip('-').strip()
            field = field.strip()
            if field in STANDARD_BUSINESS_FIELDS:
                if field == 'Company Name':
                    if 'id' in col.lower():
                        continue
                    if 'Company Name' not in mappings.values() and 'Companies' not in mappings.values():
                        mappings[col] = field
                elif field not in mappings.values():  # Only allow one column per field
                    mappings[col] = field

    # Construct output dataframe with single columns
    result_df = pd.DataFrame()
    for field in STANDARD_BUSINESS_FIELDS:
        # Find the column mapped to this field
        col = next((c for c, f in mappings.items() if f == field and c in df.columns), None)
        result_df[field] = df[col] if col else None

    return result_df, mappings

async def main():
    root = tk.Tk()
    root.withdraw()
    filepaths = filedialog.askopenfilenames(
        title="Select CSV or Excel files",
        filetypes=[("Excel and CSV files", "*.xlsx *.xls *.csv")]
    )

    if not filepaths:
        print("❌ No Excel or CSV files found in the Input files folder.")
        return

    await asyncio.gather(*(process_file(fp) for fp in filepaths))


if __name__ == "__main__":
    start_time = time.time()
    # Stage 1: Process files to identify location-related columns
    asyncio.run(main())

    stage1_time = time.time()
    # Stage 2: Process files to geocode locations
    for file in os.listdir(STAGE2_INPUT):
        if file.endswith('.xlsx'):
            process_file_2(file)

    stage2_time = time.time()
    # Stage 3: Find matches based on user input
    find_matches()

    end_time = time.time()
    print(f"\n🕒 Stage 1 time taken: {stage1_time - start_time:.2f} seconds")
    print(f"🕒 Stage 2 time taken: {stage2_time - stage1_time:.2f} seconds")
    print(f"🕒 Stage 3 time taken: {end_time - stage2_time:.2f} seconds")
    print(f"\n🕒 Total time taken: {end_time - start_time:.2f} seconds")