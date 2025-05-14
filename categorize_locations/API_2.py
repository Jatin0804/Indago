import os
from openai import OpenAI
import pandas as pd
from opencage.geocoder import OpenCageGeocode
from tqdm import tqdm
import dotenv

dotenv.load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAPI"))
OPENCAGE_KEY = os.getenv("OPENCAGE_API")

INPUT_FOLDER = 'LocSheets'
OUTPUT_FOLDER = 'enriched_LocSheets'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Get GPT ranking of columns based on their usefulness for location
def rank_columns_with_gpt(columns):
    prompt = (
        f"Rank these columns by how useful they are for determining a geographic location, keep in mind that city is the most useful, state is next, and the complete address is the least useful.\n"
        f"(city, state, country): {', '.join(columns)}.\n"
        f"Return a comma-separated list in order of usefulness."
    )
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    ranked = response.choices[0].message.content
    ranked_columns = [col.strip() for col in ranked.split(',') if col.strip() in columns]
    print(f"Ranked columns: {ranked_columns}")
    return ranked_columns

# Call OpenCage API to geocode location
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

# Process each Excel file
def process_file(filename):
    print(f"📂 Processing: {filename}")
    df = pd.read_excel(os.path.join(INPUT_FOLDER, filename))
    if df.empty:
        print(f"⚠️ Skipping empty file: {filename}")
        return

    ranked_columns = rank_columns_with_gpt(list(df.columns))
    if not ranked_columns:
        print(f"⚠️ No ranked columns found for {filename}")
        return

    # Use top N ranked columns to build the location string
    top_columns = ranked_columns[:3]  # Keep only top 3
    geocoder = OpenCageGeocode(OPENCAGE_KEY)

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Geocoding"):
        query = ', '.join(str(row[col]) for col in top_columns if pd.notna(row[col]))
        enriched = get_geocoded_info(geocoder, query)
        results.append(enriched)

    enriched_df = pd.DataFrame(results)
    output_path = os.path.join(OUTPUT_FOLDER, f"enriched_{filename}")
    enriched_df.to_excel(output_path, index=False)
    print(f"✅ Saved enriched file: {output_path}")

# Main loop
if __name__ == "__main__":
    for file in os.listdir(INPUT_FOLDER):
        if file.endswith('.xlsx'):
            process_file(file)
