import pycountry
import pandas as pd
import requests
import time
import os
import dotenv
import asyncio
from openai import AsyncOpenAI
import re

# Create a DataFrame with country codes using pycountry
data = []
for country in pycountry.countries:
    data.append({
        'name': country.name,
        'alpha_2': country.alpha_2,
        'alpha_3': country.alpha_3,
        'numeric': country.numeric
    })

df = pd.DataFrame(data)

# Save the DataFrame to Excel and CSV files
df.to_excel('country_codes.xlsx', index=False)
df.to_csv('country_codes.csv', index=False)
print("Saved country codes to 'country_codes.csv' and 'country_codes.xlsx'")


dotenv.load_dotenv()
API_KEY = os.getenv("OPENCAGE_API")

df_countries = pd.read_excel("country_codes.xlsx")
country_names = df_countries['name'].tolist()

results = []

# Loop through each country name and get the data from OpenCage API
for country in country_names:
    url = f"https://api.opencagedata.com/geocode/v1/json?q={country}&key={API_KEY}&no_annotations=1&limit=1"
    response = requests.get(url)
    data = response.json()

    if data["results"]:
        components = data["results"][0]["components"]
        print(components)
        country_name = components.get("country")
        country_code = components.get("country_code")
        alpha_2 = components.get("ISO_3166-1_alpha-2")
        alpha_3 = components.get("ISO_3166-1_alpha-3")

        results.append({
            "name": country_name,
            "alpha_1": country_code,
            "alpha_2": alpha_2,
            "alpha_3": alpha_3,
        })
    else:
        print(f"Failed to get data for {country}")
    
    time.sleep(1)  

# Save the results to a DataFrame and then to CSV and Excel files
df = pd.DataFrame(results)
df.to_csv("country_codes_opencage.csv", index=False)
df.to_excel("country_codes_opencage.xlsx", index=False)

print("Saved to country_codes_opencage.csv and .xlsx")

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAPI"))

df = pd.read_csv("country_codes_opencage.csv")

# Convert the 'alpha_1' column to string type
async def get_aliases(country, code):
    prompt = (
        f"List common alternative names, abbreviations, or aliases for the country '{country}' "
        f"with country code '{code}', such as short forms, abbreviations, historical names, or local names. "
        f"Return a list, each alias separately."
    )
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        aliases_text = response.choices[0].message.content
        # Clean and split into individual aliases
        aliases = [alias.strip("- ").strip() for alias in aliases_text.split("\n") if alias.strip()]
        return aliases
    except Exception as e:
        print(f"Error for {country}: {e}")
        return []

async def process_all():
    tasks = []
    for _, row in df.iterrows():
        tasks.append(get_aliases(row["name"], row["alpha_1"]))
    
    all_aliases = await asyncio.gather(*tasks)

    max_alias_count = max(len(a) for a in all_aliases)
    for i in range(max_alias_count):
        df[f"Alias_{i+1}"] = [aliases[i] if i < len(aliases) else "" for aliases in all_aliases]

   
    df.to_excel("countries_with_aliases.xlsx", index=False)
    print("Saved to countries_with_aliases.xlsx")

# Run the async function to get aliases
asyncio.run(process_all())

df = pd.read_excel("countries_with_aliases.xlsx")

alias_start_index = 4

# Clean the aliases by removing leading numbers and dots
for col in df.columns[alias_start_index:]:
    df[col] = df[col].astype(str).apply(lambda x: re.sub(r"^\d+\.\s*", "", x).strip() if x.strip().lower() != "nan" else "")

df.to_excel("countries_with_cleaned_aliases.xlsx", index=False)
print("✅ Cleaned and saved as countries_with_cleaned_aliases.xlsx")

# Load the Excel file
df = pd.read_excel("countries_with_cleaned_aliases.xlsx")  # Replace with your file name

# Number of leading columns to preserve (e.g., Country, Alpha-2, Alpha-3)
fixed_columns = 3

# Split into fixed and variable parts
fixed_part = df.iloc[:, :fixed_columns]
variable_part = df.iloc[:, fixed_columns:]

# Shift non-empty cells to the left row-wise
shifted_part = variable_part.apply(lambda row: pd.Series([val for val in row if pd.notna(val) and str(val).strip() != ""]), axis=1)

# Fill in missing columns if some rows have fewer non-empty entries
shifted_part = shifted_part.reindex(columns=range(variable_part.shape[1]))

# Combine fixed + shifted parts
cleaned_df = pd.concat([fixed_part, shifted_part], axis=1)

# Save to Excel
cleaned_df.to_excel("cleaned_shifted_aliases.xlsx", index=False)
print("✅ Done. Saved as cleaned_shifted_aliases.xlsx")
