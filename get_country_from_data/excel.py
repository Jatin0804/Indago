from geopy.geocoders import OpenCage
import pandas as pd
import time

OPENCAGE_API_KEY = "f2f689735dca4743b77c3e994f5683e1"

geolocator = OpenCage(api_key=OPENCAGE_API_KEY, timeout=10)

def get_country_from_location(location_name):
    try:
        location = geolocator.geocode(location_name, exactly_one=True)
        if location:
            return location.raw.get('components', {}).get('country')
    except Exception as e:
        print(f"Error for '{location_name}': {e}")
    return None

def resolve_country(row):
    # Check if the country is already provided and is not NaN
    if pd.notna(row.get('Country')):
        return row.get('Country')

    # Check if the full address is provided and is not NaN
    full_location = row.get('Location')
    if pd.notna(full_location):
        location_parts = full_location.split(",")
        return location_parts[-1].strip()  

    # Use the API if only city or state is provided
    location_city = row.get('City')
    location_state = row.get('State')

    if pd.notna(location_city):
        resolved = get_country_from_location(location_city)
        if resolved:
            return resolved

    if pd.notna(location_state):
        resolved = get_country_from_location(location_state)
        if resolved:
            return resolved

    return None

def map_locations_to_countries(df):
    df['resolved_country'] = None
    print("📦 Resolving locations to countries...\n")
    for i, row in df.iterrows():
        resolved_country = resolve_country(row)
        if resolved_country:
            df.at[i, 'resolved_country'] = resolved_country
            print(f"✅ {i} {row.get('Company Name', 'Unknown')} → {resolved_country}")
        else:
            print(f"⚠️ {i} Could not resolve country for {row.get('Company Name', 'Unknown')}")
        # time.sleep(0.5)  
    return df

def load_companies_from_excel(filepath):
    return pd.read_excel(filepath)

def find_companies_by_country(df, user_country):
    matched = df[df['resolved_country'].str.lower() == user_country.lower()]
    return matched['Company Name'].tolist()

if __name__ == "__main__":
    filepath = "company_dataset.xlsx"
    df = load_companies_from_excel(filepath)
    df = map_locations_to_countries(df)

    while True:
        user_input = input("\n🔍 Enter a country name (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break

        results = find_companies_by_country(df, user_input)

        if results:
            print(f"\n🏢 Companies in '{user_input}':")
            for company in results:
                print(f" - {company}")
        else:
            print(f"⚠️ No companies found in '{user_input}'.")
    
    # user_input = input("\n🔍 Enter a country name: ")
    # results = find_companies_by_country(df, user_input)

    # print(f"\n🏢 Companies in '{user_input}':")
    # for company in results:
    #     print(f" - {company}")
