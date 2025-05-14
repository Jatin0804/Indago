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

def map_locations_to_countries(df):
    df['resolved_country'] = None
    print("📦 Resolving locations to countries...\n")
    for i, row in df.iterrows():
        location = row['Country']
        resolved = get_country_from_location(location)
        if resolved:
            df.at[i, 'resolved_country'] = resolved
            print(f"✅ {row['Companies']} → {resolved}")
        else:
            print(f"⚠️ Could not resolve country for {row['Companies']}")
        time.sleep(0.5)  # Faster than Nominatim, but still polite
    return df

def load_companies_from_csv(filepath):
    return pd.read_csv(filepath)


def find_companies_by_country(df, user_country):
    matched = df[df['resolved_country'].str.lower() == user_country.lower()]
    return matched['Companies'].tolist()

if __name__ == "__main__":
    filepath = "companies.csv"
    df = load_companies_from_csv(filepath)
    df = map_locations_to_countries(df)

    user_input = input("\n🔍 Enter a country name: ")
    results = find_companies_by_country(df, user_input)

    print(f"\n🏢 Companies in '{user_input}':")
    for company in results:
        print(f" - {company}")
