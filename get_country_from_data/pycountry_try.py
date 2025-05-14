import pandas as pd
import pycountry
import geonamescache

# Initialize geonamescache
cache = geonamescache.GeonamesCache()

# Load cities data from the geonamescache library
cities = cache.get_cities()

# Function to normalize country name using pycountry
def normalize_country(value):
    if pd.isna(value):
        return None

    try:
        # Try pycountry for country names
        country = pycountry.countries.lookup(value)
        return country.name
    except LookupError:
        # If not a country, return None
        return None

# Function to resolve cities to countries using geonamescache
def resolve_city_to_country(city):
    # If city exists in the geonamescache cities dataset
    if city in cities:
        return cities[city]['countryName']
    return "Unresolved"

# Main resolution function that handles both country and city
def resolve_location(location):
    # First try using pycountry for country names
    country = normalize_country(location)
    if country:
        return country

    # If not a country, try resolving city/state with geonamescache
    return resolve_city_to_country(location)

# Load companies data
def load_companies_from_csv(filepath):
    return pd.read_csv(filepath)

# Normalize locations to country names
def map_locations_to_countries(df):
    df['resolved_country'] = df['Country'].apply(resolve_location)
    print("📦 Normalizing mixed locations...\n")
    for i, row in df.iterrows():
        print(f"🔍 {row['Companies']} → {row['resolved_country']}")
    return df

# Filter companies by resolved country
def find_companies_by_country(df, user_country):
    matched = df[df['resolved_country'].str.lower() == user_country.lower()]
    return matched['Companies'].tolist()

# Main execution
if __name__ == "__main__":
    filepath = "companies.csv"
    df = load_companies_from_csv(filepath)
    df = map_locations_to_countries(df)

    user_input = input("\n🔍 Enter a country name: ")
    results = find_companies_by_country(df, user_input)

    print(f"\n🏢 Companies in '{user_input}':")
    for company in results:
        print(f" - {company}")
