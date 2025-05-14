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
        location_city = row.get('City')
        location_state = row.get('State') 
        location_country = row.get('Country')
        
        resolved = get_country_from_location(location_city)
        if resolved:
            df.at[i, 'resolved_country'] = resolved
            print(f"✅ {i} {row.get('Company Name', 'Unknown')} → {resolved}")
            continue
        resolved = get_country_from_location(location_state)
        if resolved:
            df.at[i, 'resolved_country'] = resolved
            print(f"✅ {i} {row.get('Company Name', 'Unknown')} → {resolved}")
            continue
        resolved = get_country_from_location(location_country)
        if resolved:
            df.at[i, 'resolved_country'] = resolved
            print(f"✅ {i} {row.get('Company Name', 'Unknown')} → {resolved}")
            continue
        
        full_location = row.get('Location')
        if full_location:
            location_parts = full_location.split(",")
            location = location_parts[0] or location_parts[1] or location_parts[2]
            # print(f"Extracted location: {location}")
        
        resolved = get_country_from_location(location)
        if resolved:
            df.at[i, 'resolved_country'] = resolved
            print(f"✅ {i} {row.get('Company Name', 'Unknown')} → {resolved}")
        else:
            print(f"⚠️ {i} Could not resolve country for {row.get('Company Name', 'Unknown')} -> {location}")
        time.sleep(0.5)  
    return df

def load_companies_from_excel(filepath):
    return pd.read_excel(filepath)  

def find_companies_by_country(df, user_country):
    matched = df[df['resolved_country'].str.lower() == user_country.lower()]
    return matched['Company Name'].tolist()

if __name__ == "__main__":
    filepath = "companies_try.xlsx"  
    df = load_companies_from_excel(filepath)
    df = map_locations_to_countries(df)

    user_input = input("\n🔍 Enter a country name: ")
    results = find_companies_by_country(df, user_input)

    print(f"\n🏢 Companies in '{user_input}':")
    for company in results:
        print(f" - {company}")
