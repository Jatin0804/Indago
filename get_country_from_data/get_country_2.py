# import pandas as pd
# from geopy.geocoders import Nominatim
# from geopy.exc import GeocoderTimedOut
# import time

# geolocator = Nominatim(user_agent="company-location-mapper")

# def get_country_from_location(location_name):
#     try:
#         location = geolocator.geocode(location_name, addressdetails=True)
#         if location:
#             return location.raw.get('address', {}).get('country')
#     except GeocoderTimedOut:
#         return get_country_from_location(location_name)  # Retry on timeout
#     except Exception as e:
#         print(f"Error for '{location_name}': {e}")
#     return None


# def load_companies_from_csv(filepath):
#     return pd.read_csv(filepath)


# def map_locations_to_countries(df):
#     df['resolved_country'] = None
#     print("📦 Resolving locations to countries...\n")
#     for i, row in df.iterrows():
#         location = row['Country']
#         resolved = get_country_from_location(location)
#         if resolved:
#             df.at[i, 'resolved_country'] = resolved
#             print(f"✅ {row['Companies']} → {resolved}")
#         else:
#             print(f"⚠️ Could not resolve country for {row['Companies']}")
#         time.sleep(1)  # Delay to avoid API rate limiting
#     return df


# def find_companies_by_country(df, user_country):
#     matched = df[df['resolved_country'].str.lower() == user_country.lower()]
#     return matched['Companies'].tolist()

# if __name__ == "__main__":
#     filepath = "companies.csv"
#     df = load_companies_from_csv(filepath)
#     df = map_locations_to_countries(df)

#     user_input = input("\n🔍 Enter a country name: ")
#     results = find_companies_by_country(df, user_input)

#     print(f"\n🏢 Companies in '{user_input}':")
#     for company in results:
#         print(f" - {company}")

import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time

# Initialize geolocator with English language
geolocator = Nominatim(user_agent="company-location-mapper")

def get_country_from_location(location_name):
    try:
        # Request results in English
        location = geolocator.geocode(location_name, addressdetails=True, language="en")
        if location:
            return location.raw.get('address', {}).get('country')
    except GeocoderTimedOut:
        return get_country_from_location(location_name)  # Retry on timeout
    except Exception as e:
        print(f"❌ Error for '{location_name}': {e}")
    return None

def load_companies_from_csv(filepath):
    return pd.read_csv(filepath)

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
        time.sleep(1)  # Respect Nominatim's rate limit
    return df

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
