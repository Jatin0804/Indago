from geopy.geocoders import OpenCage
import pandas as pd
import time

OPENCAGE_API_KEY = "f2f689735dca4743b77c3e994f5683e1"

geolocator = OpenCage(api_key=OPENCAGE_API_KEY, timeout=10)

def get_location_details(location_name):
    """
    Fetches country and state details for a given location name using OpenCage API.
    """
    try:
        location = geolocator.geocode(location_name, exactly_one=True)
        if location:
            components = location.raw.get('components', {})
            print(f"DEBUG: Components for '{location_name}': {components}") 

            # Extract country and state using aliases in components
            country = components.get('country') or components.get('country_code')
            state = components.get('state') or components.get('state_code') or components.get('region') or components.get('county')

            return country, state
    except Exception as e:
        print(f"Error for '{location_name}': {e}")
    return None, None

def map_locations_to_details(df):
    """
    Resolves country and state for each row in the DataFrame.
    """
    df['resolved_country'] = None
    df['resolved_state'] = None
    print("📦 Resolving locations to countries and states...\n")
    for i, row in df.iterrows():
        location = row.get('Country') or row.get('State') or row.get('Location')
        if not location:
            print(f"⚠️ No valid location found for row {i}")
            continue

        resolved_country, resolved_state = get_location_details(location)
        if resolved_country:
            df.at[i, 'resolved_country'] = resolved_country
            df.at[i, 'resolved_state'] = resolved_state
            print(f"✅ {row.get('Companies', 'Unknown')} → Country: {resolved_country}, State: {resolved_state or 'N/A'}")
        else:
            print(f"⚠️ Could not resolve location for {row.get('Companies', 'Unknown')}")
        time.sleep(0.5)  # Be polite with API requests
    return df

def load_companies_from_csv(filepath):
    """
    Loads company data from a CSV file.
    """
    return pd.read_csv(filepath)

def find_companies_by_location(df, user_location=None):
    """
    Finds companies by matching the resolved country and/or state.
    """
    if not user_location:
        print("⚠️ No location provided.")
        return []

    # Ensure the location is lowercase for comparison
    user_location = user_location.lower()

    # Filter by country or state
    filtered_df = df[
        (df['resolved_country'].str.lower() == user_location) |
        (df['resolved_state'].str.lower() == user_location)
    ]

    # Return the list of companies
    return filtered_df['Companies'].tolist()

if __name__ == "__main__":
    filepath = "companies.csv"
    df = load_companies_from_csv(filepath)
    df = map_locations_to_details(df)

    while True:
        user_location = input("\n🔍 Enter a location name (country/state) to search for companies: ").strip()
        if not user_location:
            print("⚠️ Please provide a location name.")
            continue

        results = find_companies_by_location(df, user_location=user_location)

        if results:
            print(f"\n🏢 Companies matching your criteria:")
            for company in results:
                print(f" - {company}")
        else:
            print(f"⚠️ No companies found matching your criteria.")

        exit_choice = input("\nDo you want to search again? (yes/no): ").strip().lower()
        if exit_choice != 'yes':
            break
