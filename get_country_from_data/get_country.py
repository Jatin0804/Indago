from geopy.geocoders import Nominatim
import time

geolocator = Nominatim(user_agent="geoapi-example")

def get_country_from_location(location_name):
    try:
        location = geolocator.geocode(location_name, addressdetails=True)
        if location and 'country' in location.raw['address']:
            return location.raw['address']['country']
    except Exception as e:
        print(f"Error for '{location_name}': {e}")
    return None

print(get_country_from_location("Fairfield"))  
print(get_country_from_location("Gujarat"))     
print(get_country_from_location("Paris"))     

companies = {
    "Company A": "India",
    "Company B": "Fairfield",
    "Company C": "Gujarat",
    "Company D": "New York",
    "Company E": "Paris"
}

company_country_map = {}

for company, loc in companies.items():
    country = get_country_from_location(loc)
    if country:
        company_country_map[company] = country
    time.sleep(1)  # Prevent rate-limiting


def find_companies_by_country(search_country):
    return [company for company, country in company_country_map.items() if country.lower() == search_country.lower()]


print(find_companies_by_country("India"))  

