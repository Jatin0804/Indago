import googlemaps
import os
import dotenv
import requests

dotenv.load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  
geocoder = googlemaps.Client(key=GOOGLE_API_KEY)

def get_geocoded_info(query):
    try:
        result = geocoder.geocode(query)
        # print(result)
        if result and len(result) > 0:
            components = result[0].get('address_components', [])

            city = state = country = None
            for comp in components:
                types = comp.get('types', [])
                if 'locality' in types:
                    city = comp.get('long_name')
                elif 'administrative_area_level_1' in types:
                    state = comp.get('long_name')
                elif 'country' in types:
                    country = comp.get('long_name')

            return {
                'City': city,
                'State': state,
                'Country': country
            }
    except Exception:
        pass
    return {'City': None, 'State': None, 'Country': None}

def get_geolocation_details(location_input):
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={location_input}&key={GOOGLE_API_KEY}"
    response = requests.get(url)
    data = response.json()
    print(data)
    if data['status'] != 'OK':
        return {}
    address_components = data['results'][0]['address_components']
    location_data = {"city": "", "state": "", "country": ""}
    for comp in address_components:
        types = comp['types']
        if 'locality' in types:
            location_data["city"] = comp['long_name'].lower()
        elif 'administrative_area_level_1' in types:
            location_data["state"] = comp['long_name'].lower()
        elif 'country' in types:
            location_data["country"] = comp['long_name'].lower()
    return location_data

print(get_geocoded_info("Pilani"))
# print(get_geolocation_details("Pilani"))