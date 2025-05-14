import requests
import time
import csv
import os

OPENCAGE_API_KEY = "f2f689735dca4743b77c3e994f5683e1"
working_dir = os.getcwd()

def get_country_from_location(location_name):
    url = f"https://api.opencagedata.com/geocode/v1/json"
    params = {
        'q': location_name,
        'key': OPENCAGE_API_KEY,
        'no_annotations': 1,
        'limit': 1
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        results = response.json()['results']
        if results:
            components = results[0]['components']
            return components.get('country')
        else:
            print("No results found.")
    else:
        print("Error:", response.status_code, response.text)
    return None

company_country_map = {}

# fetch excel sheet 
sheet_loc = os.path.join(working_dir, "companies.csv")
if not os.path.exists(sheet_loc):
    print("companies.csv file not found in the current directory.")
    exit()
with open(sheet_loc, mode='r', encoding='utf-8') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        company = row['Companies']
        location = row['Country']
        country = get_country_from_location(location)
        if country:
            company_country_map[company] = country
        print(f"✅ {company} → {country}" if country else f"⚠️ Could not resolve country for {company}")
        time.sleep(1) # rate limit

# Find companies in a given country
def find_companies_by_country(search_country):
    return [company for company, country in company_country_map.items() if country.lower() == search_country.lower()]

location = input("Enter the location: ")
print(find_companies_by_country(location)) 