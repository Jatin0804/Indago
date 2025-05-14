import requests
from dotenv import load_dotenv
import os
import json
import csv

load_dotenv()

ACCESS_TOKEN = os.getenv("APOLLO_API_KEY")

more = "yes"
domains = ""

# Get the current working directory
working_directory = os.getcwd()

while more == "yes":
    domain = input("Enter the domain (e.g., apollo.io): ")
    if not domain:
        print("Domain cannot be empty.")
        continue
    domains += "domains[]=" + domain + "&"
    more = input("Do you want to add more domains? (yes/no): ").strip().lower()

if not domain:
    print("Domain cannot be empty.")
    exit()
url = "https://api.apollo.io/api/v1/organizations/bulk_enrich?" + domains
# print(url)

headers = {
    "accept": "application/json",
    "Cache-Control": "no-cache",
    "Content-Type": "application/json",
    "x-api-key": ACCESS_TOKEN
}

response = requests.post(url, headers=headers)
if response.status_code == 200:
    try:
        response_data = response.json()
        response_file_path = os.path.join(working_directory, "response.json")
        with open(response_file_path, "w") as json_file:
            json.dump(response_data, json_file, indent=4)
        print(f"Response saved to {response_file_path}")
    except json.JSONDecodeError:
        print("Failed to decode JSON response.")
else:
    print(f"Request failed with status code: {response.status_code}")

response_file_path = os.path.join(working_directory, "response.json")
with open(response_file_path, 'r') as file:
    data = json.load(file)

# Extract the organizations array
organizations = data.get("organizations", [])

csv_file_path = os.path.join(working_directory, "organizations_data.csv")
fields = ['Name', 'Website URL', 'LinkedIn URL', 'Phone Number', 'Number of Employees', 'Address']

with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)

    for org in organizations:
        name = org.get('name', '')
        website_url = org.get('website_url', '')
        linkedin_url = org.get('linkedin_url', '')
        phone_number = org.get('phone', '')
        num_employees = org.get('estimated_num_employees', '')
        address = org.get('raw_address', '')

        # Write the extracted data to the CSV file
        csvwriter.writerow([name, website_url, linkedin_url, phone_number, num_employees, address])

print(f"Data has been successfully extracted and saved to {csv_file_path}.")