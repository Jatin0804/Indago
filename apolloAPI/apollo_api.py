import requests
from dotenv import load_dotenv
import os
import json
import csv

load_dotenv()

ACCESS_TOKEN = os.getenv("APOLLO_API_KEY")

domain = input("Enter the domain (e.g., apollo.io): ")
if not domain:
    print("Domain cannot be empty.")
    exit()
url = "https://api.apollo.io/api/v1/organizations/enrich?domain=" + domain

headers = {
    "accept": "application/json",
    "Cache-Control": "no-cache",
    "Content-Type": "application/json",
    "x-api-key": ACCESS_TOKEN
}

response = requests.get(url, headers=headers)
if response.status_code == 200:
    data = response.json()
    
    # Remove the topmost "organization" key if it exists
    if "organization" in data:
        data = data["organization"]

    with open("temp_json.json", "w") as json_file:
        json.dump(data, json_file, indent=4)
    
    # Save as JSON (append to existing JSON file)
    json_file_path = "result.json"
    if os.path.isfile(json_file_path):
        with open(json_file_path, "r") as json_file:
            existing_data = json.load(json_file)
        if isinstance(existing_data, list):
            existing_data.append(data)
        else:
            existing_data = [existing_data, data]
    else:
        existing_data = [data]
    
    with open(json_file_path, "w") as json_file:
        json.dump(existing_data, json_file, indent=4)
    
    
    # File paths
    input_file = "temp_json.json"
    output_file = "output.csv"

    # Load data from temp_json
    with open(input_file, "r") as json_file:
        data = json.load(json_file)

    # Open CSV file for writing
    with open(output_file, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        
        # Write the header row
        writer.writerow([
            "Name", 
            "LinkedIn URL", 
            "Website URL", 
            "Mobile Number", 
            "Industry", 
            "Number of Employees", 
            "Location", 
            "Departmental Head Count"
        ])
        
        # Ensure data is a list for iteration
        if isinstance(data, dict):
            data = [data]  # Wrap single dictionary in a list

        # Iterate through the data and write rows
        for organization in data:
            if isinstance(organization, dict):  # Ensure each item is a dictionary
                name = organization.get("name", "N/A")
                linkedin_url = organization.get("linkedin_url", "N/A")
                website_url = organization.get("website_url", "N/A")
                mobile_number = organization.get("primary_phone", {}).get("number", "N/A")
                industry = organization.get("industry", "N/A")
                num_employees = organization.get("estimated_num_employees", "N/A")
                location = organization.get("raw_address", "N/A")
                departmental_head_count = organization.get("departmental_head_count", "N/A")
                
                # Write the row
                writer.writerow([
                    name, 
                    linkedin_url, 
                    website_url, 
                    mobile_number, 
                    industry, 
                    num_employees, 
                    location, 
                    departmental_head_count
                ])
            else:
                print(f"Unexpected data format: {organization}")

    print(f"Data has been successfully written to {output_file}")
else:
    print(f"Error: {response.status_code}")

# print(response.text)