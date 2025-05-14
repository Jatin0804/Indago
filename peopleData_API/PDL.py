import requests
import json
import os


API_KEY = "46dfea806d367f3bd4c29d03af0a4a6dad7175894a37fd4be05306f73c03b73c"


headers = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY
}


print("🔍 People Data Labs Company Lookup Tool")
company_input = input("Enter company domain (e.g., openai.com): ").strip()


params = {
    "website": f"https://{company_input}"
}


response = requests.get(
    "https://api.peopledatalabs.com/v5/company/enrich",
    headers=headers,
    params=params
)


if response.status_code == 200:
    new_data = response.json()


    # Load existing data if available
    file_path = "company_data.json"
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = [existing_data]
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []


    # Append the new data
    existing_data.append(new_data)


    # Write back to the file
    with open(file_path, "w") as f:
        json.dump(existing_data, f, indent=4)


    print("Company data added to company_data.json")


else:
    print(f"Failed to get company data ({response.status_code}): {response.text}")


