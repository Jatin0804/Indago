import requests
from dotenv import load_dotenv
import os

load_dotenv()

# Replace these with your LinkedIn API credentials
ACCESS_TOKEN = os.getenv("LINKEDIN_ACCESS_TOKEN")
API_BASE_URL = "https://api.linkedin.com/v2"

# Function to fetch LinkedIn profile data
def fetch_linkedin_profile():
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }

    # Endpoint to fetch profile data
    profile_url = f"{API_BASE_URL}/userinfo"

    try:
        response = requests.get(profile_url, headers=headers)
        response.raise_for_status()  # Raise an error for bad status codes
        profile_data = response.json()
        return profile_data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching profile data: {e}")
        return None

# Function to fetch email address (optional scope required)
def fetch_email_address():
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }

    # Endpoint to fetch email address
    email_url = f"{API_BASE_URL}/userinfo"

    try:
        response = requests.get(email_url, headers=headers)
        response.raise_for_status()
        email_data = response.json()
        email = email_data["email"]
        return email
    except requests.exceptions.RequestException as e:
        print(f"Error fetching email address: {e}")
        return None

if __name__ == "__main__":
    print("Fetching LinkedIn profile data...")
    profile = fetch_linkedin_profile()
    if profile:
        print("Profile Data:")
        print(profile)

    print("\nFetching LinkedIn email address...")
    email = fetch_email_address()
    if email:
        print("Email Data:")
        print(email)