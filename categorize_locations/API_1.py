import os
import asyncio
from openai import AsyncOpenAI
import pandas as pd
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Set OpenAI API key and client
client = AsyncOpenAI(api_key=os.getenv("OPENAPI"))

# Output folder
OUTPUT_DIR = "LocSheets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

async def analyze_column_with_gpt(column_name, sample_data):
    """
    Async call to GPT to analyze if a column is related to location.
    """
    prompt = f"""
    You are an AI assistant. I will provide you with a column name and some sample data from an Excel sheet. 
    Your task is to determine if the column is related to location information (e.g., city, state, country, address, etc.).

    Column Name: {column_name}
    Sample Data: {sample_data}

    Respond with "Yes" if the column is related to location, otherwise respond with "No".
    """
    try:
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return column_name, response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Error analyzing column '{column_name}': {e}")
        return column_name, "Error"

async def process_file(filepath):
    """
    Process a single Excel file, identify location-related columns, and save them.
    """
    print(f"\n📂 Processing file: {filepath}")
    try:
        df = pd.read_excel(filepath)
        tasks = []

        for column in df.columns:
            sample_data = df[column].dropna().head(5).tolist()
            if not sample_data:
                continue
            tasks.append(analyze_column_with_gpt(column, sample_data))

        results = await asyncio.gather(*tasks)
        location_cols = [col for col, result in results if result.lower() == "yes"]

        if location_cols:
            filtered_df = df[location_cols]
            filename = os.path.basename(filepath).replace(".xlsx", "_locations.xlsx")
            output_path = os.path.join(OUTPUT_DIR, filename)
            filtered_df.to_excel(output_path, index=False)
            print(f"✅ Saved to: {output_path}")
        else:
            print("ℹ️ No location-related columns found.")

    except Exception as e:
        print(f"❌ Error processing file '{filepath}': {e}")

async def main():
    filepaths = [
        "c1.xlsx",
        "c2.xlsx",
        "c3.xlsx"
    ]
    await asyncio.gather(*(process_file(fp) for fp in filepaths))

if __name__ == "__main__":
    asyncio.run(main())
