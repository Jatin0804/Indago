import pandas as pd
import tkinter as tk
from tkinter import filedialog
import json
import dotenv
import os
from openai import OpenAI
import re
from fuzzywuzzy import fuzz, process  # For fuzzy matching

# Load environment variables
dotenv.load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAPI"))

# === File selection dialog ===
def select_excel_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Excel file",
        filetypes=[("Excel Files", "*.xlsx *.xls")]
    )
    return file_path

def get_user_location():
    """Prompt the user for location input"""
    return input("Enter a location (city, state, or country): ").strip()

def perform_direct_matching(df, user_location):
    """
    Perform direct text matching first for efficiency
    
    Args:
        df: Pandas DataFrame containing company data
        user_location: String with user's location query
    
    Returns:
        List of indices that match the location criteria
    """
    print("🔍 Performing direct text matching...")
    direct_matches = []
    
    # Normalize location for searching
    loc_lower = user_location.lower().strip()
    
    # Create some basic variations for direct matching
    variations = [loc_lower]
    
    # Add common variations
    if loc_lower == "india":
        variations.extend(["indian"])
    elif loc_lower in ["usa", "united states", "us"]:
        variations.extend(["usa", "united states", "us", "u.s.", "u.s.a", "america", "american"])
    elif loc_lower in ["uk", "united kingdom"]:
        variations.extend(["uk", "united kingdom", "britain", "great britain", "england", "british"])
    
    # Check each row for matches
    for idx, row in df.iterrows():
        row_values = row.astype(str).fillna("").str.lower()
        
        # Use fuzzy matching with a high threshold for direct matches
        for value in row_values:
            for variation in variations:
                # Exact match
                if variation in value:
                    direct_matches.append(idx)
                    break
                # Fuzzy match for potential typos/slight variations (only if value is non-empty)
                elif value and fuzz.partial_ratio(variation, value) > 90:
                    direct_matches.append(idx)
                    break
            if idx in direct_matches:
                break
                
    if direct_matches:
        print(f"✅ Found {len(direct_matches)} matches through direct text matching")
    else:
        print("⚠️ No direct matches found")
        
    return direct_matches

def filter_with_ai(df, user_location, already_matched=None):
    """
    Use OpenAI to filter the dataframe based on location criteria,
    focusing on rows not already matched by direct matching
    
    Args:
        df: Pandas DataFrame containing company data
        user_location: String with user's location query
        already_matched: List of indices already matched by direct matching
    
    Returns:
        List of indices that match the location criteria
    """
    if already_matched is None:
        already_matched = []
    
    # If we have many direct matches already, we might skip AI analysis
    if len(already_matched) > len(df) * 0.5:  # If more than 50% already matched
        print("⏩ Many matches found through direct matching, skipping AI analysis")
        return already_matched
    
    # Create a mask for rows that haven't been matched yet
    unmatched_rows = [i for i in range(len(df)) if i not in already_matched]
    
    # If already matched all rows
    if not unmatched_rows:
        return already_matched
    
    # Use a subset of unmatched rows for AI analysis if dataset is very large
    analyze_indices = unmatched_rows
    df_to_analyze = df.iloc[analyze_indices].copy()
    
    # Index mapping to convert from subset indices back to original indices
    index_mapping = {i: analyze_indices[i] for i in range(len(analyze_indices))}
    
    print(f"🤖 Using AI to analyze {len(df_to_analyze)} unmatched rows...")
    
    # Convert DataFrame to JSON for GPT
    records = df_to_analyze.fillna("").astype(str).to_dict(orient='records')
    table_text = json.dumps(records, indent=2)
    
    # Get column information to help AI understand the structure
    column_info = {
        "columns": list(df.columns),
        "sample_values": {col: df_to_analyze[col].iloc[:min(3, len(df_to_analyze))].tolist() for col in df_to_analyze.columns}
    }
    column_info_text = json.dumps(column_info, indent=2)
    
    # Add information about the total number of rows
    row_count_info = f"IMPORTANT: The dataset I'm giving you contains exactly {len(df_to_analyze)} rows (indices 0 to {len(df_to_analyze)-1}). These are rows that weren't matched by direct text matching. Do not return indices outside this range."
    
    # Build GPT prompt with comprehensive location matching instructions
    prompt = f"""
    You are given a subset of a company dataset. Your task is to analyze each row and determine if it relates to the user's specified location, even through indirect connections. I need you to be thorough and catch matches that basic text matching would miss.
    
    User location: "{user_location}"
    
    {row_count_info}
    
    Column structure information:
    {column_info_text}
    
    DETAILED MATCHING CRITERIA:
    
    1. COUNTRY-LEVEL MATCHING:
       - When searching for a country, identify ALL references to that country
       - Detect cities, regions, states, or provinces within that country
       - Recognize country codes, calling codes, and domain extensions (e.g., ".in" for India)
       - Match country-specific terminology and organizations (e.g., "Rupee" for India)
       
    2. CITY/REGION MATCHING:
       - Recognize alternative spellings and historical names
       - Identify neighboring suburbs or districts
       - Detect references to famous landmarks or institutions in that location
       - Consider business districts and industrial areas
       
    3. CONTEXTUAL ANALYSIS:
       - Analyze contact information formats typical to the region
       - Consider company names that strongly imply a location (e.g., "Mumbai Textiles")
       - Look for regional industry terms, regulatory references, etc.
       - Detect postal/ZIP code patterns specific to the region
       
    4. HANDLE SPECIAL CASES:
       - For locations with many namesakes (e.g., Springfield, Portland), use contextual clues
       - For locations that have changed names (e.g., Bombay→Mumbai), check for both
       - For colonial/post-colonial contexts, check both local and colonial-era terminology
    
    EXTREMELY IMPORTANT GUIDELINES:
    
    1. Be generous with matches - include rows that have ANY reasonable connection to the location
    2. Focus on finding matches that text-based search would miss
    3. Use your understanding of world geography to make smart inferences
    4. Consider phone number country codes, currencies, and languages as location indicators
    5. Return ONLY a Python list of indices from 0 to {len(df_to_analyze)-1}
    
    YOUR RESPONSE MUST BE ONLY A PYTHON LIST OF INDICES. For example: [0, 2, 5]
    Return an empty list [] if no matches are found.
    Do not include any explanation or additional text.
    
    Here is the dataset to analyze:
    {table_text}
    """
    
    # Call OpenAI API - try multiple parsing approaches to maximize success rate
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a geographical data analysis expert. Your task is to identify location-based connections in business data, going beyond simple text matching to find subtle geographic relationships. You respond ONLY with a Python list of indices."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1  # Slight randomness to avoid getting stuck in patterns
        )
        
        reply = response.choices[0].message.content
        
        # Parse GPT response with multiple fallback methods
        try:
            # First try direct eval
            try:
                matched_indices = eval(reply.strip())
                if isinstance(matched_indices, list):
                    # Map subset indices back to original indices
                    original_indices = [index_mapping[idx] for idx in matched_indices if idx in index_mapping]
                    return list(set(already_matched + original_indices))
            except:
                pass
                
            # Second, try to extract a list pattern
            list_pattern = r'\[(?:\d+(?:,\s*\d+)*)\]'
            match = re.search(list_pattern, reply)
            
            if match:
                try:
                    matched_indices = eval(match.group(0))
                    if isinstance(matched_indices, list):
                        # Map subset indices back to original indices
                        original_indices = [index_mapping[idx] for idx in matched_indices if idx in index_mapping]
                        return list(set(already_matched + original_indices))
                except:
                    pass
            
            # Last resort, try to extract individual numbers
            number_pattern = r'\b\d+\b'
            numbers = re.findall(number_pattern, reply)
            raw_indices = [int(num) for num in numbers if num.isdigit()]
            # Ensure indices are within bounds
            valid_indices = [idx for idx in raw_indices if idx in index_mapping]
            original_indices = [index_mapping[idx] for idx in valid_indices]
            
            if original_indices:
                print(f"✓ Found {len(original_indices)} additional matches through AI analysis")
                return list(set(already_matched + original_indices))
            
            # If no new matches found
            print("⚠️ AI analysis did not find additional matches")
            return already_matched
            
        except Exception as e:
            print(f"❌ Error parsing AI response: {e}")
            print(f"Raw response: {reply}")
            return already_matched
            
    except Exception as e:
        print(f"❌ Error calling OpenAI API: {e}")
        return already_matched

def double_check_results(df, matched_indices, user_location):
    """
    Use a second AI pass to verify the final results and ensure quality
    
    Args:
        df: Original DataFrame
        matched_indices: Indices found by previous methods
        user_location: User query
        
    Returns:
        Refined list of indices
    """
    if not matched_indices:
        return matched_indices
        
    print("🔍 Double-checking matched results for accuracy...")
    
    # Get subset of matched rows
    matched_df = df.iloc[matched_indices].copy()
    
    # Sample up to 30 rows if we have many matches
    sample_size = min(len(matched_df), 30)
    if len(matched_df) > sample_size:
        print(f"⚠️ Too many matches to verify all, sampling {sample_size} rows...")
        sampled_indices = list(range(len(matched_df)))
        import random
        random.shuffle(sampled_indices)
        sampled_indices = sampled_indices[:sample_size]
        df_to_verify = matched_df.iloc[sampled_indices].copy()
        # Create mapping from sample indices to original indices
        sample_mapping = {i: matched_indices[sampled_indices[i]] for i in range(len(sampled_indices))}
    else:
        df_to_verify = matched_df
        sample_mapping = {i: matched_indices[i] for i in range(len(matched_indices))}
    
    # Convert DataFrame to JSON for GPT
    records = df_to_verify.fillna("").astype(str).to_dict(orient='records')
    table_text = json.dumps(records, indent=2)
    
    # Build verification prompt
    verification_prompt = f"""
    I need you to verify if these rows truly match the location "{user_location}".
    
    Each row has been identified as potentially related to this location, but I need you to carefully 
    check if they actually have a CLEAR and MEANINGFUL connection to {user_location}.
    
    For each row index (0 to {len(df_to_verify)-1}), tell me if it should be KEPT or REMOVED from the results.
    
    Return your answer as a Python list containing ONLY the indices that should be KEPT.
    For example: [0, 2, 5] means keep rows 0, 2, and 5, and remove others.
    
    Here are the rows to verify:
    {table_text}
    """
    
    # Call OpenAI API for verification
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a quality assurance expert specializing in geographic data verification. Your task is to review potential location matches and filter out false positives. You respond ONLY with a Python list of indices to keep."},
                {"role": "user", "content": verification_prompt}
            ],
            temperature=0.0  # Zero temperature for consistent verification
        )
        
        reply = response.choices[0].message.content
        
        # Parse the response to get verified indices
        try:
            # Try to extract a list pattern first
            list_pattern = r'\[(?:\d+(?:,\s*\d+)*)\]'
            match = re.search(list_pattern, reply)
            
            if match:
                verified_sample_indices = eval(match.group(0))
                # Map back to original indices
                verified_indices = [sample_mapping[idx] for idx in verified_sample_indices if idx in sample_mapping]
                
                # If we sampled, we need to keep the unverified matches too
                if len(matched_df) > sample_size:
                    unverified_indices = [matched_indices[i] for i in range(len(matched_indices)) if i not in sampled_indices]
                    final_indices = sorted(list(set(verified_indices + unverified_indices)))
                else:
                    final_indices = sorted(verified_indices)
                
                excluded = len(matched_indices) - len(final_indices)
                if excluded > 0:
                    print(f"⚠️ Verification removed {excluded} false positives")
                else:
                    print("✅ All matches verified as correct")
                
                return final_indices
            else:
                # If no pattern found, assume all are valid to be safe
                print("⚠️ Could not parse verification response, keeping all matches")
                return matched_indices
                
        except Exception as e:
            print(f"❌ Error in verification: {e}")
            return matched_indices
            
    except Exception as e:
        print(f"❌ API error in verification: {e}")
        return matched_indices

def main():
    try:
        # Get Excel file path (uncomment to use file dialog)
        # file_path = select_excel_file()
        file_path = "companies.xlsx"  # Hardcoded for testing
        
        print(f"📊 Processing file: {file_path}")
        
        # Get user location (uncomment to prompt user)
        # user_location = get_user_location()
        user_location = "India"  # Hardcoded for testing
        
        print(f"🔍 Searching for companies in: {user_location}")
        
        # Read Excel file
        df = pd.read_excel(file_path)
        print(f"📋 Loaded {len(df)} rows from Excel file")
        
        # Stage 1: Direct matching (fast and precise)
        direct_matches = perform_direct_matching(df, user_location)
        print(direct_matches)
        
        # Stage 2: AI-based matching for rows not caught by direct matching
        all_matches = filter_with_ai(df, user_location, direct_matches)
        print(all_matches)
        
        # Stage 3: Verify results for quality
        final_matches = double_check_results(df, all_matches, user_location)
        print(final_matches)
        
        if final_matches:
            filtered_df = df.iloc[final_matches]
            output_file = f"filtered_companies_{user_location.replace(' ', '_')}.xlsx"
            filtered_df.to_excel(output_file, index=False)
            print(f"\n✅ Found {len(filtered_df)} matching companies for '{user_location}'")
            print(f"💾 Results saved to '{output_file}'")
            
            # Summary of matching results
            print("\n📊 Match Summary:")
            if len(filtered_df) <= 5:
                for i, (_, row) in enumerate(filtered_df.iterrows()):
                    print(f"  {i+1}. {' | '.join(str(v) for v in row.values if str(v).strip())[:100]}...")
            else:
                for i, (_, row) in enumerate(filtered_df.head(3).iterrows()):
                    print(f"  {i+1}. {' | '.join(str(v) for v in row.values if str(v).strip())[:100]}...")
                print(f"  ... and {len(filtered_df) - 3} more matches")
        else:
            print(f"\n❌ No companies found matching location: '{user_location}'")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()