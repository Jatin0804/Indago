import pandas as pd
import openai
from typing import Optional, Tuple, Dict
import os
import re
import streamlit as st

# Configuration
openai.api_key = ""

# Standard business fields
STANDARD_BUSINESS_FIELDS = [
    "Company Name",
    "Business Description",
    "Website",
    "Revenue",
    "Number of Employees",
    "Company Phone Number",
    "Parent Company",
    "Active Investors",
    "Contact First Name",
    "Contact Last Name",
    "Contact Title",
    "Contact Email",
    "Geography"
]
def parse_size_criteria(criteria: str) -> Tuple[Optional[int], Optional[int], str]:
    """
    Parse size criteria from various input formats
    Returns: (min_employees, max_employees, operator)
    """
    patterns = [
        (r'employees?\s*(?:more than|greater than|over)\s*(\d+)', '>'),
        (r'employees?\s*(?:less than|fewer than|under)\s*(\d+)', '<'),
        (r'employees?\s*(?:at least|minimum|>=)\s*(\d+)', '>='),
        (r'employees?\s*(?:at most|maximum|<=)\s*(\d+)', '<='),
        (r'employees?\s*(?:exactly|equal to|=)\s*(\d+)', '='),
        (r'employees?\s*(\d+)\s*to\s*(\d+)', 'range'),
        (r'(\d+)\s*to\s*(\d+)\s*employees?', 'range'),
        (r'([<>]=?)\s*(\d+)', 'operator'),
    ]
    
    criteria = criteria.lower().strip()
    
    for pattern, operator in patterns:
        match = re.search(pattern, criteria)
        if match:
            if operator == 'range':
                min_val = int(match.group(1))
                max_val = int(match.group(2))
                return min_val, max_val, 'range'
            elif operator == 'operator':
                op = match.group(1)
                val = int(match.group(2))
                if op == '>':
                    return val, None, '>'
                elif op == '<':
                    return None, val, '<'
                elif op == '>=':
                    return val, None, '>='
                elif op == '<=':
                    return None, val, '<='
            else:
                val = int(match.group(1))
                if operator == '>':
                    return val, None, '>'
                elif operator == '<':
                    return None, val, '<'
                elif operator == '>=':
                    return val, None, '>='
                elif operator == '<=':
                    return None, val, '<='
                elif operator == '=':
                    return val, val, '='
    
    try:
        val = int(criteria)
        return val, val, '='
    except ValueError:
        pass
    
    return None, None, ''
def map_to_standard_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Map input DataFrame columns to standard business fields using GPT."""
    column_list = "\n".join([f"- {col}" for col in df.columns])
    
    mapping_prompt = f"""
    Map the following columns to standard business fields:
    {column_list}
    Standard fields to map to:
    {', '.join(STANDARD_BUSINESS_FIELDS)}
    Important mapping rules:
    1. For Company Name:
       - First priority: Exact match to "Company Name" or "Companies"
       - Only if no exact match: Consider "Company Former Name", "Company Also Known As", "Company Legal Name"
       - Never use "Company ID" for company name
    2. For revenue, look for columns containing revenue or financial metrics
    3. For Geography, map ALL columns containing location information:
       - HQ Location, HQ City, HQ State, HQ Country
       - Office Location, Address, Region
       - Any column with City, State, Country in name
    4. For contact information, look for columns with contact details (name, email, phone, etc.)
    5. For phone numbers, look for columns containing "phone", "contact", "telephone", or similar terms
    6. For Number of Employees:
       - First priority: Columns with exact employee counts (e.g., "Number of Employees", "Employee Count")
       - Only if no exact count columns: Use range columns (e.g., "Headcount Range", "Employee Range")
    Return the mapping in this format:
    Column Name -> Standard Field Name
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": """You are a helpful assistant that maps business report columns to standard fields.
                    IMPORTANT RULES:
                    1. For contact information:
                       - Map ANY of these to 'Contact First Name':
                         * 'Owner'
                         * 'First Name'
                         * 'Contact First Name'
                         * 'Name of Contact'
                         * 'Contact Name'
                         * 'Primary Contact'
                         * 'Contact Person'
                         * 'Person Name'
                         * 'Full Name'
                         * 'Contact Details'
                         * 'Personnel Name'
                         * 'Representative'
                         * 'Point of Contact'
                         * 'POC'
                         * 'Assigned To'
                         * 'Responsible Person'
                         * 'Account Manager'
                         * 'Sales Rep'
                         * 'Account Owner'
                         * Any column containing:
                           - 'first name'
                           - 'contact name'
                           - 'person name'
                           - 'contact person'
                           - 'primary contact'
                           - 'point of contact'
                           - 'poc'
                           - 'representative'
                           - 'assigned to'
                           - 'responsible'
                           - 'account manager'
                           - 'sales rep'
                           - 'account owner'
                         * If a column contains a person's name and no other contact field is mapped, use it for 'Contact First Name'
                       
                       - Map ANY of these to 'Contact Last Name':
                         * 'Last Name'
                         * 'Contact Last Name'
                         * 'Surname'
                         * 'Family Name'
                         * 'Last'
                         * Any column containing:
                           - 'last name'
                           - 'surname'
                           - 'family name'
                           - 'last'
                       
                       - Map ANY of these to 'Contact Title':
                         * 'Title'
                         * 'Job Title'
                         * 'Position'
                         * 'Role'
                         * 'Designation'
                         * 'Job Role'
                         * 'Job Position'
                         * 'Employee Title'
                         * 'Staff Title'
                         * 'Employee Role'
                         * 'Staff Role'
                         * Any column containing:
                           - 'title'
                           - 'position'
                           - 'role'
                           - 'designation'
                           - 'job'
                           - 'staff'
                    
                    2. For company name, prioritize exact matches and avoid using IDs
                    3. For employee counts, prioritize exact count columns over range columns
                    
                    IMPORTANT: If a column contains a person's name and no other contact field is mapped, use it for 'Contact First Name'"""
                },
                {"role": "user", "content": mapping_prompt}
            ],
            temperature=0.1
        )
        
        mapping_text = response.choices[0].message.content.strip()
        mappings = {}
        
        # Show GPT mapping in debug UI
        st.write("GPT Mapping Results:")
        mapping_df = pd.DataFrame(columns=['Original Column', 'Mapped To'])
        
        # Parse the mapping text
        for line in mapping_text.split('\n'):
            if '->' in line:
                col, field = line.split('->')
                col = col.strip().lstrip('-').strip()  # Remove dash and extra spaces
                field = field.strip()
                if field in STANDARD_BUSINESS_FIELDS:
                    mappings[col] = field
                    mapping_df = pd.concat([mapping_df, pd.DataFrame({'Original Column': [col], 'Mapped To': [field]})], ignore_index=True)
        
        # Debug: Show raw mapping text
        st.write("\nDebug - Raw GPT Response:")
        st.write(mapping_text)
        
        # Display the mapping results
        st.dataframe(mapping_df)
        
        # Debug: Show all mappings
        st.write("\nDebug - All Mappings:")
        for orig_col, std_field in mappings.items():
            st.write(f"{orig_col} -> {std_field}")
        
        # Create a new DataFrame with mapped columns
        result_df = pd.DataFrame()
        for standard_field in STANDARD_BUSINESS_FIELDS:
            # Find all columns that map to this standard field
            mapped_cols = [col for col, field in mappings.items() if field == standard_field]
            
            # Debug: Show mapped columns for each standard field
            st.write(f"\nDebug - Mapped columns for {standard_field}:")
            st.write(mapped_cols)
            
            if mapped_cols:
                # For Company Name, use the first mapped column (should be the best match)
                if standard_field == 'Company Name':
                    result_df[standard_field] = df[mapped_cols[0]]
                # Special handling for Contact Name fields
                elif standard_field in ['Contact First Name', 'Contact Last Name']:
                    # Get the first mapped column that exists in the DataFrame
                    valid_cols = [col for col in mapped_cols if col in df.columns]
                    if valid_cols:
                        # Debug: Show the column we're using and its values
                        st.write(f"\nDebug - Using column {valid_cols[0]} for {standard_field}")
                        st.write("Sample values from original column:")
                        st.write(df[valid_cols[0]].head())
                        
                        # For contact names, we want to preserve the original value
                        # Convert to string and handle NaN values
                        result_df[standard_field] = df[valid_cols[0]].fillna('').astype(str)
                        
                        # Debug: Show the values after mapping
                        st.write(f"\nDebug - Values after mapping to {standard_field}:")
                        st.write(result_df[standard_field].head())
                        st.write(f"Number of non-null values: {result_df[standard_field].notna().sum()}")
                        st.write(f"Total rows: {len(result_df)}")
                # Special handling for Revenue
                elif standard_field == 'Revenue':
                    valid_cols = [col for col in mapped_cols if col in df.columns]
                    if valid_cols:
                        # Function to process revenue values
                        def process_revenue(row):
                            # Get all non-empty values
                            values = []
                            for col in valid_cols:
                                val = str(row[col]).strip()
                                if val and val.lower() != 'na' and val.lower() != 'nan':
                                    try:
                                        # Convert to float, handling any formatting
                                        val = float(val.replace(',', ''))
                                        values.append(val)
                                    except (ValueError, TypeError):
                                        continue
                            
                            if not values:
                                return None
                            
                            # Remove duplicates
                            values = list(set(values))
                            
                            # If only one value remains, return it
                            if len(values) == 1:
                                return values[0]
                            
                            # Otherwise return the average
                            return sum(values) / len(values)
                        
                        # Apply the processing function
                        result_df[standard_field] = df.apply(process_revenue, axis=1)
                # Special handling for Number of Employees
                elif standard_field == 'Number of Employees':
                    valid_cols = [col for col in mapped_cols if col in df.columns]
                    if valid_cols:
                        # Prioritize exact count columns over range columns
                        exact_count_cols = [col for col in valid_cols if 'range' not in col.lower()]
                        if exact_count_cols:
                            # Use the first exact count column
                            result_df[standard_field] = df[exact_count_cols[0]]
                        else:
                            # If no exact count columns, use the first range column
                            result_df[standard_field] = df[valid_cols[0]]
                # For Geography, combine all location fields with proper separators
                elif standard_field == 'Geography':
                    valid_cols = [col for col in mapped_cols if col in df.columns]
                    if valid_cols:
                        result_df[standard_field] = df[valid_cols].fillna('').astype(str).apply(
                            lambda x: ', '.join(filter(None, x)), axis=1
                        )
                else:
                    # For other fields, combine with space separator
                    valid_cols = [col for col in mapped_cols if col in df.columns]
                    if valid_cols:
                        result_df[standard_field] = df[valid_cols].fillna('').astype(str).agg(' '.join, axis=1)
            else:
                result_df[standard_field] = None
        
        # Final debug output for contact fields
        st.write("\nDebug - Final Contact Fields:")
        contact_fields = ['Contact First Name', 'Contact Last Name']
        for field in contact_fields:
            if field in result_df.columns:
                st.write(f"\n{field} values in final output:")
                st.write(result_df[field].head())
                st.write(f"Number of non-null values: {result_df[field].notna().sum()}")
                st.write(f"Total rows: {len(result_df)}")
        
        # Debug: Show the original Owner column values
        if 'Owner' in df.columns:
            st.write("\nDebug - Original Owner column values:")
            st.write(df['Owner'].head())
            st.write(f"Number of non-null values in Owner: {df['Owner'].notna().sum()}")
        
        return result_df, mappings
        
    except Exception as e:
        st.error(f"Error in mapping columns: {str(e)}")
        return pd.DataFrame(), {}
def match_location_with_gpt(addresses: list, user_location: str) -> list:
    """
    Use GPT to match addresses with user's location criteria
    Returns list of indices that match the criteria
    """
    # Format addresses with their indices for reference
    numbered_addresses = [f"{i}: {addr}" for i, addr in enumerate(addresses)]
    addresses_text = "\n".join(numbered_addresses)
    
    # Special handling for common country searches
    country_specific_rules = {
        "INDIA": """
        For India, ONLY match:
        - Locations that start with 'IN' country code
        - Indian states and union territories
        - Indian cities
        DO NOT match:
        - Other country codes (BE, GB, TR, etc.)
        - Similar looking codes or names
        """,
        "USA": """
        For USA, ONLY match:
        - Locations that start with 'US' country code
        - US states and territories
        - US cities
        DO NOT match:
        - Other country codes
        """,
    }
    
    # Get country-specific rules if available
    specific_rules = country_specific_rules.get(user_location.upper(), "")
    
    prompt = f"""
    You are a geography expert. I have a list of company locations, and I need to find ALL locations that are in {user_location}.
    
    {specific_rules}
    
    For each location below, if it belongs to {user_location}, include its index number in your response.
    Use your knowledge of global geography to match locations hierarchically.
    BE COMPREHENSIVE but EXACT - include ALL matches but be precise with country codes.
    
    Locations:
    {addresses_text}
    
    Matching Rules:
    1. For country searches:
       - For locations with country codes (e.g., 'IN', 'US'):
         * ONLY match if the country code exactly matches
         * Example: For India, only match locations starting with 'IN'
         * DO NOT match other country codes (BE, GB, TR, etc.)
       - Include all states/provinces in that country
       - Include all cities in that country
       
    2. For state/province searches:
       - Match the state name in any format
       - Match ALL cities in that state
       - Match ONLY if the state is in the correct country
       
    3. For city searches:
       - Match the city name in any format
       - Match ONLY if the city is in the correct country
    
    BE PRECISE - Especially with country codes.
    Double-check your matches before finalizing.
    
    First, list ALL locations that match {user_location} and explain why they match.
    Then, on a new line starting with "INDICES:", list ONLY the numbers of matching locations.
    Format indices as comma-separated numbers without any other text.
    
    Example response format:
    These locations match because they are in {user_location}:
    12: IN Maharashtra Mumbai (matches: Indian state and city)
    45: IN Karnataka Bangalore (matches: Indian state and city)
    
    INDICES: 12,45
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": """You are a geography expert with comprehensive knowledge of global locations.
                    Your task is to find ALL locations that belong to the searched location.
                    Be thorough but EXACT - especially with country codes.
                    For countries, ONLY match the exact country code (e.g., 'IN' for India).
                    Double-check your matches before responding.
                    Format indices as comma-separated numbers without any other text."""
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=2000
        )
        
        # Get the response content
        response_text = response.choices[0].message.content.strip()
        
        # Debug - show full response
        st.write("Debug - GPT Analysis:", response_text)
        
        # Extract indices from the response
        try:
            # Look for the INDICES: line and extract only the numbers
            if "INDICES:" in response_text:
                indices_part = response_text.split("INDICES:")[1].strip()
                # Extract numbers, handling both comma-separated and space-separated formats
                indices = [int(num.strip()) for num in re.findall(r'\d+', indices_part)]
                
                # Validate indices are within bounds
                valid_indices = [idx for idx in indices if 0 <= idx < len(addresses)]
                
                if not valid_indices:
                    st.warning(f"No valid indices found in range 0-{len(addresses)-1}")
                    return []
                
                # For specific countries, validate country codes
                if user_location.upper() == "INDIA":
                    valid_indices = [
                        idx for idx in valid_indices 
                        if addresses[idx].strip().upper().startswith("IN ")
                    ]
                elif user_location.upper() == "USA":
                    valid_indices = [
                        idx for idx in valid_indices 
                        if addresses[idx].strip().upper().startswith("US ")
                    ]
                
                # Show matched locations for debugging
                st.write("Matched Locations:")
                for idx in valid_indices:
                    st.write(f"- {addresses[idx]}")
                
                return valid_indices
            else:
                st.warning("No 'INDICES:' section found in GPT response")
                return []
                
        except Exception as e:
            st.error(f"Error extracting indices: {str(e)}")
            return []
            
    except Exception as e:
        st.error(f"Error in location matching: {str(e)}")
        return []
def filter_companies_with_gpt(df: pd.DataFrame, location_criteria: str) -> pd.DataFrame:
    """
    Process DataFrame and filter based on location using GPT's natural language understanding
    """
    try:
        st.write(f"Finding companies in {location_criteria}...")
        
        # Process in chunks of 25 rows
        chunk_size = 25
        all_indices = []
        
        for i in range(0, len(df), chunk_size):
            chunk_df = df.iloc[i:i + chunk_size].copy()
            # Reset index for the chunk to make it easier for GPT
            chunk_df = chunk_df.reset_index(drop=True)
            chunk_str = chunk_df.to_string()
            
            st.write(f"Processing rows {i} to {i + len(chunk_df)}...")
            
            # Simple, direct prompt
            prompt = f"""Here's a list of companies. Tell me which ones are in {location_criteria}.
            
Data:
{chunk_str}
Important: Only return row numbers between 0 and {len(chunk_df) - 1}.
Return ONLY the row numbers of matching companies, comma-separated (e.g., "5,8,12")."""
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # Using standard model for smaller chunks
                messages=[
                    {"role": "system", "content": "You are a geography expert. Return ONLY the row numbers of companies in the specified location."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            
            try:
                # Extract indices
                indices_str = response.choices[0].message.content.strip()
                # Extract all numbers from the response
                chunk_indices = [int(idx.strip()) for idx in re.findall(r'\d+', indices_str)]
                
                # Validate indices are within chunk bounds
                valid_chunk_indices = [idx for idx in chunk_indices if 0 <= idx < len(chunk_df)]
                
                if valid_chunk_indices:
                    # Adjust indices for the current chunk
                    adjusted_indices = [idx + i for idx in valid_chunk_indices]
                    # Validate adjusted indices are within full DataFrame bounds
                    valid_adjusted_indices = [idx for idx in adjusted_indices if 0 <= idx < len(df)]
                    all_indices.extend(valid_adjusted_indices)
                    
                    # Show matches from this chunk
                    st.write(f"Found {len(valid_chunk_indices)} matches in this chunk")
                    
            except Exception as chunk_error:
                st.warning(f"Error processing chunk: {chunk_error}")
                continue
        
        if all_indices:
            # Remove any duplicates and sort
            all_indices = sorted(list(set(all_indices)))
            
            # Final validation of indices
            valid_indices = [idx for idx in all_indices if 0 <= idx < len(df)]
            
            # Filter DataFrame using validated indices
            filtered_df = df.iloc[valid_indices].copy()
            
            st.success(f"✅ Found {len(filtered_df)} companies in {location_criteria}")
            st.dataframe(filtered_df)
            
            # Add download button
            csv = filtered_df.to_csv(index=False)
            button_key = f"download_{location_criteria}{len(filtered_df)}{hash(str(filtered_df.index))}"
            st.download_button(
                label=f"📥 Download {location_criteria} Matches",
                data=csv,
                file_name=f"{location_criteria.lower().replace(' ', '_')}_matches.csv",
                mime="text/csv",
                key=button_key
            )
            
            return filtered_df
        
        st.warning("No matching companies found")
        return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error in filtering: {str(e)}")
        return pd.DataFrame()
def process_location_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Intelligently process location data to create standardized city, state, country columns
    """
    try:
        # Create new columns
        df['Standardized_City'] = None
        df['Standardized_State'] = None
        df['Standardized_Country'] = None
        
        # Use GPT to identify location-related columns
        columns = "\n".join(df.columns.tolist())
        location_prompt = f"""
        Identify location-related columns from this list that might contain city, state, country information:
        {columns}
        
        Important:
        - Include columns with: city, state, province, country, region, HQ, headquarters, address
        - DO NOT include these technical columns:
          * Google Maps URL
          * Google Knowledge URL
          * Latitude
          * Longitude
          * Plus Code
          * Time Zone
          * Place Id
        
        Return only the column names, one per line.
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Identify columns that contain location information."},
                {"role": "user", "content": location_prompt}
            ],
            temperature=0
        )
        
        location_columns = response.choices[0].message.content.strip().split('\n')
        st.write("Processing Location Columns:", location_columns)
        
        all_conversions = []  # Store all conversions for display
        
        # Process each location column
        for col in location_columns:
            if col in df.columns:
                st.write(f"\nProcessing: {col}")
                
                # Sample unique values for GPT analysis
                unique_locations = df[col].dropna().unique()
                locations_text = "\n".join(map(str, unique_locations[:100]))  # First 100 for GPT
                
                parse_prompt = f"""
                Parse these locations into city, state, country components.
                For each location, return: original|city|state|country
                
                Important rules:
                1. For Indian locations:
                   - Always include state for Indian cities
                   - Use standard state names (e.g., Maharashtra, Karnataka)
                   - Set country as 'India' when location is Indian
                
                2. For US locations:
                   - Always include state for US cities
                   - Use standard state abbreviations (e.g., CT, NY, CA)
                   - Set country as 'United States'
                   Examples:
                   - Stamford|Stamford|CT|United States
                   - Morton Grove|Morton Grove|IL|United States
                
                3. For international locations:
                   - Include state/province when known
                   - Always include country if identifiable
                   - For ambiguous names, use most prominent location
                   Examples:
                   - Ba|Ba|Western|Fiji
                   - Sydney|Sydney|NSW|Australia
                
                Examples:
                Mumbai, Maharashtra|Mumbai|Maharashtra|India
                New Delhi|New Delhi|Delhi|India
                Bengaluru|Bengaluru|Karnataka|India
                Maharashtra||Maharashtra|India
                Stamford|Stamford|CT|United States
                Ba|Ba|Western|Fiji
                
                Locations:
                {locations_text}
                """
                
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Parse location strings into components, ensuring state is included for Indian cities."},
                        {"role": "user", "content": parse_prompt}
                    ],
                    temperature=0
                )
                
                # Store conversions for this column
                conversions = []
                for line in response.choices[0].message.content.strip().split('\n'):
                    try:
                        if '|' in line:
                            parts = line.split('|')
                            # Ensure we have exactly 4 parts, pad with empty strings if needed
                            parts = (parts + [''] * 4)[:4]
                            orig, city, state, country = parts
                            conversions.append({
                                'Original': orig.strip(),
                                'City': city.strip(),
                                'State': state.strip(),
                                'Country': country.strip()
                            })
                    except Exception as e:
                        st.write(f"Skipped invalid line: {line}")
                        continue
                
                # Display conversions for this column
                if conversions:
                    conv_df = pd.DataFrame(conversions)
                    st.write(f"Conversions for {col}:")
                    st.dataframe(conv_df)
                    all_conversions.extend(conversions)
                
                # Create mapping dictionary and apply to DataFrame
                mappings = {conv['Original']: conv for conv in conversions}
                for idx, row in df.iterrows():
                    if pd.notna(row[col]) and str(row[col]) in mappings:
                        mapping = mappings[str(row[col])]
                        if mapping['City'] and pd.isna(df.at[idx, 'Standardized_City']):
                            df.at[idx, 'Standardized_City'] = mapping['City']
                        if mapping['State'] and pd.isna(df.at[idx, 'Standardized_State']):
                            df.at[idx, 'Standardized_State'] = mapping['State']
                        if mapping['Country'] and pd.isna(df.at[idx, 'Standardized_Country']):
                            df.at[idx, 'Standardized_Country'] = mapping['Country']
        
        # Show final standardized columns
        st.write("\nFinal Standardized Locations (All Records):")
        location_data = df[['Standardized_City', 'Standardized_State', 'Standardized_Country']]
        st.dataframe(location_data, height=400)  # Added height for better scrolling
        
        # Show statistics
        total_locations = len(df)
        mapped_locations = df['Standardized_Country'].notna().sum()
        st.write(f"\nMapping Statistics:")
        st.write(f"- Total records: {total_locations}")
        st.write(f"- Successfully mapped: {mapped_locations}")
        st.write(f"- Mapping rate: {(mapped_locations/total_locations*100):.1f}%")
        
        return df
        
    except Exception as e:
        st.error(f"Error processing location data: {str(e)}")
        return df
def process_excel_file(file_path_or_df, min_employees: Optional[int] = None, max_employees: Optional[int] = None, operator: str = '', geography_criteria: str = None):
    """
    Process an Excel file or DataFrame and filter based on size and geography criteria
    """
    try:
        # Handle both file path and DataFrame input
        if isinstance(file_path_or_df, (str, pd.DataFrame)):
            if isinstance(file_path_or_df, str):
                df = pd.read_excel(file_path_or_df)
            else:
                df = file_path_or_df.copy()
        else:
            df = pd.read_excel(file_path_or_df)
        
        st.write(f"Original records: {len(df)}")
        
        # Debug: Show original columns
        st.write("\n📊 Original Columns:")
        st.dataframe(pd.DataFrame({'Original Columns': df.columns}))
        
        # Get column mappings first
        mapping_prompt = f"""
        Map these columns to standard business fields:
        {', '.join(df.columns)}
        
        Standard fields:
        {', '.join(STANDARD_BUSINESS_FIELDS)}
        
        Important:
        - Map ALL location/geography related columns (HQ, City, State, Country, Address, etc.) to 'Geography'
        - DO NOT map these technical columns:
          * Google Maps URL
          * Google Knowledge URL
          * Latitude
          * Longitude
          * Plus Code
          * Time Zone
          * Place Id
        - Map company name columns but never use ID columns
        - Map all contact and revenue fields appropriately
        
        Return only in format:
        Original Column -> Standard Field
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Map columns to standard fields, ensuring all location-related columns are mapped to Geography"},
                {"role": "user", "content": mapping_prompt}
            ],
            temperature=0
        )
        
        # Parse and display mappings
        mappings = {}
        st.write("\n🔄 Column Mappings:")
        mapping_data = []
        
        for line in response.choices[0].message.content.strip().split('\n'):
            if '->' in line:
                orig, standard = line.split('->')
                orig = orig.strip()
                standard = standard.strip()
                mappings[orig] = standard
                mapping_data.append({
                    'Original Column': orig,
                    'Mapped To': standard
                })
        
        # Display mappings in a table
        mapping_df = pd.DataFrame(mapping_data)
        st.dataframe(mapping_df)
        
        # Process location data
        df = process_location_data(df)
        
        # Apply geography filtering if criteria provided
        if geography_criteria:
            st.write(f"Filtering for location: {geography_criteria}")
            
            # Load country aliases dataset
            country_aliases = pd.read_excel('country_aliases_dataset.xlsx')
            
            # Split the geography criteria by comma and clean each part
            location_parts = [part.strip().lower() for part in geography_criteria.split(',')]
            st.write(f"Processing location parts: {location_parts}")
            
            # Initialize empty mask
            final_mask = pd.Series(False, index=df.index)
            
            # Process each location part
            for part in location_parts:
                # Get all variations for this part
                search_variations = set()
                search_variations.add(part)
                
                # Find matching country in aliases dataset
                is_country = False
                for _, row in country_aliases.iterrows():
                    if any(str(val).lower() == part for val in row if pd.notna(val)):
                        is_country = True
                        for val in row:
                            if pd.notna(val):
                                search_variations.add(str(val).lower())
                
                st.write(f"\nSearching for '{part}' with variations: {search_variations}")
                
                if is_country:
                    # If it's a country, only check the country column and ensure exact match
                    country_mask = df['Standardized_Country'].fillna('').str.lower().isin(search_variations)
                    part_mask = country_mask
                    st.write(f"'{part}' identified as a country - only checking country column")
                else:
                    # If it's not a country, check all columns
                    country_mask = df['Standardized_Country'].fillna('').str.lower().str.contains('|'.join(search_variations), na=False)
                    state_mask = df['Standardized_State'].fillna('').str.lower().str.contains('|'.join(search_variations), na=False)
                    city_mask = df['Standardized_City'].fillna('').str.lower().str.contains('|'.join(search_variations), na=False)
                    part_mask = country_mask | state_mask | city_mask
                
                # Show matches for this part
                if part_mask.any():
                    st.write(f"\nMatches found for '{part}':")
                    matches_df = df[part_mask][['Standardized_City', 'Standardized_State', 'Standardized_Country']]
                    st.write(matches_df)
                
                # Add to final mask
                final_mask = final_mask | part_mask
            
            # Apply the final mask
            df = df[final_mask]
            
            if df.empty:
                st.warning("No matches found for the location criteria")
                return pd.DataFrame()
                
            st.write(f"\nTotal records after location filtering: {len(df)}")
            st.write("Final matches:")
            st.write(df[['Standardized_City', 'Standardized_State', 'Standardized_Country']])
        
        # Then map to standard fields
        mapped_df, mappings = map_to_standard_fields(df)
        
        # Apply size filtering if criteria provided
        if min_employees is not None or max_employees is not None:
            size_column = 'Number of Employees'
            if size_column in mapped_df.columns:
                mapped_df[size_column] = pd.to_numeric(
                    mapped_df[size_column].astype(str).str.replace(',', '').str.strip(),
                    errors='coerce'
                )
                
                filtered_df = mapped_df[pd.notna(mapped_df[size_column])].copy()
                
                if operator == '>':
                    filtered_df = filtered_df[filtered_df[size_column] > min_employees]
                elif operator == '<':
                    filtered_df = filtered_df[filtered_df[size_column] < max_employees]
                elif operator == '>=':
                    filtered_df = filtered_df[filtered_df[size_column] >= min_employees]
                elif operator == '<=':
                    filtered_df = filtered_df[filtered_df[size_column] <= max_employees]
                elif operator == '=':
                    filtered_df = filtered_df[filtered_df[size_column] == min_employees]
                elif operator == 'range':
                    filtered_df = filtered_df[
                        (filtered_df[size_column] >= min_employees) & 
                        (filtered_df[size_column] <= max_employees)
                    ]
                
                mapped_df = filtered_df
                st.write(f"Records after size filtering: {len(mapped_df)}")
        
        if len(mapped_df) == 0:
            st.warning("⚠ No matching records found")
            return pd.DataFrame()
        
        st.success("✅ Data processed successfully!")
        st.write("Filtered Data Preview:")
        st.dataframe(mapped_df.head())
        
        return mapped_df
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")
        return pd.DataFrame()
def main():
    st.title("Business Research Tool")
    st.write("Upload Excel/CSV reports and filter by company size")
    
    # File upload
    uploaded_files = st.file_uploader("Upload Excel/CSV files", type=['xlsx', 'csv'], accept_multiple_files=True)
    
    if uploaded_files:
        st.write("Files uploaded successfully!")
        
        # Create two columns for inputs
        col1, col2 = st.columns(2)
        
        with col1:
            # Geography criteria input
            geography_criteria = st.text_input("Geography", 
                                             placeholder="Enter geography criteria (examples: 'US', 'Europe', 'Asia Pacific')")
        
        with col2:
            # Size criteria
            size_criteria = st.text_input("Size Criteria",
                                        placeholder="Enter size criteria (examples: 'employees > 10', '10 to 50 employees', 'more than 100')")
        
        # Add a divider
        st.markdown("---")
        
        # Create a container for the button
        button_container = st.container()
        with button_container:
            # Make the button more prominent
            st.markdown("""
                <style>
                .stButton>button {
                    width: 100%;
                    height: 3em;
                    font-size: 1.2em;
                    background-color: #4CAF50;
                    color: white;
                }
                </style>
            """, unsafe_allow_html=True)
            
            # Process button
            process_button = st.button("🚀 PROCESS FILES", key="process_button")
        
        # Add some space
        st.write("")
        
        if process_button:
            st.write("Processing started...")
            
            # Create a progress bar
            progress_bar = st.progress(0)
            
            # List to store all filtered DataFrames
            all_filtered_dfs = []
            total_records_processed = 0
            total_matches_found = 0
            
            # Process each file
            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    # Update progress
                    progress = (i + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    
                    st.write(f"Processing {uploaded_file.name}...")
                    
                    # Parse size criteria if provided
                    min_emp = None
                    max_emp = None
                    operator = ''
                    
                    if size_criteria:
                        min_emp, max_emp, operator = parse_size_criteria(size_criteria)
                    
                    # Process the file
                    result_df = process_excel_file(
                        uploaded_file,
                        min_emp,
                        max_emp,
                        operator,
                        geography_criteria if geography_criteria else None
                    )
                    
                    # Add source file information and track matches
                    if result_df is not None and not result_df.empty:
                        result_df['Source File'] = uploaded_file.name
                        all_filtered_dfs.append(result_df)
                        matches_in_file = len(result_df)
                        total_matches_found += matches_in_file
                        st.success(f"✅ {uploaded_file.name} processed successfully!")
                        st.write(f"Found {matches_in_file} matching records")
                        
                        # Show sample of matches from this file
                        st.write(f"Sample matches from {uploaded_file.name}:")
                        st.dataframe(result_df.head())
                    else:
                        st.warning(f"❌ No matching records found in {uploaded_file.name}")
                        
                    total_records_processed += len(result_df) if result_df is not None else 0
                        
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    import traceback
                    st.error(f"Detailed error: {traceback.format_exc()}")
            
            progress_bar.progress(1.0)
            
            # Combine all filtered DataFrames
            if all_filtered_dfs:
                combined_df = pd.concat(all_filtered_dfs, ignore_index=True)
                st.success("Processing completed!")
                
                # Show detailed statistics
                st.write("📊 Processing Summary:")
                st.write(f"- Total files processed: {len(uploaded_files)}")
                st.write(f"- Total records processed: {total_records_processed}")
                st.write(f"- Total matching records: {len(combined_df)}")
                st.write(f"- Matches by file:")
                for df in all_filtered_dfs:
                    file_name = df['Source File'].iloc[0]
                    st.write(f"  • {file_name}: {len(df)} matches")
                
                # Show all results
                st.write("Complete Results:")
                st.dataframe(combined_df)
                
                # Add download button for the combined results
                csv = combined_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Combined Results",
                    data=csv,
                    file_name="combined_results.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No matching records found in any of the files.")
if __name__ == "__main__":
    main()