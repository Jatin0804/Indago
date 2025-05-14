import pandas as pd

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
    Return the mapping in this format:
    Column Name -> Standard Field Name
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that maps business report columns to standard fields. For company name, prioritize exact matches and avoid using IDs."},
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
                    # Special handling for Company Name
                    if field == 'Company Name':
                        # Skip if it's an ID
                        if 'id' in col.lower():
                            continue
                        # Only map if we don't already have a better match
                        if 'Company Name' not in mappings.values() and 'Companies' not in mappings.values():
                            mappings[col] = field
                            mapping_df = pd.concat([mapping_df, pd.DataFrame({'Original Column': [col], 'Mapped To': [field]})], ignore_index=True)
                    else:
                        mappings[col] = field
                        mapping_df = pd.concat([mapping_df, pd.DataFrame({'Original Column': [col], 'Mapped To': [field]})], ignore_index=True)
        # Display the mapping results
        st.dataframe(mapping_df)
        # Create a new DataFrame with mapped columns
        result_df = pd.DataFrame()
        for standard_field in STANDARD_BUSINESS_FIELDS:
            # Find all columns that map to this standard field
            mapped_cols = [col for col, field in mappings.items() if field == standard_field]
            if mapped_cols:
                # For Company Name, use the first mapped column (should be the best match)
                if standard_field == 'Company Name':
                    result_df[standard_field] = df[mapped_cols[0]]
                else:
                    # For Geography, combine all location fields with proper separators
                    if standard_field == 'Geography':
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
        return result_df, mappings
    except Exception as e:
        st.error(f"Error in mapping columns: {str(e)}")
        return pd.DataFrame(), {}