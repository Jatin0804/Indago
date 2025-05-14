import streamlit as st
import pandas as pd
import openai
from typing import Optional, Tuple, Dict
import os
import re
from io import BytesIO
from main import (
    parse_size_criteria,
    process_excel_file
)


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
    "Contact Email"
]
def main():
    st.title("Business Research Tool")
    st.write("Upload Excel/CSV reports and filter by company size")
    
    # File upload
    uploaded_files = st.file_uploader("Upload Excel/CSV files", type=['xlsx', 'xls', 'csv'], accept_multiple_files=True)
    
    if uploaded_files:
        # Create two columns for inputs
        col1, col2 = st.columns(2)
        
        with col1:
            # Size criteria input
            st.subheader("Size Criteria")
            criteria = st.text_input(
                "Enter size criteria (examples: 'employees > 10', '10 to 50 employees', 'more than 100')",
                help="Supports various formats like 'employees > 10', '10 to 50 employees', 'more than 100'"
            )
        
        with col2:
            # Geography criteria input
            st.subheader("Geography")
            geography = st.text_input(
                "Enter geography criteria (examples: 'US', 'Europe', 'Asia Pacific')",
                help="Enter the geographic region to filter by"
            )
        
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
            # List to store all filtered DataFrames
            all_filtered_dfs = []
            
            # Process each uploaded file
            for uploaded_file in uploaded_files:
                # Create a container for each file's processing
                with st.container():
                    st.subheader(f"📄 Processing: {uploaded_file.name}")
                    
                    # Read file based on extension
                    if uploaded_file.name.lower().endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    # Show initial stats
                    st.write(f"📊 Initial records: {len(df)}")
                    
                    # Process and filter data
                    if criteria or geography:
                        min_emp = None
                        max_emp = None
                        operator = ''
                        
                        if criteria:
                            min_emp, max_emp, operator = parse_size_criteria(criteria)
                        
                        # Process the file with both size and geography criteria
                        result_df = process_excel_file(df, min_emp, max_emp, operator, geography)
                        
                        if result_df is not None and not result_df.empty:
                            # Add source file information
                            result_df['Source File'] = uploaded_file.name
                            all_filtered_dfs.append(result_df)
                            st.write(f"✅ Filtered records: {len(result_df)}")
                        else:
                            st.warning("⚠ No matching records found")
                
                # Add a divider between files
                st.markdown("---")
            
            # Show combined results if any files were processed successfully
            if all_filtered_dfs:
                combined_df = pd.concat(all_filtered_dfs, ignore_index=True)
                st.success("✅ Processing completed!")
                st.subheader("📊 Combined Results")
                st.write(f"Total matching records across all files: {len(combined_df)}")
                
                # Show combined data in an expander
                with st.expander("📋 View Combined Data", expanded=True):
                    st.dataframe(combined_df)
                
                # Add download button for combined results
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    combined_df.to_excel(writer, index=False)
                st.download_button(
                    label="⬇ Download Combined Results",
                    data=output.getvalue(),
                    file_name="combined_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.warning("⚠ No matching records found in any of the files.")

if __name__ == "__main__":
    main()