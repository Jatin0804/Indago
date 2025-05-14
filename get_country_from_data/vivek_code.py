import pandas as pd
from tkinter import filedialog, Tk
import os
class ExcelCellMatcher:
    def __init__(self):
        self.df = None
        self.original_indices = []
    def load_file(self):
        """Load Excel/CSV file and store original indices"""
        root = Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Select Excel or CSV File",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv")]
        )
        if not file_path:
            return False
        try:
            if file_path.endswith('.csv'):
                self.df = pd.read_csv(file_path, dtype=str, keep_default_na=False)
            else:
                self.df = pd.read_excel(file_path, dtype=str, keep_default_na=False)
            self.df.fillna("", inplace=True)
            self.original_indices = self.df.index.tolist()
            return True
        except Exception as e:
            print(f"Error loading file: {e}")
            return False
    def excel_style_search(self, search_term):
        """Replicate Excel's Find All (cell-level substring matching)"""
        search_term = str(search_term).lower().strip()
        matches = []
        for idx in self.original_indices:
            for col in self.df.columns:
                cell_value = str(self.df.at[idx, col]).lower().strip()
                if not cell_value or search_term not in cell_value:
                    continue
                # Convert to Excel-style reference (e.g., $A$2)
                col_idx = list(self.df.columns).index(col)
                col_letter = chr(ord('A') + col_idx)
                cell_ref = f"${col_letter}${idx + 1}"
                matches.append({
                    'row_index': idx,
                    'column': col,
                    'cell_ref': cell_ref,
                    'value': self.df.at[idx, col]
                })
        return matches
def main():
    print("=== Indexing Method For the Search Function ===")
    matcher = ExcelCellMatcher()
    if not matcher.load_file():
        return
    while True:
        search_term = input("\nEnter search term (or 'quit'): ").strip()
        if search_term.lower() == 'quit':
            break
        matches = matcher.excel_style_search(search_term)
        print(f"\n Found {len(matches)} matching **cells** for '{search_term}'")
        if matches:
            for match in matches[:10]:  # Show first 10
                print(f"{match['cell_ref']} ({match['column']}): {match['value']}")
            if len(matches) > 10:
                print("... (showing first 10 only)")
            # Save all matches to Excel
            output_file = f"excel_cell_matches_for_{search_term.replace(' ', '_')}.xlsx"
            pd.DataFrame(matches).to_excel(output_file, index=False)
            print(f"\nSaved full results to: {output_file}")
        else:
            print("No matches found.")
if __name__ == "__main__":
    main()