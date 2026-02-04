import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from azure.cosmos import CosmosClient
import json
from collections import defaultdict
from src.utils.config import COSMOSDB_ENDPOINT,COSMOSDB_KEY,DIGIHUB_DBNAME
import json
import pandas as pd
import re
from openpyxl.utils import get_column_letter

# Constants
INPUT_JSON_FILE = "FilesReport.json"
OUTPUT_EXCEL_FILE = "FilesReport.xlsx"
CONTAINER_NAME = "dh-chatbot-sharepoint-sync"

# OUTPUT FILE NAME
OUTPUT_MD_FILE = "FilesReport.md"
OUTPUT_JSON_FILE = "FilesReport.json"

def export_unique_files_to_md():
    client = CosmosClient(COSMOSDB_ENDPOINT, COSMOSDB_KEY)
    database = client.get_database_client(DIGIHUB_DBNAME)
    container = database.get_container_client(CONTAINER_NAME)

    # Simplified query to get the metadata object or the root properties
    # This selects the whole metadata object to be safe
    query = "SELECT c.metadata, c.foldername, c.pathwithfilename, c.processedstatus, c.dateofcreated, c.data_extracted FROM c"

    try:
        print("Fetching data...")
        items = list(container.query_items(query=query, enable_cross_partition_query=True))
        print(f"Total records retrieved: {len(items)}")

        grouped_data = defaultdict(list)

        for item in items:
            # 1. Try to get data from 'metadata' block, if not, look at root level
            meta = item.get('metadata', {})
            
            # Helper to check meta first, then root
            def get_val(key, default=None):
                return meta.get(key) or item.get(key) or default

            folder = get_val('foldername', 'Uncategorized')
            
            file_info = {
                "foldername": folder,
                "pathwithfilename": get_val('pathwithfilename', 'N/A'),
                "dateofcreated": get_val('dateofcreated', 'N/A'),
                "processedstatus": get_val('processedstatus', 'N/A'),
                "data_extracted": get_val('data_extracted', 0)
            }

            grouped_data[folder].append(file_info)

        # --- 1. JSON Export ---
        with open(OUTPUT_JSON_FILE, "w", encoding="utf-8") as jf:
            json.dump(grouped_data, jf, indent=4)

        # --- 2. Markdown Export ---
        md_content = []
        md_content.append("# Cosmos DB Folder-wise Report")
        md_content.append(f"**Total Valid Records Found:** {sum(len(v) for v in grouped_data.values())}")
        md_content.append("\n---\n")

        if not grouped_data:
            md_content.append("### ⚠️ No matching data found.")
            md_content.append("Check if the fields 'foldername' or 'metadata' exist in your documents.")
        else:
            for folder_name in sorted(grouped_data.keys()):
                md_content.append(f"## 📁 Folder: {folder_name}")
                md_content.append("| Status | Created Date | Extracted | Path |")
                md_content.append("| :--- | :--- | :--- | :--- | :--- |")
                
                for info in grouped_data[folder_name]:
                    md_content.append(
                        f"| {info['processedstatus']} | "
                        f"{info['dateofcreated']} | {info['data_extracted']} | `{info['pathwithfilename']}` |"
                    )
                md_content.append("\n")

        with open(OUTPUT_MD_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(md_content))

        print(f"Success! MD saved to {OUTPUT_MD_FILE}, JSON saved to {OUTPUT_JSON_FILE}")

    except Exception as e:
        print(f"An error occurred: {e}")


def clean_sheet_name(name):
    """
    Excel sheet names have a 31-character limit and cannot contain 
    special characters like: / \ ? * [ ] :
    """
    sanitized = re.sub(r'[\\/*?:\[\]]', '_', str(name))
    return sanitized[:31]

def json_to_excel():
    try:
        with open(INPUT_JSON_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not data:
            print("No data found in JSON file.")
            return

        with pd.ExcelWriter(OUTPUT_EXCEL_FILE, engine='openpyxl') as writer:
            print(f"Creating Excel file: {OUTPUT_EXCEL_FILE}")
            
            for folder_name, files in data.items():
                if not files:
                    continue

                # 1. Create initial DataFrame
                df = pd.DataFrame(files)

                # 2. Process the 'pathwithfilename' into multiple columns
                # Split the path by '/'
                path_parts = df['pathwithfilename'].str.split('/')
                
                # Determine the maximum depth of folders
                max_depth = path_parts.map(len).max()

                # Create lists for the new path columns
                path_data = []
                for parts in path_parts:
                    # parts[:-1] are the folders, parts[-1] is the filename
                    folders = parts[:-1]
                    filename = parts[-1]
                    
                    # Pad the folder list with None so they all have the same length
                    # (max_depth - 1) because the last part is the filename
                    padding = [None] * ((max_depth - 1) - len(folders))
                    row = folders + padding + [filename]
                    path_data.append(row)

                # Create column names: Dir_1, Dir_2, ..., Filename
                path_columns = [f"Level_{i+1}" for i in range(max_depth - 1)] + ["Actual_Filename"]
                
                # Create a temporary DataFrame for the split paths
                df_paths = pd.DataFrame(path_data, columns=path_columns)

                # 3. Combine with other metadata
                # We drop the old 'filename' and 'pathwithfilename' to avoid redundancy
                df_metadata = df.drop(columns=['filename', 'pathwithfilename'], errors='ignore')
                df_final = pd.concat([df_paths, df_metadata], axis=1)

                # 4. Clean sheet name and write to Excel
                sheet_name = clean_sheet_name(folder_name)
                df_final.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # 5. Auto-adjust column widths
                worksheet = writer.sheets[sheet_name]
                for idx, col in enumerate(df_final.columns):
                    # Get the max length of the content in this column
                    series = df_final[col].astype(str)
                    max_len = max(series.map(len).max(), len(col)) + 2
                    # Limit width to 50 for readability
                    col_letter = get_column_letter(idx + 1)
                    worksheet.column_dimensions[col_letter].width = min(max_len, 50)

                print(f" - Added sheet: {sheet_name} ({len(files)} rows, {max_depth} path levels)")

        print(f"\nSuccessfully exported data to {OUTPUT_EXCEL_FILE}")

    except FileNotFoundError:
        print(f"Error: {INPUT_JSON_FILE} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    export_unique_files_to_md()
    json_to_excel()