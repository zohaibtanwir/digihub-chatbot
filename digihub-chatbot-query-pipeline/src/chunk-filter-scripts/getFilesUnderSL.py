from azure.cosmos import CosmosClient
from src.utils.config import COSMOSDB_ENDPOINT, COSMOS_ACCOUNT_KEY
import sys

# THIS file is invoked from terminal getFilesUnderSL folderName

DATABASE_NAME = "DigiHubChatBot"
CONTAINER_NAME = "dh-chatbot-sharepoint-sync"
OUTPUT_FILE = "extracted_files.md"
def query_and_generate_md(target_folder):
    client = CosmosClient(COSMOSDB_ENDPOINT, COSMOS_ACCOUNT_KEY)
    database = client.get_database_client(DATABASE_NAME)
    container = database.get_container_client(CONTAINER_NAME)

    # SQL Query using parameters for safety
    # Filters by the folder passed in the command line and data_extracted != 0
    query = """
    SELECT c.foldername, c.pathwithfilename 
    FROM c 
    WHERE c.foldername = @folder 
    AND c.data_extracted != 0
    """
    
    parameters = [{"name": "@folder", "value": target_folder}]

    results = container.query_items(
        query=query,
        parameters=parameters,
        enable_cross_partition_query=True
    )

    items = list(results)
    output_filename = f"{target_folder}_report.md"

    with open(output_filename, "w") as f:
        f.write(f"# Folder Report: {target_folder}\n")
        f.write(f"Showing files where `data_extracted` is not 0.\n\n")

        if not items:
            f.write(f"No records found for folder: **{target_folder}**")
            print(f"No records found for '{target_folder}'. MD file created.")
            return

        f.write(f"### Files in '{target_folder}'\n")
        for item in items:
            path = item.get('pathwithfilename', 'N/A')
            f.write(f"- {path}\n")

    print(f"Successfully generated {output_filename}")

if __name__ == "__main__":
    # Check if a folder name was provided as an argument
    if len(sys.argv) < 2:
        print("Error: Please provide a folder name.")
        print("Usage: python getFilesUnderService.py <foldername>")
        sys.exit(1)

    folder_input = sys.argv[1]
    try:
        query_and_generate_md(folder_input)
    except Exception as e:
        print(f"An error occurred: {e}")