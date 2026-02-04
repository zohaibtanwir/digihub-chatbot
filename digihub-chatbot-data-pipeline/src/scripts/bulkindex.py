import requests
from pathlib import Path
from src.utils.config import site_id,database,COSMOSDB_SHAREPOINT_DATA_CONTAINER_NAME,COSMOSDB_VECTOR_INDEX,COSMOSDB_SERVICE_NAME_MAPPING_CONTAINER_NAME
from src.utils.sharepoint_spn_login import TokenRetriever
from src.utils.logger import logger  
from datetime import datetime
import os
from src.scripts.doclingmodel import download_docling
from src.service.dataprocessor import DocumentProcessor
document_processor = DocumentProcessor() 
docling_model_path = "/app/docling/docling-model"

ROOT_FOLDER = Path("./input_docs")
token_retriever = TokenRetriever()
container = database.get_container_client(COSMOSDB_SHAREPOINT_DATA_CONTAINER_NAME)
index_container = database.get_container_client(COSMOSDB_VECTOR_INDEX)
def get_drive_info(site_id, token):
    logger.info("Fetching drive information...")
    try:
        drive_url = f'https://graph.microsoft.com/v1.0/sites/{site_id}/drives'
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        drive_response = requests.get(drive_url, headers=headers)
        drive_response.raise_for_status()
        logger.info("Drive information retrieved successfully.")
        return drive_response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP error while fetching drive info: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error while fetching drive info: {e}", exc_info=True)
        return None

def list_files_and_folders(site_id, drive_id, folder_id, token):
    token = token_retriever.get_token()
    try:
        list_url = f'https://graph.microsoft.com/v1.0/sites/{site_id}/drives/{drive_id}/items/{folder_id}/children'
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        list_response = requests.get(list_url, headers=headers)

        list_response.raise_for_status()

        logger.info("Files and folders listed successfully.")
        return list_response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP error while listing files/folders: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error while listing files/folders: {e}", exc_info=True)
        return None

def download_file(site_id, drive_id, item, token, file_path, listid, current_path):
    token = token_retriever.get_token()
    logger.info(f"Downloading file: {file_path}")
    file_data = {
        'id': item['id'],
        'foldername': current_path,
        'pathwithfilename': file_path,
        'dateofcreated': item.get('createdDateTime'),
        'lastModifiedDateTime': item.get('lastModifiedDateTime'),
        'dateofprocessed': datetime.utcnow().isoformat(),
        'processedstatus': 'processing',
        'data_extracted' : 0
    }
 
    try:
        file_id = item['id']
        download_url = f'https://graph.microsoft.com/v1.0/sites/{site_id}/drives/{drive_id}/items/{file_id}/content'
        headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
        download_response = requests.get(download_url, headers=headers)
        download_response.raise_for_status()
 
        full_path = ROOT_FOLDER / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, 'wb') as f:
            f.write(download_response.content)
 
        container.upsert_item(body=file_data)  # Save as 'processing'
 
        folder_name = current_path
        result = None
        chunks_len = 0
        try:
            result,chunks_len = document_processor.process_file(full_path, folder_name, listid, file_path)
            logger.info(f"File {full_path} processed successfully.")
        except Exception as e:
            logger.error(f"Processing error for {file_path}: {e}", exc_info=True)
        
        file_data['processedstatus'] = 'Processed' if result else 'UnProcessed' 
        file_data['data_extracted'] = chunks_len

        container.upsert_item(body=file_data)
        
    except Exception as e:
        logger.error(f"Unexpected error while downloading or processing file {file_path}: {e}", exc_info=True)

def delete_existing_index(file_path):
    """Deletes metadata and vector chunks for a specific file path."""
    try:
        # 1. Find and delete metadata record
        query = "SELECT * FROM c WHERE c.pathwithfilename = @path"
        parameters = [{"name": "@path", "value": file_path}]
        items = list(container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True))
        
        for item in items:
            logger.info(f"Deleting existing metadata record for: {file_path}")
            container.delete_item(item=item['id'], partition_key=item['foldername'])

        # 2. Find and delete vector index chunks
        index_query = "SELECT * FROM c WHERE c.metadata.filepath = @filepath"
        index_params = [{"name": "@filepath", "value": file_path}]
        index_items = list(index_container.query_items(
            query=index_query,
            parameters=index_params,
            enable_cross_partition_query=True
        ))
        
        if index_items:
            logger.info(f"Deleting {len(index_items)} existing vector chunks for: {file_path}")
            for index_item in index_items:
                # Construct partition key logic as per your event hub reference
                # partitionkey = f"{serviceName}-{filename}"
                p_key = index_item.get('partitionKey') or index_item.get('serviceName') # Fallback logic
                index_container.delete_item(item=index_item['id'], partition_key=p_key)
        
        return True
    except Exception as e:
        logger.error(f"Error during index cleanup for {file_path}: {e}")
        return False

def get_db_record(item_path):
    """Returns the DB record if it exists, else None."""
    query = "SELECT * FROM c WHERE c.pathwithfilename = @path"
    parameters = [{"name": "@path", "value": item_path}]
    items = list(container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True))
    return items[0] if items else None

def process_folder(site_id, drive_id, folder_id, current_path, listid):
    token = token_retriever.get_token()
    items = list_files_and_folders(site_id, drive_id, folder_id, token)
    
    if not items:
        return

    for item in items.get('value', []):
        item_path = f"{current_path}/{item['name']}"
        
        if 'file' in item and item['name'].lower().endswith(('.pdf', '.docx')):
            existing_record = get_db_record(item_path) # Function that queries Cosmos for pathwithfilename
            
            if existing_record:
                # UPDATE LOGIC: Compare timestamps
                if item['lastModifiedDateTime'] > existing_record['lastModifiedDateTime']:
                    logger.info(f"Update detected for {item_path}. Re-indexing...")
                    delete_existing_index(item_path) # Deletes old chunks and meta
                    download_file(site_id, drive_id, item, token, item_path, listid, current_path.split('/')[0])
            else:
                # NEW FILE LOGIC
                logger.info(f"New file detected: {item_path}")
                download_file(site_id, drive_id, item, token, item_path, listid, current_path.split('/')[0])

        elif 'folder' in item:
            process_folder(site_id, drive_id, item['id'], item_path, listid)

# ... (rest of the functions like download_file and sync_deleted_files remain similar)

def list_all_files_and_folders_flat(site_id, drive_id, folder_id, token, current_path, all_items):
    """Recursively fetches a flat list of all file metadata from SharePoint."""
    items_response = list_files_and_folders(site_id, drive_id, folder_id, token)
    if not items_response:
        return

    for item in items_response.get('value', []):
        item_path = f"{current_path}/{item['name']}"
        if 'file' in item and item['name'].lower().endswith(('.pdf', '.docx')):
            all_items.append({'pathwithfilename': item_path, 'id': item['id']})
        elif 'folder' in item:
            list_all_files_and_folders_flat(site_id, drive_id, item['id'], token, item_path, all_items)

def sync_deleted_files(site_id, drive_id, drive_name):
    """Compares SharePoint with Cosmos DB and deletes records for files that no longer exist."""
    logger.info(f"Starting deletion sync for drive: {drive_name}...")
    token = token_retriever.get_token()

    # 1. Get all file paths currently in SharePoint for this drive
    sharepoint_files_list = []
    list_all_files_and_folders_flat(site_id, drive_id, 'root', token, drive_name, sharepoint_files_list)
    sharepoint_paths = {item['pathwithfilename'] for item in sharepoint_files_list}
    logger.info(f"Found {len(sharepoint_paths)} files in SharePoint for drive '{drive_name}'.")

    # 2. Get all file paths from Cosmos DB for this drive
    query = "SELECT c.id, c.pathwithfilename, c.foldername FROM c WHERE c.foldername = @drive_name"
    parameters = [{"name": "@drive_name", "value": drive_name}]
    db_items = list(container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True))
    db_paths = {item['pathwithfilename'] for item in db_items}
    logger.info(f"Found {len(db_paths)} file records in Cosmos DB for drive '{drive_name}'.")

    # 3. Find files that are in the DB but not in SharePoint
    deleted_paths = db_paths - sharepoint_paths
    
    if not deleted_paths:
        logger.info(f"No deleted files found for drive '{drive_name}'. Sync complete.")
        return

    logger.warning(f"Found {len(deleted_paths)} deleted files to remove from database.")
    
    # 4. Delete records from both containers
    for path in deleted_paths:
        # Find the specific item to get its ID and partition key
        item_to_delete = next((item for item in db_items if item['pathwithfilename'] == path), None)
        if not item_to_delete:
            continue

        try:
            # Delete from the main metadata container
            logger.info(f"Deleting metadata record for: {path}")
            container.delete_item(item=item_to_delete['id'], partition_key=item_to_delete['foldername'])

            # Delete corresponding vector index entries
            logger.info(f"Querying for vector chunks to delete for: {path}")
            index_query = "SELECT c.id, c.serviceName, c.metadata.filename, c.partitionKey FROM c WHERE c.metadata.filepath = @filepath"
            index_params = [{"name": "@filepath", "value": path}]
            index_items_to_delete = list(index_container.query_items(query=index_query, parameters=index_params, enable_cross_partition_query=True))
            
            if index_items_to_delete:
                logger.info(f"Deleting {len(index_items_to_delete)} vector chunks for: {index_items_to_delete}")
                for index_item in index_items_to_delete:
                    partition_key=index_item['partitionKey']
                    logger.info(f"partition_key {partition_key}.")
                    index_container.delete_item(item=index_item['id'], partition_key=partition_key)
            else:
                logger.info(f"No vector chunks found for {path}.")

        except Exception as e:
            logger.error(f"Failed to delete records for {path}: {e}", exc_info=True)


# --- MAIN EXECUTION ---

# Initialize mapping container
mapping_container = database.get_container_client(COSMOSDB_SERVICE_NAME_MAPPING_CONTAINER_NAME)

def get_allowed_drives_from_db():
    """Fetches active drive configurations from the mapping container."""
    logger.info("Fetching allowed drives from Mapping Container...")
    try:
        query = "SELECT * FROM c"
        items = list(mapping_container.query_items(query=query, enable_cross_partition_query=True))
        logger.info(f"Found {len(items)} active drives in mapping database.")
        return items
    except Exception as e:
        logger.error(f"Error fetching drive mappings: {e}", exc_info=True)
        return []

def main():
    """Main function to orchestrate the SharePoint processing workflow."""
    logger.info("Starting SharePoint file processing job...")
    
    # 1. Check Model
    if not os.path.exists(docling_model_path):
        logger.info(f"Model not found at {docling_model_path}. Downloading...")
        download_docling()
    else:
        logger.info(f"Model already exists at {docling_model_path}.")

    # 2. Check for Environment Variables (passed by the K8s Job)
    env_drive_id = os.getenv("DRIVE_ID")
    env_drive_name = os.getenv("DRIVE_NAME")

    # 3. Get allowed drives from Cosmos DB
    all_allowed_drives = get_allowed_drives_from_db()
    
    if not all_allowed_drives:
        logger.warning("No active drives found in mapping configuration. Task finished.")
        return

    # 4. Decide if we process one specific drive or all of them
    drives_to_process = []
    
    if env_drive_id:
        logger.info(f"Environment variable detected. Filtering for Drive ID: {env_drive_id}")
        # Find the specific drive config from the DB list that matches the ID
        drives_to_process = [d for d in all_allowed_drives if d.get('drive_id') == env_drive_id]
        
        if not drives_to_process:
            logger.error(f"Drive ID {env_drive_id} ({env_drive_name}) was passed but is not in the allowed drives list in DB.")
            return
    else:
        logger.info("No specific drive ID provided. Processing all allowed drives.")
        drives_to_process = all_allowed_drives

    # 5. Get Auth Token
    token = token_retriever.get_token()
    if not token:
        logger.error("Failed to retrieve authentication token. Aborting.")
        return

    # 6. Process the selected drives
    for drive_config in drives_to_process:
        drive_id = drive_config.get('drive_id')
        drive_name = drive_config.get('name')
        list_id = drive_config.get('list_id')
        
        if not drive_id or not drive_name:
            logger.error(f"Missing drive_id or name in config: {drive_config}")
            continue

        logger.info(f"--- Processing Drive: {drive_name} (ID: {drive_id}) ---")
        
        try:
            # Refresh token for each drive to prevent expiry
            token = token_retriever.get_token() 
            
            # Step 1: Crawl for new/updated files
            logger.info(f"Step 1: Crawling for new/updated files in drive '{drive_name}'...")
            process_folder(site_id, drive_id, 'root', drive_name, list_id)
            
            # Step 2: Sync and remove deleted files
            logger.info(f"Step 2: Syncing deleted files for drive '{drive_name}'...")
            sync_deleted_files(site_id, drive_id, drive_name)

        except Exception as e:
            logger.error(f"An unexpected error occurred while processing drive {drive_name}: {e}", exc_info=True)

    logger.info("SharePoint file processing job finished.")

if __name__ == "__main__":
    main()