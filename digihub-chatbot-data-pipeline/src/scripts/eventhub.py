from azure.eventhub.aio import EventHubConsumerClient
import requests
import json
from datetime import datetime, timedelta
from pathlib import Path
from datetime import datetime
from azure.cosmos import CosmosClient, PartitionKey
import os
from src.utils.config import EVENT_HUB_CONNECTION_STRING,EVENT_HUB_CONSUMER_GROUP,EVENT_HUB_NAME,site_id,COSMOSDB_NAME,cosmos_client,COSMOSDB_SHAREPOINT_DATA_CONTAINER_NAME,COSMOSDB_VECTOR_INDEX,COSMOSDB_DEBOUNCER_CONTAINER_NAME
from src.utils.logger import logger
from src.utils.sharepoint_spn_login import TokenRetriever
from src.service.dataprocessor import DocumentProcessor
document_processor = DocumentProcessor() 

token_retriever = TokenRetriever()

ROOT_FOLDER = Path("./input_docs")
 
async def on_event(partition_context, event):
    logger.info(f"Received event from partition: {partition_context.partition_id}")
    logger.debug(f"Raw event data: {event}")

    try:
        process_event(event)
        logger.info("Event processed successfully.")
    except Exception as e:
        logger.error(f"Failed to process event: {e}")
    finally:
        await partition_context.update_checkpoint(event)

async def start_eventhub_listener():
    logger.info("Starting EventHubConsumerClient...")
    consumer_client = EventHubConsumerClient.from_connection_string(
        conn_str=EVENT_HUB_CONNECTION_STRING,
        consumer_group=EVENT_HUB_CONSUMER_GROUP,
        eventhub_name=EVENT_HUB_NAME,
    )

    try:
        logger.info("Listening for events...")
        await consumer_client.receive(
            on_event=on_event,
        )
    except Exception as e:
        logger.error(f"Error in EventHub listener: {e}")
    finally:
        await consumer_client.close()

def process_event(event):
    token=token_retriever.get_token()

    try:
        logger.info("Processing event...")
        event_data = json.loads(event.body_as_str())
        drive_id = event_data["value"][0]["resource"].split('/')[-2]
        logger.info(f"Extracted drive_id: {drive_id}")
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        drive_info_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}"
        drive_info_response = requests.get(drive_info_url, headers=headers)
        drive_info_response.raise_for_status()

        Drivename = drive_info_response.json()["name"]
        logger.info(f"Drive name: {Drivename}")

        all_items = list_all_files_and_folders(site_id, drive_id, "root", token, Drivename)
        logger.info(f"Total items fetched from SharePoint: {len(all_items)}")
    
        database = cosmos_client.get_database_client(COSMOSDB_NAME)
        container = database.get_container_client(COSMOSDB_SHAREPOINT_DATA_CONTAINER_NAME)
        logger.info("Connected to Cosmos DB container.")

        query = f"""SELECT * FROM c WHERE c.foldername = @drivename"""
        parameters = [{"name": "@drivename", "value": Drivename}]
        db_items = list(container.query_items(query=query,parameters=parameters,enable_cross_partition_query=True))
        logger.info(f"Items fetched from DB: {len(db_items)}")

        sharepoint_files = {item["pathwithfilename"] for item in all_items}
        db_files = {item["pathwithfilename"] for item in db_items}


        created_files = sharepoint_files - db_files
        updated_files = sharepoint_files & db_files
        deleted_files = db_files - sharepoint_files

        logger.info(f"Created files: {created_files}")
        logger.info(f"Deleted files: {deleted_files}")

        index_database = cosmos_client.get_database_client(COSMOSDB_NAME)
        index_container = index_database.get_container_client(COSMOSDB_VECTOR_INDEX)

        logger.info(created_files)
        for item in all_items:
            item_path = item["pathwithfilename"]
            
            if item_path in created_files:
                logger.info(f"Creating new file record: {item_path}")
                file_data = {
                    'id': item['id'],
                    'foldername': Drivename,
                    'pathwithfilename': item_path,
                    'dateofcreated': item['createdDateTime'],
                    'lastModifiedDateTime': item['lastModifiedDateTime'],
                    'dateofprocessed': datetime.utcnow().isoformat(),
                    'processedstatus': 'processing',
                    'data_extracted' : 0
                }
                container.upsert_item(body=file_data)
                chunks_len = 0
                try:
                    chunks_len=download_file(drive_id, item["id"],item_path,Drivename)
                    logger.info(f"file processed: {item_path}")
                    file_data['processedstatus'] = 'Processed'
                    file_data['data_extracted'] = chunks_len
                    container.upsert_item(body=file_data)
                except Exception as e:
                    file_data['processedstatus'] = 'UnProcessed'
                    file_data['data_extracted'] = chunks_len
                    container.upsert_item(body=file_data)
                    logger.error(f"An error occurred during conversion: {e}")
                
            elif item_path in updated_files:
                db_item = next(db_item for db_item in db_items if db_item["pathwithfilename"] == item_path)
                if item['lastModifiedDateTime'] > db_item['lastModifiedDateTime']:
                    logger.info(f"Updating file record: {item_path}")    
                    db_item['lastModifiedDateTime'] = item['lastModifiedDateTime']
                    db_item['dateofprocessed'] = datetime.utcnow().isoformat()
                    container.upsert_item(body=db_item)

                    debouncer_container = database.create_container_if_not_exists(id=COSMOSDB_DEBOUNCER_CONTAINER_NAME, partition_key=PartitionKey(path='/drive_id'))
                    debouncer_item = {
                        "id": Drivename,
                        "drive_id": drive_id,
                        "item_id" : item["id"],
                        "item_path" : item_path,
                        "last_event_time": datetime.utcnow().isoformat(),
                        "scheduled_trigger_time": (datetime.utcnow() + timedelta(minutes=5)).isoformat(),
                        "status": "pending"
                    }
                    debouncer_container.upsert_item(body=debouncer_item)
                    logger.info(f"Debouncer entry updated for: {Drivename}")


        for item in db_items:
            if item["pathwithfilename"] in deleted_files:
                logger.info(f"Deleting file record: {item['pathwithfilename']}")
                container.delete_item(item=item['id'], partition_key=item['foldername'])
                
                # Delete corresponding vector index entries
                file_path = item["pathwithfilename"]
                # new_file_path = "input_docs\\" + file_path
                index_query = "SELECT * FROM c WHERE c.metadata.filepath = @filepath"
                index_params = [{"name": "@filepath", "value": file_path}]
                index_items = list(index_container.query_items(
                    query=index_query,
                    parameters=index_params,
                    enable_cross_partition_query=True
                ))
                logger.info(f"Deleting vector index for: {file_path}")
                for index_item in index_items:
                    partitionkey=f"{index_item['serviceName']}-{index_item['metadata']['filename'].replace(' ', '')}"
                    index_container.delete_item(item=index_item['id'], partition_key=partitionkey)

    except KeyError as e:
        logger.info(f"KeyError: {e}. Event data structure might be different than expected.")
    except Exception as e:
        logger.info(f"An error occurred: {e}")

def list_all_files_and_folders(site_id, drive_id, folder_id, token, current_path):
    
    folder_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives/{drive_id}/items/{folder_id}/children"
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    response = requests.get(folder_url, headers=headers)
    response.raise_for_status()
    items = response.json().get('value', [])
    all_items = []
    for item in items:
        item_path = f"{current_path}/{item['name']}"
        if 'file' in item:
            if item['name'].lower().endswith(('.pdf', '.docx')):
                logger.debug(f"Found file: {item_path}")
                file_data = {
                    'id': item['id'],
                    'pathwithfilename': item_path,
                    'createdDateTime': item['createdDateTime'],
                    'lastModifiedDateTime': item['lastModifiedDateTime'],
                    'webUrl': item.get('webUrl', 'URL not available')  
                }
                all_items.append(file_data)
        elif 'folder' in item:
            logger.debug(f"Found folder: {item_path}")
            all_items.extend(list_all_files_and_folders(site_id, drive_id, item['id'], token, item_path))
    
    return all_items


def download_file(drive_id, item_id,item_path,Drivename):
    token=token_retriever.get_token()
    logger.info(f"Downloading file: {item_path}")
    file_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives/{drive_id}/items/{item_id}/content"
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    download_response = requests.get(file_url, headers=headers)
    download_response.raise_for_status()
    List_url = f'https://graph.microsoft.com/v1.0/sites/{site_id}/drives/{drive_id}/list'  
    List_response = requests.get(List_url, headers=headers)     

    List_response.raise_for_status()

    List_info = List_response.json()  
    listid=List_info["id"]
    local_folder = ROOT_FOLDER
    os.makedirs(local_folder, exist_ok=True)

    full_path = os.path.join(local_folder, item_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)


    # Save the file to the specified directory
    with open(full_path, 'wb') as f:
        f.write(download_response.content)
    file_name = os.path.basename(item_path)
    logger.info(f"File saved to: {full_path} in {Drivename}")
    result,chunks_len=document_processor.process_file(full_path, Drivename,listid,item_path)
    logger.info(f"File {file_name} processed successfully.")
    return chunks_len