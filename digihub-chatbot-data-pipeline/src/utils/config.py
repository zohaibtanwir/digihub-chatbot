import os
import requests
from azure.cosmos._retry_options import RetryOptions
from src.utils.logger import logger
from src.utils.vault_manager import get_secret
from azure.storage.blob import BlobServiceClient  
from azure.cosmos import CosmosClient
from langchain_openai import AzureOpenAIEmbeddings
from pathlib import Path
# from src.utils.action_log import logging_container
# Initialize configuration variables
AZURE_OPENAI_ENDPOINT = None
AZURE_OPENAI_API_KEY = None
OPENAI_API_VERSION = None
AZURE_OPENAI_DEPLOYMENT = None
OPENAI_DEPLOYMENT_NAME = None
COSMOSDB_ENDPOINT = None
COSMOS_ACCOUNT_KEY = None
COSMOSDB_KEY = None



#Azure Storage Account details
AZURE_STORAGE_ACCOUNT_NAME=None
AZURE_STORAGE_ACCOUNT_URL=None
AZURE_STORAGE_CONTAINER_NAME=None
AZURE_STORAGE_BLOB_CONNECTION_STRING=None

#SharePoint details 
SP_SPN_TENANT_ID=None
SP_SPN_CLIENT_ID=None
SP_SPN_CLIENT_SECRET=None
SP_SVC_USERNAME=None
SP_SVC_PASSWORD=None

#Event Hub details
EVENT_HUB_CONNECTION_STRING=None
EVENT_HUB_CONSUMER_GROUP=None
EVENT_HUB_NAME=None
EVENT_HUB_NOTIFICATION_URL=None

COSMOSDB_NAME = None
COSMOSDB_VECTOR_INDEX = None
site_id = None
BLOB_CONTAINER_NAME = None
COSMOS_LOGGING_CONTAINER_NAME = None
COSMOSDB_SHAREPOINT_DATA_CONTAINER_NAME = None
COSMOSDB_SERVICE_NAME_MAPPING_CONTAINER_NAME=None
SESSION_CONTAINER_NAME = "Session"
KEY_VAULT_URL = None
SHAREPOINT_HOSTNAME = None
SHAREPOINT_SITENAME = None
COSMOSDB_DEBOUNCER_CONTAINER_NAME= None
file_processing_time=None
ROOT_FOLDER = Path("./input_docs")
OUTPUT_FOLDER = Path("./output_docs")
origins = None
OCR_STATUS = None
MAX_MEMORY= None
CHUNK_SIZE= None
DOCLING_ARTIFACTS_PATH = None

# Image extraction settings
ENABLE_IMAGE_TEXT_EXTRACTION = None
MIN_WORDS_FOR_VISION = None
MAX_CONCURRENT_VISION_CALLS = None

# PPTX hybrid processing
ENABLE_PPTX_HYBRID = None
PPTX_SLIDE_IMAGE_THRESHOLD = None
PPTX_SLIDE_IMAGE_AREA_THRESHOLD = None

# Chunking settings
ENABLE_TOKEN_AWARE_CHUNKING = None
CHUNK_SIZE_TOKENS = None
CHUNK_OVERLAP_TOKENS = None
ENABLE_TABLE_CONVERSION = None

# Processing improvements (error handling & resilience)
ENABLE_IDEMPOTENCY_CHECK = None
ENABLE_TRANSACTION_ROLLBACK = None
RETRY_ATTEMPTS = None
try:
    ENVIRONMENT = os.getenv("CONFIG_URL")
    # Fetch configuration details from the API
    logger.info(f"Fetching configuration details from ENV URL")
    logger.info(f"Sending GET request to ENV URL")
    response = requests.get(ENVIRONMENT)
    logger.info(f"Received response: {response.status_code} from ENV URL")

    # Check if the request was successful
    if response.status_code == 200:
        config_data = response.json()

        # Extract the required configuration details
        property_sources = config_data.get("propertySources", [])
        if property_sources:
            source = property_sources[0].get("source", {})
            KEY_VAULT_URL = source.get('KEY_VAULT_URL')
            AZURE_OPENAI_ENDPOINT = source.get("AZURE_OPENAI_ENDPOINT")
            AZURE_OPENAI_API_KEY = get_secret(source.get("AZURE_OPENAI_API_KEY"))
            OPENAI_API_VERSION = source.get("OPENAI_API_VERSION")
            AZURE_OPENAI_DEPLOYMENT = source.get("AZURE_OPENAI_DEPLOYMENT")
            OPENAI_DEPLOYMENT_NAME = source.get("OPENAI_DEPLOYMENT_NAME")
            COSMOSDB_ENDPOINT = source.get("COSMOSDB_ENDPOINT")
            COSMOS_ACCOUNT_KEY = get_secret(source.get("COSMOS_ACCOUNT_KEY"))
            COSMOSDB_KEY = get_secret(source.get("COSMOSDB_KEY"))
            AZURE_OPENAI_API_BASE = source.get("AZURE_OPENAI_ENDPOINT")
            AZURE_STORAGE_ACCOUNT_NAME=source.get("AZURE_STORAGE_ACCOUNT_NAME")
            AZURE_STORAGE_ACCOUNT_URL=source.get("AZURE_STORAGE_ACCOUNT_URL")
            AZURE_STORAGE_CONTAINER_NAME=source.get("AZURE_STORAGE_CONTAINER_NAME")
            AZURE_STORAGE_BLOB_CONNECTION_STRING=get_secret(source.get("AZURE_STORAGE_BLOB_CONNECTION_STRING"))
             #SharePoint details 
            SP_SPN_TENANT_ID=source.get("SP_SPN_TENANT_ID")
            SP_SPN_CLIENT_ID=get_secret(source.get("SP_SPN_CLIENT_ID"))
            SP_SPN_CLIENT_SECRET=get_secret(source.get("SP_SPN_CLIENT_SECRET"))
            SP_SVC_USERNAME=get_secret(source.get("SP_SVC_USERNAME"))
            SP_SVC_PASSWORD=get_secret(source.get("SP_SVC_PASSWORD"))    

            #Event Hub details
            EVENT_HUB_CONNECTION_STRING=get_secret(source.get("EVENT_HUB_CONNECTION_STRING"))
            EVENT_HUB_CONSUMER_GROUP=source.get("EVENT_HUB_CONSUMER_GROUP")
            EVENT_HUB_NAME=source.get("EVENT_HUB_NAME")
            EVENT_HUB_NOTIFICATION_URL=source.get("EVENT_HUB_NOTIFICATION_URL")

            COSMOSDB_NAME = source.get("COSMOSDB_NAME")
            COSMOSDB_VECTOR_INDEX = source.get("COSMOSDB_VECTOR_INDEX")
            BLOB_CONTAINER_NAME = source.get("BLOB_CONTAINER_NAME")
            COSMOS_LOGGING_CONTAINER_NAME = source.get("COSMOS_LOGGING_CONTAINER_NAME") 
            COSMOSDB_SHAREPOINT_DATA_CONTAINER_NAME = source.get("COSMOSDB_SHAREPOINT_DATA_CONTAINER_NAME")
            COSMOSDB_SERVICE_NAME_MAPPING_CONTAINER_NAME= source.get("COSMOSDB_SERVICE_NAME_MAPPING_CONTAINER_NAME")
            SHAREPOINT_HOSTNAME=source.get("SHAREPOINT_HOSTNAME")
            SHAREPOINT_SITENAME=source.get("SHAREPOINT_SITENAME")
            COSMOSDB_DEBOUNCER_CONTAINER_NAME=source.get("COSMOSDB_DEBOUNCER_CONTAINER_NAME")
            file_processing_time=source.get("file_processing_time")
            origins=source.get("origins")
            OCR_STATUS=source.get("OCR_STATUS")
            MAX_MEMORY=source.get("MAX_MEMORY")
            CHUNK_SIZE=source.get("CHUNK_SIZE")
            DOCLING_ARTIFACTS_PATH=source.get("DOCLING_ARTIFACTS_PATH", "/app/docling/docling-model")

            # Image extraction settings
            ENABLE_IMAGE_TEXT_EXTRACTION=source.get("ENABLE_IMAGE_TEXT_EXTRACTION", False)
            MIN_WORDS_FOR_VISION=source.get("MIN_WORDS_FOR_VISION", 10)
            MAX_CONCURRENT_VISION_CALLS=source.get("MAX_CONCURRENT_VISION_CALLS", 5)

            # PPTX hybrid processing
            ENABLE_PPTX_HYBRID=source.get("ENABLE_PPTX_HYBRID", False)
            PPTX_SLIDE_IMAGE_THRESHOLD=source.get("PPTX_SLIDE_IMAGE_THRESHOLD", 5)
            PPTX_SLIDE_IMAGE_AREA_THRESHOLD=source.get("PPTX_SLIDE_IMAGE_AREA_THRESHOLD", 30.0)

            # Chunking settings
            ENABLE_TOKEN_AWARE_CHUNKING=source.get("ENABLE_TOKEN_AWARE_CHUNKING", True)
            CHUNK_SIZE_TOKENS=source.get("CHUNK_SIZE_TOKENS", 1000)
            CHUNK_OVERLAP_TOKENS=source.get("CHUNK_OVERLAP_TOKENS", 100)
            ENABLE_TABLE_CONVERSION=source.get("ENABLE_TABLE_CONVERSION", False)

            # Processing improvements (error handling & resilience)
            ENABLE_IDEMPOTENCY_CHECK=source.get("ENABLE_IDEMPOTENCY_CHECK", True)
            ENABLE_TRANSACTION_ROLLBACK=source.get("ENABLE_TRANSACTION_ROLLBACK", True)
            RETRY_ATTEMPTS=source.get("RETRY_ATTEMPTS", 3)

            logger.info(f"Configuration details fetched successfully from ENV URL")
             
        else:
            logger.error("No property sources found in the configuration response.")
            raise ValueError("Invalid configuration response format.")
    else:
        logger.error(f"Failed to fetch configuration details. Status code: {response.status_code}")
        raise Exception(f"Failed to fetch configuration details. Status code: {response.status_code}")

except Exception as e:
    logger.critical(f"Error while fetching configuration details: {e}")
    raise

from src.utils.sharepoint_site import SharePointSiteHelper
site_helper = SharePointSiteHelper()
site_id = site_helper.get_sharepoint_site_id()
logger.info(f"site id fetched successfully")

blob_service = BlobServiceClient.from_connection_string(AZURE_STORAGE_BLOB_CONNECTION_STRING) 
cosmos_client = CosmosClient(COSMOSDB_ENDPOINT, COSMOS_ACCOUNT_KEY, connection_verify=False,retry_options=RetryOptions(
        max_retry_attempt_count=5  
    ))
database = cosmos_client.get_database_client(COSMOSDB_NAME)
# container_log=logging_container()
# Azure OpenAI Embeddings configuration  
embeddings = AzureOpenAIEmbeddings(  
    azure_endpoint=AZURE_OPENAI_API_BASE,  
    azure_deployment=AZURE_OPENAI_DEPLOYMENT,  
    openai_api_version=OPENAI_API_VERSION,  
    api_key=AZURE_OPENAI_API_KEY  
)
