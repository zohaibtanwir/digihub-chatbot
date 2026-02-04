import os
import requests
from src.utils.logger import logger
from src.utils.vault_manager import get_secret
from dotenv import load_dotenv

load_dotenv()


# Initialize configuration variables
AZURE_OPENAI_ENDPOINT = None
AZURE_OPENAI_API_KEY = None
OPENAI_API_VERSION = None
AZURE_OPENAI_DEPLOYMENT = None
OPENAI_DEPLOYMENT_NAME = None
COSMOSDB_ENDPOINT = None
COSMOS_ACCOUNT_KEY = None
COSMOSDB_KEY = None
BLACKLISTED_WORDS = None
SESSION_CONTAINER_NAME = "Session"
AZURE_STORAGE_ACCOUNT_NAME = None
AZURE_STORAGE_ACCOUNT_URL = None
AZURE_STORAGE_CONTAINER_NAME = None
AZURE_STORAGE_CONTAINER_SAS_URL_VALIDITY = None
DIGIHUB_USER_MANAGEMENT_URL = None
COSMOSDB_SERVICE_NAME_MAPPING_CONTAINER_NAME="ServiceNameMapping"

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
            AZURE_OPENAI_ENDPOINT = source.get("AZURE_OPENAI_ENDPOINT")
            KEY_VAULT_URL = source.get('KEY_VAULT_URL')
            AZURE_OPENAI_API_KEY = get_secret(source.get("AZURE_OPENAI_API_KEY"))

            OPENAI_API_VERSION = source.get("OPENAI_API_VERSION")
            AZURE_OPENAI_DEPLOYMENT = source.get("AZURE_OPENAI_DEPLOYMENT")
            OPENAI_DEPLOYMENT_NAME = source.get("OPENAI_DEPLOYMENT_NAME")
            COSMOSDB_ENDPOINT = source.get("COSMOSDB_ENDPOINT")
            COSMOS_ACCOUNT_KEY = get_secret(source.get("COSMOS_ACCOUNT_KEY"))
            COSMOSDB_KEY = get_secret(source.get("COSMOSDB_KEY"))
            AZURE_OPENAI_API_BASE = source.get("AZURE_OPENAI_ENDPOINT")
            AZURE_STORAGE_ACCOUNT_NAME = source.get("AZURE_STORAGE_ACCOUNT_NAME")
            AZURE_STORAGE_ACCOUNT_URL = source.get("AZURE_STORAGE_ACCOUNT_URL")
            AZURE_STORAGE_CONTAINER_NAME = source.get("AZURE_STORAGE_CONTAINER_NAME")
            AZURE_STORAGE_CONTAINER_SAS_URL_VALIDITY = source.get("AZURE_STORAGE_CONTAINER_SAS_URL_VALIDITY")
            DIGIHUB_USER_MANAGEMENT_URL = source.get("DIGIHUB_USER_MANAGEMENT_URL")
            CORS_ORIGINS = source.get("CORS_ORIGINS")
            KNOWLEDGE_BASE_CONTAINER = source.get("KNOWLEDGE_BASE_CONTAINER")
            COSMOSDB_SHAREPOINT_DATA_CONTAINER_NAME = source.get("COSMOSDB_SHAREPOINT_DATA_CONTAINER_NAME")
            DIGIHUB_DBNAME = source.get("DIGIHUB_DBNAME")
            blacklisted_string = property_sources[1].get("source", {}).get("app.security.xss.blacklist", "")
            if(blacklisted_string):
                BLACKLISTED_WORDS = blacklisted_string.split(",")
            logger.info("Configuration details fetched successfully.")
        else:
            logger.error("No property sources found in the configuration response.")
            raise ValueError("Invalid configuration response format.")
    else:
        logger.error(f"Failed to fetch configuration details. Status code: {response.status_code}")
        raise Exception(f"Failed to fetch configuration details. Status code: {response.status_code}")

except Exception as e:
    logger.critical(f"Error while fetching configuration details: {e}")
    raise