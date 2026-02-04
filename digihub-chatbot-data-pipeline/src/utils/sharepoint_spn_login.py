import requests
import time
from src.utils.config import (
    SP_SPN_CLIENT_SECRET,
    SP_SPN_TENANT_ID,
    SP_SPN_CLIENT_ID,
    SP_SVC_USERNAME,
    SP_SVC_PASSWORD
)
from src.utils.logger import logger

class TokenRetriever:
    _global_token = None
    _token_expiry = 0  # Epoch time in seconds

    def __init__(self,
                 tenant_id=SP_SPN_TENANT_ID,
                 client_id=SP_SPN_CLIENT_ID,
                 client_secret=SP_SPN_CLIENT_SECRET,
                 username=SP_SVC_USERNAME,
                 password=SP_SVC_PASSWORD):
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.username = username
        self.password = password

    def get_token(self):
        current_time = time.time()

        if TokenRetriever._global_token and TokenRetriever._token_expiry > current_time:
            logger.debug("Returning SP Cached token")
            return TokenRetriever._global_token

        logger.info("Token expired or not found. Requesting new token...")

        token_url = f'https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token'
        payload = {
            'client_id': self.client_id,
            'grant_type': 'password',
            'username': self.username,
            'password': self.password,
            'scope': 'Sites.Read.All',
            'client_secret': self.client_secret
        }

        try:
            response = requests.post(token_url, data=payload)
            response.raise_for_status()
            token_data = response.json()

            if 'access_token' not in token_data:
                raise Exception("Access token not found in response.")

            # Cache token and expiry
            TokenRetriever._global_token = token_data['access_token']
            expires_in = int(token_data.get('expires_in', 3600))  # seconds until expiry
            TokenRetriever._token_expiry = current_time + expires_in - 60  # buffer of 60s

            logger.info(f"New token retrieved. Expires in {expires_in} seconds.")
            return TokenRetriever._global_token

        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP error during token retrieval: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error during token retrieval: {e}", exc_info=True)
            raise
