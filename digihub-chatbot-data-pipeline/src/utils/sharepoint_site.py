import requests
from src.utils.config import SHAREPOINT_HOSTNAME,SHAREPOINT_SITENAME
from src.utils.sharepoint_spn_login import TokenRetriever
from src.utils.logger import logger
class SharePointSiteHelper:
    def __init__(self):
        self.token_retriever = TokenRetriever()

    def get_sharepoint_site_id(self) -> str:
        """
        Retrieves the SharePoint site ID using Microsoft Graph API.

        Returns:
            str: The site ID if successful, None otherwise.
        """
        try:
            token = self.token_retriever.get_token()
            headers = {
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json'
            }

            url = f'https://graph.microsoft.com/v1.0/sites/{SHAREPOINT_HOSTNAME}:/sites/{SHAREPOINT_SITENAME}/'
            logger.debug(f"Requesting SharePoint site ID from URL: {url}")
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            site_info = response.json()
            return site_info.get('id')

        except Exception as e:
            logger.error(f"Error retrieving SharePoint site ID: {e}", exc_info=True)
            return None
