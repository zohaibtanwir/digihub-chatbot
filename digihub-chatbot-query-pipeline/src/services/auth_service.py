from typing import Optional
from urllib.parse import quote
import requests
from fastapi import HTTPException
from src.utils.logger import logger
from src.dto.models import BaseHeader
from src.utils.config import DIGIHUB_USER_MANAGEMENT_URL

class AuthorizationService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(AuthorizationService, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def get_subscriptions(self, header : BaseHeader) -> Optional[list[int]]:
        """
        Retrieves a list of subscriptions associated with the user.

        Args:
            header (BaseHeader): A Pydantic model containing the user's authentication and request details.

        Returns:
            list[str]: A list of subscription names that the user has.

        Description:
            This function takes a Pydantic model `header` as input, which includes necessary authentication and request information.
            It processes this header to fetch and return a list of subscriptions that the user is currently subscribed to.
        """
        try:
            email_id = header.x_digihub_emailid
            logger.info(f"Checking subscription for email: {email_id}")

            # Make a request to fetch user subscriptions
            url = f"{DIGIHUB_USER_MANAGEMENT_URL}/v1/users/subscriptions?emailid={email_id}"
            request_headers = self.get_headers(header)

            # If user impersonation is allowed, return None to skip subscription flow and allow all subscriptions access to user
            if(self.is_user_impersonation_allowed(email_id, request_headers)):
                return None
           
            auth_response = requests.get(url, headers = request_headers, verify=False)
            if(auth_response.status_code != 200):
                logger.error(f"Failed to fetch subscriptions for {email_id}. Status code: {auth_response.status_code}")
                raise HTTPException(status_code=401, detail="Unauthorized: Unable to verify user subscriptions.")
            
            user_subscriptions = [
                {'id': category['id'], 'name': category['name'], 'status': category['status']} 
                for category in auth_response.json().get("payload", [])
            ]
            return user_subscriptions
        
        except requests.RequestException as e:
            logger.error(f"Error while checking subscription: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    
    def get_headers(self, header : BaseHeader) -> dict:
        """
        Converts header keys from underscores to hyphens and returns a headers dictionary.

        Args:
            header (BaseHeader): A Pydantic model containing the header information.

        Returns:
            dict: A dictionary with header keys converted from underscores to hyphens.

        Description:
            This function takes a Pydantic model `header` as input. It processes the header to convert all key names from underscores to hyphens.
            The resulting dictionary with modified keys is then returned.
        """
        header_json = header.model_dump()
        result_json = {}
        for key,value in header_json.items():
            result_json[key.replace("_", "-")] = value
        return result_json

    def is_user_impersonation_allowed(self,email_id, request_headers):
        """
        Checks if impersonation is allowed for a given user.

        Args:
            email_id (str): The email ID of the user.
            request_headers (dict): Headers to include in the request.

        Returns:
            bool: True if impersonation is allowed, False otherwise.

        Raises:
            HTTPRequestFailed: If the HTTP request fails.
        """

        # Make a request to fetch user details to check if impersonation is allowed
        impersonation_url = f"{DIGIHUB_USER_MANAGEMENT_URL}/v1/users?emailid={email_id}"

        try:
            response = requests.get(url=impersonation_url, headers=request_headers, verify=False)
            if response.status_code != 200:
                return False
            data = response.json()
            logger.info(f"User Impersonation Allowed : {data["payload"]["isImpersonationAllowed"]}")
            return data["payload"]["isImpersonationAllowed"]
        
        except requests.RequestException as e:
            logger.error(f"Error while checking impersonation: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")