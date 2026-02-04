from fastapi import Header, Depends, HTTPException, APIRouter
from pydantic import EmailStr
from src.utils.logger import logger
from src.chatbot.azure_sas_url_generator import generate_container_sas_url
from src.utils.request_utils import validate_incoming_request

router = APIRouter(dependencies=[Depends(validate_incoming_request)])

class SASView:

    @staticmethod
    @router.post("/generate-sas-url", tags=["Azure Blob SAS Generator"])
    def post(x_digihub_emailid: EmailStr = Header(convert_underscores=True)):
        """
        Generate a container-level SAS URL for accessing blobs.

        Headers:
            email-id: Email of the requesting user.

        Returns:
            dict: SAS URL, expiry, and container name.
        """
        logger.info(f"Generating SAS URL for container by user: {x_digihub_emailid}")

        try:
            result = generate_container_sas_url()
            return result
        except Exception:
            raise HTTPException(status_code=500, detail="Failed to generate container SAS URL.")