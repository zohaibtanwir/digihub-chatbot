from typing import Annotated
from fastapi import Header, Depends, HTTPException, APIRouter
from src.dto.models import BaseHeader
from src.services.session_service import SessionDBService
from src.utils.logger import logger
from src.utils.request_utils import validate_incoming_request
from pydantic import EmailStr

router = APIRouter()


class SessionView:

    @staticmethod
    @router.get("/session/{chat_session_id}", tags=["DigiHub ChatBot"], dependencies=[Depends(validate_incoming_request)])
    def get(chat_session_id: str,
            query_header: Annotated[BaseHeader, Header(convert_underscores=True)],
            emailid: EmailStr | None = None,
            ):
        try:
            # Call the ResponseGeneratorAgent to get the structured response
            chat_session_id = chat_session_id.strip()
            if not chat_session_id:
                raise Exception("Mandatory Parameter Missing")
            converstations = SessionDBService().retrieve_session(chat_session_id)
            # Return the response in the desired JSON format
            return {
                "conversation": converstations
            }
        except Exception as e:
            logger.error(f"Unable to create response for user query: {e}")
            raise HTTPException(status_code=500, detail="Internal Server Error")