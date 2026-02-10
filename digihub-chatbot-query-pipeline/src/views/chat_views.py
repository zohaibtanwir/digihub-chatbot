import traceback
import json

from fastapi import APIRouter, Depends, Header, Query, HTTPException
from fastapi.responses import StreamingResponse
from typing import Annotated, Optional
import uuid
from pydantic import BaseModel, Field, EmailStr
from src.chatbot.response_generator import ResponseGeneratorAgent
from src.dto.models import QueryRequest, BaseHeader
from src.services.auth_service import AuthorizationService
from src.services.session_service import SessionDBService
from src.utils.config import KNOWLEDGE_BASE_CONTAINER
from src.utils.hash_util import HashUtils
from src.utils.logger import logger
import datetime
from src.services.session_service import SessionDBService
from src.utils.request_utils import validate_incoming_request, validate_request_body

router = APIRouter()


# Define the request body model for the feedback endpoint
class FeedbackRequest(BaseModel):
    feedback_score: int = Field(..., description="The feedback score provided by the user.")


class ChatView:

    @staticmethod
    @router.post("/chat", tags=["DigiHub ChatBot"], dependencies=[Depends(validate_incoming_request)])
    def post(query_request: Annotated[QueryRequest, Depends(validate_request_body)],
             query_header: Annotated[BaseHeader, Header(convert_underscores=True)],
             emailid: EmailStr | None = None,
             ):
        """
        Chatbot API endpoint to interact with the AI chatbot.

        Args:
            query_header:
            query_request (QueryRequest): The user query.
            emailid (EmailStr, optional): Impersonation email.

        Returns:
            dict: The chatbot's response.
        """
        start_time = datetime.datetime.utcnow()
        ## Impersonation Util ## START
        if emailid and emailid != query_header.x_digihub_emailid:
            query_header.x_digihub_emailid = emailid
            logger.info(f"Impersonated Email:{query_header.x_digihub_emailid}")
            impersonated_user_id = query_header.x_digihub_emailid
        else:
            logger.info(f"Non Impersonation User Flow Email: {query_header.x_digihub_emailid}")
            impersonated_user_id = "Null"
        ## Impersonation Util ## END

        user_id = HashUtils.hash_user_id(str(query_header.x_digihub_emailid))
        chat_session_id = query_request.chat_session_id.strip()
        if not chat_session_id or chat_session_id == "":
            chat_session_id = SessionDBService().get_session_id(user_id)

        if chat_session_id:
            chat_session_userhash = chat_session_id.split('-')[0]
            if user_id != chat_session_userhash:
                raise Exception("Invalid Session Id Passed")

        logger.info("Starting chat with user")
        query = query_request.query.strip()
        if not query:
            logger.warning("Empty query received. Returning default response.")
            return {
                "session_id": chat_session_id,
                "response": "Hi, I am DigiHub AI Bot. How may I help you?",
                "citation": None,
                "score": None,
                "timestamp": datetime.datetime.utcnow().isoformat()
            }

        try:
            # Call the ResponseGeneratorAgent to get the structured response
            user_subscriptions = AuthorizationService().get_subscriptions(query_header)
            logger.info(f"User Subscriptions : {user_subscriptions}")
            before_response_time = datetime.datetime.utcnow()
            response_data = ResponseGeneratorAgent(
                user_id=user_id,
                session_id=chat_session_id,
                impersonated_user_id=impersonated_user_id
            ).generate_response(query, KNOWLEDGE_BASE_CONTAINER, user_subscriptions)

            end_time = datetime.datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            before_response_duration = (before_response_time - start_time).total_seconds()
            logger.info(
                f"Request processed in {duration:.2f} seconds Before Rag Initialization: {before_response_duration:.2f}")
            # Return the response in the desired JSON format
            return {
                "session_id": chat_session_id,
                "message_id": response_data.get("message_id"),
                "response": response_data["response"],
                "citation": response_data.get("citation"),
                "disclaimer": response_data.get('disclaimer', None),
                "score": response_data.get("score", 0),
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "confidence": response_data.get("confidence", 0)
            }
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Unable to create response for user query: {e}")
            raise HTTPException(status_code=500, detail="Internal Server Error")

    @staticmethod
    @router.post("/chat/stream", tags=["DigiHub ChatBot"], dependencies=[Depends(validate_incoming_request)])
    def post_stream(query_request: Annotated[QueryRequest, Depends(validate_request_body)],
             query_header: Annotated[BaseHeader, Header(convert_underscores=True)],
             emailid: EmailStr | None = None,
             ):
        """
        Streaming Chatbot API endpoint - streams tokens as they are generated.

        Uses Server-Sent Events (SSE) format:
        - data: {"type": "token", "content": "..."} for each token
        - data: {"type": "metadata", "data": {...}} for final metadata
        - data: {"type": "error", "message": "..."} for errors
        - data: [DONE] when complete

        Args:
            query_header: Request headers
            query_request (QueryRequest): The user query.
            emailid (EmailStr, optional): Impersonation email.

        Returns:
            StreamingResponse: SSE stream of tokens and metadata
        """
        start_time = datetime.datetime.utcnow()

        # Impersonation handling
        if emailid and emailid != query_header.x_digihub_emailid:
            query_header.x_digihub_emailid = emailid
            logger.info(f"Impersonated Email:{query_header.x_digihub_emailid}")
            impersonated_user_id = query_header.x_digihub_emailid
        else:
            logger.info(f"Non Impersonation User Flow Email: {query_header.x_digihub_emailid}")
            impersonated_user_id = "Null"

        user_id = HashUtils.hash_user_id(str(query_header.x_digihub_emailid))
        chat_session_id = query_request.chat_session_id.strip()
        if not chat_session_id or chat_session_id == "":
            chat_session_id = SessionDBService().get_session_id(user_id)

        if chat_session_id:
            chat_session_userhash = chat_session_id.split('-')[0]
            if user_id != chat_session_userhash:
                raise HTTPException(status_code=400, detail="Invalid Session Id Passed")

        logger.info("Starting streaming chat with user")
        query = query_request.query.strip()
        if not query:
            logger.warning("Empty query received. Returning default response.")
            async def empty_response():
                yield f"data: {json.dumps({'type': 'token', 'content': 'Hi, I am DigiHub AI Bot. How may I help you?'})}\n\n"
                yield f"data: {json.dumps({'type': 'metadata', 'data': {'session_id': chat_session_id, 'response': 'Hi, I am DigiHub AI Bot. How may I help you?', 'citation': None, 'score': None, 'confidence': 0}})}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(empty_response(), media_type="text/event-stream")

        def generate_stream():
            try:
                user_subscriptions = AuthorizationService().get_subscriptions(query_header)
                logger.info(f"User Subscriptions : {user_subscriptions}")

                agent = ResponseGeneratorAgent(
                    user_id=user_id,
                    session_id=chat_session_id,
                    impersonated_user_id=impersonated_user_id
                )

                # Yield session_id first
                yield f"data: {json.dumps({'type': 'session', 'session_id': chat_session_id})}\n\n"

                # Stream the response
                for event in agent.generate_response_streaming(query, KNOWLEDGE_BASE_CONTAINER, user_subscriptions):
                    yield f"data: {json.dumps(event)}\n\n"

                # Signal completion
                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.error(f"Streaming error: {e}")
                traceback.print_exc()
                yield f"data: {json.dumps({'type': 'error', 'message': 'Internal Server Error'})}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    @staticmethod
    @router.post("/chat/{session_id}/message/{message_id}/feedback", tags=["DigiHub ChatBot"])
    def post_feedback(session_id: str, message_id: str, feedback: FeedbackRequest):
        """
        API endpoint to submit feedback for a specific message.

        Args:
            session_id (str): The ID of the chat session from the URL path.
            message_id (str): The ID of the message from the URL path.
            feedback (FeedbackRequest): The feedback data from the request body.

        Returns:
            dict: A confirmation message.
        """
        try:
            logger.info(f"Storing feedback for session '{session_id}' and message '{message_id}'")
            
            session_db_service = SessionDBService()

            session_db_service.store_feedback(
                session_id=session_id,
                message_id=message_id,
                feedback_score=feedback.feedback_score,
            )
            logger.info("Feedback stored successfully.")
            return {"message": "Feedback stored successfully"}

        except Exception as e:
            logger.error(f"Failed to store feedback: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail="Failed to store feedback.")