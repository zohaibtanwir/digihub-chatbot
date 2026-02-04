import json
import uuid
from dataclasses import asdict
from datetime import datetime

# NOTE: Assumes the Session DTO has been updated with a `disclaimer` field.
from src.dto.session_object import Session
from src.services.cosmos_db_service import CosmosDBClientSingleton
from src.utils.config import SESSION_CONTAINER_NAME
from src.utils.logger import logger

from typing import Any, Dict, List, Optional

from src.utils.request_utils import timing_decorator


class SessionDBService():
    def __init__(self):
        self.database = CosmosDBClientSingleton().get_database()
        self.container = self.database.get_container_client(SESSION_CONTAINER_NAME)

    def add_session_details(self,session_instance: Any):
        """
        Add session details to the session_container in CosmosDB.

        Args:
        session_data (dict): The session data to be added.
        """
        try:
            session_data = asdict(session_instance)
            self.container.create_item(body=session_data)
        except Exception as e:
            logger.error(f"Failed to add session data: {e}")
            raise Exception(f"Failed to add session data: {e}")

    def get_session_id(self,user_id):
        incremental_no = self.get_incremental_number(user_id)
        session_id = f"{user_id}-{datetime.now().strftime('%Y-%m-%d')}-{incremental_no + 1}"
        logger.info("[SessionService] New Session Created")
        return session_id

    # MODIFIED: Added `disclaimer`, `entities`, and `chunk_service_line` parameters
    def add_user_assistant_session(self, impersonated_user_id, user_id: str, user_content: str, assistant_content: str, session_id: str, citation, score: float, confidence: float, disclaimer: Optional[str] = None, entities: Optional[List[str]] = None, chunk_service_line: Optional[List[int]] = None):

        # Use provided entities or empty list
        session_entities = entities or []
        session_chunk_service_line = chunk_service_line or []

        user_session = self.create_session(
            user_id, impersonated_user_id, "user", user_content, session_id, citation=[{}],
            entities=[],
            chunk_service_line=[]  # User message doesn't have chunk_service_line
        )

        assistant_session = self.create_session(
            user_id, impersonated_user_id, "assistant", assistant_content, session_id,
            citation=citation, score=score, confidence=confidence, disclaimer=disclaimer,
            entities=session_entities,
            chunk_service_line=session_chunk_service_line  # Store service lines from retrieved chunks
        )

        session_list = [user_session, assistant_session]

        for session in session_list:
            logger.info(f"#$#-session:{session}")
            self.add_session_details(session)
            logger.info("#$#-END ##########################")


        return assistant_session.messageId

    def get_incremental_number(self,user_id: str) -> int:
        query = f"""SELECT c.sessionId FROM c WHERE c.userId = '{user_id}'
        ORDER BY c.timestamp DESC 
            OFFSET 0 LIMIT 5
        """
        items = list(self.container.query_items(query=query, enable_cross_partition_query=True))
        if len(items)>0:
            increment_no = int(items[0].get('sessionId').split('-')[-1])
        else:
            increment_no = 0
        return increment_no

    # MODIFIED: Added `disclaimer`, `entities`, and `chunk_service_line` parameters with default values
    @staticmethod
    def create_session(user_id: str, impersonated_user_id: str, sender: str, text: str, session_id: str, citation: List[dict], score: float = 0, confidence: float = 0, disclaimer: Optional[str] = None, entities: Optional[List[str]] = None, chunk_service_line: Optional[List[int]] = None) -> Session:
        logger.info(f"Create Session for {sender} ")
        message_id = str(uuid.uuid4())
        timestamp = int(datetime.now().timestamp())

        return Session(
            id=str(uuid.uuid4()),
            messageId=message_id,
            sessionId=session_id,
            userId=user_id,
            impersonated_user_id=impersonated_user_id,
            sender=sender,
            timestamp=timestamp,
            text=text,
            citation=citation,
            entities=entities or [],
            chunk_service_line=chunk_service_line or [],
            score=score,
            confidence=confidence,
            disclaimer=disclaimer
        )

    @timing_decorator
    def retrieve_session_details(self,user_id:str,session_id:str,limit: int = 10):
        # This method is used for building context and only needs text/sender, so no changes are required here.
        query_stmt = """
            SELECT c.text,c.sender FROM c 
            WHERE c.userId = @userId AND c.sessionId = @sessionId
            ORDER BY c.timestamp ASC 
            OFFSET 0 LIMIT @limit
            """
        parameters = [
                {"name": "@userId", "value": user_id},
            {"name": "@sessionId", "value": session_id},
                {"name": "@limit", "value": limit}
           ]
        try:
            sessions = list(self.container.query_items(
                query=query_stmt,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            logger.info(f"Retrieved {len(sessions)} sessions for user_id: {user_id}")
            session_list = [{"content":session.get("text"),"role":session.get("sender")} for session in sessions]
            return session_list
        except Exception as e:
            logger.error(f"Failed to retrieve session details for user_id {user_id}: {e}")
            raise Exception(f"Failed to retrieve session details for user_id {user_id}: {e}")

    def retrieve_session(self,session_id:str):
        # MODIFIED: Updated query to also retrieve the `disclaimer` field
        query_stmt = """
            SELECT c.text,c.sender,c.messageId,c.timestamp,c.entities,c.citation,c.score,c.confidence,c.disclaimer,c.feedback_score FROM c 
            WHERE c.sessionId = @sessionId
            ORDER BY c.timestamp ASC 
            """
        parameters = [
            {"name": "@sessionId", "value": session_id},
           ]
        try:
            sessions = list(self.container.query_items(
                query=query_stmt,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            logger.info(f"Retrieved {len(sessions)} sessions for session_id: {session_id}")
            return sessions
        except Exception as e:
            logger.error(f"Failed to retrieve session details for session_id {session_id}: {e}")
            raise Exception(f"Failed to retrieve session details for session_id {session_id}: {e}")

    @timing_decorator
    def retrieve_session_entities(self, user_id: str, session_id: str, limit: int = 5) -> Dict[str, List[str]]:
        """
        Retrieve entities from recent session history for context.

        Args:
            user_id: User's hashed ID
            session_id: Current session ID
            limit: Number of recent messages to retrieve entities from (default: 5)

        Returns:
            Dictionary with categorized entities:
            {
                "services": ["WorldTracer", "Bag Manager"],
                "topics": ["lost baggage", "billing"],
                "technical_terms": ["Type B messages", "LNI code"]
            }
        """
        query_stmt = """
            SELECT c.entities FROM c
            WHERE c.userId = @userId AND c.sessionId = @sessionId
            ORDER BY c.timestamp DESC
            OFFSET 0 LIMIT @limit
            """
        parameters = [
            {"name": "@userId", "value": user_id},
            {"name": "@sessionId", "value": session_id},
            {"name": "@limit", "value": limit}
        ]

        try:
            sessions = list(self.container.query_items(
                query=query_stmt,
                parameters=parameters,
                enable_cross_partition_query=True
            ))

            # Collect all entities from recent messages
            all_entities = []
            for session in sessions:
                entities = session.get("entities", [])
                if entities:
                    all_entities.extend(entities)

            # Remove duplicates while preserving order
            unique_entities = []
            seen = set()
            for entity in all_entities:
                if entity not in seen:
                    unique_entities.append(entity)
                    seen.add(entity)

            logger.info(f"Retrieved {len(unique_entities)} unique entities from last {limit} messages")

            # Use ContextManager to group entities (import at top if not already imported)
            from src.chatbot.context_manager import ContextManager
            context_mgr = ContextManager()
            grouped_entities = context_mgr.group_entities_from_flat(unique_entities)

            return grouped_entities

        except Exception as e:
            logger.error(f"Failed to retrieve session entities for user_id {user_id}: {e}")
            # Return empty structure on error
            return {"services": [], "topics": [], "technical_terms": []}

    @timing_decorator
    def retrieve_session_service_lines(self, user_id: str, session_id: str, limit: int = 1) -> List[int]:
        """
        Retrieve chunk_service_line from recent assistant messages for contextual filtering.

        Args:
            user_id: User's hashed ID
            session_id: Current session ID
            limit: Number of recent assistant messages to retrieve service lines from (default: 1)

        Returns:
            List of unique service line IDs from previous responses
        """
        query_stmt = """
            SELECT c.chunk_service_line FROM c
            WHERE c.userId = @userId AND c.sessionId = @sessionId AND c.sender = 'assistant'
            ORDER BY c.timestamp DESC
            OFFSET 0 LIMIT @limit
            """
        parameters = [
            {"name": "@userId", "value": user_id},
            {"name": "@sessionId", "value": session_id},
            {"name": "@limit", "value": limit}
        ]

        try:
            sessions = list(self.container.query_items(
                query=query_stmt,
                parameters=parameters,
                enable_cross_partition_query=True
            ))

            # Collect all service lines from recent assistant messages
            all_service_lines = []
            for session in sessions:
                service_lines = session.get("chunk_service_line", [])
                if service_lines:
                    all_service_lines.extend(service_lines)

            # Remove duplicates while preserving integers
            unique_service_lines = list(set(all_service_lines))

            logger.info(f"Retrieved {len(unique_service_lines)} unique service lines from last {limit} assistant messages")

            return unique_service_lines

        except Exception as e:
            logger.error(f"Failed to retrieve session service lines for user_id {user_id}: {e}")
            return []

    def store_feedback(self, session_id: str, message_id: str, feedback_score: int):
        """
        Store feedback for a specific message in a session.
        Args:
        session_id (str): The ID of the session.
        message_id (str): The ID of the message to update.
        feedback (bool): The feedback value.
        """
        try:
            # Retrieve the item from Cosmos DB
            items = list(self.container.query_items(
                query="SELECT * FROM c WHERE c.sessionId = @sessionId AND c.messageId = @messageId",
                parameters=[
                    {"name": "@sessionId", "value": session_id},
                    {"name": "@messageId", "value": message_id}
                ],
                enable_cross_partition_query=True
            ))

            if not items:
                raise Exception(f"Message with ID '{message_id}' not found in session '{session_id}'")

            item = items[0]
            item['feedback_score'] = feedback_score

            # Replace the item in Cosmos DB
            self.container.replace_item(item=item, body=item)
            logger.info(f"Feedback stored for message_id: {message_id} in session_id: {session_id}")
        except Exception as e:
            logger.error(f"Failed to store feedback for message_id {message_id}: {e}")
            raise Exception(f"Failed to store feedback for message_id {message_id}: {e}")