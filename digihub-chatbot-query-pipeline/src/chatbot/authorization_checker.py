"""
Authorization Checker Module

Handles user authorization validation, cross-checking service line access,
and generating appropriate error messages for unauthorized access.
"""

import json
from typing import Optional
from src.enums.subscriptions import get_service_names
from src.exceptions.service_line_exception import (
    UnAuthorizedServiceLineException,
    PartialAccessServiceLineException
)
from src.services.retrieval_service import RetreivalService
from src.chatbot.relevance_judge import RelevanceJudge
from src.utils.logger import logger


class AuthorizationChecker:
    """
    Validates user authorization for service lines and handles authorization exceptions.
    """

    # Sensitive service line that requires special authorization
    SENSITIVE_SERVICE_LINE_ID = 460

    # Multi-language unauthorized access messages
    UNAUTHORIZED_SERVICE_LINE_MESSAGES = {
        "english": (
            "For more information, you'll need access to:<br><br>{services}<br><br>"
            "If you believe you should have access to this information, please navigate to the Services page and submit a subscription request."
        ),
        "german": (
            "Für weitere Informationen benötigen Sie Zugriff auf:<br><br>{services}<br><br>"
            "Falls Sie der Meinung sind, dass Sie Zugriff haben sollten, gehen Sie bitte zur Services-Seite und stellen Sie dort eine Zugriffsanfrage."
        ),
        "french": (
            "Pour plus d'informations, vous devez avoir accès à :<br><br>{services}<br><br>"
            "Si vous considérez être éligible, veuillez soumettre une demande d'accès via la page Services pour chaque service correspondant."
        ),
        "spanish": (
            "Para más información, necesitarás acceso a:<br><br>{services}<br><br>"
            "Si consideras que deberías tener acceso, por favor dirígete a la página de Servicios y solicita la suscripción correspondiente."
        )
    }

    UNAUTHORIZED_DIRECT_MESSAGES = {
        "english": (
            "It looks like the information you're asking for is part of a service you currently are not subscribed to.<br><br>"
            "To view this content, you'll need access to:<br><br>{services}<br><br>"
            "If you believe you should have access to this information, please navigate to the Services page and submit a subscription request."
        ),
        "german": (
            "Die angeforderten Informationen gehören zu einem Service, der noch nicht von Ihnen bezogen wird, oder der in ihrem Profil fehlt.<br><br>"
            "Um auf diesen Inhalt zuzugreifen, benötigen Sie Zugriff auf:<br><br>{services}<br><br>"
            "Falls Sie der Meinung sind, dass Sie Zugriff haben sollten, gehen Sie bitte zur Services-Seite und stellen Sie dort eine Zugriffsanfrage."
        ),
        "french": (
            "Les informations que vous recherchez font apparemment partie d'un service auquel vous n'êtes pas encore abonné(e).<br><br>"
            "Pour y accéder, vous devez disposer des droits d'accès au(x) service(s) suivant(s) :<br><br>{services}<br><br>"
            "Si vous considérez être éligible, veuillez soumettre une demande d'accès via la page Services pour chaque service correspondant."
        ),
        "spanish": (
            "Parece que la información que estás buscando forma parte de un servicio al que actualmente no estás suscrito.<br><br>"
            "Para acceder a este contenido, necesitas tener acceso a:<br><br>{services}<br><br>"
            "Si consideras que deberías tener acceso, por favor dirígete a la página de Servicios y solicita la suscripción correspondiente."
        )
    }

    def __init__(self):
        """Initialize the AuthorizationChecker."""
        self.relevance_judge = RelevanceJudge()

    def cross_check_authorization(
        self,
        prompt: str,
        service_line: Optional[list[dict]],
        detected_language: str,
        is_out_of_scope: bool = False,
        final_response: str = ""
    ):
        """
        Cross-checks user authorization for all queries and appends a disclaimer for partially accessible information.

        Args:
            prompt (str): The user query
            service_line (Optional[list[dict]]): List of service line objects with 'id', 'name', and 'status' fields
            detected_language (str): Detected language for error messages
            is_out_of_scope (bool): Whether the query is out of scope
            final_response (str): The generated response text
            used_service_lines (Optional[list[int]]): Service line IDs used in the response context

        Raises:
            UnAuthorizedServiceLineException: When user lacks access to all relevant service lines
            PartialAccessServiceLineException: When user has partial access but is missing some service lines
        """
        if not isinstance(service_line, list):
            return

        # --- STEP 1: Process service_line (List of Dicts) ---
        # We need to separate Authorized IDs for checking, and a Map for looking up names later.
        authorized_ids = [0]
        id_name_map = {}

        for item in service_line:
            # Map ID to Name for all categories (Subscribed or Not)
            id_name_map[item['id']] = item['name']

            # Determine Authorization: Check status or if it's General Info (0)
            # Assuming 'status' key exists based on previous prompt.
            # If id is 0, it's always authorized (General Info).
            if item.get('status') == 'SUBSCRIBED' or item['id'] == 0:
                authorized_ids.append(item['id'])

        # --- STEP 2: Logic Checks using authorized_ids ---
        service_lines_to_exclude = []

        # Check against the list of integers (authorized_ids), not the list of dicts
        if self.SENSITIVE_SERVICE_LINE_ID not in authorized_ids:
            logger.info(f"User lacks access to service line {self.SENSITIVE_SERVICE_LINE_ID}. Excluding it from relevance search.")
            service_lines_to_exclude.append(self.SENSITIVE_SERVICE_LINE_ID)

        retrieved_chunks = RetreivalService().get_ranked_service_line_chunk(
            prompt,
            exclude_service_lines=service_lines_to_exclude
        )

        if not retrieved_chunks:
            logger.info("No chunks were retrieved for authorization cross-check.")
            return

        relevant_service_lines = self.relevance_judge.judge_chunks_relevance(prompt, retrieved_chunks)

        if not relevant_service_lines:
            logger.info("No relevant service lines identified for authorization cross-check.")
            return

        unique_relevant_service_lines = list(set(relevant_service_lines))
        logger.info(f"Relevant Service Lines IDs: {unique_relevant_service_lines} | User's Authorized IDs: {authorized_ids}")

        # Check relevant items against authorized_ids
        unauthorized_service_line_ids = [item for item in unique_relevant_service_lines if item not in authorized_ids]

        if unauthorized_service_line_ids:
            # --- STEP 3: Get Names from service_line (id_name_map) ---
            # We assume service_line contains ALL categories (Subscribed & Unsubscribed), so the ID should exist.
            # If not found, fallback to string of ID.
            name_overrides = {
                "BILLING_DOCUMENT_MANAGEMENT": "BILLING",
                "BILLING": "BILLING",
                "SERVICE_MANAGEMENT_DOCUMENT_MANAGEMENT": "Operational Support"
            }

            # --- Build the List with Renaming Logic ---
            service_names_list = []
            for uid in unauthorized_service_line_ids:
                # 1. Get the original name from your ID map
                raw_name = id_name_map.get(uid, str(uid))

                # 2. Check if there is an override, otherwise keep the raw name
                final_name = name_overrides.get(raw_name, raw_name)

                service_names_list.append(final_name)

            # Join them as before
            service_names = "<br>".join(service_names_list)

            message_template = self.UNAUTHORIZED_SERVICE_LINE_MESSAGES.get(
                detected_language.lower(),
                self.UNAUTHORIZED_SERVICE_LINE_MESSAGES["english"]
            )
            formatted_message = message_template.format(services=service_names)

            # --- STEP 4: Create JSON Payload ---
            disclaimer_json = json.dumps({
                "message": formatted_message,
                "service_names": service_names
            })

            if is_out_of_scope:
                # If the original query was out-of-scope, we can definitively say the user lacks access
                raise UnAuthorizedServiceLineException(disclaimer_json)
            else:
                # For in-scope queries, the user gets an answer but is also alerted about content they're missing
                raise PartialAccessServiceLineException(message=final_response, disclaimar=disclaimer_json)

    def cross_check_authorization_direct(
        self,
        prompt: str,
        service_line: Optional[list[int]],
        chunk_service_line: list,
        content: str,
        detected_language: str,
        access_context: list,
        chunk_acess_services: list,
        final_response: str,
        is_generic: bool
    ):
        """
        Cross Checks Chunks Service Line Access is Authorized for User Or Not

        Args:
            prompt (str): User query
            service_line (Optional[list[int]]): User's authorized service line IDs
            chunk_service_line (list): Service line IDs from retrieved chunks
            content (str): Retrieved content
            detected_language (str): Detected language for error messages
            access_context (list): Context chunks the user has access to
            chunk_acess_services (list): Service lines user has access to from chunks
            final_response (str): Generated response text
            is_generic (bool): Whether query is generic

        Raises:
            UnAuthorizedServiceLineException: When user lacks access to required service lines
            PartialAccessServiceLineException: When user has partial access
        """
        if isinstance(service_line, list):  # For Non Admin User WE get List of Service Line
            # This fetches Top 4 Chunks Ranked List of ServiceNameId
            logger.info(
                f"Service Line Authorization Chunk Service Line :{chunk_service_line} User Service Line: {service_line}"
            )
            if not is_generic and not set(chunk_service_line).issubset(service_line):
                # Checks if Chunks Service Line is Present in User Service Line
                unauthorized_service_line = [item for item in chunk_service_line if item not in service_line]

                service_names = "<br>".join(get_service_names(unauthorized_service_line))

                message_template = self.UNAUTHORIZED_DIRECT_MESSAGES.get(
                    detected_language.lower(),
                    self.UNAUTHORIZED_DIRECT_MESSAGES["english"]
                )
                message = message_template.format(services=service_names)

                if chunk_acess_services and len(chunk_acess_services) < len(chunk_service_line):
                    # If the user has partial access
                    raise PartialAccessServiceLineException(
                        message=final_response,
                        disclaimar=message
                    )

                raise UnAuthorizedServiceLineException(message)
