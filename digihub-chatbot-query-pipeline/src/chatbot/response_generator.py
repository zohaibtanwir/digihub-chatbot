"""
Response Generator Module

Main orchestrator for the RAG (Retrieval-Augmented Generation) chatbot.
Coordinates query analysis, retrieval, authorization, and response generation.
"""

import json
import re
import traceback
from openai import BadRequestError
from src.enums.prompt_template import PromptTemplate
from src.exceptions.service_line_exception import (
    UnAuthorizedServiceLineException,
    OutOfScopeException,
    PartialAccessServiceLineException
)
from src.services.azure_openai_service import AzureOpenAIService
from src.services.retrieval_service import RetreivalService
from src.services.session_service import SessionDBService
from src.utils.config import (
    OPENAI_DEPLOYMENT_NAME,
    SESSION_CONTEXT_WINDOW_SIZE,
    ENABLE_RELEVANCE_FILTERING,
    OUT_OF_SCOPE_CONFIDENCE_THRESHOLD
)
from src.utils.logger import logger, log_context
from src.utils.response_utils import replace_spaces_in_image_urls, get_keyword_aware_message
from src.utils.metrics import RetrievalMetrics, LatencyTracker, PipelineMetrics
import time
import random

# Import the new modular components
from src.chatbot.query_analyzer import QueryAnalyzer
from src.chatbot.relevance_judge import RelevanceJudge
from src.chatbot.authorization_checker import AuthorizationChecker
from src.chatbot.response_formatter import ResponseFormatter
from src.chatbot.context_manager import ContextManager


class ResponseGeneratorAgent:
    """
    Main orchestrator for generating chatbot responses using RAG pipeline.

    This agent coordinates:
    - Query analysis and classification
    - Document retrieval from vector store
    - Authorization validation
    - Response generation using LLM
    - Session management
    """

    def __init__(self, user_id, session_id, impersonated_user_id, model=OPENAI_DEPLOYMENT_NAME, max_tokens=16384):
        """
        Initialize the ResponseGeneratorAgent.

        Args:
            user_id (str): The user's ID
            session_id (str): The current session ID
            impersonated_user_id (str): ID of impersonated user if applicable
            model (str): The OpenAI model to use
            max_tokens (int): Maximum tokens for responses
        """
        self.model = model
        self.max_tokens = max_tokens
        self.user_id = user_id
        self.session_id = session_id
        self.impersonated_user_id = impersonated_user_id
        self.client = AzureOpenAIService().get_client()

        # Initialize modular components
        self.query_analyzer = QueryAnalyzer(model=model, max_tokens=max_tokens)
        self.relevance_judge = RelevanceJudge(model=model, max_tokens=max_tokens)
        self.authorization_checker = AuthorizationChecker()
        self.response_formatter = ResponseFormatter(model=model, max_tokens=max_tokens)
        self.context_manager = ContextManager()

    def get_filtered_context(self, query, container_name, service_line, query_embedding):
        """
        Retrieves and filters context chunks by service line.

        Args:
            query (str): The user query
            container_name (str): CosmosDB container name
            service_line (list): Authorized service line IDs
            query_embedding: Query embedding vector

        Returns:
            tuple: (filtered_context_dict, citations)
        """
        context, _, _, _, citation = RetreivalService().retrieve_general_info_chunks(
            query,
            container_name,
            service_line,
            query_embedding
        )

        chunk_filtered_context = {}

        for cnt_session in context:
            if cnt_session.get('serviceNameid') in list(chunk_filtered_context.keys()):
                chunk_filtered_context[cnt_session.get('serviceNameid')].append(cnt_session)
            else:
                chunk_filtered_context[cnt_session.get('serviceNameid')] = [cnt_session]

        return chunk_filtered_context, citation

    def get_response_from_agent(
        self,
        trace_id,
        prompt,
        user_chat_history,
        context,
        context_session,
        top_doc,
        chunk_service_line,
        retrieved_context,
        user_service_line,
        detected_language,
        is_generic,
        expanded_queries,
        citations
    ):
        """
        Generates a response from the LLM agent based on retrieved context.

        Args:
            trace_id (str): Trace ID for logging
            prompt (str): User query
            user_chat_history (list): Previous conversation messages
            context (dict): Retrieved context organized by service line
            context_session (dict): Session-specific context
            top_doc (dict): Top retrieved document metadata
            chunk_service_line (list): Service line IDs from chunks
            retrieved_context (list): All retrieved context chunks
            user_service_line (list): User's authorized service lines
            detected_language (str): Detected query language
            is_generic (bool): Whether query is generic
            expanded_queries (list): Expanded query variations
            citations (list): Citation information

        Returns:
            tuple: (structured_response dict, exception_type or None)
        """
        log_context.set(trace_id)
        start = time.time()
        current_date_str = time.strftime("%A, %B %d, %Y", time.localtime())

        # Format context as structured markdown for better LLM comprehension
        formatted_context = self._format_context_for_llm(context)

        # Format session context with clear structure
        formatted_session_context = json.dumps(context_session, indent=2, default=str)

        enriched_prompt = PromptTemplate.RESPONSE_TEMPLATE.value.format(
            Date=current_date_str,
            prompt=str(prompt),
            retrieved_data_source_1=formatted_context,
            retrieved_data_source_2=formatted_session_context,
            language=detected_language,
        )
        logger.info(f"enriched_prompt : {enriched_prompt}")

        # Call the Azure OpenAI API
        response = self.client.chat.completions.create(
            temperature=0,
            model=OPENAI_DEPLOYMENT_NAME,
            messages=user_chat_history + [{"role": "user", "content": enriched_prompt}],
            max_tokens=self.max_tokens
        )

        end = time.time()
        logger.info(f"[Latency] get_response_from_agent.response : {end - start:.2f}s")
        logger.info(f"response : {response}")
        message_content = response.choices[0].message.content
        response_object = {}

        try:
            # First, try to parse the content as JSON
            response_object = json.loads(message_content)
        except json.JSONDecodeError:
            # If it fails, the response is a plain string.
            # Manually create the object to match the expected structure.
            response_object = {
                "Answer": str(message_content),
                "Source": [],
                "Confidence": "0.0"  # Default confidence for plain text out-of-scope
            }

        if "Answer" in response_object:
            response_object["Answer"] = (
                response_object["Answer"]
                .replace("![Image](image_path)!", " ")
                .replace("![Image](image_path)", " ")
                .replace("https://my.sita.aero", "https://digihub.sita.aero")
                .replace("my.sita", "digihub.sita")
            )

        logger.info(f"Response generated successfully. {response_object}")
        citation = response_object.get("Source", None)

        try:
            OUT_OF_SCOPE_MESSAGES = {
                "english": [
                    "I'm here to support you with SITA documentation and services. For anything outside DigiHub, a quick web search might be your best bet. Let me know how I can assist further!",
                    "I'm designed to help with work-related questions about SITA and DigiHub services. For other topics, I recommend using your preferred search engine. Happy to help with anything work-related!",
                    "I specialize in SITA-related topics here on DigiHub. For other questions, a quick web search might be more helpful. Feel free to ask me anything work-related!",
                    "I'd love to help! My scope is focused on SITA documentation and services. For anything else, your favorite search engine might be the best place to look. Let me know what you need!"
                ],
                "german": [
                    "Ich bin hier, um Sie bei der SITA-Dokumentation und den SITA-Diensten zu unterstützen. Für alles außerhalb von DigiHub ist eine schnelle Websuche vielleicht die beste Lösung. Lassen Sie mich wissen, wie ich Ihnen weiterhelfen kann!",
                    "Ich bin dafür konzipiert, bei arbeitsbezogenen Fragen zu SITA- und DigiHub-Diensten zu helfen. Für andere Themen empfehle ich die Verwendung Ihrer bevorzugten Suchmaschine. Bei allem, was mit der Arbeit zu tun hat, helfe ich Ihnen gerne!",
                    "Ich bin hier auf DigiHub auf SITA-bezogene Themen spezialisiert. Bei anderen Fragen könnte eine schnelle Websuche hilfreicher sein. Zögern Sie nicht, mich alles zu fragen, was mit Ihrer Arbeit zu tun hat!",
                    "Ich würde Ihnen gerne helfen! Mein Schwerpunkt liegt auf der SITA-Dokumentation und den SITA-Diensten. Für alles andere ist Ihre bevorzugte Suchmaschine vielleicht die beste Anlaufstelle. Sagen Sie mir einfach, was Sie brauchen!"
                ],
                "french": [
                    "Je suis là pour vous aider avec la documentation et les services de SITA. Pour tout ce qui ne concerne pas DigiHub, une recherche rapide sur le web est sans doute la meilleure solution. N'hésitez pas à me dire comment je peux vous aider davantage !",
                    "Je suis conçu pour répondre aux questions professionnelles concernant les services SITA et DigiHub. Pour d'autres sujets, je vous recommande d'utiliser votre moteur de recherche préféré. Je serai ravi de vous aider pour toute question d'ordre professionnel !",
                    "Je suis spécialisé dans les sujets liés à SITA ici sur DigiHub. Pour d'autres questions, une recherche rapide sur le web pourrait être plus utile. N'hésitez pas à me poser des questions d'ordre professionnel !",
                    "J'adorerais vous aider ! Mon domaine de compétence se concentre sur la documentation et les services de SITA. Pour toute autre chose, votre moteur de recherche préféré est sans doute le meilleur endroit où chercher. Dites-moi ce dont vous avez besoin !"
                ],
                "spanish": [
                    "Estoy aquí para ayudarte con la documentación y los servicios de SITA. Para cualquier cosa fuera de DigiHub, una búsqueda rápida en la web podría ser tu mejor opción. ¡Dime cómo puedo ayudarte!",
                    "Estoy diseñado para ayudar con preguntas de trabajo sobre los servicios de SITA y DigiHub. Para otros temas, te recomiendo que uses tu motor de búsqueda preferido. ¡Estaré encantado de ayudarte con cualquier cosa relacionada con el trabajo!",
                    "Me especializo en temas relacionados con SITA aquí en DigiHub. Para otras preguntas, una búsqueda rápida en la web podría ser más útil. ¡No dudes en preguntarme cualquier cosa relacionada con el trabajo!",
                    "¡Me encantaría ayudarte! Mi especialidad es la documentación y los servicios de SITA. Para cualquier otra cosa, tu motor de búsqueda favorito podría ser el mejor lugar para buscar. ¡Dime qué necesitas!"
                ]
            }

            # Structured out-of-scope detection using multiple signals
            is_out_of_scope = self._detect_out_of_scope(response_object, prompt, detected_language)

            if is_out_of_scope:
                llm_answer = response_object.get("Answer", "")
                # Check if LLM provided a contextual response (mentions specific services/products)
                # These responses are more helpful than generic canned messages
                service_keywords = ["WorldTracer", "Billing", "Airport", "Bag Manager", "BagManager",
                                    "AeroPerformance", "Messaging", "CUTE", "document", "documentation"]
                is_contextual_response = any(kw.lower() in llm_answer.lower() for kw in service_keywords)

                if is_contextual_response and llm_answer:
                    # LLM provided a contextual response based on actual document search - use it
                    logger.info("Using LLM's contextual out-of-scope response instead of canned message")
                    final_response = self.response_formatter.parse_response(llm_answer)
                else:
                    # Use canned messages for truly generic out-of-scope queries
                    keyword_message = get_keyword_aware_message(prompt, detected_language)
                    if keyword_message:
                        final_response = keyword_message
                    else:
                        # Randomly select a message for the detected language
                        language_messages = OUT_OF_SCOPE_MESSAGES.get(detected_language.lower(), OUT_OF_SCOPE_MESSAGES["english"])
                        final_response = random.choice(language_messages)
                is_out_of_scope = True
            else:
                final_response = self.response_formatter.parse_response(response_object.get("Answer"))

            structured_response = {
                "response": replace_spaces_in_image_urls(final_response),
                "citation": citation,
                "confidence": response_object.get("Confidence", 0),
                "score": top_doc.get("question_score", 0)
            }

            translations = [
                "This question is outside the scope of the documents provided",
                "Cette question est en dehors du champ d'application des documents fournis.",
                "Esta pregunta está fuera del alcance de los documentos proporcionados",
                "Diese Frage liegt außerhalb des Umfangs der bereitgestellten Dokumente."
            ]
            if any(phrase in final_response for phrase in translations):
                # Randomly select a message for the detected language
                language_messages = OUT_OF_SCOPE_MESSAGES.get(detected_language.lower(), OUT_OF_SCOPE_MESSAGES["english"])
                final_response = random.choice(language_messages)

            # Flatten the list of all out-of-scope messages to check if the final response is one of them
            all_out_of_scope_phrases = [phrase for lang_phrases in OUT_OF_SCOPE_MESSAGES.values() for phrase in lang_phrases]

            # Use the authorization checker
            self.authorization_checker.cross_check_authorization(
                prompt=prompt,
                service_line=user_service_line,
                detected_language=detected_language,
                is_out_of_scope=is_out_of_scope,
                final_response=final_response
            )

            if any(phrase in final_response for phrase in all_out_of_scope_phrases):
                raise OutOfScopeException(final_response)

            return structured_response, None

        except OutOfScopeException as e:
            if structured_response:
                final_response = str(e)
                structured_response["response"] = str(e)
                structured_response["citation"] = []
            return structured_response, OutOfScopeException

        except UnAuthorizedServiceLineException as e:
            if structured_response:
                final_response = str(e)
                structured_response["response"] = str(e)
                structured_response["citation"] = []
            return structured_response, UnAuthorizedServiceLineException

        except PartialAccessServiceLineException as e:
            if structured_response:
                final_response = str(e.args[0])
                structured_response["response"] = str(e.args[0])
                structured_response["disclaimer"] = str(e.args[1])
            return structured_response, PartialAccessServiceLineException

        except BadRequestError as e:
            logger.error(f"Unexpected error while generating response: {e}")
            raise Exception(f"OpenAI API Error: {e}")

    def _retrieve_session_context(self):
        """
        Retrieves session history and chat context.

        Uses configurable SESSION_CONTEXT_WINDOW_SIZE (default: 5) for the number
        of Q&A pairs to include in context.

        Returns:
            tuple: (user_chat_history, session_context_window)
        """
        start = time.time()

        # Use configurable context window size (default: 5 Q&A pairs = 10 messages)
        context_window_size = SESSION_CONTEXT_WINDOW_SIZE if SESSION_CONTEXT_WINDOW_SIZE else 5
        messages_to_fetch = context_window_size * 2  # Each Q&A pair is 2 messages

        user_chat_history = SessionDBService().retrieve_session_details(
            user_id=self.user_id,
            session_id=self.session_id,
            limit=messages_to_fetch
        )
        end = time.time()
        logger.info(f"[Latency] SessionDBService.retrieve_session_details: {end - start:.2f}s")
        logger.info(f"Retrieved {len(user_chat_history)} messages (context window: {context_window_size} Q&A pairs)")

        # Use configured number of Q&A pairs for context window
        session_context_window = str(" ".join([e.get('content') for e in user_chat_history[-messages_to_fetch:]]))

        # Remove image markdown references to prevent them from polluting the context
        session_context_window = re.sub(r'!\[Image\]\([^)]+\)!?', '', session_context_window)

        return user_chat_history[-messages_to_fetch:], session_context_window

    def _analyze_query(self, prompt: str, session_context_window: str):
        """
        Analyzes and classifies the query.

        Args:
            prompt (str): User query
            session_context_window (str): Recent session messages

        Returns:
            dict: Query analysis result containing language, service lines, etc.
        """
        start = time.time()
        result = self.query_analyzer.query_classifer(prompt, session_context_window)
        end = time.time()
        logger.info(f"[Latency] detect_language: {end - start:.2f}s")
        logger.info(f"result detect_language: {result}")
        return result

    def _determine_service_lines(self, result: dict, service_line: list[dict] | None, prompt: str):
        """
        Determines which service lines to use for retrieval based on user authorization.

        Note: We always retrieve from ALL authorized service lines, not just the ones
        detected from keywords. Keyword-based detection is unreliable (e.g., "billing
        problem with my bag" might only detect "Billing", missing Bag Manager docs).
        The LLM relevance judge handles filtering after retrieval.

        Args:
            result (dict): Query analysis result
            service_line (list[dict] | None): User's authorized service lines
            prompt (str): User query

        Returns:
            tuple: (final_id_list, suppress_disclaimer)
        """
        all_mappings = RetreivalService().get_all_service_line()
        service_lines_requested = result.get("service_lines") or []
        suppress_disclaimer = False

        # Determine allowed service line IDs based on user authorization
        if isinstance(service_line, list):
            allowed_ids = [0] + [
                s['id'] for s in service_line
                if s.get('status') == 'SUBSCRIBED'
            ]
            suppress_disclaimer = False
        else:
            # No authorization info - allow all service lines (impersonation mode)
            allowed_ids = [item['id'] for item in all_mappings]
            suppress_disclaimer = False

        # Always use ALL authorized service lines for retrieval
        # This ensures we don't miss relevant documents due to imprecise keyword detection
        # The relevance judge (LLM) will filter chunks after retrieval
        final_id_list = [item['id'] for item in all_mappings if item.get('id') in allowed_ids]

        if not final_id_list and 0 in allowed_ids:
            final_id_list = [0]

        # Log detected service lines for debugging (but don't filter by them)
        if service_lines_requested:
            logger.info(f"Detected service lines from query: {service_lines_requested} (used for logging only)")
        final_id_list = [i for i in final_id_list if i is not None]
        logger.info(f"Final ID List (all authorized): {final_id_list}")
        final_id_list = list(set([int(i) for i in final_id_list] + [0]))

        return final_id_list, suppress_disclaimer
    
    def _handle_session_entities(self, is_session_dependent: bool):
        """
        Retrieves entities from session history for reference resolution.

        Args:
            is_session_dependent (bool): Whether query depends on session context

        Returns:
            dict: Session entities (services, topics, technical_terms)
        """
        session_entities = {"services": [], "topics": [], "technical_terms": []}
        if is_session_dependent:
            try:
                session_entities = SessionDBService().retrieve_session_entities(
                    user_id=self.user_id,
                    session_id=self.session_id,
                    limit=5
                )
                logger.info(f"Retrieved session entities: {session_entities}")
            except Exception as e:
                logger.error(f"Failed to retrieve session entities: {e}")
        return session_entities

    def _get_contextual_service_lines(self, is_session_dependent: bool, final_id_list: list) -> list:
        """
        Get service lines for retrieval.

        Note: We no longer restrict retrieval based on previous session's service lines.
        This was causing issues where follow-up questions about different topics would
        miss relevant documents. The relevance judge (LLM) handles filtering after
        retrieval, which is more reliable than pre-filtering.

        Args:
            is_session_dependent (bool): Whether the query depends on session context
            final_id_list (list): User's authorized service line IDs

        Returns:
            list: Service line IDs for retrieval (always returns full authorized list)
        """
        # Log previous service lines for debugging, but don't filter by them
        if is_session_dependent:
            try:
                previous_service_lines = SessionDBService().retrieve_session_service_lines(
                    user_id=self.user_id,
                    session_id=self.session_id,
                    limit=1
                )
                if previous_service_lines:
                    logger.info(f"Previous session service lines: {previous_service_lines} (for context, not filtering)")
            except Exception as e:
                logger.debug(f"Could not retrieve previous service lines: {e}")

        # Always return full authorized list - let relevance judge handle filtering
        return final_id_list

    def _resolve_query_references(self, translated_text: str, is_session_dependent: bool,
                                   session_entities: dict, user_chat_history: list):
        """
        Resolves references (pronouns, etc.) in the query.

        Reference resolution runs whenever the query contains pronouns or references
        (e.g., "it", "that", "this"), regardless of is_session_dependent status.
        This ensures queries like "Tell me more about it" get resolved even when
        service lines change (which sets is_session_dependent=False).

        Args:
            translated_text (str): Translated query text
            is_session_dependent (bool): Whether query depends on session (not used for resolution trigger)
            session_entities (dict): Session entities for resolution
            user_chat_history (list): Previous conversation messages

        Returns:
            str: Resolved query
        """
        resolved_query = translated_text
        has_references = self.context_manager.has_references(translated_text)

        if has_references:
            # Always attempt reference resolution when references are detected,
            # regardless of is_session_dependent flag
            try:
                resolved_query = self.context_manager.resolve_references(
                    query=translated_text,
                    entities=session_entities,
                    history=user_chat_history[-3:]
                )
                logger.info(f"Reference resolution: '{translated_text}' -> '{resolved_query}' "
                           f"(session_dependent={is_session_dependent})")
            except Exception as e:
                logger.error(f"Reference resolution failed, using original query: {e}")
                resolved_query = translated_text
        else:
            logger.debug(f"No references detected in query: '{translated_text}'")

        return resolved_query

    def _build_retrieval_query(self, resolved_query: str, is_session_dependent: bool,
                               session_entities: dict, user_chat_history: list,
                               expanded_queries: list, query_type: str):
        """
        Builds the query for retrieval based on session dependency.

        Args:
            resolved_query (str): Query after reference resolution
            is_session_dependent (bool): Whether query depends on session
            session_entities (dict): Session entities
            user_chat_history (list): Previous messages
            expanded_queries (list): Expanded query variations
            query_type (str): Type of query (e.g., 'Acronym')

        Returns:
            str: Final retrieval query
        """
        if query_type == 'Acronym':
            logger.info("The query is an Acronym type.")
            return resolved_query

        if is_session_dependent:
            try:
                retrieval_query = self.context_manager.build_smart_retrieval_query(
                    query=resolved_query,
                    history=user_chat_history,
                    is_dependent=True,
                    entities=session_entities
                )
                logger.info(f"Built smart retrieval query (length: {len(retrieval_query)})")
                return retrieval_query
            except Exception as e:
                logger.error(f"Smart query building failed, using fallback: {e}")
                user_messages = [msg for msg in user_chat_history if msg['role'] == 'user']
                user_text_history = " ".join([msg['content'] for msg in user_messages])
                return f"{user_text_history} {resolved_query}"
        else:
            return f"{resolved_query} {' '.join(expanded_queries)}"

    def _deduplicate_context(self, retrieved_context: list, chunk_filtered_context: dict):
        """
        Deduplicates and merges context from multiple sources.

        Args:
            retrieved_context (list): Initial retrieved chunks
            chunk_filtered_context (dict): Additional filtered chunks by service line

        Returns:
            dict: Deduplicated context organized by service line
        """
        # Combine all retrieved chunks
        all_retrieved_chunks = retrieved_context + [
            chunk for chunks in chunk_filtered_context.values() for chunk in chunks
        ]

        # Deduplicate based on content
        unique_chunks = []
        seen_content = set()
        for chunk in all_retrieved_chunks:
            content = chunk.get("content")
            if content not in seen_content:
                unique_chunks.append(chunk)
                seen_content.add(content)

        # Rebuild context dictionary from unique chunks
        final_context = {}
        for chunk in unique_chunks:
            service_id = chunk.get('serviceNameid')
            if service_id not in final_context:
                final_context[service_id] = []
            final_context[service_id].append(chunk)

        return final_context

    def _format_context_for_llm(self, retrieved_context: list) -> str:
        """
        Formats retrieved context chunks as structured markdown for better LLM comprehension.

        Each chunk is formatted with clear labels including:
        - Source file path
        - Section heading
        - Service name
        - Chunk content

        Args:
            retrieved_context (list): List of retrieved chunk dictionaries

        Returns:
            str: Formatted markdown string with numbered sections
        """
        if not retrieved_context:
            return "No relevant context found."

        formatted_sections = []

        for idx, chunk in enumerate(retrieved_context, 1):
            # Extract metadata
            file_path = chunk.get("citation", chunk.get("metadata", {}).get("filepath", "Unknown source"))
            heading = chunk.get("heading", "Untitled Section")
            service_name = chunk.get("serviceName", "General")
            content = chunk.get("content", "")

            # Calculate relevance indicators if available
            hybrid_score = chunk.get("hybrid_score", 0)
            question_sim = chunk.get("question_similarity", 0)

            # Build formatted section
            section = f"""### Context {idx}
**Source:** {file_path}
**Section:** {heading}
**Service:** {service_name}
**Relevance Score:** {hybrid_score:.2f}

{content}

---"""
            formatted_sections.append(section)

        formatted_context = "\n\n".join(formatted_sections)

        logger.info(f"Formatted {len(retrieved_context)} chunks into structured context")
        return formatted_context

    def _filter_relevant_chunks(self, query: str, chunks: list, detected_service_names: list = None) -> list:
        """
        Filters chunks using LLM-based relevance judgment.

        Uses RelevanceJudge to evaluate each chunk's relevance to the query
        and filters out non-relevant chunks while ensuring a minimum number
        of chunks are retained.

        Args:
            query (str): The user's query
            chunks (list): List of retrieved chunks to filter
            detected_service_names (list): Service names detected from user query (e.g., ['Bag Manager', 'WorldTracer'])

        Returns:
            list: Filtered list of relevant chunks
        """
        if detected_service_names is None:
            detected_service_names = []
        if not ENABLE_RELEVANCE_FILTERING:
            logger.info("Relevance filtering disabled, returning all chunks")
            return chunks

        if not chunks:
            return chunks

        start = time.time()
        try:
            # Get relevant service line IDs from the judge
            relevant_service_lines = self.relevance_judge.judge_chunks_relevance(query, chunks)

            # Filter chunks to only include relevant ones
            relevant_chunks = [
                chunk for chunk in chunks
                if chunk.get("serviceNameid") in relevant_service_lines
            ]

            # Log filtering results using metrics utility
            RetrievalMetrics.log_filtering_results(
                original_count=len(chunks),
                filtered_count=len(relevant_chunks),
                filter_type="llm_relevance"
            )

            # Additional filter: If user explicitly asked about specific services,
            # prefer chunks from those services over semantically similar but wrong services
            if relevant_chunks and detected_service_names:
                detected_names_lower = [name.lower().replace(" ", "") for name in detected_service_names]
                service_matched_chunks = [
                    chunk for chunk in relevant_chunks
                    if chunk.get('serviceName', '').lower().replace(" ", "") in detected_names_lower
                ]
                if service_matched_chunks:
                    logger.info(f"Service-filtered {len(relevant_chunks)} -> {len(service_matched_chunks)} chunks to match detected services: {detected_service_names}")
                    relevant_chunks = service_matched_chunks
                else:
                    # User asked about specific services but no relevant chunks match those services
                    # Clear relevant_chunks to trigger fallback which will search in original chunks
                    logger.info(f"No chunks from detected services {detected_service_names} found in relevant chunks - triggering fallback")
                    relevant_chunks = []

            # Fallback: If all chunks are filtered out, prefer chunks from detected services.
            # This prevents complete context loss when the relevance judge is too strict.
            if len(relevant_chunks) == 0 and len(chunks) > 0:
                logger.info("No relevant chunks found after LLM filtering - applying fallback")

                # Normalize detected service names for case-insensitive matching
                detected_names_lower = [name.lower().replace(" ", "") for name in detected_service_names]

                # Try to find chunks from detected service lines first
                if detected_service_names:
                    matching_chunks = [
                        chunk for chunk in chunks
                        if chunk.get('serviceName', '').lower().replace(" ", "") in detected_names_lower
                    ]
                    if matching_chunks:
                        # Sort matching chunks by similarity and take top 2
                        matching_chunks = sorted(
                            matching_chunks,
                            key=lambda x: x.get('hybrid_score', x.get('question_similarity', 0)),
                            reverse=True
                        )
                        relevant_chunks = matching_chunks[:2]
                        logger.info(f"Fallback: keeping top {len(relevant_chunks)} chunks from detected services: {detected_service_names}")

                # If no matching chunks found, fall back to top 2 by similarity
                if len(relevant_chunks) == 0:
                    sorted_chunks = sorted(
                        chunks,
                        key=lambda x: x.get('hybrid_score', x.get('question_similarity', 0)),
                        reverse=True
                    )
                    relevant_chunks = sorted_chunks[:2]
                    logger.info(f"Fallback: keeping top {len(relevant_chunks)} chunks by similarity (no service match)")

            end = time.time()
            logger.info(f"[Latency] relevance_filtering: {end - start:.2f}s")

            return relevant_chunks

        except Exception as e:
            logger.error(f"Relevance filtering failed, returning all chunks: {e}")
            return chunks

    def _detect_out_of_scope(self, response_object: dict, prompt: str, detected_language: str) -> bool:
        """
        Detects if a query is out of scope using a clear hierarchy of signals.

        Signal priority (first match wins):
        1. Explicit is_out_of_scope boolean from LLM (authoritative when present)
        2. Confidence score below threshold (reliable secondary signal)
        3. Text-based matching (legacy fallback only)

        Note: We deliberately avoid using multiple thresholds or combining signals
        in ways that could produce contradictory results.

        Args:
            response_object (dict): The parsed LLM response containing Answer, Confidence, is_out_of_scope
            prompt (str): The original user query
            detected_language (str): The detected language of the query

        Returns:
            bool: True if the query is determined to be out of scope
        """
        # Signal 1: Explicit is_out_of_scope boolean from LLM (authoritative)
        # When the LLM explicitly sets this field, trust it as the primary signal
        explicit_out_of_scope = response_object.get("is_out_of_scope")
        if explicit_out_of_scope is not None:
            # Handle both boolean and string values
            is_oos = False
            if isinstance(explicit_out_of_scope, bool):
                is_oos = explicit_out_of_scope
            elif isinstance(explicit_out_of_scope, str):
                is_oos = explicit_out_of_scope.lower() == "true"

            if is_oos:
                logger.info("Out-of-scope detected via explicit is_out_of_scope=true")
                return True
            else:
                # LLM explicitly said it's in scope - trust this decision
                logger.info("Query confirmed in scope via explicit is_out_of_scope=false")
                return False

        # Signal 2: Confidence threshold (only checked if is_out_of_scope not set)
        # This handles cases where older prompts don't return is_out_of_scope
        confidence_threshold = OUT_OF_SCOPE_CONFIDENCE_THRESHOLD if OUT_OF_SCOPE_CONFIDENCE_THRESHOLD else 0.4
        try:
            confidence = float(response_object.get("Confidence", 1.0))
            if confidence < confidence_threshold:
                logger.info(
                    f"Out-of-scope detected via low confidence: {confidence:.2f} < {confidence_threshold}"
                )
                return True
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not parse confidence score: {e}")

        # Signal 3: Text-based matching (legacy fallback for backwards compatibility)
        # Only used when is_out_of_scope is not set and confidence is above threshold
        answer = response_object.get("Answer", "")
        out_of_scope_phrases = [
            "This question is outside the scope",
            "outside the scope of the documents",
            "cannot be answered from the context",
            "I don't have information about",
            "not covered in the available documents",
            # German
            "Diese Frage liegt außerhalb des Umfangs",
            "außerhalb des Umfangs der bereitgestellten Dokumente",
            # French
            "Cette question est en dehors du champ d'application",
            "en dehors du champ d'application des documents",
            # Spanish
            "Esta pregunta está fuera del alcance",
            "fuera del alcance de los documentos"
        ]

        for phrase in out_of_scope_phrases:
            if phrase.lower() in answer.lower():
                logger.info(f"Out-of-scope detected via text matching: '{phrase}'")
                return True

        logger.info("Query determined to be in scope")
        return False

    def _process_citations(self, citation_data: list) -> list:
        """
        Processes citations by enriching with document IDs and drive names.

        Enriches each citation with:
        - id: Document ID from CosmosDB lookup
        - drivename: First part of the file path (drive/folder name)

        Args:
            citation_data (list): Raw citation data with 'File' keys

        Returns:
            list: Processed citation data with IDs and drive names added.
                  Caller should capture and use this return value.
        """
        # Extract list of files
        file_list = [item["File"] for item in citation_data if "File" in item]

        if file_list:
            # Get the IDs from service
            ids_found = RetreivalService().get_ids_from_file_paths(file_list)

            # Create lookup map
            file_id_map = {entry['pathwithfilename']: entry['id'] for entry in ids_found}

            # Update citations with ID and drivename
            for item in citation_data:
                file_path = item.get("File")
                if file_path:
                    item["id"] = file_id_map.get(file_path, None)
                    parts = file_path.split('/')
                    item["drivename"] = parts[0] if len(parts) > 0 else ""

        return citation_data

    def _create_error_response(self, message: str):
        """
        Creates a standardized error response structure.

        Args:
            message (str): Error message

        Returns:
            dict: Error response structure
        """
        return {
            "response": message,
            "citation": [],
            "confidence": 0,
            "score": 0,
            "disclaimer": "",
        }

    def _append_disclaimer(self, structured_response: dict, suppress_disclaimer: bool):
        """
        Appends disclaimer to response if applicable.

        Args:
            structured_response (dict): Response to modify
            suppress_disclaimer (bool): Whether to suppress disclaimer

        Returns:
            dict: Modified response with disclaimer appended
        """
        if suppress_disclaimer:
            structured_response["disclaimer"] = None
            logger.info(f"structured_response disclaimer None")
            return structured_response

        disclaimer = structured_response.get("disclaimer", "")
        if disclaimer:
            disclaimer_data = json.loads(disclaimer)
            message_text = disclaimer_data.get("message")
            if message_text:
                current_response = structured_response.get('response', '')
                structured_response["response"] = f"{current_response}<br><br><hr>{message_text}"

        return structured_response

    def generate_response(self, prompt: str, container_name: str, service_line: list[int] | None = None):
        """
        Main entry point for generating a response to a user query.

        This method orchestrates the entire RAG pipeline:
        1. Retrieves session history
        2. Analyzes and classifies the query
        3. Determines authorized service lines
        4. Retrieves relevant context from vector store
        5. Generates response using LLM
        6. Validates authorization
        7. Saves session

        Args:
            prompt (str): User query
            container_name (str): CosmosDB container name for retrieval
            service_line (list[int] | None): User's authorized service line IDs

        Returns:
            dict: Structured response containing:
                - response (str): The generated answer
                - citation (list): Source citations
                - confidence (float): Confidence score
                - score (float): Cosine similarity score
                - disclaimer (str): Authorization disclaimer if applicable
                - message_id (str): Session message ID
        """
        structured_response = None
        message_id = None
        chunk_service_line = []

        try:
            # Step 1: Retrieve session context
            user_chat_history, session_context_window = self._retrieve_session_context()

            # Step 2: Analyze query
            result = self._analyze_query(prompt, session_context_window)

            # Step 3: Determine service lines - Nazeel change method name to get Autorized Service line per user
            final_service_lines, suppress_disclaimer = self._determine_service_lines(result, service_line, prompt)

            # Step 4: Extract query metadata
            detected_language = result.get("language", "unknown")
            is_generic = result.get("is_generic", False)
            translated_text = result.get("translation", prompt)
            is_prompt_vulnerable = result.get("is_prompt_vulnerable")
            query_type = result.get("Query_classifier")
            acronyms = result.get("acronyms")
            expanded_queries = result.get("expanded_queries", [])
            is_session_dependent = result.get("is_session_dependent", False)
            detected_service_names = result.get("service_lines") or []

            # Filter out excluded acronyms
            exclude_words = {"SITA", "DIGIHUB"}
            if acronyms:
                acronyms = [a for a in acronyms if a.upper() not in exclude_words]
            logger.info(f"The query is an Acronym type. {query_type} {acronyms}")

            # Fetch acronym definitions from database if acronyms are detected
            acronym_definitions = []
            if query_type == 'Acronym' or acronyms:
                acronym_definitions = RetreivalService().get_acronym_definitions(acronyms)
                logger.info(f"Fetched acronym definitions: {acronym_definitions}")

            # Check for prompt injection
            if is_prompt_vulnerable:
                raise OutOfScopeException(
                    message="", final_response="", detected_language=detected_language, auto_msg=True
                )

            # Step 5: Handle session entities
            session_entities = self._handle_session_entities(is_session_dependent)

            # Step 6: Resolve references in query
            resolved_query = self._resolve_query_references(
                translated_text, is_session_dependent, session_entities, user_chat_history
            )

            # Step 6.5: Get contextual service lines for session-dependent queries
            retrieval_service_lines = self._get_contextual_service_lines(
                is_session_dependent, final_service_lines
            )

            # Step 6.6: Handle user guide service lines based on query type
            # General Info (0) and Operational Support (440) contain DigiHub user guides
            # that answer common questions about the platform.
            USER_GUIDE_SERVICE_LINES = [0, 440]
            authorized_user_guides = [sl for sl in USER_GUIDE_SERVICE_LINES if sl in final_service_lines]

            if not detected_service_names and not is_session_dependent:
                # GENERIC QUERY: No service line detected, not a follow-up
                # RESTRICT search to only user guide service lines to prevent
                # high-volume service lines (WorldTracer: 2,968 chunks) from drowning out
                # relevant user guide content (General Info: 10 chunks)
                if authorized_user_guides:
                    retrieval_service_lines = authorized_user_guides
                    logger.info(f"Generic query detected - restricting to user guide service lines: {authorized_user_guides}")
                else:
                    logger.info(f"Generic query but user not authorized for user guide service lines - using full list")
            elif authorized_user_guides:
                # SPECIFIC QUERY: Service line detected or session-dependent
                # ADD user guide service lines alongside detected service lines
                # This ensures user guides compete with service-specific content
                retrieval_service_lines = list(set(retrieval_service_lines + authorized_user_guides))
                logger.info(f"Including user guide service lines in search: {authorized_user_guides}")

            # Step 7: Retrieval from vector store using resolved query
            retrieved_context, _, top_doc, chunk_service_line, query_embedding, citations = (
                RetreivalService().rag_retriever_agent(resolved_query, container_name, retrieval_service_lines)
            )

            # Step 7.5: Filter chunks by relevance using LLM judge
            # Pass detected service names to help fallback prefer relevant chunks
            retrieved_context = self._filter_relevant_chunks(resolved_query, retrieved_context, detected_service_names)

            # Update top_doc after filtering
            if retrieved_context:
                top_doc = retrieved_context[0]

            # Step 8: Deduplicate citations
            citation_keys = []
            for cit in citations:
                if cit.get("File") not in citation_keys:
                    citation_keys.append(cit.get("File"))
            citations = [{"File": cit, "Section": ""} for cit in citation_keys]

            # Step 9: Prepare context for response generation
            populated_context_session = {
                "entities": session_entities,
                "resolved_query": resolved_query if is_session_dependent else prompt,
                "is_session_dependent": is_session_dependent,
                "previous_service_lines": list(set([
                    chunk.get('serviceNameid') for chunk in retrieved_context
                    if chunk.get('serviceNameid') is not None
                ])),
                "acronym_definitions": acronym_definitions
            }
            logger.info(
                f"Populated context_session with {len(session_entities.get('services', []))} services, "
                f"{len(session_entities.get('topics', []))} topics, "
                f"{len(acronym_definitions)} acronym definitions"
            )

            # Step 10: Generate response
            start = time.time()
            trace_id = log_context.get()
            structured_response, _ = self.get_response_from_agent(
                trace_id=trace_id,
                prompt=prompt,
                user_chat_history=user_chat_history,
                context=retrieved_context,
                context_session=populated_context_session,
                top_doc=top_doc,
                chunk_service_line=chunk_service_line,
                retrieved_context=retrieved_context,
                user_service_line=service_line,
                detected_language=detected_language,
                is_generic=is_generic,
                expanded_queries=expanded_queries,
                citations=citations
            )
            end = time.time()
            logger.info(f"[Latency] get_response_from_agent: {end - start:.2f}s")

            # Step 11: Process citations - explicitly capture and reassign result
            citation_data = structured_response.get("citation", [])
            processed_citations = self._process_citations(citation_data)
            structured_response["citation"] = processed_citations

            # Step 12: Handle disclaimer
            structured_response = self._append_disclaimer(structured_response, suppress_disclaimer)

        except UnAuthorizedServiceLineException as e:
            structured_response = self._create_error_response(str(e))
        except OutOfScopeException as e:
            structured_response = self._create_error_response(str(e))
        except Exception as e:
            logger.error(f"Unexpected Error: {e}")
            structured_response = self._create_error_response("An unexpected error occurred. Please try again.")
            logger.error(traceback.format_exc())

        finally:
            message_id = self.save_session_in_background(
                self.user_id, self.impersonated_user_id, prompt,
                structured_response.get("response"),
                self.session_id,
                structured_response.get("citation"),
                structured_response.get("score"),
                structured_response.get("confidence"),
                structured_response.get("disclaimer", ""),
                chunk_service_line=chunk_service_line
            )
            structured_response["message_id"] = message_id
            # logger.info(f"#$#- response : {structured_response.get("response")}")
        return structured_response

    def save_session_in_background(self, user_id, impersonated_user_id, prompt, final_response, session_id, citation, score, confidence, disclaimer, chunk_service_line=None):
        """
        Saves the current conversation turn to the session database with extracted entities.

        Args:
            user_id (str): User ID
            impersonated_user_id (str): Impersonated user ID if applicable
            prompt (str): User query
            final_response (str): Generated response
            session_id (str): Session ID
            citation (list): Citation information
            score (float): Cosine similarity score
            confidence (float): Confidence score
            disclaimer (str): Disclaimer message
            chunk_service_line (list): Service line IDs from retrieved chunks

        Returns:
            str: Message ID from the database
        """
        # Extract entities from the Q&A pair before saving
        extracted_entities_dict = {"services": [], "topics": [], "technical_terms": []}
        try:
            extracted_entities_dict = self.context_manager.extract_entities(
                query=prompt,
                response=final_response
            )
            logger.info(f"Extracted entities for session: {extracted_entities_dict}")
        except Exception as e:
            logger.error(f"Entity extraction failed during session save: {e}")

        # Flatten entities for storage
        flat_entities = self.context_manager.get_session_entities_flat(extracted_entities_dict)

        message_id = SessionDBService().add_user_assistant_session(
            user_id=user_id,
            impersonated_user_id=impersonated_user_id,
            user_content=prompt,
            assistant_content=final_response,
            session_id=session_id,
            citation=citation,
            score=score,
            confidence=confidence,
            disclaimer=disclaimer,
            entities=flat_entities,
            chunk_service_line=chunk_service_line or []
        )
        return message_id
