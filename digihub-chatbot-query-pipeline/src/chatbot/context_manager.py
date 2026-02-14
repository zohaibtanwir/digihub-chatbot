"""
Context Manager for Chatbot Contextual Continuity

This module provides functionality for:
- Entity extraction from Q&A pairs
- Reference resolution (pronouns to entities)
- Smart context window building
- Smart retrieval query construction
"""

import re
import json
import os
from typing import Dict, List, Optional, Tuple
from src.services.azure_openai_service import AzureOpenAIService
from src.enums.prompt_template import PromptTemplate
from src.utils.config import OPENAI_DEPLOYMENT_NAME
from src.utils.logger import logger


class ContextManager:
    """
    Manages contextual continuity for chatbot conversations.
    Handles entity tracking, reference resolution, and smart query merging.
    """

    # Pronoun patterns for reference detection
    PRONOUN_PATTERNS = r'\b(it|that|them|those|this|the service|the product|the system|the feature|the solution)\b'

    # Incomplete/implicit query patterns - queries with actions but missing objects
    # These indicate the user is referring to something from context
    INCOMPLETE_QUERY_PATTERNS = r'(?i)\b(show\s+me|tell\s+me|explain|help\s+me|give\s+me|send\s+me|get\s+me|find|display|view|see|download|access|open)\s*[\?\.\!]?\s*$'

    # Follow-up continuation patterns - these need reference resolution to identify the topic
    CONTINUATION_PATTERNS = r'(?i)^(tell\s+me\s+more|more\s+details?|explain\s+(more|further)|go\s+on|continue|elaborate|can\s+you\s+elaborate|what\s+else|more\s+info|lets?\s+explore|explore\s+more|dig\s+deeper|further\s+details?|can\s+you\s+explain|please\s+explain|expand\s+on|clarify|what\s+do\s+you\s+mean|how\s+so|why\s+is\s+that|give\s+me\s+more|and\??)\s*[\?\.\!]?\s*$'

    def __init__(self):
        """Initialize the ContextManager with Azure OpenAI client."""
        self.client = AzureOpenAIService().get_client()
        self.model = OPENAI_DEPLOYMENT_NAME

        # Feature flags from environment variables
        self.entity_tracking_enabled = os.getenv("ENABLE_ENTITY_TRACKING", "true").lower() == "true"
        self.reference_resolution_enabled = os.getenv("ENABLE_REFERENCE_RESOLUTION", "true").lower() == "true"
        self.smart_query_merging_enabled = os.getenv("ENABLE_SMART_QUERY_MERGING", "true").lower() == "true"

    def extract_entities(self, query: str, response: str) -> Dict[str, List[str]]:
        """
        Extract key entities from a Q&A pair using LLM.

        Args:
            query: User's question
            response: Assistant's response

        Returns:
            Dictionary with extracted entities:
            {
                "services": ["WorldTracer", "Bag Manager"],
                "topics": ["lost baggage", "billing"],
                "technical_terms": ["Type B messages", "LNI code"]
            }
        """
        if not self.entity_tracking_enabled:
            logger.info("Entity tracking is disabled via feature flag")
            return {"services": [], "topics": [], "technical_terms": []}

        try:
            prompt = PromptTemplate.ENTITY_EXTRACTION_TEMPLATE.value.format(
                query=query,
                response=response
            )

            llm_response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": "You are an entity extraction assistant. Extract entities and return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )

            content = llm_response.choices[0].message.content
            entities = json.loads(content)

            # Validate structure
            if not all(key in entities for key in ["services", "topics", "technical_terms"]):
                logger.warning("Entity extraction returned invalid structure, using defaults")
                return {"services": [], "topics": [], "technical_terms": []}

            logger.info(f"Extracted entities: {entities}")
            return entities

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}", exc_info=True)
            return {"services": [], "topics": [], "technical_terms": []}

    def build_context_window(self, session_history: List[Dict], window_size: int = 5) -> str:
        """
        Build a recency-weighted context window from session history.

        Args:
            session_history: List of message dicts with 'role' and 'content'
            window_size: Number of recent messages to include (default: 5)

        Returns:
            Formatted context string with recent messages
        """
        if not session_history:
            return ""

        # Take last N messages
        recent_messages = session_history[-window_size:]

        # Build formatted context
        context_parts = []
        for msg in recent_messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            context_parts.append(f"{role.capitalize()}: {content}")

        return "\n".join(context_parts)

    def has_references(self, query: str) -> bool:
        """
        Detect if a query contains pronouns or references that need resolution.

        This method detects two types of references:
        1. Explicit pronouns (it, that, this, etc.)
        2. Incomplete/implicit queries (e.g., "Can you show me?" without specifying what)

        Args:
            query: User's query text

        Returns:
            True if pronouns/references detected, False otherwise
        """
        if not query:
            return False

        # Check for pronoun patterns (case-insensitive)
        pronoun_match = re.search(self.PRONOUN_PATTERNS, query, re.IGNORECASE)

        if pronoun_match:
            logger.info(f"Detected pronoun reference in query: '{pronoun_match.group()}' in '{query}'")
            return True

        # Check for incomplete/implicit queries (e.g., "Can you show me?", "Please help me")
        incomplete_match = re.search(self.INCOMPLETE_QUERY_PATTERNS, query)

        if incomplete_match:
            logger.info(f"Detected incomplete/implicit query: '{incomplete_match.group()}' in '{query}'")
            return True

        # Check for continuation patterns (e.g., "tell me more", "explain further")
        continuation_match = re.search(self.CONTINUATION_PATTERNS, query)

        if continuation_match:
            logger.info(f"Detected continuation pattern in query: '{continuation_match.group()}' in '{query}'")
            return True

        return False

    def resolve_references(
        self,
        query: str,
        entities: Dict[str, List[str]],
        history: List[Dict]
    ) -> str:
        """
        Resolve pronouns and references in query to explicit entities using LLM.

        Args:
            query: Current query with potential pronouns
            entities: Dictionary of entities discussed (services, topics, technical_terms)
            history: Recent conversation history (last 2-3 turns)

        Returns:
            Query with pronouns replaced by explicit entities, or original query if resolution fails
        """
        if not self.reference_resolution_enabled:
            logger.info("Reference resolution is disabled via feature flag")
            return query

        if not self.has_references(query):
            logger.info("No references detected in query")
            return query

        try:
            # Build history context
            history_text = self.build_context_window(history, window_size=3)

            # Flatten entities for prompt
            services = ", ".join(entities.get("services", []))
            topics = ", ".join(entities.get("topics", []))
            technical_terms = ", ".join(entities.get("technical_terms", []))

            prompt = PromptTemplate.REFERENCE_RESOLUTION_TEMPLATE.value.format(
                query=query,
                history=history_text,
                services=services or "None",
                topics=topics or "None",
                technical_terms=technical_terms or "None"
            )

            llm_response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": "You are a reference resolution assistant. Resolve pronouns to entities and return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )

            content = llm_response.choices[0].message.content
            result = json.loads(content)

            resolved_query = result.get("resolved_query", query)
            entities_referenced = result.get("entities_referenced", [])

            logger.info(f"Reference resolution: '{query}' -> '{resolved_query}' (entities: {entities_referenced})")
            return resolved_query

        except Exception as e:
            logger.error(f"Reference resolution failed: {e}", exc_info=True)
            return query  # Fallback to original query

    def build_smart_retrieval_query(
        self,
        query: str,
        history: List[Dict],
        is_dependent: bool,
        entities: Dict[str, List[str]]
    ) -> str:
        """
        Build a smart retrieval query with recency weighting.

        For session-dependent queries:
        - Merge last 2-3 user messages (not ALL messages)
        - Apply recency weighting (recent = 1.0, older = 0.5)
        - Include resolved entities explicitly

        For independent queries:
        - Return the current query as-is

        Args:
            query: Current user query (may already be resolved)
            history: List of previous user messages
            is_dependent: Whether query is session-dependent
            entities: Extracted entities for explicit inclusion

        Returns:
            Smart retrieval query string
        """
        if not self.smart_query_merging_enabled:
            logger.info("Smart query merging is disabled via feature flag")
            # Fallback to simple concatenation
            if is_dependent and history:
                user_messages = [msg for msg in history if msg.get('role') == 'user']
                user_text_history = " ".join([msg.get('content', '') for msg in user_messages])
                return f"{user_text_history} {query}"
            return query

        if not is_dependent:
            logger.info("Query is not session-dependent, using as-is")
            return query

        if not history:
            logger.info("No history available, using current query")
            return query

        try:
            # Extract user messages only
            user_messages = [msg for msg in history if msg.get('role') == 'user']

            # Take last 3 user messages max
            recent_user_messages = user_messages[-3:]

            # Build weighted query
            query_parts = []

            # Add recent messages with labels
            if len(recent_user_messages) > 1:
                for idx, msg in enumerate(recent_user_messages[:-1]):
                    content = msg.get('content', '')
                    query_parts.append(f"Previous context: {content}")

            # Add current query with highest weight
            query_parts.append(f"Current query: {query}")

            # Add explicit entity mentions if available
            all_entities = []
            for entity_type, entity_list in entities.items():
                all_entities.extend(entity_list)

            if all_entities:
                # Add top 3 most relevant entities
                entity_mentions = " ".join(all_entities[:3])
                query_parts.append(f"Related to: {entity_mentions}")

            smart_query = " ".join(query_parts)
            logger.info(f"Built smart retrieval query: {smart_query[:200]}...")
            return smart_query

        except Exception as e:
            logger.error(f"Smart query building failed: {e}", exc_info=True)
            # Fallback to simple concatenation
            user_text_history = " ".join([msg.get('content', '') for msg in user_messages])
            return f"{user_text_history} {query}"

    def get_session_entities_flat(self, entities: Dict[str, List[str]]) -> List[str]:
        """
        Flatten entities dictionary to a list for storage.

        Args:
            entities: Dictionary with services, topics, technical_terms

        Returns:
            Flat list of all entities
        """
        flat_list = []
        for entity_type, entity_list in entities.items():
            flat_list.extend(entity_list)
        return flat_list

    def group_entities_from_flat(self, flat_entities: List[str]) -> Dict[str, List[str]]:
        """
        Group flat entity list back into categorized dictionary.

        Note: This is a best-effort reconstruction. The original categorization
        is lost when entities are flattened, so this uses heuristics.

        Args:
            flat_entities: Flat list of entity strings

        Returns:
            Dictionary with services, topics, technical_terms
        """
        # Known SITA service names from documentation
        known_services = [
            "WorldTracer", "World Tracer", "Bag Manager", "Airport Solutions",
            "Community Messaging", "SITA AeroPerformance", "AeroPerformance",
            "Billing", "Airport Committee", "APAC Customer Advisory Board",
            "Euro Customer Advisory Board", "Operational Support", "General Info"
        ]

        services = []
        topics = []
        technical_terms = []

        for entity in flat_entities:
            # Check if it's a known service (case-insensitive)
            if any(service.lower() in entity.lower() for service in known_services):
                services.append(entity)
            # Check if it looks like a technical term (contains special chars or all caps)
            elif re.search(r'[A-Z]{2,}|[_-]', entity):
                technical_terms.append(entity)
            else:
                topics.append(entity)

        return {
            "services": services,
            "topics": topics,
            "technical_terms": technical_terms
        }
