"""
Relevance Judge Module

Evaluates the relevance of retrieved chunks to user queries using LLM-based judgment.
"""

import json
from openai import AzureOpenAI
from src.enums.prompt_template import PromptTemplate
from src.services.azure_openai_service import AzureOpenAIService
from src.utils.config import OPENAI_DEPLOYMENT_NAME
from src.utils.logger import logger


class RelevanceJudge:
    """
    Uses LLM to judge if retrieved chunks are relevant to user queries.
    """

    def __init__(self, model=OPENAI_DEPLOYMENT_NAME, max_tokens=16384):
        """
        Initialize the RelevanceJudge.

        Args:
            model (str): The OpenAI model to use for relevance judgment
            max_tokens (int): Maximum tokens for API responses
        """
        self.model = model
        self.max_tokens = max_tokens
        self.client = AzureOpenAIService().get_client()

    def judge_chunks_relevance(self, prompt: str, chunks: list[dict]) -> list[int]:
        """
        Uses an LLM to judge if a list of chunks is relevant to the user's query in a single call.
        Returns a list of service line IDs for the chunks that are deemed relevant.

        Args:
            prompt (str): The user's query
            chunks (list[dict]): List of chunks to evaluate, each containing 'content' and 'serviceNameid'

        Returns:
            list[int]: List of service line IDs for chunks deemed relevant to the query

        Example:
            >>> chunks = [
            ...     {"content": "WorldTracer is a baggage tracing system", "serviceNameid": 11},
            ...     {"content": "Airport solutions help with operations", "serviceNameid": 2}
            ... ]
            >>> judge_chunks_relevance("What is WorldTracer?", chunks)
            [11]
        """
        if not chunks:
            return []

        # Prepare the chunks for the prompt by formatting them as a JSON string.
        # We only need the content and the serviceNameid for the judgment.
        chunks_for_prompt = [
            {"content": chunk.get("content", ""), "serviceNameid": chunk.get("serviceNameid")}
            for chunk in chunks
        ]

        try:
            judge_prompt = PromptTemplate.RELEVANCE_JUDGE_TEMPLATE_BULK.value.format(
                prompt=prompt,
                chunks_json=json.dumps(chunks_for_prompt, indent=2)
            )

            response = self.client.chat.completions.create(
                model=OPENAI_DEPLOYMENT_NAME,
                messages=[{"role": "system", "content": judge_prompt}],
                temperature=0,
                max_tokens=1024,  # Adjust token limit as needed for larger chunk lists
                response_format={"type": "json_object"}  # Use JSON mode for reliability
            )

            parsed_response = json.loads(response.choices[0].message.content)

            # Extract the service line IDs from the relevant chunks identified by the LLM
            relevant_chunks = parsed_response.get("relevant_chunks", [])
            relevant_service_lines = [chunk['serviceNameid'] for chunk in relevant_chunks if 'serviceNameid' in chunk]

            logger.info(f"Judged {len(chunks)} chunks. Found {len(relevant_service_lines)} relevant service lines: {relevant_service_lines}")
            return relevant_service_lines
        except Exception as e:
            logger.error(f"Error judging relevance for a batch of chunks: {e}. Defaulting to no relevant chunks.")
            return []  # Default to an empty list in case of any error
