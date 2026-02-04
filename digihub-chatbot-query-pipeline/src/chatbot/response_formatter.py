"""
Response Formatter Module

Handles parsing and formatting of LLM responses, including output cleaning,
response rephrasing, and text transformations.
"""

import json
import time
from openai import AzureOpenAI
from src.enums.prompt_template import PromptTemplate
from src.services.azure_openai_service import AzureOpenAIService
from src.utils.config import OPENAI_DEPLOYMENT_NAME
from src.utils.logger import logger


class ResponseFormatter:
    """
    Parses and formats responses from the LLM for presentation to users.
    """

    def __init__(self, model=OPENAI_DEPLOYMENT_NAME, max_tokens=16384):
        """
        Initialize the ResponseFormatter.

        Args:
            model (str): The OpenAI model to use for formatting
            max_tokens (int): Maximum tokens for API responses
        """
        self.model = model
        self.max_tokens = max_tokens
        self.client = AzureOpenAIService().get_client()

    def parse_response(self, response: str) -> str:
        """
        Parses and cleans the response from the LLM.

        This method takes a raw response and uses the LLM to parse it into a clean,
        well-formatted output suitable for presentation to users.

        Args:
            response (str): The raw response text from the LLM

        Returns:
            str: The parsed and formatted response
        """
        current_date_str = time.strftime("%A, %B %d, %Y", time.localtime())
        prompt = PromptTemplate.OUTPUT_PARSING_TEMPLATE.value.format(
            message=response,
        )

        # Call the Azure OpenAI API
        response = self.client.chat.completions.create(
            model=OPENAI_DEPLOYMENT_NAME,
            messages=[{"role": "system", "content": prompt}],
            max_tokens=self.max_tokens
        )
        response = response.choices[0].message.content
        logger.info(f"Parsed Response: {response}")
        return response

    def get_important_information(self, prompt: str, context: str, language: str) -> dict:
        """
        Rephrases a response from multiple service lines into a structured format.

        This function takes a user prompt, contextual information, and a target language,
        then formats them into a predefined prompt template. It sends the formatted prompt
        to the Azure OpenAI API to generate a rephrased and structured response, which is
        expected to be in JSON format. The function is primarily used to extract and
        rephrase important information from responses across different service lines
        for consistency and clarity.

        Args:
            prompt (str): The original user query or input.
            context (str): Additional context or background information relevant to the prompt.
            language (str): The language in which the response should be rephrased.

        Returns:
            dict: A dictionary containing the rephrased response. If an error occurs,
                  returns a fallback dictionary with language set to "unknown".
        """
        start = time.time()
        formatted_prompt = PromptTemplate.RESPONSE_REPHRASE_AGENT.value.format(
            prompt=prompt,
            context=context,
            language=language
        )
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_DEPLOYMENT_NAME,
                messages=[{"role": "system", "content": formatted_prompt}],
                max_tokens=self.max_tokens,
            )
            end = time.time()
            logger.info(f"[Latency] get_important_information.response: {end - start:.2f}s")
            parsed_response = json.loads(response.choices[0].message.content)
            logger.info(f"Rephrased Response: {parsed_response}")
            return parsed_response
        except Exception as e:
            logger.error(f"Error during response rephrasing: {e}")
            return {"language": "unknown", "translation": language}
