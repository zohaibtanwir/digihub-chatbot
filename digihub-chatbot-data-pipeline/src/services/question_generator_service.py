"""
Question Generator Service for the Data Pipeline.

Generates synthetic questions for document chunks to enable
question-first retrieval in the RAG system.
"""

import json
from typing import List, Dict, Tuple
from src.services.azure_openai_service import AzureOpenAIService
from src.services.embedding_service import AzureEmbeddingService
from src.utils.config import OPENAI_DEPLOYMENT_NAME
from src.utils.logger import logger


class QuestionGeneratorService:
    """
    Service for generating synthetic questions from document chunks.

    This service uses Azure OpenAI to generate questions that a chunk
    could answer, which are then embedded to create questionsEmbedding
    for improved question-first retrieval.
    """

    # Minimum content length for valid chunks (in characters)
    MIN_CONTENT_LENGTH = 100

    # Minimum non-heading lines for valid chunks
    MIN_NON_HEADING_LINES = 2

    def __init__(self):
        """Initialize the question generator service."""
        self.openai_service = AzureOpenAIService()
        self.embedding_service = AzureEmbeddingService()
        self.client = self.openai_service.get_client()
        # Use gpt-4o-mini for cost-effective question generation
        self.model = OPENAI_DEPLOYMENT_NAME if OPENAI_DEPLOYMENT_NAME else "gpt-4o-mini"

    def validate_chunk_quality(self, content: str, heading: str = "") -> Tuple[bool, str]:
        """
        Validate chunk quality to determine if it should be indexed.

        Args:
            content: The chunk content to validate.
            heading: The chunk heading (optional).

        Returns:
            Tuple of (is_valid, reason).
        """
        if not content or not content.strip():
            return False, "Empty content"

        # Check minimum content length
        if len(content.strip()) < self.MIN_CONTENT_LENGTH:
            return False, f"Content too short ({len(content.strip())} chars < {self.MIN_CONTENT_LENGTH})"

        # Check if content is more than just headings
        lines = content.strip().split('\n')
        non_heading_lines = [
            line for line in lines
            if line.strip() and not line.strip().startswith('#')
        ]

        if len(non_heading_lines) < self.MIN_NON_HEADING_LINES:
            return False, f"Content contains only headings ({len(non_heading_lines)} non-heading lines)"

        # Check for placeholder or meaningless content
        placeholder_indicators = [
            "lorem ipsum",
            "placeholder",
            "todo:",
            "tbd",
            "[insert",
            "coming soon"
        ]
        content_lower = content.lower()
        for indicator in placeholder_indicators:
            if indicator in content_lower and len(content) < 500:
                return False, f"Content appears to be placeholder text"

        return True, "valid"

    def generate_questions(self, content: str, heading: str = "", num_questions: int = 5) -> List[str]:
        """
        Generate synthetic questions that a chunk could answer.

        Args:
            content: The chunk content.
            heading: The chunk heading for context.
            num_questions: Number of questions to generate (default: 5).

        Returns:
            List of generated questions.
        """
        # Truncate content if too long to avoid token limits
        max_content_length = 3000
        truncated_content = content[:max_content_length] if len(content) > max_content_length else content

        prompt = f"""You are an expert at generating search queries. Given the following document chunk, generate {num_questions} diverse questions that this content could answer.

Document Heading: {heading if heading else "N/A"}

Document Content:
---
{truncated_content}
---

Instructions:
1. Generate questions that someone might ask when searching for this information
2. Include a mix of:
   - Direct factual questions (What is...? How does...?)
   - Procedural questions (How to...? What are the steps to...?)
   - Contextual questions (When should I...? Why does...?)
3. Questions should be specific enough to match this content
4. Avoid overly generic questions
5. Each question should be on a single line

Return ONLY a valid JSON array of question strings, nothing else.
Example format: ["Question 1?", "Question 2?", "Question 3?"]"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a question generation expert. Generate relevant questions that the provided content could answer. Always return valid JSON array of strings."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )

            result_text = response.choices[0].message.content.strip()

            # Clean up response - remove markdown code blocks if present
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.startswith("```"):
                result_text = result_text[3:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]

            questions = json.loads(result_text.strip())

            # Validate response
            if isinstance(questions, list) and all(isinstance(q, str) for q in questions):
                logger.info(f"Generated {len(questions)} questions for chunk")
                return questions[:num_questions]
            else:
                logger.warning("Invalid question format received, using fallback")
                return self._generate_fallback_questions(heading, content)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse questions JSON: {e}")
            return self._generate_fallback_questions(heading, content)
        except Exception as e:
            logger.error(f"Error generating questions: {e}")
            return self._generate_fallback_questions(heading, content)

    def _generate_fallback_questions(self, heading: str, content: str) -> List[str]:
        """
        Generate simple fallback questions when LLM fails.

        Args:
            heading: The chunk heading.
            content: The chunk content.

        Returns:
            List of fallback questions.
        """
        questions = []

        if heading and heading != "Untitled":
            questions.append(f"What is {heading}?")
            questions.append(f"How does {heading} work?")
            questions.append(f"Tell me about {heading}")

        # Extract first sentence for context
        first_line = content.split('\n')[0].strip()
        if first_line and not first_line.startswith('#'):
            questions.append(f"What information is available about {first_line[:50]}...?")

        # Add generic questions if we don't have enough
        if len(questions) < 3:
            questions.extend([
                "What does this document explain?",
                "What are the key points in this section?"
            ])

        return questions[:5]

    def generate_questions_embedding(self, questions: List[str]) -> List[float]:
        """
        Generate embedding vector from a list of questions.

        Concatenates all questions and generates a single embedding
        for question-first retrieval.

        Args:
            questions: List of questions to embed.

        Returns:
            Embedding vector for the concatenated questions.
        """
        if not questions:
            return []

        # Concatenate questions with newlines for better semantic representation
        questions_text = "\n".join(questions)

        try:
            embedding = self.embedding_service.embed_text(questions_text)
            logger.info(f"Generated questions embedding from {len(questions)} questions")
            return embedding
        except Exception as e:
            logger.error(f"Error generating questions embedding: {e}")
            return []

    def process_chunk(self, content: str, heading: str = "") -> Dict:
        """
        Process a chunk to generate questions, embeddings, and validation.

        This is the main entry point for chunk processing that combines
        all the functionality of this service.

        Args:
            content: The chunk content.
            heading: The chunk heading.

        Returns:
            Dictionary containing:
            - validChunk: "yes" or "no"
            - validationReason: Reason for validation result
            - questions: List of generated questions
            - questionsEmbedding: Embedding vector for questions
        """
        # Validate chunk quality
        is_valid, reason = self.validate_chunk_quality(content, heading)

        result = {
            "validChunk": "yes" if is_valid else "no",
            "validationReason": reason,
            "questions": [],
            "questionsEmbedding": []
        }

        if is_valid:
            # Generate questions for valid chunks
            questions = self.generate_questions(content, heading)
            result["questions"] = questions

            # Generate questions embedding
            if questions:
                questions_embedding = self.generate_questions_embedding(questions)
                result["questionsEmbedding"] = questions_embedding

        return result
