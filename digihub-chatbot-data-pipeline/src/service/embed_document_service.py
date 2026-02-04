from typing import List
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import tiktoken
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from src.enums.chunking_params import ChunkingParams
from src.utils.logger import logger
from src.utils.nested_tables_utils import MarkdownTableExtractor


class Chunk:
    """Data class for representing a document chunk with metadata."""

    def __init__(self, content: str, heading_context: dict, token_count: int):
        self.content = content
        self.heading_context = heading_context
        self.token_count = token_count


class EmbedDocumentService:
    """
    Service for intelligent document chunking with token awareness and table conversion.

    Two-stage chunking algorithm:
    1. Stage 1: Split on markdown headers (H1, H2, H3) to preserve document structure
    2. Stage 2: Verify token counts using tiktoken; recursive split if exceeds limit

    Features:
    - Accurate token counting prevents embedding dimension mismatches
    - Preserves document structure through header hierarchy
    - Converts unreadable markdown tables to natural language using LLM
    - Creates dual entries: original chunk + expanded table version
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        """
        Initializes token encoding based on the model configuration.

        Args:
            model_name: Name of the model for token encoding (default: gpt-4o-mini)
        """
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Fallback to cl100k_base encoding if model not found
            logger.warning(f"Model {model_name} not found, using cl100k_base encoding")
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """
        Counts the number of tokens in the given text using the model's encoding.

        Args:
            text: The input text to tokenize

        Returns:
            Number of tokens in the text
        """
        if not text:
            return 0
        return len(self.encoding.encode(text))

    def _add_section_headers(self, current_chunk_headers: dict, new_chunk_headers: dict) -> str:
        """
        Generates markdown-style header context based on changes between current and new headers.

        Args:
            current_chunk_headers: Headers from the current chunk
            new_chunk_headers: Headers from the new chunk

        Returns:
            String containing updated markdown headers
        """
        header_section_context = ""
        for i in range(1, 4):
            header_key = f"Header {i}"
            if (header_key in new_chunk_headers and
                    header_key in current_chunk_headers and
                    current_chunk_headers[header_key] == new_chunk_headers[header_key]):
                continue
            elif header_key in new_chunk_headers:
                header_section_context += '#' * i + f" {new_chunk_headers[header_key]}\n"
        return header_section_context

    def get_document_chunks(self, input_text: str, chunk_size: int = ChunkingParams.chunk_size.value) -> List[str]:
        """
        Splits the input document into chunks based on markdown headers and token limits.

        Two-stage algorithm:
        1. Split on markdown headers (H1, H2, H3) to preserve structure
        2. Verify token counts; recursively split chunks exceeding the limit

        Args:
            input_text: The full document text to split
            chunk_size: Maximum token size per chunk (default from ChunkingParams)

        Returns:
            List of processed document chunks as strings
        """
        if not input_text:
            return []

        # Stage 1: Split on markdown headers
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

        header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        header_docs = header_splitter.split_text(input_text)

        # Build chunks while tracking token counts
        chunked_docs: List[Chunk] = []
        current_chunk_content = ""
        header_section_context = ""
        current_chunk_headers = {}
        current_tokens = 0

        for doc in header_docs:
            doc_tokens = self.count_tokens(doc.page_content)
            new_chunk_headers = doc.metadata.copy()

            # If adding this doc would exceed chunk size, save current chunk and start new one
            if current_tokens + doc_tokens > chunk_size and current_chunk_content:
                chunked_docs.append(Chunk(
                    content=current_chunk_content.strip(),
                    heading_context=current_chunk_headers,
                    token_count=current_tokens
                ))
                header_section_context = self._add_section_headers({}, new_chunk_headers)
                current_chunk_content = header_section_context + doc.page_content
                current_tokens = doc_tokens

            else:
                # Add to current chunk
                header_section_context = self._add_section_headers(current_chunk_headers, new_chunk_headers)

                if current_chunk_content:
                    current_chunk_content += "\n\n" + header_section_context + doc.page_content
                elif doc.page_content.strip():
                    current_chunk_content = header_section_context + doc.page_content
                current_tokens += doc_tokens

            current_chunk_headers = new_chunk_headers

        # Add final chunk
        if current_chunk_content:
            chunked_docs.append(Chunk(
                content=current_chunk_content.strip(),
                heading_context=current_chunk_headers,
                token_count=current_tokens
            ))

        # Stage 2: Verify token counts and recursively split if needed
        final_chunks = []
        chunk_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4o-mini",  # Use consistent model
            chunk_size=chunk_size,
            chunk_overlap=ChunkingParams.chunk_overlap.value
        )

        for chunk_data in chunked_docs:
            if chunk_data.token_count <= chunk_size:
                # Chunk is within limits
                final_chunks.append(chunk_data.content)
            else:
                # Chunk too large, recursively split
                sub_chunks = chunk_splitter.split_text(chunk_data.content)
                header_section_context = self._add_section_headers({}, chunk_data.heading_context)
                is_first_chunk = True

                for sub_chunk in sub_chunks:
                    if is_first_chunk:
                        chunk_content = sub_chunk
                        is_first_chunk = False
                    else:
                        # Add header context to subsequent sub-chunks
                        chunk_content = header_section_context + sub_chunk
                    final_chunks.append(chunk_content)

        return final_chunks

    async def markdown_loading_splitting(
        self,
        markdown_document: str,
        filename: str,
        file_id: str
    ) -> List[Document]:
        """
        Processes a markdown document by splitting it into chunks and creating Document objects.

        Includes markdown table extraction and conversion to natural language for better retrieval.

        Args:
            markdown_document: The markdown content to be processed
            filename: Name of the file being processed
            file_id: Unique identifier for the file

        Returns:
            List of Document objects containing chunked content with metadata
        """
        # Get token-aware chunks
        chunks = self.get_document_chunks(markdown_document)
        logger.info(f"Split {filename} into {len(chunks)} chunks.")

        list_chunks = []
        for chunk in chunks:
            # Check if chunk contains tables
            tables = MarkdownTableExtractor.extract_tables_from_markdown(chunk)

            if tables:
                # Add original chunk with table markdown
                list_chunks.append(Document(
                    page_content=chunk,
                    metadata={"filename": filename, "fileId": file_id}
                ))

                # Convert table to readable format and add as separate chunk
                table_expanded_report = await self.markdown_table_to_word(tables)
                list_chunks.append(Document(
                    page_content=table_expanded_report,
                    metadata={"filename": filename, "fileId": file_id}
                ))
            else:
                # No tables, just add the chunk
                list_chunks.append(Document(
                    page_content=chunk,
                    metadata={"filename": filename, "fileId": file_id}
                ))

        return list_chunks

    async def markdown_table_to_word(self, markdown_tables: List[str]) -> str:
        """
        Converts markdown tables to highly readable natural language descriptions using LLM.

        This improves retrieval quality for tables, as embedding models work better with
        natural language than raw markdown table syntax.

        Args:
            markdown_tables: List of markdown table strings

        Returns:
            Natural language description of the tables
        """
        # Import here to avoid circular dependencies
        from src.services.azure_services import AzureOpenAIService

        markdown_table_str = str(markdown_tables).replace('{', '').replace('}', '')

        system_prompt = """
You are an expert AI assistant specialized in analyzing structured Markdown documents containing GitHub-Flavored Markdown (GFM) tables, including nested tables.

Your task is to convert raw Markdown tables into a clear, well-organized, and highly readable Markdown report.

You must:
- Preserve all information from the original tables, including every row and column.
- Accurately interpret nested tables and include their context in the final output.
- Rephrase each row's data into brief, natural-language summaries that are easy to understand.
- Highlight any dates, timelines, or chronological details clearly.
- Identify and emphasize key points, relationships, and important information.
- Use Markdown formatting with logical headings, bullet points, and sub-sections where appropriate.

Do not summarize or omit any content. Instead, present the full data in a structured and insightful way.
"""

        user_prompt = f"""
Please analyze the following Markdown-formatted tables and convert them into a meaningful Markdown report.

- Include every row and column.
- Do not omit or summarize any content.
- Highlight any dates, timelines, or chronological entries clearly.
- Present the data in a readable, structured format using natural language.
- If there are nested tables, include their context and explain their relevance.

**Markdown Table Data:**
{markdown_table_str}
"""

        # Create prompt template and invoke LLM
        prompt_template = ChatPromptTemplate.from_messages([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "{input}"}
        ])

        chain = prompt_template | AzureOpenAIService().get_client()
        response = await chain.ainvoke({"input": user_prompt})

        return response.content
