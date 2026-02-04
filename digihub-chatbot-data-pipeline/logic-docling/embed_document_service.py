from typing import List
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import tiktoken
from langchain_core.prompts import ChatPromptTemplate

from src.enums.chunking_params import ChunkingParams
from src.models.doc_chunk_model import Chunk
from src.services.azure_services import AzureOpenAIService
from src.utils.config import AZURE_OPENAI_DEPLOYMENT_TEXT
from src.utils.logger import logger
from langchain_core.documents import Document

from src.utils.nested_tables_utils import MarkdownTableExtractor


class EmbedDocumentService:
    def __init__(self):
        """
        Initializes token encoding
        based on the Azure OpenAI deployment configuration.
        """
        self.encoding = tiktoken.encoding_for_model(AZURE_OPENAI_DEPLOYMENT_TEXT)

    def count_tokens(self, text):
        """
        Counts the number of tokens in the given text using the model's encoding.

        Args:
            text (str): The input text to tokenize.

        Returns:
            int: The number of tokens in the text.
        """
        return len(self.encoding.encode(text))

    def _add_section_headers(self, current_chunk_headers, new_chunk_headers):
        """
        Generates markdown-style header context based on changes between current and new headers.

        Args:
            current_chunk_headers (dict): Headers from the current chunk.
            new_chunk_headers (dict): Headers from the new chunk.

        Returns:
            str: A string containing updated markdown headers.
        """
        header_section_context = ""
        for i in range(1, 4):
            if (f"Header {i}" in new_chunk_headers and f"Header {i}" in current_chunk_headers and current_chunk_headers[
                f"Header {i}"] == new_chunk_headers[f"Header {i}"]):
                continue
            elif f"Header {i}" in new_chunk_headers:
                header_section_context += '#' * i + f" {new_chunk_headers[f'Header {i}']}\n"
        return header_section_context

    def get_document_chunks(self, input_text: str, chunk_size=ChunkingParams.chunk_size.value) -> list:
        """
        Splits the input document into chunks based on markdown headers and token limits.

        Args:
            input_text (str): The full document text to split.
            chunk_size (int): Maximum token size per chunk.

        Returns:
            List[str]: A list of processed document chunks.
        """

        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

        header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        header_docs = header_splitter.split_text(input_text)

        chunked_docs: List[Chunk] = []
        current_chunk_content = ""
        header_section_context = ""
        current_chunk_headers = {}
        current_tokens = 0

        for doc in header_docs:
            doc_tokens = self.count_tokens(doc.page_content)
            new_chunk_headers = doc.metadata.copy()
            if current_tokens + doc_tokens > chunk_size and current_chunk_content:
                chunked_docs.append(Chunk(content=current_chunk_content.strip(),
                                          heading_context=current_chunk_headers,
                                          token_count=current_tokens))
                header_section_context = self._add_section_headers({}, new_chunk_headers)
                current_chunk_content = header_section_context + doc.page_content
                current_tokens = doc_tokens

            else:
                header_section_context = self._add_section_headers(current_chunk_headers, new_chunk_headers)

                if current_chunk_content:
                    current_chunk_content += "\n\n" + header_section_context + doc.page_content
                elif (doc.page_content.strip()):
                    current_chunk_content = header_section_context + doc.page_content
                current_tokens += doc_tokens
            current_chunk_headers = new_chunk_headers

        if current_chunk_content:
            chunked_docs.append(
                Chunk(
                    content=current_chunk_content.strip(),
                    heading_context=current_chunk_headers,
                    token_count=current_tokens,
                )
            )

        final_chunks = []
        chunk_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name=AZURE_OPENAI_DEPLOYMENT_TEXT,
            chunk_size=chunk_size,
            chunk_overlap=100
        )

        for chunk_data in chunked_docs:
            if chunk_data.token_count <= chunk_size:
                final_chunks.append(chunk_data.content)
            else:
                sub_chunks = chunk_splitter.split_text(chunk_data.content)
                header_section_context = self._add_section_headers({}, chunk_data.heading_context)
                is_firstchunk = True
                for sub_chunk in sub_chunks:
                    if (is_firstchunk):
                        chunk_content = sub_chunk
                        is_firstchunk = False
                    else:
                        chunk_content = header_section_context + sub_chunk
                    final_chunks.append(chunk_content)

        return final_chunks

    async def markdown_loading_splitting(self, markdown_document: str, filename: str, file_id: str) -> List[Document]:
        """
        Processes a markdown document by splitting it into chunks and creating Document objects.

        Args:
            markdown_document (str): The markdown content to be processed
            filename (str): Name of the file being processed
            file_id (str): Unique identifier for the file

        Returns:
            List[Document]: List of Document objects containing chunked content with metadata
        """
        chunks = self.get_document_chunks(markdown_document)
        logger.info(f"Split {filename} into {len(chunks)} chunks.")
        list_chunks = []
        for chunk in chunks:
            tables = MarkdownTableExtractor().extract_tables_from_markdown(chunk)
            if tables:
                table_expanded_report = await self.markdown_table_to_word(tables)
                list_chunks.append(Document(page_content=chunk,
                                            metadata={"filename": filename, "fileId": file_id}))
                list_chunks.append(Document(page_content=table_expanded_report,
                                            metadata={"filename": filename, "fileId": file_id}))
            else:
                list_chunks.append(Document(page_content=chunk, metadata={"filename": filename, "fileId": file_id}))
        return list_chunks

    async def markdown_table_to_word(self, markdown_table: list):
        """
        Gives highly readable word data from markdown tables
        Args:
            excel_bytes:

        Returns:

        """

        markdown_table_str = str(markdown_table).replace('{', '').replace('}', '')
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

        # Define user prompt
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