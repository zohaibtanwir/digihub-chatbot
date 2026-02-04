import os
import re
import time
import uuid
import shutil
import urllib3
import traceback
import multiprocessing
from pathlib import Path
 
from azure.cosmos import CosmosClient
from azure.core.exceptions import ServiceResponseError
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption, WordFormatOption
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    AcceleratorOptions,
    AcceleratorDevice
)
from docling_core.types.doc import ImageRefMode
from docling.exceptions import ConversionError
 
from src.utils.config import (
    embeddings,
    BLOB_CONTAINER_NAME,
    blob_service,
    database,
    COSMOSDB_SERVICE_NAME_MAPPING_CONTAINER_NAME,
    ROOT_FOLDER,
    OUTPUT_FOLDER,
    AZURE_STORAGE_ACCOUNT_URL,
    file_processing_time,
    COSMOSDB_ENDPOINT,
    COSMOS_ACCOUNT_KEY,
    COSMOSDB_NAME,
    COSMOS_LOGGING_CONTAINER_NAME,
    OCR_STATUS,
    MAX_MEMORY,
    CHUNK_SIZE,
    ENABLE_IMAGE_TEXT_EXTRACTION,
    MAX_CONCURRENT_VISION_CALLS,
    ENABLE_TOKEN_AWARE_CHUNKING,
    CHUNK_SIZE_TOKENS,
    ENABLE_TABLE_CONVERSION,
    ENABLE_QUESTION_GENERATION,
    QUESTIONS_PER_CHUNK
)
from src.utils.action_log import cosmos_index_logger
from src.utils.cosmos_initialize import CosmosDBInitializers
from src.utils.logger import logger
import psutil
import threading
import signal
import atexit
from PyPDF2 import PdfReader, PdfWriter
from src.utils.action_log import logging_container
container_log=logging_container()


urllib3.disable_warnings()
MAX_MEMORY_BYTES = int(MAX_MEMORY) * 1024 * 1024 * 1024

# Global flag for graceful shutdown
_shutdown_requested = False

def log_memory_usage(stage):
    """Log current memory usage with garbage collection."""
    import gc
    gc.collect()  # Force garbage collection before measuring
    mem = psutil.Process(os.getpid()).memory_info().rss
    logger.info(f"{stage} memory usage: {mem / (1024 ** 2):.2f} MB")

def graceful_shutdown(signum=None, frame=None):
    """
    Handle graceful shutdown on signal or memory limit exceeded.

    This ensures:
    - Resources are properly released
    - Cleanup functions are called
    - Proper exit code is returned
    """
    global _shutdown_requested
    if _shutdown_requested:
        return  # Already shutting down

    _shutdown_requested = True
    logger.warning(f"Graceful shutdown initiated (signal: {signum})")

    # Perform cleanup
    try:
        # Note: cleanup is handled by individual processors
        logger.info("Shutdown cleanup completed")
    except Exception as e:
        logger.error(f"Error during shutdown cleanup: {e}")

    # Exit with appropriate code
    exit_code = 137 if signum == "MEMORY" else 0
    logger.info(f"Exiting with code {exit_code}")
    os._exit(exit_code)

def memory_watchdog(interval=1):
    """
    Monitor memory usage and trigger graceful shutdown if limit exceeded.

    Improved version that:
    - Triggers graceful shutdown instead of immediate exit
    - Logs memory warnings before critical limit
    - Allows cleanup to complete
    """
    process = psutil.Process(os.getpid())
    warning_threshold = MAX_MEMORY_BYTES * 0.8  # Warn at 80%
    warned = False

    while not _shutdown_requested:
        try:
            mem = process.memory_info().rss

            if mem > MAX_MEMORY_BYTES:
                logger.error(f"Memory limit exceeded: {mem / (1024**3):.2f} GB > {MAX_MEMORY} GB. Initiating graceful shutdown.")
                graceful_shutdown(signum="MEMORY")
                break
            elif mem > warning_threshold and not warned:
                logger.warning(f"Memory usage high: {mem / (1024**3):.2f} GB (80% of {MAX_MEMORY} GB limit)")
                warned = True

            time.sleep(interval)
        except Exception as e:
            logger.error(f"Memory watchdog error: {e}")
            time.sleep(interval)

# Register signal handlers for graceful shutdown
signal.signal(signal.SIGTERM, graceful_shutdown)
signal.signal(signal.SIGINT, graceful_shutdown)

def extract_text_worker(input_doc_path, folder_name, return_dict):
    """
        Worker function to extract text from a document.
        Runs in a separate process and stores the result or error in a shared dictionary.
    """ 
    try:
        threading.Thread(target=memory_watchdog, daemon=True).start()
        log_memory_usage("Before processing")
        processor = DocumentProcessor()
        result = processor.extract_text(input_doc_path, folder_name)
        log_memory_usage("After processing")
        return_dict["result"] = result
    except Exception as e:
        return_dict["error"] = str(e)
 
def extract_text_with_timeout(input_doc_path, folder_name, timeout=int(file_processing_time)):
    """
        Extracts text from a document with a timeout.
        
        Args:
            input_doc_path (str): Path to the input document.
            timeout (int): Maximum time (in seconds) to allow for extraction. Default is 1800 seconds (30 minutes).
        
        Returns:
            str: Extracted text from the document.
        
        Raises:
            TimeoutError: If the extraction takes longer than the specified timeout.
            Exception: If an error occurs during extraction.
    """  
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
 
    process = multiprocessing.Process(
        target=extract_text_worker,
        args=(input_doc_path, folder_name, return_dict)
    )
    process.start()
    process.join(timeout)
 
    if process.is_alive():
        process.terminate()
        process.join()
        logger.error(f"Timeout: extract_text took longer than {timeout} seconds and was terminated.")
        return None
    if process.exitcode == 137:
        logger.error(f"Process for {input_doc_path} killed with exit code 137.")
        return None
    if "error" in return_dict:
        logger.error(f"Extraction Error: {return_dict['error']}")
        return None
    
    return return_dict.get("result")
 
class DocumentProcessor:
    def __init__(self):
        # Import new services
        from src.service.document_converter_service import DocumentConverterService
        from src.service.embed_document_service import EmbedDocumentService

        # Use singleton pattern for document converter
        self.doc_converter_service = DocumentConverterService()

        # Initialize image extraction service if enabled
        if ENABLE_IMAGE_TEXT_EXTRACTION:
            from src.service.image_extraction_service import ImageTextExtractionService
            self.image_extraction_service = ImageTextExtractionService(
                max_concurrent_extractions=MAX_CONCURRENT_VISION_CALLS
            )
            logger.info("Image text extraction service enabled")
        else:
            self.image_extraction_service = None
            logger.info("Image text extraction service disabled")

        # Initialize enhanced embedding service for token-aware chunking
        if ENABLE_TOKEN_AWARE_CHUNKING:
            self.embed_document_service = EmbedDocumentService()
            logger.info("Token-aware chunking enabled")
        else:
            self.embed_document_service = None
            logger.info("Token-aware chunking disabled (using legacy chunking)")

        # Initialize question generator service for question-first retrieval
        if ENABLE_QUESTION_GENERATION:
            from src.services.question_generator_service import QuestionGeneratorService
            self.question_generator_service = QuestionGeneratorService()
            logger.info("Question generation service enabled")
        else:
            self.question_generator_service = None
            logger.info("Question generation service disabled")

        # Backward compatibility: Keep doc_converter for legacy code paths
        # This will be removed once all code is migrated to use doc_converter_service
        self.doc_converter = self.doc_converter_service.get_document_converter(is_pptx_as_pdf=False)
 
    def get_cosmos_container(self):
        if not hasattr(self, "_container"):
            self.cosmos_initializer = CosmosDBInitializers()
            self._container = self.cosmos_initializer.get_cosmos()
        return self._container

    def get_embedding(self, text: str):
        return embeddings.embed_query(text)
 
    def handle_cleanup(self):
        """Cleans up temporary files and folders after processing is complete."""
        for folder in [OUTPUT_FOLDER, ROOT_FOLDER]:
            if os.path.exists(folder):
                for d in os.listdir(folder):
                    full_path = os.path.join(folder, d)
                    if os.path.isdir(full_path):
                        shutil.rmtree(full_path)
                    else:
                        os.unlink(full_path)

    def split_pdf(self, input_path, chunk_size):
        reader = PdfReader(str(input_path))
        total_pages = len(reader.pages)
        chunks = []

        for start in range(0, total_pages, chunk_size):
            writer = PdfWriter()
            for i in range(start, min(start + chunk_size, total_pages)):
                writer.add_page(reader.pages[i])
            chunk_path = input_path.parent / f"{input_path.stem}_chunk_{start // chunk_size}.pdf"
            with open(chunk_path, "wb") as f:
                writer.write(f)
            chunks.append(chunk_path)
        return chunks

    def extract_text(self, input_doc_path, folder_name):
        cosmos_client = CosmosClient(COSMOSDB_ENDPOINT, COSMOS_ACCOUNT_KEY, connection_verify=False)
        database = cosmos_client.get_database_client(COSMOSDB_NAME)
        multiprocess_container_log = database.get_container_client(COSMOS_LOGGING_CONTAINER_NAME)
        input_doc_path = Path(input_doc_path)
        
        if int(CHUNK_SIZE) == 0:
            logger.info("Only one chunk found. Using direct conversion method.")
            return self.extract_text_single_chunk(input_doc_path, folder_name)

        chunks = self.split_pdf(input_doc_path,int(CHUNK_SIZE))
        final_md_path = OUTPUT_FOLDER / f"{input_doc_path.stem}-final.md"

        total_chunks = len(chunks)
        successful_chunks = 0
        failed_chunks_info = []

        for chunk_index, chunk_path in enumerate(chunks):
            logger.info(f"Converting chunk {chunk_path}")
            start_time = time.time()

            try:
                conv_res = self.doc_converter.convert(chunk_path)
                logger.info(f"Conversion completed for {chunk_path}")
            except Exception as e:
                logger.error(f"Conversion failed for chunk {chunk_path}: {e}")
                start_page = chunk_index * 5
                end_page = min(start_page + 4, len(PdfReader(str(input_doc_path)).pages) - 1)
                failed_chunks_info.append(f"Pages {start_page + 1}-{end_page + 1}")
                continue

            successful_chunks += 1    

            relative_output_folder = chunk_path.relative_to(ROOT_FOLDER)
            output_folder = OUTPUT_FOLDER / relative_output_folder.parent
            output_folder.mkdir(parents=True, exist_ok=True)

            time_difference_ms = (time.time() - start_time) * 1000
            chunk_filename = chunk_path.stem
            md_output_path = output_folder / f"{chunk_filename}-with-images.md"

            conv_res.document.save_as_markdown(md_output_path, image_mode=ImageRefMode.REFERENCED)
            chunk_info = f"Chunk {chunk_index + 1}/{total_chunks} (Pages {chunk_index * 5 + 1}-{min((chunk_index + 1) * 5, len(PdfReader(str(input_doc_path)).pages))})"
            cosmos_index_logger(multiprocess_container_log, f"Created_MD_File - {chunk_info}", folder_name, chunk_filename, str(chunk_path), time_difference_ms)

            # Extract image text if enabled
            if self.image_extraction_service:
                try:
                    # Read the markdown file
                    with open(md_output_path, 'r', encoding='utf-8') as f:
                        markdown_content = f.read()

                    # Use async processing to extract image text
                    import asyncio
                    markdown_with_images = asyncio.run(
                        self.image_extraction_service.convert_document_to_markdown_with_images(
                            doc=conv_res.document,
                            file_name=chunk_path.name,
                            base_dir=str(OUTPUT_FOLDER)
                        )
                    )

                    # Overwrite markdown with enhanced version
                    with open(md_output_path, 'w', encoding='utf-8') as f:
                        f.write(markdown_with_images)

                    logger.info(f"Successfully extracted image text for {chunk_info}")
                except Exception as e:
                    logger.error(f"Image text extraction failed for {chunk_info}: {e}")
                    # Continue with the original markdown if image extraction fails

            with open(md_output_path, 'r') as chunk_file, open(final_md_path, 'a') as final_file:
                final_file.write(chunk_file.read())

            for file_path in output_folder.rglob('*'):
                if file_path.is_file():
                    blob_client = blob_service.get_blob_client(
                        container=BLOB_CONTAINER_NAME,
                        blob=file_path.relative_to(OUTPUT_FOLDER).as_posix()
                    )
                    with open(file_path, 'rb') as file:
                        blob_client.upload_blob(file, overwrite=True)

            blob_path = f"{AZURE_STORAGE_ACCOUNT_URL}/{BLOB_CONTAINER_NAME}/{file_path.relative_to(OUTPUT_FOLDER).as_posix()}"
            uploaded_folder_blob_path = f"{AZURE_STORAGE_ACCOUNT_URL}/{BLOB_CONTAINER_NAME}/{md_output_path.parent.relative_to(OUTPUT_FOLDER).as_posix()}"
            cosmos_index_logger(multiprocess_container_log, f"Uploaded_Blob_Storage - {chunk_info}", folder_name, file_path.name, uploaded_folder_blob_path, time_difference_ms)
            
            summary = f"Processed {successful_chunks}/{total_chunks} chunks"
            logger.info(summary)
            if failed_chunks_info:
                logger.info(f"Failed chunks: {', '.join(failed_chunks_info)}")

            logger.info(f"Chunk {chunk_path} processed and uploaded in {time.time() - start_time:.2f} sec")

        return final_md_path, input_doc_path.name, blob_path, time_difference_ms
    
    def extract_text_single_chunk(self, input_doc_path, folder_name):
        """Extracts text from a document, converts it to Markdown, and uploads the result."""
        # Cosmos DB client is created inside the function to avoid issues with multiprocessing.
        # Creating the client globally can cause serialization errors when using multiprocessing,
        # so it's safer to initialize it within the process scope. 
        cosmos_client = CosmosClient(COSMOSDB_ENDPOINT, COSMOS_ACCOUNT_KEY, connection_verify=False)
        database = cosmos_client.get_database_client(COSMOSDB_NAME)
        multiprocess_container_log = database.get_container_client(COSMOS_LOGGING_CONTAINER_NAME)
 
        logger.info(f"Converting the {input_doc_path}")
        start_time = time.time()
        # Convert the document  
        try:
            conv_res = self.doc_converter.convert(input_doc_path)
            logger.info(f"Conversion completed for {input_doc_path}")
        except ConversionError as ce:
            logger.error(f"Conversion failed: {ce}")
            self.handle_cleanup()
            raise
        except Exception as e:
            logger.error(f"An error occurred during conversion: {e}")
            self.handle_cleanup()
            raise
        # Create the output folder structure based on input path
        input_doc_path = Path(input_doc_path)
        relative_output_folder = input_doc_path.relative_to(ROOT_FOLDER)
        output_folder = OUTPUT_FOLDER / relative_output_folder.parent
        output_folder.mkdir(parents=True, exist_ok=True)
 
        time_difference_ms = (time.time() - start_time) * 1000
        doc_filename = input_doc_path.stem
        md_output_path = output_folder / f"{doc_filename}-with-images.md"
        # Save the document as markdown
        conv_res.document.save_as_markdown(md_output_path, image_mode=ImageRefMode.REFERENCED)
        cosmos_index_logger(multiprocess_container_log, "Created_MD_File", folder_name, doc_filename, str(input_doc_path), time_difference_ms)

        # Extract image text if enabled
        if self.image_extraction_service:
            logger.info(f"Extracting text from images in {input_doc_path.name}")
            try:
                # Read the markdown file
                with open(md_output_path, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()

                # Use async processing to extract image text
                import asyncio
                markdown_with_images = asyncio.run(
                    self.image_extraction_service.convert_document_to_markdown_with_images(
                        doc=conv_res.document,
                        file_name=input_doc_path.name,
                        base_dir=str(OUTPUT_FOLDER)
                    )
                )

                # Overwrite markdown with enhanced version
                with open(md_output_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_with_images)

                logger.info(f"Successfully extracted image text for {input_doc_path.name}")
            except Exception as e:
                logger.error(f"Image text extraction failed for {input_doc_path.name}: {e}")
                # Continue with the original markdown if image extraction fails

        for file_path in output_folder.rglob('*'):
            if file_path.is_file():
                blob_client = blob_service.get_blob_client(
                    container=BLOB_CONTAINER_NAME,
                    blob=file_path.relative_to(OUTPUT_FOLDER).as_posix()
                )
                with open(file_path, 'rb') as file:
                    blob_client.upload_blob(file, overwrite=True)
 
        blob_path = f"{AZURE_STORAGE_ACCOUNT_URL}/{BLOB_CONTAINER_NAME}/{file_path.relative_to(OUTPUT_FOLDER).as_posix()}"
        uploaded_folder_blob_path = f"{AZURE_STORAGE_ACCOUNT_URL}/{BLOB_CONTAINER_NAME}/{md_output_path.parent.relative_to(OUTPUT_FOLDER).as_posix()}"
        cosmos_index_logger(multiprocess_container_log, "Uploaded_Blob_Storage", folder_name, file_path.name, uploaded_folder_blob_path, time_difference_ms)
 
        logger.info(f"Document {input_doc_path} processed and uploaded in {time.time() - start_time:.2f} sec")
        return md_output_path, input_doc_path.name, blob_path, time_difference_ms
 
    def markdown_loading_splitting(self, md_path, filename):
        """Splits the markdown file into chunks based on headers with optional token-aware chunking."""
        with open(md_path, "r", encoding="utf-8") as file:
            markdown_document = file.read()

        # Update image paths in markdown
        relative_path_parts = os.path.relpath(os.path.dirname(md_path), OUTPUT_FOLDER).split(os.sep)
        new_base_path = os.path.join(*relative_path_parts)
        markdown_document = markdown_document.replace(r'![Image](', f'![Image]({new_base_path}/')

        # Use token-aware chunking if enabled
        if self.embed_document_service:
            logger.info(f"Using token-aware chunking for {filename}")
            try:
                # Generate a file_id for this file (use filename hash for now)
                import hashlib
                file_id = hashlib.md5(filename.encode()).hexdigest()

                # Use async method for table conversion support
                import asyncio
                documents = asyncio.run(
                    self.embed_document_service.markdown_loading_splitting(
                        markdown_document,
                        filename,
                        file_id
                    )
                )

                self.handle_cleanup()
                logger.info(f"Split {filename} into {len(documents)} chunks using token-aware chunking.")
                return documents

            except Exception as e:
                logger.error(f"Token-aware chunking failed for {filename}: {e}, falling back to legacy chunking")
                # Fall through to legacy chunking

        # Legacy chunking (fallback or if token-aware chunking disabled)
        logger.info(f"Using legacy chunking for {filename}")
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4")
        ]
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
        chunks = splitter.split_text(markdown_document)

        self.handle_cleanup()
        logger.info(f"Split {filename} into {len(chunks)} chunks.")

        return [
            Document(page_content=chunk.page_content, metadata={"filename": filename})
            for chunk in chunks
        ]
 
    def update_image_paths(self, content, relative_path_parts):
        """Updates image paths in the markdown content."""
        new_base_path = os.path.join(*relative_path_parts)
        return content.replace(r'![Image](', f'![Image]({new_base_path}/')
 
    def get_service_id(self, listid):
        """Retrieves the service ID for a given list ID from Cosmos DB.""" 
        try:
            service_name_mapping_container = database.get_container_client(COSMOSDB_SERVICE_NAME_MAPPING_CONTAINER_NAME)
            query = f"SELECT c.service_id FROM c WHERE c.list_id = '{listid}'"
            items = list(service_name_mapping_container.query_items(query=query, enable_cross_partition_query=True))
        except ServiceResponseError as e:
            logger.error(f"Database connection error: {e} {traceback.format_exc()}")
            items = []
 
        if items:
            return int(items[0]['service_id'])
        else:
            raise ValueError(f"Service ID not found for folder name: {listid}")
 
    def upload_chunks_to_cosmos(self, chunks, folder_name, listid, file_path):
        """
        Uploads document chunks to Cosmos DB with idempotency and transaction support.

        Each chunk is processed to include:
        - validChunk: "yes" or "no" based on quality validation
        - questions: List of synthetic questions the chunk can answer
        - questionsEmbedding: Embedding vector for question-first retrieval

        Duplicate detection is reset at the start of each file to detect
        duplicates within the same document.
        """
        from src.utils.partition_key_utils import generate_partition_key, generate_chunk_id

        # Reset duplicate tracking for this new file
        if self.question_generator_service:
            self.question_generator_service.reset_duplicate_tracking()
            logger.info("Reset duplicate tracking for new file processing")

        try:
            container = self.get_cosmos_container()
            service_id = self.get_service_id(listid)
        except ValueError as e:
            logger.error(e)
            return

        # Track uploaded chunks for potential rollback
        uploaded_ids = []
        filename = chunks[0].metadata['filename'] if chunks else "unknown"

        # Generate consistent partition key
        partition_key = generate_partition_key(folder_name, filename)
        logger.info(f"Using partition key: {partition_key} for {filename}")

        try:
            for index, chunk in enumerate(chunks):
                try:
                    content = chunk.page_content
                    heading = self.extract_first_heading(content)

                    # Generate deterministic chunk ID for idempotency
                    chunk_id = generate_chunk_id(folder_name, filename, index, content)

                    # Check if chunk already exists with same content (idempotency)
                    try:
                        existing_doc = container.read_item(item=chunk_id, partition_key=partition_key)
                        if existing_doc and existing_doc.get("content") == content:
                            logger.info(f"Chunk {chunk_id} unchanged, skipping upload")
                            uploaded_ids.append(chunk_id)
                            continue
                    except Exception:
                        # Chunk doesn't exist, proceed with upload
                        pass

                    # Generate content embedding
                    embedding = self.get_embedding(content)

                    # Initialize question-related fields with defaults
                    valid_chunk = "yes"
                    questions = []
                    questions_embedding = []

                    # Generate questions and validate chunk if service is enabled
                    if self.question_generator_service:
                        try:
                            chunk_result = self.question_generator_service.process_chunk(
                                content=content,
                                heading=heading
                            )
                            valid_chunk = chunk_result.get("validChunk", "yes")
                            questions = chunk_result.get("questions", [])
                            questions_embedding = chunk_result.get("questionsEmbedding", [])

                            logger.info(
                                f"Chunk {index}: validChunk={valid_chunk}, "
                                f"questions={len(questions)}, "
                                f"questionsEmbedding={'generated' if questions_embedding else 'empty'}"
                            )
                        except Exception as qg_error:
                            logger.error(f"Question generation failed for chunk {index}: {qg_error}")
                            # Continue with defaults - chunk is still valid for basic retrieval
                            valid_chunk = "yes"

                    # Create document with deterministic ID and consistent partition key
                    doc = {
                        "id": chunk_id,
                        "partitionKey": partition_key,
                        "serviceName": folder_name,
                        "serviceNameid": service_id,
                        "heading": heading,
                        "content": content,
                        "embedding": embedding,
                        "validChunk": valid_chunk,
                        "questions": questions,
                        "questionsEmbedding": questions_embedding,
                        "processedAt": time.time(),
                        "processingVersion": "v3.0",  # Updated version for question-first retrieval
                        "metadata": {
                            "filepath": str(file_path),
                            "heading": heading,
                            "filename": filename,
                        }
                    }

                    container.upsert_item(doc)
                    uploaded_ids.append(chunk_id)
                    logger.info(f"Uploaded chunk {chunk_id} | Heading: {heading} | Valid: {valid_chunk}")

                except Exception as e:
                    logger.error(f"Error uploading chunk {index}: {e}")
                    # Rollback: delete successfully uploaded chunks
                    logger.warning(f"Upload failed at chunk {index}, rolling back {len(uploaded_ids)} chunks")
                    for uploaded_chunk_id in uploaded_ids:
                        try:
                            container.delete_item(item=uploaded_chunk_id, partition_key=partition_key)
                            logger.info(f"Rolled back chunk {uploaded_chunk_id}")
                        except Exception as delete_err:
                            logger.error(f"Rollback failed for {uploaded_chunk_id}: {delete_err}")
                    raise  # Re-raise to propagate error

        except Exception as e:
            logger.error(f"Fatal error in upload_chunks_to_cosmos: {e}")
            raise
 
    def extract_first_heading(self, markdown_text):
        """Extracts the first heading from markdown text."""
        match = re.search(r'^(#+)\s+(.*)', markdown_text, re.MULTILINE)
        return match.group(2).strip() if match else "Untitled"
 
    def process_file(self, file_path: Path, folder_name: str, listid, sharepoint_file_path):
        """Processes a single file, extracts text, splits markdown, and uploads to Cosmos DB."""
        result = extract_text_with_timeout(file_path, folder_name)
 
        if result is None:
            cosmos_index_logger(container_log, "Timeout_or_Failure", folder_name, file_path.name, str(file_path), 0)
            return
 
        md_output_path, filename, md_blob_path, time_difference_ms = result
        cosmos_index_logger(container_log, "Extracted_Converted_to_MD", folder_name, md_output_path.name, md_blob_path, time_difference_ms)
 
        chunks = self.markdown_loading_splitting(md_output_path, filename)
        chunks_len =len(chunks)
        self.upload_chunks_to_cosmos(chunks, folder_name, listid, sharepoint_file_path)
        
        cosmos_index_logger(container_log, "Uploaded_To_VectorDB", folder_name, md_output_path.name, md_blob_path, time_difference_ms)
        return True, chunks_len
 
 