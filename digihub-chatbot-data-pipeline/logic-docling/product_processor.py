# Import necessary modules and libraries
import asyncio
import io
import os
import traceback
import uuid
from typing import List
import json
from docling.document_converter import DocumentStream
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from src.models.product_info import ProductTag
from src.services.azure_services import AzureOpenAIService
from src.services.embed_document_service import EmbedDocumentService
from src.utils.config import PRODUCT_DOCS_EMBEDDINGS, PRODUCT_DOCS_LIBRARY
from src.models.document_payload import GenerateSummaryPayload
from src.services.cosmos_db_service import CosmosDBClientSingleton
from src.services.document_convertor import DocumentConverterService
from src.services.embedding_service import AzureEmbeddingService
from src.utils.decorators import timing_decorator
from src.utils.indexing_utils import extract_first_heading
from src.utils.logger import logger
from src.services.image_extraction_service import ImageTextExtractionService
import time
import urllib3
import base64

# Disable SSL warnings
urllib3.disable_warnings()


class ProductProcessorService:
    def __init__(self):
        # Initialize services (DocumentConverter is now created dynamically per file type)
        self.doc_converter_service = DocumentConverterService()
        self.embedding_client = AzureEmbeddingService()
        self.cosmos_client = CosmosDBClientSingleton()
        self.embed_document_service = EmbedDocumentService()
        self.image_extraction_service = ImageTextExtractionService()

    # Entry point for processing files asynchronously
    async def process_files_async(self, documents: List[dict]):
        return await self.process_files(documents)

    # Process a list of documents concurrently
    async def process_files(self, documents: List[dict]):
        tasks = [
            self.process_file(document=document)
            for document in documents
        ]
        contents = await asyncio.gather(*tasks)
        logger.info(f"[get_file_contents] No of Files Converted To Markdown: {len(contents)}")
        return contents

    # Process a single document: extract text, split into chunks, and upload to Cosmos DB
    async def process_file(self, document: dict):
        # Pass pptx_as_pdf flag to control whether to use hybrid processing
        pptx_as_pdf = document.get("pptx_as_pdf", False)
        markdown_string = await self.extract_text(document.get("stream"), pptx_as_pdf=pptx_as_pdf)

        chunks = await self.embed_document_service.markdown_loading_splitting(
            markdown_string,
            document.get("stream").name,
            file_id=document.get("fileId"),
        )
        document_tag = await self.get_document_tag(document.get('filePath'))
        await self.upload_chunks_to_cosmos(
            chunks,
            document.get("stream").name,
            document.get("createdDate"),
            document.get("lastModifiedDate"),
            document.get("product"),
            document.get("spnUrl"),
            document_tag,
            document.get('product-tag'),
            document.get('internalUsage'),
            document.get('division'),
        )
        logger.info(f"Uploaded File : {document.get('stream').name}")


    async def get_document_tag(self,filename):

        content_mapping = {
            "Customer Presentation": "Product Desc",
            "Battlecard": "Positioning",
            "Use Cases": "Product Desc",
            "Product Webinar": "Product Desc",
            "Bid Boilerplate": "Positioning",
            "Brochure": "Product Desc",
            "Top Tips": "Positioning",
            "Solution Design Guidelines": "Positioning",
            "User Guide": "Tech Spec",
            "FAQs": "Product Desc"
        }
        for key,value in content_mapping.items():
            if key in filename:
                return value
        return "Other"

    # Extract markdown text from a document stream
    @timing_decorator
    async def extract_text(self, doc_stream: DocumentStream, pptx_as_pdf: bool = False):
        file_name = doc_stream.name
        file_ext = os.path.splitext(file_name)[1].lower()
        is_pptx = file_ext == '.pptx'
        
        logger.info(f"Converting {file_name} (format: {file_ext}, pptx_as_pdf: {pptx_as_pdf})")
        
        try:
            logger.info(f"Markdown to text conversion started for {file_name}")
            
            # Get appropriate converter based on file type
            # PPTX files (downloaded as PDF) need generate_page_images=True
            # Only use PDF converter if pptx_as_pdf=True (successful PDF download)
            converter = self.doc_converter_service.get_document_converter(is_pptx_as_pdf=pptx_as_pdf)
            conv_res = converter.convert(doc_stream)
            
            logger.info(f"Conversion completed for {file_name}")
        except Exception as e:
            logger.error(f"An error occurred during conversion: {e}")
            raise

        # Route based on file format AND whether PPTX was successfully downloaded as PDF
        # PPTX files downloaded as PDF (with generate_page_images=True) generate page-0.png, page-1.png, etc.
        # Only use hybrid processing if we successfully downloaded PPTX as PDF
        if is_pptx and pptx_as_pdf:
            # PPTX downloaded as PDF: Use hybrid processing (Vision for slides with many images, standard for others)
            logger.info(f"Using PPTX hybrid processing for {file_name} (downloaded as PDF)")
            markdown_with_ocr = await self.image_extraction_service.convert_pptx_to_markdown_hybrid(
                doc=conv_res.document,
                file_name=file_name
            )
        else:
            # PDF/DOCX or PPTX downloaded as-is: Use standard image-by-image processing
            if is_pptx and not pptx_as_pdf:
                logger.info(f"Using standard image extraction for {file_name} (PPTX fallback - not downloaded as PDF)")
            else:
                logger.info(f"Using standard image extraction for {file_name}")
            markdown_with_ocr = await self.image_extraction_service.convert_document_to_markdown_with_images(
                doc=conv_res.document,
                file_name=file_name
            )
        
        return markdown_with_ocr

    # Upload the document chunks and their embeddings to Cosmos DB
    async def upload_chunks_to_cosmos(self, chunks, file_name, created_at, modified_at, product_name, spnUrl,document_tag,product_tag, internal_usage = None, division = None):
        try:
            contents = [chunk.page_content for chunk in chunks]
            embeddings = await self.embedding_client.get_document_embedding(contents)

            for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                content = chunk.page_content
                heading = await extract_first_heading(content)
                file_id = chunk.metadata.get("fileId")

                doc = {
                    "id": str(file_id) + '_' + str(index),
                    "splitType": "chunk",
                    "groupType": "product_docs",
                    "content": content,
                    "embedding": embedding,
                    "chunkOrder": str(index),
                    "fileId": str(file_id),
                    "createdAT": created_at,
                    "lastModifiedAt": modified_at,
                    "documentType":document_tag,
                    "product-tag":product_tag,
                    "metadata": {
                        "heading": heading,
                        "filename": file_name,
                        'productName': product_name,
                        'spnUrl': spnUrl,
                    },
                    "internalUsage" : internal_usage,
                    "division" : division
                }

                container = await self.cosmos_client.create_product_doc_embeddings(PRODUCT_DOCS_EMBEDDINGS)
                container.upsert_item(body=doc)

                logger.info(f"Uploaded chunk from {file_name} | Heading: {heading} | chunkOrder: {index}")

        except Exception as e:
            logger.error(f"Error occurred")

    # Upload the final classifier output to Cosmos DB
    async def upload_product_docs_classifier_to_cosmos(self, response, items):
        try:
            hardwares_used = {}
            keywords = []

            # Extract hardware and technology mapping
            for hardware_data in response.technology_hardware_map:
                hardwares_used[hardware_data.technology] = hardware_data.hardware
                keywords += hardware_data.hardware
                keywords.append(hardware_data.technology)

            chunks = response.use_case
            embeddings = await self.embedding_client.get_embedding(chunks)

            meta = dict(items[0]["metadata"])
            metadata = {
                "filename": meta["filename"],
                "spnUrl": meta["spnUrl"]
            }

            final_data = {
                "id": items[0]["fileId"],
                "useCase": response.use_case,
                "embeddings": embeddings,
                "hardware": hardwares_used,
                "technologiesUsed": list(set(keywords)),
                "productName": response.product_name,
                "createdAt": items[0]["createdAT"],
                "lastModifiedAt": items[0]["lastModifiedAt"],
                "metaData": metadata
            }

            container = await self.cosmos_client.get_product_docs_classifier()
            container.create_item(body=final_data)
            logger.info(f"Uploaded classifier data from {meta['filename']}")
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error uploading classifier data: {e}")

    def update_product_doc_in_cosmos(
        self,index_status, indexedAt, error_message,file_id=None,doc_id=None,document_tag=None,product_tag=None
    ):
        try:
            logger.info(f"updating the product doc library:{index_status}")
            container = self.cosmos_client.get_container_sync(PRODUCT_DOCS_LIBRARY,'masterId')

            # Step 1: Read the existing item
            query = f"""
                   SELECT * FROM c
                   WHERE c.id = '{doc_id}'
                   """
            items = list(container.query_items(
                query=query,
                enable_cross_partition_query=True
            ))
            item = items[0]
            # Step 2: Update fields
            item["indexedStatus"] = index_status
            item["indexedAt"] = indexedAt
            item["errorMessage"] = error_message
            item['document-type'] = document_tag
            if file_id:
                item["fileId"] = file_id
            if product_tag:
                item['product-tag'] = product_tag

            # Step 3: Replace the item
            container.upsert_item(body=item)
            logger.info(f"Item updated successfully: {doc_id}")

            # Step 4: Read back and return the updated item
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error updating product doc data: {e}")
            return None

    async def get_product_from_folder_path(self, file_path, decoded_url):
        """
        Retrieves product tags based on the file path and decoded URL.

        This asynchronous function extracts the product tag from the decoded URL
        and maps it to predefined tags. If the product tag corresponds to 
        a "smart path", it generates a prompt to suggest additional product 
        tags based on the file name and the existing product tags.

        Args:
            file_path (str): The path of the file from which the product name can be derived.
            decoded_url (str): The decoded URL containing the product tag.
        
        Returns:
            List[str]: A deduplicated list of product tags, potentially including 
                        suggestions based on the file name.
        """

        file_name = file_path.split("/")[-1]
        product_tag = decoded_url.split("/")[-1]

        product_tag_mapping = {
            "SITA Smart Path Gates": "Smart Path Gates",
            "SITA Maestro DCS": "Maestro",
            "SITA Smart Path Kiosks": "SITA Smart Path Check In Kiosk",
            "FlexBox": "SITA FlexBox",
            "Bag Fast": "SITA Bag Fast",
            "Bag Journey": "SITA Bag Journey",
            "Bag Manager": "SITA Bag Manager",
            "Bag Message": "SITA Bag Message"
        }

        if(product_tag in product_tag_mapping):
            product_tags = [product_tag_mapping[product_tag]]
        else:
            product_tags = [product_tag]

        if("smart path" in product_tag.lower()):
            prompt = """
            You are an expert in analyzing product names and suggest a product name if the file name matches any of the following product names and its missing in the list of product names, suggest only if the product name is present in the filename, if not able to find from file name and one more product tag exists, then leave as empty list, else, add the respective product mentioned below list

            If "SITA Smart Path" is present in the PRODUCT TAGS, suggest one of the following tags based on file name
                "SITA Smart Path Bag Drop",
                "Smart Path Biometrics (SITA Smart Path Hub)",
                "SITA Smart Path Check In Kiosk",
                "SITA Smart Path Scan & Fly",
                "SITA Smart Path Drop & Fly",
                "Smart Path Gates",
                "Smart Path Mobile"

            Note : If `FILE NAME` contains word TS6 - prefer adding tag "SITA Smart Path Check In Kiosk"

            FILE NAME : {file_name}
            PRODUCT TAGS : {product_tags}
            """
            structured_llm = AzureOpenAIService().get_client().with_structured_output(ProductTag)
            suggested_tags : ProductTag = await structured_llm.ainvoke(prompt.format(file_name = file_name, product_tags = product_tags[0]))
            product_tags.extend(suggested_tags.product_tags)
            product_tags = list(set(product_tags)) 

        return product_tags


    async def get_product_from_file(self,filepath) -> str:
        """
        Uses LLM to determine which product name is most relevant to the provided file path.

        Parameters:
        ----------
        file_path : str
            The full path of the file (e.g., 'Passenger Processing Collateral/SITA Smart Path Gates/UC-25Q1-Smart Path Gates Use Case.pdf').
        product_names : list
            List of known product names from SITA.

        Returns:
        -------
        str
            The predicted product name that best matches the file path.
        """

        file_path = str(filepath)
        product_names = [
        "API (Advanced Passenger Information)",
        "Bag Drop",
        "Bag Fast",
        "Bag Journey",
        "Bag Manager (V7)",
        "Bag Message",
        "Drop.Go",
        "FIDS (Flight Information Display System)",
        "Face Pods",
        "Flex Essentials",
        "Flex On Prem",
        "Flex as a Service",
        "Flex.Go",
        "FlexBox",
        "Kiosk",
        "Maestro",
        "Passenger Intelligent Insights",
        "Print.Go",
        "SD-WAN",
        "SITA A-CDM",
        "SITA Airport Management",
        "SITA Bag Fast",
        "SITA Bag Journey",
        "SITA Bag Manager",
        "SITA Bag Message",
        "SITA BlipTrack",
        "SITA Border Management",
        "SITA Business Talk",
        "SITA CUSS",
        "SITA Cargo Manager",
        "SITA Connect",
        "SITA Connect Global Messaging",
        "SITA Control Centre",
        "SITA Crew Management",
        "SITA Data Connect",
        "SITA Flex",
        "SITA Flex Hybrid",
        "SITA Flight Planning",
        "SITA In-flight Communications",
        "SITA Managed SBC",
        "SITA Mobile Resource Manager",
        "SITA OCS (Omnichannel Contact Services)",
        "SITA Passenger Flow",
        "SITA Pay",
        "SITA Queue Management",
        "SITA Smart Path",
        "SITA WorldTracer",
        "SITA e-freight",
        "SITATEX IP Resiliency",
        "SITATEX Online",
        "Smart Path Biometrics",
        "Smart Path Gates",
        "Smart Path Mobile",
        "WorldTracer Auto Notify",
        "WorldTracer Auto Reflight",
        "WorldTracer Bag Delivery Service",
        "WorldTracer Desktop",
        "WorldTracer Lost & Found Property",
        "WorldTracer Self Service",
        "WorldTracer Tablet"
        ]  # Assume this returns the product list
        product_names = str(product_names)

        system_prompt = """You are a product classification expert. Your task is to analyze a file path and determine which product name from a known list it most likely refers to.

    Instructions:
    1. Carefully examine the file path and extract any keywords or product references.
    2. Match these keywords against the list of known product names.
    3. Return all relevant product name from the list.

    Output Format: JSON — Directly give JSON response. Do not add any prefix or suffix like ```json or text blocks.

    {{
        "product": ["Predicted Product Name 1","Predicted Product Name 1"]
    }}
    """

        user_prompt = f"""Analyze the following file path and return the most relevant product name:

        **File Path:**
        {file_path}
    
        **Product Names:**
        {product_names}
    
        Return the product names that best matches the file path.
    
        Output Format: JSON — Directly give JSON response. Do not add any prefix or suffix like ```json or text blocks.
        product : predicted product names list
        """

        prompt_template = ChatPromptTemplate.from_messages([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content":  "{input}"}
        ])

        chain = prompt_template | AzureOpenAIService().get_client()
        response = await chain.ainvoke({"input": user_prompt})

        return json.loads(response.content).get('product')

