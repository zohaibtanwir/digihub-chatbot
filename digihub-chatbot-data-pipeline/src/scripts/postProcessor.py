from azure.cosmos import CosmosClient, exceptions
from src.utils.config import COSMOSDB_ENDPOINT, COSMOS_ACCOUNT_KEY, COSMOSDB_SERVICE_NAME_MAPPING_CONTAINER_NAME, DIGIHUB_DBNAME
from src.services.azure_openai_service import AzureOpenAIService
from src.services.embedding_service import AzureEmbeddingService
import sys
import json
import argparse


# This file is invoked from terminal: python -m src.chunk-filter-scripts.getChunkDetails <serviceNameid> <partitionKey> [--enrich-metadata]

DATABASE_NAME = "DigiHubChatBot"
CONTAINER_NAME = "dh-chatbot-documents"

# GPT-4o-mini deployment name - update this based on your Azure OpenAI deployment
GPT4O_MINI_DEPLOYMENT = "gpt-4o-mini"  # Replace with actual deployment name

# Container for acronym storage
ACRONYM_CONTAINER_NAME = "dh-chatbot-documents"

# Default subdomain if no mapping is found
DEFAULT_SUBDOMAIN = "UserGuide"

# Default SubDomain path mappings (fallback if CosmosDB is unavailable)
DEFAULT_SUBDOMAIN_PATH_MAPPINGS = {
    "WorldTracer/Customer Conference/": "Customer Conference",
    "WorldTracer/Performance Reporting/": "Performance Reporting",
    "WorldTracer/Trouble shooting/": "Trouble shooting",
    "Bag Manager/Announcements/": "Announcements",
}

# Cache for subdomain path mappings loaded from CosmosDB
_subdomain_path_mappings_cache = None


def load_subdomain_path_mappings() -> dict:
    """
    Load subdomain path mappings from Azure Cosmos DB.

    Returns:
        dict: Mapping of file path patterns to subdomain names
    """
    global _subdomain_path_mappings_cache

    if _subdomain_path_mappings_cache is not None:
        return _subdomain_path_mappings_cache

    try:
        client = CosmosClient(COSMOSDB_ENDPOINT, COSMOS_ACCOUNT_KEY)
        database = client.get_database_client(DIGIHUB_DBNAME)
        container = database.get_container_client(COSMOSDB_SERVICE_NAME_MAPPING_CONTAINER_NAME)

        response = container.read_item(
            item="subdomain_mappings",
            partition_key="keyword"
        )

        # Extract path_mappings from the document
        _subdomain_path_mappings_cache = response.get("path_mappings", DEFAULT_SUBDOMAIN_PATH_MAPPINGS)

        print(f"Loaded {len(_subdomain_path_mappings_cache)} subdomain path mappings from Cosmos DB")
        return _subdomain_path_mappings_cache

    except exceptions.CosmosResourceNotFoundError:
        print("subdomain_mappings document not found in Cosmos DB, using default mappings")
        _subdomain_path_mappings_cache = DEFAULT_SUBDOMAIN_PATH_MAPPINGS
        return _subdomain_path_mappings_cache
    except Exception as e:
        print(f"Error loading subdomain_mappings from Cosmos DB: {e}")
        _subdomain_path_mappings_cache = DEFAULT_SUBDOMAIN_PATH_MAPPINGS
        return _subdomain_path_mappings_cache


def get_subdomain_from_filepath(filepath: str) -> str:
    """
    Determine the subDomain based on file path.

    Args:
        filepath: The full file path of the source document

    Returns:
        The appropriate subDomain value based on path matching
    """
    if not filepath:
        return DEFAULT_SUBDOMAIN

    # Load subdomain path mappings (from CosmosDB or defaults)
    subdomain_path_mappings = load_subdomain_path_mappings()

    # Check each path pattern for a match
    for path_pattern, subdomain in subdomain_path_mappings.items():
        if path_pattern in filepath:
            return subdomain

    return DEFAULT_SUBDOMAIN


# Service line to content type mapping
SERVICE_LINE_CONTENT_TYPES = {
    "Airport Solutions": ["Others"],
    "World Tracer": ["Conference", "ProductDocumentation", "Report", "RCA", "Training", "TroubleShooting", "News", "WorkingGroups"],
    "Community Messaging KB": ["Others"],
    "Airport Committee": ["MeetingNotes"],
    "Operational Support": ["UserGuide"],
    "Bag Manager": ["Report", "UserGuide", "ProductDocumentation", "Conference", "Announcements"],
    "Euro Customer Advisory Board": ["CAB", "Survey", "Report"],
    "Billing": ["Survey", "DigiHubUserGuide", "BillingUpdate", "Invoice"],
    "Airport Management Solution": ["Others"],
    "SITA AeroPerformance": ["Others"],
    "APAC Customer Advisory Board": ["CAB", "Survey", "Report", "MeetingMinutes"],
    "General Info": ["DigiHubUserGuide", "ProductDocumentation"],
}


def generate_question_embeddings(questions: list) -> list:
    """
    Generate embeddings for a list of questions.

    Args:
        questions: List of question strings

    Returns:
        List of embedding vectors (one per question)
    """
    if not questions:
        return []

    try:
        embedding_service = AzureEmbeddingService()
        embeddings = embedding_service.get_embeddings()

        question_embeddings = []
        for question in questions:
            embedding = embeddings.embed_query(question)
            question_embeddings.append(embedding)

        print(f"  Generated {len(question_embeddings)} embeddings for questions")
        return question_embeddings
    except Exception as e:
        print(f"  Error generating question embeddings: {e}")
        return []


def generate_combined_question_embedding(questions: list) -> list:
    """
    Generate a single embedding for all questions combined.

    This allows CosmosDB to perform native vector search on questions,
    enabling question-first retrieval instead of content-first.

    Args:
        questions: List of question strings

    Returns:
        Single embedding vector for the combined questions text
    """
    if not questions:
        return []

    try:
        # Combine all questions into a single text
        combined_text = " ".join(questions)

        embedding_service = AzureEmbeddingService()
        embeddings = embedding_service.get_embeddings()

        combined_embedding = embeddings.embed_query(combined_text)
        print(f"  Generated combined embedding for {len(questions)} questions")
        return combined_embedding
    except Exception as e:
        print(f"  Error generating combined question embedding: {e}")
        return []


def classify_chunk_metadata(content: str, file_path: str = None, service_name: str = None, heading: str = None) -> dict:
    """
    Use GPT-4o-mini to extract metadata from chunk content.

    Args:
        content: The text content to analyze
        file_path: The full file path of the source document
        service_name: The service line name for service-specific content type classification
        heading: The heading/title of the chunk section

    Returns:
        Dictionary containing contentType, year, month, validChunk, questions, and products
    """
    # Get service-line-specific content types or use default
    content_types = SERVICE_LINE_CONTENT_TYPES.get(service_name, ["Others"])
    # Always include "Others" as a fallback option
    if "Others" not in content_types:
        content_types = content_types + ["Others"]
    content_type_options = "|".join(content_types)

    # Build file path context if available
    file_path_context = ""
    if file_path:
        file_path_context = f"""
File Path: {file_path}
(Use this file path to help identify the year, content type, or other metadata. File names often contain dates like "2023" or "2024", or content indicators like "Report", "UserGuide", "Minutes", etc.)
"""

    # Build heading context if available and meaningful
    heading_context = ""
    # List of generic/meaningless heading words to ignore
    generic_headings = ["untitled", "none", "n/a", "na", "heading", "title", "section", "chapter", "page", "document", "content", "text", "body", "main"]
    if heading and heading.strip():
        heading_lower = heading.strip().lower()
        print("##Heading",heading_lower)
        # Check if heading is meaningful (not generic)
        is_generic = heading_lower in generic_headings or len(heading_lower) < 3
        if not is_generic:
            heading_context = f"""
Heading/Title: {heading} - This heading provides important context about the section topic. Give appropriate weightage to this heading when generating questions, as it often contains key subject matter that SITA users would reference in their queries.
"""

    # Service-specific question generation instructions
    base_question_instructions = """Generate 2-3 specific questions that SITA users would realistically ask about this content.

   IMPORTANT GUIDELINES FOR QUESTION GENERATION:
   - Questions should mimic EXACT queries that SITA users would ask about Products and Services
   - Use natural language that matches how real users phrase questions (e.g., "How do I...", "What is...", "Why does...", "How to configure...")
   - If the heading contains a meaningful product name, feature, or topic, incorporate it into the questions, contents are in markdown format and titles are marked with # symbols
   - Focus on practical, actionable questions that users seeking help would ask
   - Avoid overly generic questions - be specific to the content and context
   - Questions should be answerable based ONLY on the provided text content"""

    if service_name == "Operational Support":
        question_instructions = f"""{base_question_instructions}
   ADDITIONAL: Since this is DigiHub user guide documentation, include "DigiHub" in the questions where relevant (e.g., "How do I configure X in DigiHub?" instead of just "How do I configure X?")."""
    else:
        question_instructions = base_question_instructions

    prompt = f"""You are an expert document analyzer. Your task is to extract specific metadata from the provided text and format it as a JSON object.
{file_path_context}
Instructions:
{heading_context}
Analyze the text provided and determine the following fields:

1. contentType: Classify the document into one of these categories: {content_type_options}.
   Choose the most appropriate category based on the content and file path.

2. year: Extract the four-digit year (YYYY) the document was created or refers to. Check both the text content AND the file path for year indicators. If not found, return null.

3. month: Extract the month the document was created or refers to. If not found, return null.

4. validChunk: Return "yes" if the text contains meaningful, coherent information relevant to the content type. Return "no" if the text is fragmented, contains only gibberish, table of contents or lacks enough context to be useful.

5. questions: {question_instructions}

6. products: Extract all SITA products/service lines mentioned in the text. Return an array of matching products from this list:
   - Airport Solutions
   - WorldTracer
   - Community Messaging KB
   - Airport Committee
   - Operational Support
   - Bag Manager
   - Euro Customer Advisory Board
   - Billing
   - Airport Management Solution
   - SITA AeroPerformance
   - APAC Customer Advisory Board
   - General Info

   Only include products that are explicitly mentioned or clearly referenced in the text. Return empty array if none found.

Text to analyze:
---
{content}
---

Return ONLY a valid JSON object with the following structure:
{{
  "contentType": "{content_type_options}",
  "year": "YYYY or null",
  "month": "Month name or null",
  "validChunk": "yes|no",
  "questions": ["question1", "question2", "question3"],
  "products": ["product1", "product2"]
}}"""

    try:
        openai_service = AzureOpenAIService()
        client = openai_service.get_client()

        response = client.chat.completions.create(
            model=GPT4O_MINI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are a document metadata extraction expert. Always return valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )

        result_text = response.choices[0].message.content.strip()

        # Parse JSON response
        # Remove markdown code blocks if present
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.startswith("```"):
            result_text = result_text[3:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]

        metadata = json.loads(result_text.strip())

        # Validate and clean the response
        return {
            "contentType": metadata.get("contentType", "Others"),
            "year": metadata.get("year"),
            "month": metadata.get("month"),
            "validChunk": metadata.get("validChunk", "yes"),
            "questions": metadata.get("questions", []),
            "products": metadata.get("products", [])
        }
    except Exception as e:
        print(f"Error classifying chunk metadata: {e}")
        return {
            "contentType": "Others",
            "year": None,
            "month": None,
            "validChunk": "yes",
            "questions": [],
            "products": []
        }


def extract_acronyms_from_content(content: str, metadata: dict = None) -> dict:

    """
    Use GPT-4o-mini to extract acronyms and their definitions from chunk content.

    Args:
        content: The text content to analyze for acronyms

    Returns:
        Dictionary mapping acronyms to their definitions (e.g., {"AFRAA": "African Airlines Association"})
    """
    prompt = f"""You are an expert at extracting acronyms and their definitions from text.

Analyze the provided text and extract ALL acronyms along with their full forms/definitions.

Instructions:
1. Look for patterns like "ABC (Full Form)" or "Full Form (ABC)" or "ABC - Full Form" or "ABC: Full Form"
2. Also identify acronyms that are defined in context (e.g., "The African Airlines Association, commonly known as AFRAA...")
3. Only include acronyms that have clear definitions in the text
4. Do NOT include generic words or terms that are not acronyms
5. Acronyms are typically 2-10 capital letters (some may have numbers)

Text to analyze:
---
{content}
---

Return ONLY a valid JSON object with acronyms as keys and their definitions as values.
Example format:
{{
  "AFRAA": "African Airlines Association",
  "PAX": "Passenger",
  "IATA": "International Air Transport Association"
}}

If no acronyms with definitions are found, return an empty object: {{}}"""

    try:
        openai_service = AzureOpenAIService()
        client = openai_service.get_client()

        response = client.chat.completions.create(
            model=GPT4O_MINI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are an acronym extraction expert. Always return valid JSON with acronyms as keys and definitions as values."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=5000
        )

        result_text = response.choices[0].message.content.strip()

        # Remove markdown code blocks if present
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.startswith("```"):
            result_text = result_text[3:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]

        acronyms = json.loads(result_text.strip())
        return acronyms if isinstance(acronyms, dict) else {}

    except Exception as e:
        print(f"Error extracting acronyms: {e}")
        return {}


def store_acronyms_to_cosmos(acronyms: dict, container):
    """
    Store extracted acronyms to CosmosDB in key-value format.

    Each acronym is stored as a separate document with:
    - partitionKey: "acronym"
    - id: the acronym (e.g., "AFRAA")
    - value: the definition (e.g., "African Airlines Association")
    - metadata: source chunk metadata (filepath, filename, heading)

    Args:
        acronyms: Dictionary mapping acronyms to {value, metadata}
        container: CosmosDB container client
    """
    stored_count = 0
    updated_count = 0

    for acronym, acronym_data in acronyms.items():
        if not acronym or not acronym_data:
            continue

        # Handle both old format (string) and new format (dict with value and metadata)
        if isinstance(acronym_data, dict):
            definition = acronym_data.get("value", "")
            metadata = acronym_data.get("metadata", {})
        else:
            definition = acronym_data
            metadata = {}

        if not definition:
            continue

        # Normalize the acronym key (uppercase, strip whitespace)
        acronym_key = acronym.strip().upper()

        document = {
            "id": acronym_key,
            "partitionKey": "acronym",
            "value": definition.strip(),
            "type": "acronym",
            "metadata": metadata
        }

        try:
            # Try to read existing document
            try:
                existing = container.read_item(item=acronym_key, partition_key="acronym")
                # Document exists, check if we need to update
                if existing.get("value") != definition.strip():
                    container.upsert_item(document)
                    updated_count += 1
                    print(f"  ↻ Updated: {acronym_key} = {definition}")
                else:
                    print(f"  ○ Exists (unchanged): {acronym_key}")
            except Exception:
                # Document doesn't exist, create it
                container.create_item(document)
                stored_count += 1
                print(f"  ✓ Created: {acronym_key} = {definition}")

        except Exception as e:
            print(f"  ✗ Error storing {acronym_key}: {e}")

    return stored_count, updated_count


def extract_and_store_acronyms(service_name_id: int = 0, batch_size=100):
    """
    Extract acronyms from chunks and store them in CosmosDB using batch processing.

    Args:
        service_name_id: Service name ID to query (default 0 for General Info)
        batch_size: Number of items to process in each batch (default: 100)
    """
    client = CosmosClient(COSMOSDB_ENDPOINT, COSMOS_ACCOUNT_KEY)
    database = client.get_database_client(DATABASE_NAME)
    container = database.get_container_client(CONTAINER_NAME)

    # Query chunks from General Info (serviceNameid=0)
    query = """
    SELECT * FROM c
    WHERE c.serviceNameid = @serviceNameid
    """
    parameters = [
        {"name": "@serviceNameid", "value": int(service_name_id)}
    ]

    print(f"\nQuerying chunks for serviceNameid={service_name_id}...")
    results = container.query_items(
        query=query,
        parameters=parameters,
        enable_cross_partition_query=True
    )

    # Extract acronyms from all chunks using batch processing
    all_acronyms = {}
    total_processed = 0
    batch = []

    print("\nExtracting acronyms from chunks in batches...")

    for item in results:
        batch.append(item)
        total_processed += 1

        if len(batch) >= batch_size:
            # Process batch
            process_acronym_batch(batch, all_acronyms, total_processed - len(batch) + 1)
            batch = []  # Clear batch to free memory

    # Process remaining items
    if batch:
        process_acronym_batch(batch, all_acronyms, total_processed - len(batch) + 1)

    print(f"\n{'='*50}")
    print(f"Total chunks processed: {total_processed}")
    print(f"Total unique acronyms extracted: {len(all_acronyms)}")
    print(f"{'='*50}")

    if total_processed == 0:
        print("No chunks found. Exiting.")
        return

    if not all_acronyms:
        print("No acronyms found to store.")
        return

    # Store acronyms to CosmosDB
    print("\nStoring acronyms to CosmosDB...")
    stored, updated = store_acronyms_to_cosmos(all_acronyms, container)

    print(f"\n{'='*50}")
    print(f"Summary:")
    print(f"  - New acronyms created: {stored}")
    print(f"  - Existing acronyms updated: {updated}")
    print(f"  - Total in database: {stored + updated}")
    print(f"{'='*50}")

    # Save extracted acronyms to JSON file for reference
    output_file = f"acronyms_extracted_SL{service_name_id}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_acronyms, f, indent=2, ensure_ascii=False)
    print(f"\nAcronyms also saved to: {output_file}")


def process_acronym_batch(batch, all_acronyms, start_idx):
    """
    Process a batch of items for acronym extraction.

    Args:
        batch: List of items to process
        all_acronyms: Dictionary to accumulate all acronyms
        start_idx: Starting index for progress display
    """
    batch_size = len(batch)
    print(f"\nProcessing batch: items {start_idx} to {start_idx + batch_size - 1}")

    for idx, item in enumerate(batch, 1):
        global_idx = start_idx + idx - 1
        content = item.get('content', '')
        chunk_id = item.get('id', 'unknown')
        chunk_metadata = item.get('metadata', {})

        if not content:
            continue

        print(f"Processing chunk {global_idx}: {chunk_id}")
        chunk_acronyms = extract_acronyms_from_content(content)

        if chunk_acronyms:
            print(f"  Found {len(chunk_acronyms)} acronyms: {list(chunk_acronyms.keys())}")
            # Merge with existing, preferring longer definitions
            for key, value in chunk_acronyms.items():
                existing_entry = all_acronyms.get(key.upper(), {})
                existing_value = existing_entry.get("value", "") if isinstance(existing_entry, dict) else existing_entry
                if len(value) > len(existing_value):
                    all_acronyms[key.upper()] = {
                        "value": value,
                        "metadata": chunk_metadata
                    }

    print(f"Batch completed: {batch_size} items processed")


def format_chunk_as_markdown(chunk):
    """Format a single chunk document as markdown."""
    md_lines = []

    md_lines.append("---")
    md_lines.append("")
    md_lines.append(f"## Chunk ID: {chunk.get('id', 'N/A')}")
    md_lines.append("")
    md_lines.append("### Basic Information")
    md_lines.append(f"- **Partition Key**: {chunk.get('partitionKey', 'N/A')}")
    md_lines.append(f"- **Service Name**: {chunk.get('serviceName', 'N/A')}")
    md_lines.append(f"- **Service Name ID**: {chunk.get('serviceNameid', 'N/A')}")
    md_lines.append(f"- **Heading**: {chunk.get('heading', 'N/A')}")
    md_lines.append("")

    md_lines.append("### Content")
    md_lines.append("```")
    md_lines.append(chunk.get('content', 'N/A'))
    md_lines.append("```")
    md_lines.append("")

    # Metadata section
    metadata = chunk.get('metadata', {})
    if metadata:
        md_lines.append("### Metadata")
        md_lines.append(f"- **File Path**: {metadata.get('filepath', 'N/A')}")
        md_lines.append(f"- **File Name**: {metadata.get('filename', 'N/A')}")
        md_lines.append(f"- **Heading**: {metadata.get('heading', 'N/A')}")
        md_lines.append("")

    # Additional fields if they exist
    if 'product' in chunk:
        md_lines.append("### Product")
        md_lines.append(f"- {', '.join(chunk['product']) if isinstance(chunk['product'], list) else chunk['product']}")
        md_lines.append("")

    # Enriched metadata section
    has_enriched_metadata = any(key in chunk for key in ['contentType', 'year', 'month', 'validChunk', 'questions', 'products', 'subDomain'])
    if has_enriched_metadata:
        md_lines.append("### Enriched Metadata")
        if 'subDomain' in chunk:
            md_lines.append(f"- **Sub Domain**: {chunk.get('subDomain', 'N/A')}")
        if 'contentType' in chunk:
            md_lines.append(f"- **Content Type**: {chunk.get('contentType', 'N/A')}")
        if 'year' in chunk:
            md_lines.append(f"- **Year**: {chunk.get('year', 'N/A')}")
        if 'month' in chunk:
            md_lines.append(f"- **Month**: {chunk.get('month', 'N/A')}")
        if 'validChunk' in chunk:
            valid_status = chunk.get('validChunk', 'N/A')
            valid_emoji = "✅" if valid_status == "yes" else "❌" if valid_status == "no" else ""
            md_lines.append(f"- **Valid Chunk**: {valid_status} {valid_emoji}")
        if 'products' in chunk and chunk.get('products'):
            md_lines.append(f"- **Products**: {', '.join(chunk['products'])}")
        md_lines.append("")

        if 'questions' in chunk and chunk.get('questions'):
            md_lines.append("### Generated Questions")
            for idx, q in enumerate(chunk.get('questions', []), 1):
                md_lines.append(f"{idx}. {q}")
            md_lines.append("")

            # Show question embeddings info if available
            if 'questionEmbeddings' in chunk and chunk.get('questionEmbeddings'):
                embeddings_count = len(chunk.get('questionEmbeddings', []))
                md_lines.append(f"**Question Embeddings**: {embeddings_count} individual embeddings generated")
            if 'questionsEmbedding' in chunk and chunk.get('questionsEmbedding'):
                md_lines.append(f"**Combined Questions Embedding**: Yes (single vector for all questions)")
            md_lines.append("")

    md_lines.append("")
    md_lines.append("### Raw JSON")
    md_lines.append("```json")
    # Create a copy without embedding fields for cleaner output
    chunk_copy = {k: v for k, v in chunk.items() if k not in ['embedding', 'questionEmbeddings', 'questionsEmbedding']}
    # Add info about excluded fields
    if 'embedding' in chunk or 'questionEmbeddings' in chunk or 'questionsEmbedding' in chunk:
        excluded_fields = []
        if 'embedding' in chunk:
            excluded_fields.append('embedding')
        if 'questionEmbeddings' in chunk:
            excluded_fields.append(f'questionEmbeddings ({len(chunk.get("questionEmbeddings", []))} vectors)')
        if 'questionsEmbedding' in chunk:
            excluded_fields.append('questionsEmbedding (combined vector)')
        chunk_copy['_excluded_fields'] = excluded_fields
    md_lines.append(json.dumps(chunk_copy, indent=2))
    md_lines.append("```")
    md_lines.append("")

    return "\n".join(md_lines)


def remove_valid_chunk_field(service_name_id, partition_key=None, batch_size=100):
    """
    Remove the validChunk field from CosmosDB documents using batch processing.

    Args:
        service_name_id: Service name ID to filter by. Use "all" to process ALL documents.
        partition_key: Optional partition key to filter by specific document.
        batch_size: Number of items to process in each batch (default: 100)
    """
    client = CosmosClient(COSMOSDB_ENDPOINT, COSMOS_ACCOUNT_KEY)
    database = client.get_database_client(DATABASE_NAME)
    container = database.get_container_client(CONTAINER_NAME)

    # Build query based on parameters
    if service_name_id == "all":
        query = """
        SELECT * FROM c
        WHERE IS_DEFINED(c.validChunk)
        """
        parameters = []
        print(f"\nQuerying ALL chunks with validChunk field...")
    elif partition_key:
        query = """
        SELECT * FROM c
        WHERE c.serviceNameid = @serviceNameid
        AND c.partitionKey = @partitionKey
        AND IS_DEFINED(c.validChunk)
        """
        parameters = [
            {"name": "@serviceNameid", "value": int(service_name_id)},
            {"name": "@partitionKey", "value": partition_key}
        ]
        print(f"\nQuerying chunks for serviceNameid={service_name_id}, partitionKey={partition_key}...")
    else:
        query = """
        SELECT * FROM c
        WHERE c.serviceNameid = @serviceNameid
        AND IS_DEFINED(c.validChunk)
        """
        parameters = [
            {"name": "@serviceNameid", "value": int(service_name_id)}
        ]
        print(f"\nQuerying chunks for serviceNameid={service_name_id}...")

    results = container.query_items(
        query=query,
        parameters=parameters,
        enable_cross_partition_query=True
    )

    # Process in batches to avoid OOM
    updated_count = 0
    failed_count = 0
    total_count = 0
    batch = []

    print("\nProcessing chunks in batches to avoid OOM...")

    for item in results:
        batch.append(item)
        total_count += 1

        if len(batch) >= batch_size:
            # Process batch
            batch_updated, batch_failed = process_removal_batch(batch, container, total_count - len(batch) + 1)
            updated_count += batch_updated
            failed_count += batch_failed
            batch = []  # Clear batch to free memory

    # Process remaining items
    if batch:
        batch_updated, batch_failed = process_removal_batch(batch, container, total_count - len(batch) + 1)
        updated_count += batch_updated
        failed_count += batch_failed

    print(f"\n{'='*50}")
    print(f"Summary:")
    print(f"  - Total chunks processed: {total_count}")
    print(f"  - Successfully updated: {updated_count}")
    print(f"  - Failed: {failed_count}")
    print(f"{'='*50}")

    if total_count == 0:
        print("No chunks found with validChunk field. Nothing to remove.")


def process_removal_batch(batch, container, start_idx):
    """
    Process a batch of items for validChunk field removal.

    Returns: (updated_count, failed_count)
    """
    batch_size = len(batch)
    print(f"\nProcessing batch: items {start_idx} to {start_idx + batch_size - 1}")

    updated = 0
    failed = 0

    for idx, item in enumerate(batch, 1):
        global_idx = start_idx + idx - 1
        chunk_id = item.get('id', 'unknown')
        valid_chunk_value = item.get('validChunk', 'N/A')

        print(f"Processing chunk {global_idx}: {chunk_id} (validChunk={valid_chunk_value})")

        # Remove the validChunk field
        if 'validChunk' in item:
            del item['validChunk']

        # Update in CosmosDB
        try:
            container.upsert_item(item)
            updated += 1
            print(f"  ✓ Removed validChunk field from chunk {chunk_id}")
        except Exception as e:
            failed += 1
            print(f"  ✗ Error updating chunk {chunk_id}: {e}")

    print(f"Batch completed: {updated} updated, {failed} failed")
    return updated, failed


def query_and_generate_md(service_name_id, partition_key=None, enrich_metadata=False, batch_size=100):
    """
    Query chunks from CosmosDB and generate markdown report with batch processing.

    Args:
        service_name_id: The serviceNameid to filter by
        partition_key: Optional partition key to filter by specific document
        enrich_metadata: If True, use GPT-4o-mini to classify and enrich chunks with metadata
        batch_size: Number of items to process in each batch (default: 100)
    """
    client = CosmosClient(COSMOSDB_ENDPOINT, COSMOS_ACCOUNT_KEY)
    database = client.get_database_client(DATABASE_NAME)
    container = database.get_container_client(CONTAINER_NAME)

    # Build query based on parameters
    if partition_key:
        query = """
        SELECT * FROM c
        WHERE c.serviceNameid = @serviceNameid
        AND c.partitionKey = @partitionKey
        AND NOT IS_DEFINED(c.validChunk)
        """
        parameters = [
            {"name": "@serviceNameid", "value": int(service_name_id)},
            {"name": "@partitionKey", "value": partition_key}
        ]
        output_filename = f"chunks_SL{service_name_id}_{partition_key.replace('/', '_')}_report.md"
    else:
        query = """
        SELECT * FROM c
        WHERE c.serviceNameid = @serviceNameid
        AND NOT IS_DEFINED(c.validChunk)
        """
        parameters = [
            {"name": "@serviceNameid", "value": int(service_name_id)}
        ]
        output_filename = f"chunks_SL{service_name_id}_report.md"

    results = container.query_items(
        query=query,
        parameters=parameters,
        enable_cross_partition_query=True
    )

    # Process in batches to avoid OOM
    total_processed = 0
    batch = []

    # Statistics tracking
    content_types = {}
    valid_chunks = {"yes": 0, "no": 0}
    years_found = set()
    service_names = set()
    partition_keys = set()

    # Open file once and write incrementally
    with open(output_filename, "w", encoding="utf-8") as f:
        # Write header (will update total count later)
        f.write(f"# Chunk Details Report\n\n")
        f.write(f"**Service Name ID**: {service_name_id}\n\n")
        if partition_key:
            f.write(f"**Partition Key**: {partition_key}\n\n")

        # Placeholder for total count (will be updated at the end)
        f.write(f"**Total Chunks Found**: [Counting...]\n\n")

        if enrich_metadata:
            f.write(f"**Metadata Enriched**: Yes (using GPT-4o-mini)\n\n")

        # Reserve space for summary (will be written at the end)
        f.write("\n---\n\n")
        f.write("## Chunk Details\n\n")

        # Process results in batches
        for item in results:
            batch.append(item)

            # Collect statistics
            service_names.add(item.get('serviceName'))
            partition_keys.add(item.get('partitionKey'))

            if len(batch) >= batch_size:
                # Process batch
                total_processed += process_batch(
                    batch, container, enrich_metadata, f, total_processed,
                    content_types, valid_chunks, years_found
                )
                batch = []  # Clear batch to free memory

        # Process remaining items
        if batch:
            total_processed += process_batch(
                batch, container, enrich_metadata, f, total_processed,
                content_types, valid_chunks, years_found
            )

    # Update file with summary statistics
    if total_processed > 0:
        prepend_summary_to_file(
            output_filename, total_processed, service_names, partition_keys,
            enrich_metadata, content_types, valid_chunks, years_found
        )
    else:
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(f"# Chunk Details Report\n\n")
            f.write(f"**Service Name ID**: {service_name_id}\n\n")
            if partition_key:
                f.write(f"**Partition Key**: {partition_key}\n\n")
            f.write(f"\nNo chunks found for the given criteria.\n")

    print(f"\nSuccessfully generated {output_filename}")
    print(f"Total chunks retrieved: {total_processed}")


def process_batch(batch, container, enrich_metadata, file_handle, start_idx, content_types, valid_chunks, years_found):
    """
    Process a batch of items and write to file immediately.

    Returns: Number of items processed
    """
    batch_size = len(batch)
    print(f"\nProcessing batch: items {start_idx + 1} to {start_idx + batch_size}")

    for idx, item in enumerate(batch, 1):
        global_idx = start_idx + idx

        if enrich_metadata:
            content = item.get('content', '')
            if content:
                print(f"Processing chunk {global_idx}: {item.get('id', 'unknown')}")

                # Get file path, service name, and heading for context
                file_path = item.get('metadata', {}).get('filepath', '')
                service_name = item.get('serviceName', '')
                heading = item.get('heading', '') or item.get('metadata', {}).get('heading', '')

                # Get metadata from GPT-4o-mini
                metadata = classify_chunk_metadata(content, file_path, service_name, heading)
                print(json.dumps(metadata, indent=4))

                # Update the item with new metadata
                item['contentType'] = metadata['contentType']
                item['year'] = metadata['year']
                item['month'] = metadata['month']
                item['validChunk'] = metadata['validChunk']
                item['questions'] = metadata['questions']
                item['products'] = metadata['products']

                # Set subDomain based on file path
                item['subDomain'] = get_subdomain_from_filepath(file_path)
                print(f"  SubDomain: {item['subDomain']} (from path: {file_path})")

                # Generate embeddings for questions
                if metadata['questions']:
                    # Generate individual question embeddings (keep for fallback/debugging)
                    question_embeddings = generate_question_embeddings(metadata['questions'])
                    item['questionEmbeddings'] = question_embeddings

                    # Generate combined question embedding (for native CosmosDB vector search)
                    combined_embedding = generate_combined_question_embedding(metadata['questions'])
                    item['questionsEmbedding'] = combined_embedding
                else:
                    item['questionEmbeddings'] = []
                    item['questionsEmbedding'] = []

                # Update in CosmosDB
                try:
                    container.upsert_item(item)
                    print(f"  ✓ Updated chunk {item.get('id')} with metadata, {len(item.get('questionEmbeddings', []))} question embeddings, and combined embedding")
                except Exception as e:
                    print(f"  ✗ Error updating chunk {item.get('id')}: {e}")
            else:
                print(f"Skipping chunk {global_idx}: No content found")

        # Collect statistics
        ct = item.get('contentType', 'Others')
        content_types[ct] = content_types.get(ct, 0) + 1

        vc = item.get('validChunk', 'yes')
        if vc in valid_chunks:
            valid_chunks[vc] += 1

        year = item.get('year')
        if year:
            years_found.add(str(year))

        # Write to file immediately
        file_handle.write(format_chunk_as_markdown(item))
        if idx < batch_size:
            file_handle.write("\n")

    print(f"Batch completed: {batch_size} items processed")
    return batch_size


def prepend_summary_to_file(filename, total_count, service_names, partition_keys,
                            enrich_metadata, content_types, valid_chunks, years_found):
    """
    Read the file, update summary section, and rewrite.
    This is done once at the end after all batches are processed.
    """
    # Read the existing content
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract service_name_id and partition_key from content
    lines = content.split('\n')
    service_name_id = None
    partition_key = None

    for line in lines[:10]:  # Check first 10 lines
        if '**Service Name ID**:' in line:
            service_name_id = line.split(':')[-1].strip()
        if '**Partition Key**:' in line:
            partition_key = line.split(':')[-1].strip()

    # Build new header with summary
    header_lines = []
    header_lines.append(f"# Chunk Details Report\n")
    header_lines.append(f"**Service Name ID**: {service_name_id}\n")
    if partition_key:
        header_lines.append(f"**Partition Key**: {partition_key}\n")
    header_lines.append(f"\n**Total Chunks Found**: {total_count}\n")

    if enrich_metadata:
        header_lines.append(f"**Metadata Enriched**: Yes (using GPT-4o-mini)\n")

    # Add summary section
    header_lines.append("\n## Summary\n\n")

    service_names_clean = {s for s in service_names if s}
    if service_names_clean:
        header_lines.append(f"- **Service Names**: {', '.join(service_names_clean)}\n")

    partition_keys_clean = {p for p in partition_keys if p}
    if partition_keys_clean:
        header_lines.append(f"- **Unique Documents**: {len(partition_keys_clean)}\n")

    # Add metadata statistics if enriched
    if enrich_metadata and content_types:
        header_lines.append("\n### Metadata Statistics\n\n")
        header_lines.append(f"- **Content Types**:\n")
        for ct, count in sorted(content_types.items(), key=lambda x: x[1], reverse=True):
            header_lines.append(f"  - {ct}: {count}\n")

        header_lines.append(f"\n- **Valid Chunks**: {valid_chunks['yes']} valid, {valid_chunks['no']} invalid\n")

        if years_found:
            header_lines.append(f"- **Years Found**: {', '.join(sorted(years_found))}\n")

    header_lines.append("\n---\n\n")
    header_lines.append("## Chunk Details\n\n")

    # Find where "## Chunk Details" starts in original content
    chunk_details_start = content.find("## Chunk Details")
    if chunk_details_start != -1:
        # Skip the header line itself
        chunk_content_start = content.find("\n\n", chunk_details_start) + 2
        chunks_content = content[chunk_content_start:]
    else:
        chunks_content = ""

    # Write the new file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(''.join(header_lines))
        f.write(chunks_content)


if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="Query chunks from CosmosDB and generate markdown report with optional metadata enrichment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate report for service line 240
  python -m src.chunk-filter-scripts.getChunkDetails 240

  # Generate report for specific partition key
  python -m src.chunk-filter-scripts.getChunkDetails 240 'WorldTracer-Agenda2019_InternetMinutesfinal.docx'

  # Enrich chunks with metadata using GPT-4o-mini and update CosmosDB
  python -m src.chunk-filter-scripts.getChunkDetails 240 --enrich-metadata

  # Enrich specific document
  python -m src.chunk-filter-scripts.getChunkDetails 240 'WorldTracer-Agenda2019_InternetMinutesfinal.docx' --enrich-metadata

  # Enrich with smaller batch size for large datasets (reduces memory usage)
  python -m src.chunk-filter-scripts.getChunkDetails 240 --enrich-metadata --batch-size 50

  # Extract acronyms from General Info (serviceNameid=0) and store in CosmosDB
  python -m src.chunk-filter-scripts.getChunkDetails 0 --extract-acronyms

  # Extract acronyms from a different service line
  python -m src.chunk-filter-scripts.getChunkDetails 240 --extract-acronyms

  # Remove validChunk field from ALL documents
  python -m src.chunk-filter-scripts.getChunkDetails all --remove-valid-chunk

  # Remove validChunk field from specific service line with custom batch size
  python -m src.chunk-filter-scripts.getChunkDetails 240 --remove-valid-chunk --batch-size 50

  # Remove validChunk field from specific document
  python -m src.chunk-filter-scripts.getChunkDetails 240 'WorldTracer-Agenda2019_InternetMinutesfinal.docx' --remove-valid-chunk
        """
    )

    parser.add_argument(
        'service_name_id',
        type=str,
        help='Service name ID to filter chunks'
    )

    parser.add_argument(
        'partition_key',
        type=str,
        nargs='?',
        default=None,
        help='Optional partition key to filter by specific document'
    )

    parser.add_argument(
        '--enrich-metadata',
        action='store_true',
        help='Enrich chunks with metadata (contentType, year, month, validChunk, questions, products) using GPT-4o-mini and update CosmosDB'
    )

    parser.add_argument(
        '--extract-acronyms',
        action='store_true',
        help='Extract acronyms from General Info chunks (serviceNameid=0) and store them in CosmosDB with partitionKey="acronym"'
    )

    parser.add_argument(
        '--remove-valid-chunk',
        action='store_true',
        help='Remove the validChunk field from all documents. Use service_name_id="all" to process all documents, or specify a service_name_id to filter by service line.'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Number of items to process in each batch (default: 100). Reduce this if you encounter memory issues.'
    )

    args = parser.parse_args()

    try:
        if args.extract_acronyms:
            # Extract acronyms mode - uses serviceNameid=0 (General Info) by default
            service_id = int(args.service_name_id) if args.service_name_id != '0' else 0
            extract_and_store_acronyms(service_id, args.batch_size)
        elif args.remove_valid_chunk:
            # Remove validChunk field mode
            remove_valid_chunk_field(
                args.service_name_id,
                args.partition_key,
                args.batch_size
            )
        else:
            query_and_generate_md(
                args.service_name_id,
                args.partition_key,
                args.enrich_metadata,
                args.batch_size
            )
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
