
# Unique list of Service line

1. Airport Solutions
2. World Tracer
3. Community Messaging KB
4. Airport Committee
5. Operational Support
6. Bag Manager
7. Euro Customer Advisory Board
8. Billing
9. Airport Management Solution
10. SITA AeroPerformance
11. APAC Customer Advisory Board
12. General Info

# Filter the chunks

## Current Data storage structure
```
{
    "id": "chunk-6",
    "partitionKey": "WorldTracer-Agenda2019_InternetMinutesfinal.docx",
    "serviceName": "WorldTracer",
    "serviceNameid": 240,
    "heading": "6 Close of Meeting",
    "content": "## 6 Close of Meeting",
    "embedding":[],
    "metadata": {
        "filepath": "WorldTracer/Customer Conference/2019 Customer Conference Bangkok/Agenda Item Minutes/Agenda 2019_Internet Minutes final.docx",
        "heading": "6 Close of Meeting",
        "filename": "Agenda 2019_Internet Minutes final.docx"
    }
  "product": ["DigiHub", "Billing"],
  "contentType": "UserGuide/Marketing/MeetingMinutes/ReleaseNotes/APIDocs/Others",
  "year": "2024",
  "month": "January",
  "validChunk": "yes/no",
  "questions": ["Question 1?", "Question 2?", "Question 3?"],
  "questionEmbeddings": [
    [0.123, 0.456, ...],  // Embedding vector for Question 1
    [0.789, 0.012, ...],  // Embedding vector for Question 2
    [0.345, 0.678, ...]   // Embedding vector for Question 3
  ],
  "questionsEmbedding": [0.234, 0.567, ...]  // Single combined embedding for all questions (for native CosmosDB vector search)
}
```

### Embedding Fields Explained

| Field | Type | Purpose |
|-------|------|---------|
| `embedding` | Single vector | Content embedding for content-based retrieval |
| `questionEmbeddings` | Array of vectors | Individual embeddings per question (kept for fallback/debugging) |
| `questionsEmbedding` | Single vector | Combined embedding of all questions concatenated (enables native question-first CosmosDB vector search) |

# DB DetailsF
Azure cosmos with Vector
## dh-chatbot-sharepoint-sync - Keeps all the folderName under this

```
{
    "id": "01LU75EAXTJBRK5QFPM5GJHX5QEKTO6EME",
    "foldername": "Billing",
    "pathwithfilename": "Billing/Billing Updates/2025/09_2025_Billing Update.pdf",
    "dateofcreated": "2025-11-06T08:35:01Z",
    "lastModifiedDateTime": "2025-11-06T08:35:01Z",
    "dateofprocessed": "2025-11-07T09:10:14.361247",
    "processedstatus": "Processed",
    "data_extracted": 6,
    "_rid": "i5hFAOApuUDnBAAAAAAAAA==",
    "_self": "dbs/i5hFAA==/colls/i5hFAOApuUA=/docs/i5hFAOApuUDnBAAAAAAAAA==/",
    "_etag": "\"680129bc-0000-0d00-0000-690db7950000\"",
    "_attachments": "attachments/",
    "_ts": 1762506645
}
```

## Contianer 2- dh-chatbot-documents

dh-chatbot-sharepoint-sync.foldername = serviceName

Chunks of the above documents

```
{
    "id": "chunk-6",
    "partitionKey": "WorldTracer-Agenda2019_InternetMinutesfinal.docx",
    "serviceName": "WorldTracer",
    "serviceNameid": 240,
    "heading": "6 Close of Meeting",
    "content": "## 6 Close of Meeting",
    "embedding":[],
    "metadata": {
        "filepath": "WorldTracer/Customer Conference/2019 Customer Conference Bangkok/Agenda Item Minutes/Agenda 2019_Internet Minutes final.docx",
        "heading": "6 Close of Meeting",
        "filename": "Agenda 2019_Internet Minutes final.docx"
    }
}

```

## Acronym Storage Structure

Acronyms are stored as separate documents in the same container with a special format for fast key-value lookups:

```json
{
    "id": "AFRAA",
    "partitionKey": "acronym",
    "value": "African Airlines Association",
    "type": "acronym"
}
```

### Acronym Document Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | The acronym key in uppercase (e.g., "AFRAA", "PAX", "IATA") |
| `partitionKey` | string | Always "acronym" for all acronym documents |
| `value` | string | The full definition/expansion of the acronym |
| `type` | string | Always "acronym" to identify document type |

### How Acronyms Are Used

When a user query is classified as an "Acronym" type (e.g., "What is AFRAA?", "What does PAX mean?"):
1. The query analyzer detects the acronym pattern
2. The system routes to `serviceNameid=0` (General Info)
3. Acronym documents can be queried directly by `partitionKey="acronym"`

# Usage: getChunkDetails.py

## Basic Usage - Generate Markdown Report Only

Query chunks by serviceNameid only (gets all chunks for that service):
```bash
python -m src.chunk-filter-scripts.getChunkDetails 240
```

Query chunks by both serviceNameid and partitionKey (specific document):
```bash
python -m src.chunk-filter-scripts.getChunkDetails 240 'WorldTracer-Agenda2019_InternetMinutesfinal.docx'
```

## Enhanced Usage - With Metadata Enrichment

**NEW FEATURE**: The script now supports automatic metadata enrichment using GPT-4o-mini model.

Enrich all chunks for a service line with metadata:
```bash
python -m src.chunk-filter-scripts.getChunkDetails 240 --enrich-metadata
```

Enrich chunks for a specific document:
```bash
python -m src.chunk-filter-scripts.getChunkDetails 240 'WorldTracer-Agenda2019_InternetMinutesfinal.docx' --enrich-metadata
```

### What the `--enrich-metadata` flag does:

1. **Reads content** from each chunk's `content` property
2. **Analyzes content** using GPT-4o-mini to extract:
   - `contentType`: UserGuide, Marketing, MeetingMinutes, ReleaseNotes, APIDocs, or Others
   - `year`: Four-digit year (YYYY) or null
   - `month`: Month name or null
   - `validChunk`: "yes" if meaningful content, "no" if fragmented/gibberish/TOC only
   - `questions`: 2-3 specific questions answerable from the content
3. **Generates embeddings** for questions using Azure OpenAI embedding service:
   - `questionEmbeddings`: Array with one embedding vector per question (for fallback/debugging)
   - `questionsEmbedding`: Single combined embedding of all questions concatenated (for native CosmosDB vector search)
4. **Updates CosmosDB** with the enriched metadata and question embeddings for each chunk
5. **Generates report** with metadata statistics and enriched fields

### Configuration

Before using the enrichment feature, update the deployment name in the script:

```python
# In getChunkDetails.py, line 13
GPT4O_MINI_DEPLOYMENT = "gpt-4o-mini"  # Replace with your actual Azure OpenAI deployment name
```

### Output

The generated markdown report includes:
- Summary statistics with metadata distribution
- Content type breakdown
- Valid/invalid chunk counts
- Years found in documents
- Each chunk displays enriched metadata and generated questions

### Important Considerations

⚠️ **Cost & Performance**:
- Each chunk requires one GPT-4o-mini API call (~500 tokens)
- For large service lines with hundreds/thousands of chunks, this can be costly and time-consuming
- Consider testing with a specific `partitionKey` first before enriching entire service lines
- The script processes chunks sequentially to avoid rate limiting

⚠️ **Error Handling**:
- If GPT-4o-mini fails for a chunk, default values are used (contentType: "Others", validChunk: "yes")
- Check console output for any errors during processing
- Failed updates to CosmosDB are logged but don't stop the process

💡 **Best Practices**:
- Start with a small test set (specific partition key) to verify deployment name and results
- Review the generated markdown report before running large-scale enrichment
- Consider running enrichment during off-peak hours for large datasets

## Acronym Extraction - `--extract-acronyms` Flag

**NEW FEATURE**: Extract acronyms from document chunks and store them as key-value pairs in CosmosDB.

### Basic Usage

Extract acronyms from General Info (serviceNameid=0):
```bash
python -m src.chunk-filter-scripts.getChunkDetails 0 --extract-acronyms
```

Extract acronyms from a different service line:
```bash
python -m src.chunk-filter-scripts.getChunkDetails 240 --extract-acronyms
```

### What the `--extract-acronyms` flag does:

1. **Queries chunks** from CosmosDB for the specified serviceNameid
2. **Analyzes content** using GPT-4o-mini to extract acronyms and their definitions
3. **Merges duplicates** - keeps the longer/better definition when the same acronym appears in multiple chunks
4. **Stores to CosmosDB** with the following format:
   ```json
   {
       "id": "AFRAA",
       "partitionKey": "acronym",
       "value": "African Airlines Association",
       "type": "acronym"
   }
   ```
5. **Saves JSON file** - exports all extracted acronyms to `acronyms_extracted_SL{id}.json`

### Extraction Logic

The GPT-4o-mini model identifies acronyms using these patterns:
- `ABC (Full Form)` or `Full Form (ABC)`
- `ABC - Full Form` or `ABC: Full Form`
- Contextual definitions (e.g., "The African Airlines Association, commonly known as AFRAA...")

### Output

Console output includes:
- Progress for each chunk processed
- List of acronyms found per chunk
- Summary of new vs updated acronyms
- Path to exported JSON file

Example output:
```
Querying chunks for serviceNameid=0...
Found 15 chunks to process.

Extracting acronyms from chunks...

Processing chunk 1/15: chunk-abc123
  Found 3 acronyms: ['AFRAA', 'IATA', 'PAX']

Processing chunk 2/15: chunk-def456
  Found 2 acronyms: ['ACI', 'ICAO']
...

==================================================
Total unique acronyms extracted: 25
==================================================

Storing acronyms to CosmosDB...
  ✓ Created: AFRAA = African Airlines Association
  ✓ Created: IATA = International Air Transport Association
  ○ Exists (unchanged): PAX
  ↻ Updated: ACI = Airports Council International (updated definition)

==================================================
Summary:
  - New acronyms created: 20
  - Existing acronyms updated: 3
  - Total in database: 23
==================================================

Acronyms also saved to: acronyms_extracted_SL0.json
```

### Important Considerations

⚠️ **Duplicate Handling**:
- If the same acronym appears in multiple chunks, the longer definition is kept
- Existing acronyms in CosmosDB are only updated if the new definition differs

⚠️ **Cost & Performance**:
- Each chunk requires one GPT-4o-mini API call
- Recommended to run on General Info (serviceNameid=0) which typically has acronym glossary documents

💡 **Best Practices**:
- Run on General Info first to extract the main acronym glossary
- Review the generated JSON file before relying on the extracted data
- Can be re-run safely - existing acronyms are preserved unless definitions change




# Task TODO

1. dh-chatbot-documents - Need python script - getFilesUnderSL.py GeneralInfo

xyz.docx
file.pdf


## Sample Prompt for Meta-Classification
Task
You are an expert document analyzer. Your task is to extract specific metadata from the provided text and format it as a JSON object.

Instructions
Analyze the text provided and determine the following fields:

contentType: Classify the document into one of these categories: UserGuide, Marketing, MeetingMinutes, ReleaseNotes, APIDocs, or Others.

year: Extract the four-digit year (YYYY) the document was created or refers to. If not found, return null.

month: Extract the month the document was created or refers to. If not found, return null.

validChunk: Return yes if the text contains meaningful, coherent information relevant to the content type. Return no if the text is fragmented, contains only gibberish, table of contents or lacks enough context to be useful.

questions: Generate 2-3 specific questions that can be answered based only on the provided text content.

Output Format
Return ONLY a JSON object with the following structure:

JSON

{
  "contentType": "string",
  "year": "integer/null",
  "month": "string/null",
  "validChunk": "yes/no",
  "questions": ["string", "string"]
}
Text to Analyze:
[INSERT YOUR TEXT HERE]



-----------------------

# Using Question Embeddings for Retrieval

After enriching metadata with question embeddings, you can use the new retrieval method for better semantic matching.

## Retrieval Method: `retrieve_with_question_matching()`

Located in `src/services/retrieval_service.py`, this method provides **question-first hybrid retrieval** using native CosmosDB vector search.

### How It Works (Question-First Approach)

1. **Native question-first query**: CosmosDB orders results by `VectorDistance(c.questionsEmbedding, query_embedding)`
2. **Fetches both scores**: Gets question similarity and content similarity from the database
3. **Hybrid re-ranking**: Combines both scores with configurable weighting
   - Default: 70% question similarity + 30% content similarity
   - Configurable via `question_boost_weight` parameter

### Key Advantage

By using the `questionsEmbedding` field (single combined vector), CosmosDB performs native vector search on questions **first**. This ensures chunks with great question matches are found even if their content similarity is lower.

### Usage Example

```python
from src.services.retrieval_service import RetreivalService

retrieval_service = RetreivalService()

# Hybrid retrieval with question matching
docs, query_embedding, citations = retrieval_service.retrieve_with_question_matching(
    query="How do I track a lost bag?",
    container_name="dh-chatbot-documents",
    service_line=[240, 241],  # WorldTracer, Bag Manager
    top_k=7,
    question_boost_weight=0.3  # 30% weight to question matching
)

# Each citation includes the matched question
for citation in citations:
    print(f"File: {citation['File']}")
    print(f"Section: {citation['Section']}")
    print(f"Matched Question: {citation['MatchedQuestion']}")
```

### Parameters

- `query` (str): User's query
- `container_name` (str): CosmosDB container name
- `service_line` (list[int]): Service line IDs to filter by
- `top_k` (int): Number of results to return (default: 7)
- `question_boost_weight` (float): Weight for question matching
  - `0.0` = Only use content matching (same as standard retrieval)
  - `0.3` = 70% content, 30% questions (recommended default)
  - `0.5` = Equal weight to content and questions
  - `1.0` = Only use question matching

### Benefits

1. **Better semantic understanding**: Matches user queries to pre-generated questions
2. **Improved relevance**: Chunks with matching questions get boosted in rankings
3. **Explainability**: Citations include which question matched, helping users understand why a chunk was retrieved
4. **Flexible weighting**: Adjust the balance between content and question matching

### Performance Notes

- Native CosmosDB vector search on `questionsEmbedding` - no Python-side similarity calculation needed
- Retrieves top 50 candidates ordered by question similarity
- Hybrid scoring calculated in Python (fast - just arithmetic)
- Similar RU cost to standard retrieval (uses same VectorDistance function)
- No additional latency compared to content-first retrieval

### Integration with ResponseGeneratorAgent

To use question matching in your chatbot, update `src/chatbot/response_generator.py`:

**Before (line 536):**
```python
retrieved_context, top_doc_metadata, top_doc, chunk_service_line, query_embedding, citations = \
    RetreivalService().rag_retriever_agent(prompt, container_name, final_id_list)
```

**After (with question matching):**
```python
# Use question-enhanced retrieval
retrieval_service = RetreivalService()
retrieved_context, query_embedding, citations = \
    retrieval_service.retrieve_with_question_matching(
        query=prompt,
        container_name=container_name,
        service_line=final_id_list,
        top_k=7,
        question_boost_weight=0.3  # Adjust based on your needs
    )

# Extract metadata for backward compatibility
top_doc = retrieved_context[0] if retrieved_context else {}
top_doc_metadata = {"file": top_doc.get("citation", "")}
chunk_service_line = list(set([doc.get('serviceNameid') for doc in retrieved_context]))
```

**Gradual Rollout Strategy:**

You can A/B test the question matching feature:

```python
# Feature flag for question matching
USE_QUESTION_MATCHING = True  # or False, or from config

if USE_QUESTION_MATCHING:
    retrieved_context, query_embedding, citations = \
        retrieval_service.retrieve_with_question_matching(
            query=prompt,
            container_name=container_name,
            service_line=final_id_list,
            top_k=7,
            question_boost_weight=0.3
        )
    # Convert to expected format...
else:
    # Use existing method
    retrieved_context, top_doc_metadata, top_doc, chunk_service_line, query_embedding, citations = \
        retrieval_service.rag_retriever_agent(prompt, container_name, final_id_list)
```

-----------------------

In 