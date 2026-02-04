# Metadata Enrichment Feature - Implementation Guide

## Overview

The `getChunkDetails.py` script has been enhanced with GPT-4o-mini powered metadata enrichment capabilities. This feature automatically classifies and enriches chunk documents with structured metadata.

## What's New

### 1. Metadata Classification Function

A new `classify_chunk_metadata()` function that:
- Analyzes chunk content using GPT-4o-mini
- Extracts structured metadata (contentType, year, month, validChunk, questions)
- Returns validated JSON with default fallbacks for errors

### 2. Command-Line Flag: `--enrich-metadata`

The script now supports an optional flag to enable metadata enrichment:

```bash
# Without enrichment (original behavior)
python -m src.chunk-filter-scripts.getChunkDetails 240

# With enrichment (NEW)
python -m src.chunk-filter-scripts.getChunkDetails 240 --enrich-metadata
```

### 3. CosmosDB Auto-Update

When `--enrich-metadata` is enabled, the script:
1. Retrieves chunks from CosmosDB
2. Analyzes each chunk's content field
3. Enriches the document with metadata
4. Updates the document back to CosmosDB using `upsert_item()`

### 4. Enhanced Markdown Reports

Reports now include:
- Metadata statistics (content type distribution, valid/invalid counts)
- Enriched metadata section for each chunk
- Generated questions for each chunk
- Visual indicators (✅/❌) for valid chunks

## Metadata Schema

Each chunk is enriched with the following fields:

```json
{
  "contentType": "UserGuide|Marketing|MeetingMinutes|ReleaseNotes|APIDocs|Others",
  "year": "YYYY or null",
  "month": "Month name or null",
  "validChunk": "yes|no",
  "questions": ["question1", "question2", "question3"]
}
```

### Field Descriptions

- **contentType**: Document category classification
- **year**: Four-digit year extracted from content or metadata
- **month**: Month name extracted from content or metadata
- **validChunk**:
  - `"yes"` = Meaningful, coherent content
  - `"no"` = Fragmented, gibberish, or table-of-contents only
- **questions**: 2-3 specific questions answerable from the chunk content

## Usage Examples

### Example 1: Enrich All Chunks for a Service Line

```bash
python -m src.chunk-filter-scripts.getChunkDetails 240 --enrich-metadata
```

**Output:**
```
Enriching 150 chunks with metadata using GPT-4o-mini...
Processing chunk 1/150: chunk-123
  ✓ Updated chunk chunk-123 with metadata
Processing chunk 2/150: chunk-124
  ✓ Updated chunk chunk-124 with metadata
...
Metadata enrichment completed for 150 chunks.
Successfully generated chunks_SL240_report.md
```

### Example 2: Enrich Specific Document

```bash
python -m src.chunk-filter-scripts.getChunkDetails 240 'WorldTracer-Agenda2019_InternetMinutesfinal.docx' --enrich-metadata
```

### Example 3: Generate Report Without Enrichment

```bash
python -m src.chunk-filter-scripts.getChunkDetails 240 'WorldTracer-Agenda2019_InternetMinutesfinal.docx'
```

## Configuration

### Azure OpenAI Deployment Name

**IMPORTANT**: Update the deployment name before using enrichment:

Edit `src/chunk-filter-scripts/getChunkDetails.py` line 13:

```python
# Replace with your actual GPT-4o-mini deployment name
GPT4O_MINI_DEPLOYMENT = "gpt-4o-mini"
```

You can find your deployment name in:
- Azure Portal > Azure OpenAI > Deployments
- Or ask your Azure administrator

### Environment Setup

The script uses existing configuration from `src.utils.config`:
- `COSMOSDB_ENDPOINT`
- `COSMOS_ACCOUNT_KEY`
- Azure OpenAI credentials from `AzureOpenAIService`

No additional environment variables needed.

## Implementation Details

### GPT-4o-mini Prompt

The classification uses a carefully crafted prompt that:
1. Provides clear instructions for metadata extraction
2. Defines valid categories and formats
3. Requests JSON-only output for reliable parsing
4. Uses temperature=0.3 for consistent results
5. Limits to 500 tokens for cost efficiency

### Error Handling

- **API Failures**: Returns default values and logs error
- **JSON Parse Errors**: Falls back to safe defaults
- **CosmosDB Update Errors**: Logs error but continues processing
- **Missing Content**: Skips chunk with warning

### Performance Characteristics

- **Sequential Processing**: One chunk at a time to avoid rate limits
- **Progress Tracking**: Console output for every chunk processed
- **Cost per Chunk**: ~500 tokens = ~$0.0001 per chunk (estimate)
- **Time per Chunk**: ~1-2 seconds depending on content length

## Sample Output

### Console Output

```
Enriching 3 chunks with metadata using GPT-4o-mini...
Processing chunk 1/3: chunk-6
  ✓ Updated chunk chunk-6 with metadata
Processing chunk 2/3: chunk-7
  ✓ Updated chunk chunk-7 with metadata
Processing chunk 3/3: chunk-8
  ✓ Updated chunk chunk-8 with metadata

Metadata enrichment completed for 3 chunks.
Successfully generated chunks_SL240_WorldTracer-Agenda2019_InternetMinutesfinal.docx_report.md
Total chunks retrieved: 3
```

### Markdown Report Excerpt

```markdown
# Chunk Details Report

**Service Name ID**: 240

**Partition Key**: WorldTracer-Agenda2019_InternetMinutesfinal.docx

**Total Chunks Found**: 3

**Metadata Enriched**: Yes (using GPT-4o-mini)

## Summary

- **Service Names**: WorldTracer
- **Unique Documents**: 1

### Metadata Statistics

- **Content Types**:
  - MeetingMinutes: 2
  - Others: 1

- **Valid Chunks**: 3 valid, 0 invalid
- **Years Found**: 2019

---

## Chunk Details

---

## Chunk ID: chunk-6

### Basic Information
- **Partition Key**: WorldTracer-Agenda2019_InternetMinutesfinal.docx
- **Service Name**: WorldTracer
- **Service Name ID**: 240
- **Heading**: 6 Close of Meeting

### Content
\`\`\`
The meeting was formally closed at 5:30 PM...
\`\`\`

### Enriched Metadata
- **Content Type**: MeetingMinutes
- **Year**: 2019
- **Month**: null
- **Valid Chunk**: yes ✅

### Generated Questions
1. What time was the meeting closed?
2. What was discussed in the final agenda item?
3. Who chaired the closing session?
```

## Cost Estimation

For reference (using GPT-4o-mini approximate pricing):

| Chunks | Est. Tokens | Est. Cost |
|--------|-------------|-----------|
| 10     | 5,000       | $0.001    |
| 100    | 50,000      | $0.01     |
| 1,000  | 500,000     | $0.10     |
| 10,000 | 5,000,000   | $1.00     |

*Note: Actual costs depend on Azure OpenAI pricing and content length*

## Best Practices

### ✅ DO:
- Test with small batches first (use partition key filter)
- Review generated metadata for quality before large-scale runs
- Run during off-peak hours for large datasets
- Monitor Azure OpenAI usage and costs
- Keep deployment name updated in the script

### ❌ DON'T:
- Run enrichment on entire knowledge base without testing
- Run multiple instances simultaneously (rate limits)
- Ignore error messages in console output
- Forget to configure the deployment name

## Troubleshooting

### Issue: "Error classifying chunk metadata"

**Cause**: GPT-4o-mini deployment name incorrect or API unavailable

**Solution**:
1. Verify deployment name in Azure Portal
2. Check Azure OpenAI service status
3. Verify credentials in `src.utils.config`

### Issue: "Error updating chunk"

**Cause**: CosmosDB write permissions or connection issue

**Solution**:
1. Verify COSMOS_ACCOUNT_KEY has write permissions
2. Check CosmosDB connection in Azure Portal
3. Verify container name "dh-chatbot-documents" exists

### Issue: Slow processing

**Cause**: API latency or large content chunks

**Solution**:
- Normal behavior for many chunks (1-2 sec each)
- Consider parallel processing for very large datasets (future enhancement)

## Future Enhancements

Potential improvements for consideration:

1. **Batch Processing**: Process multiple chunks in parallel with rate limiting
2. **Dry Run Mode**: Preview classifications without updating CosmosDB
3. **Selective Re-enrichment**: Only process chunks missing metadata
4. **Custom Prompts**: Allow custom classification categories per service line
5. **Quality Scoring**: Add confidence scores to metadata
6. **Retry Logic**: Automatic retry for failed API calls

## Code Changes Summary

### Files Modified

1. **`src/chunk-filter-scripts/getChunkDetails.py`**
   - Added `classify_chunk_metadata()` function
   - Added `--enrich-metadata` CLI argument
   - Enhanced `query_and_generate_md()` with enrichment logic
   - Updated `format_chunk_as_markdown()` for metadata display
   - Switched to `argparse` for better CLI handling

2. **`src/chunk-filter-scripts/Approch.md`**
   - Added usage examples for metadata enrichment
   - Added configuration instructions
   - Added important considerations and best practices

3. **`src/chunk-filter-scripts/METADATA_ENRICHMENT_GUIDE.md`** (NEW)
   - Comprehensive implementation guide
   - Usage examples and troubleshooting

## Version History

- **v1.0** (2026-01-22): Initial metadata enrichment feature
  - GPT-4o-mini integration
  - Auto-update to CosmosDB
  - Enhanced markdown reports
  - Metadata statistics

---

For questions or issues, refer to the main project README or contact the development team.
