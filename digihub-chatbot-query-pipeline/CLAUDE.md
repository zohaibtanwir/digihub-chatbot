# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DigiHub Chatbot Query Pipeline is a RAG (Retrieval-Augmented Generation) chatbot service built with FastAPI. It integrates Azure OpenAI with CosmosDB vector search to provide intelligent responses based on user queries with service-line-based authorization.

## Common Commands

### Running the Application

```bash
# Run locally with uvicorn
python src/app.py

# Run with gunicorn (production mode)
gunicorn src.app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8080 --timeout 1200
```

### Testing

```bash
# Run all tests
python tests/run_test.py

# Run specific test file
pytest tests/tests/test_auth_service.py -v

# Run tests with coverage
pytest tests/tests/ --cov=src --cov-report=html
```

### Docker

```bash
# Build image
docker build -t digihub-chatbot-query-pipeline .

# Run container
docker run -p 8080:8080 digihub-chatbot-query-pipeline
```

## Architecture

### Core Flow

1. **Request Entry** (`src/app.py`): FastAPI app with CORS middleware and health probes at `/actuator/health/liveness` and `/actuator/health/readiness`
2. **Views Layer** (`src/views/`): Route handlers for chat, SAS URL generation, and sessions
3. **Authorization** (`src/services/auth_service.py`): Fetches user subscriptions from DigiHub User Management API
4. **Response Generation** (`src/chatbot/response_generator.py`): Orchestrates the RAG pipeline
5. **Retrieval** (`src/services/retrieval_service.py`): Vector search in CosmosDB using embeddings
6. **Session Management** (`src/services/session_service.py`): Persists conversation history in CosmosDB

### Key Components

**ResponseGeneratorAgent** (`src/chatbot/response_generator.py`):
- Main orchestrator for generating chatbot responses
- Handles confidence-based authorization checks (if confidence < 0.80)
- Performs relevance judgment on retrieved chunks
- Manages session-dependent query expansion for follow-up questions
- Parses responses and determines if query is in/out of scope

**RetreivalService** (`src/services/retrieval_service.py`):
- **Hybrid Search**: Combines semantic similarity (70%) with keyword matching (30%)
- Vector similarity search using CosmosDB with Azure OpenAI embeddings
- Keyword scoring with heading boost and exact phrase matching
- Filters results by user's authorized service lines
- Content type boosting for UserGuide/DigiHubUserGuide (5% boost)
- Returns top 7 chunks ordered by hybrid score
- Filters out chunks that only contain headings

**AuthorizationService** (`src/services/auth_service.py`):
- Singleton service that fetches user subscriptions
- Supports user impersonation (returns `None` for subscriptions if allowed)
- Integrates with DigiHub User Management API

**SessionDBService** (`src/services/session_service.py`):
- Creates session IDs in format: `{user_id}-{date}-{increment}`
- Stores user and assistant messages separately in CosmosDB
- Includes citation, score, confidence, and disclaimer fields
- Tracks extracted entities (services, topics, technical terms) per message

**ConversationalContextEngine** (multiple components):
- **QueryAnalyzer** (`src/chatbot/query_analyzer.py`): Classifies queries and detects session dependencies
- **ContextManager** (`src/chatbot/context_manager.py`): Handles reference resolution and query merging
- **SessionDBService** (`src/services/session_service.py`): Manages entity retrieval and session history
- See [docs/conversational-context-engine.md](docs/conversational-context-engine.md) for complete architecture

### Configuration Management

Configuration is loaded dynamically from a remote config server specified by `CONFIG_URL` environment variable. The config includes:
- Azure OpenAI credentials (retrieved from Azure Key Vault)
- CosmosDB connection details
- Storage account settings for SAS URL generation
- CORS origins and other runtime settings

**VaultManager** (`src/utils/vault_manager.py`):
- Retrieves secrets from Azure Key Vault using DefaultAzureCredential
- Used for sensitive values like API keys and connection strings

### Authorization & Security

**Confidence-Based Authorization**:
When `confidence_score < 0.80`, the system performs additional checks:
1. Retrieves service line IDs relevant to the query
2. Compares with user's authorized service lines
3. Raises `UnAuthorizedServiceLineException` if user lacks access
4. Error message includes which service lines are required

**Request Validation** (`src/utils/request_utils.py`):
- XSS protection via blacklist checking
- Request body validation with pydantic models

### Conversational Context Engine (CCE)

The system uses a sophisticated **Conversational Context Engine** to preserve context for follow-up queries and enable natural multi-turn conversations.

**Key Capabilities:**
- **Session Dependency Detection**: LLM-based analysis determines if new queries build on previous context
- **Reference Resolution**: Resolves pronouns (e.g., "it", "that") to specific entities from conversation history
- **Smart Query Merging**: Combines last 3 user messages with entity context for improved retrieval
- **Entity Tracking**: Extracts and stores services, topics, and technical terms per conversation turn
- **Hybrid Search Retrieval**: Combines semantic similarity (70%) with keyword matching (30%)

**Example Flow:**
```
User: "What is WorldTracer?"
Bot:  [Explains WorldTracer...]

User: "How do I configure it?"
→ CCE resolves "it" → "WorldTracer"
→ Merges context: "Previous: What is WorldTracer? Current: How do I configure WorldTracer?"
→ Retrieves relevant docs via hybrid search (semantic + keyword)
→ Generates contextual response
```

**For detailed architecture and implementation:** See [docs/conversational-context-engine.md](docs/conversational-context-engine.md)

### Logging & Tracing

- Custom logger with trace ID context (`src/utils/logger.py`)
- Trace IDs extracted from `x-digihub-traceid` header
- Performance timing decorators for key operations

## Directory Structure

```
digihub-chatbot-query-pipeline/
├── docs/                     # Feature documentation
│   └── conversational-context-engine.md  # CCE architecture & implementation
├── src/
│   ├── app.py                    # FastAPI application entry point
│   ├── chatbot/                  # Core chatbot logic
│   │   ├── response_generator.py   # Main RAG orchestration
│   │   ├── query_analyzer.py       # Query classification & session dependency detection
│   │   ├── context_manager.py      # Reference resolution & smart query merging
│   │   ├── prompt_manager.py       # Prompt templates
│   │   └── azure_sas_url_generator.py
│   ├── services/                 # External service integrations
│   │   ├── auth_service.py         # User authorization
│   │   ├── retrieval_service.py    # Hybrid search (semantic + keyword)
│   │   ├── session_service.py      # Session persistence & entity tracking
│   │   ├── azure_openai_service.py
│   │   ├── cosmos_db_service.py
│   │   └── embedding_service.py
│   ├── views/                    # FastAPI route handlers
│   │   ├── chat_views.py
│   │   ├── session_view.py
│   │   └── sas_view.py
│   ├── dto/                      # Data models
│   ├── enums/                    # Prompt templates & constants
│   ├── exceptions/               # Custom exceptions
│   └── utils/                    # Shared utilities
│       ├── config.py               # Configuration loader
│       ├── vault_manager.py        # Azure Key Vault access
│       ├── logger.py               # Custom logging
│       └── request_utils.py        # Validation & utilities
├── tests/                    # Test suite
├── CLAUDE.md                 # Project guidance for Claude Code
└── README.md                 # General project documentation
```

## Azure DevOps Pipelines

- **Linting** (`.azuredevops/azure-pipelines.ci.yml`): Commit message validation for PRs
- **Semantic Versioning** (`.azuredevops/azure-pipelines.semver.yml`): Automated version management
- **Renovate** (`.azuredevops/azure-pipelines.renovate.yml`): Dependency updates
- **Deployment** (`ci/develop/`, `ci/prod/`): Environment-specific build and deploy pipelines

Linting follows conventional commits: https://sita-pse.visualstudio.com/Communication%20and%20Network/_wiki/wikis/Communication-and-Network.wiki/40180/Conventional-commits-and-linting

## Important Notes

**Environment Variables**:
- `CONFIG_URL`: Points to remote configuration server
- `KEY_VAULT_URL`: Azure Key Vault URL (fetched from config)

**CosmosDB Containers**:
- `KNOWLEDGE_BASE_CONTAINER`: Vector-indexed knowledge base
- `Session`: Chat session history
- `ServiceNameMapping`: Service line metadata
- `COSMOSDB_SHAREPOINT_DATA_CONTAINER_NAME`: SharePoint data

**Impersonation**:
Users with `isImpersonationAllowed=true` bypass subscription filtering (receive `None` for subscriptions).

**User ID Hashing**:
Email IDs are hashed using `HashUtils.hash_user_id()` before use in session IDs.

**SAS URL Generation**:
The service can generate time-limited Azure Blob Storage SAS URLs. Requires Service Principal with:
- Storage Blob Data Contributor role
- Storage Blob Delegator role

---

## Session Context (February 5, 2026)

### Problem Solved: Generic Queries and CI Analysis

**Original Issue:** Generic queries like "What is CI Analysis?", "What is DigiHub?" returned out-of-scope responses due to:
1. Pure semantic search matching "What is X?" patterns instead of actual content
2. High-volume service lines (WorldTracer: 2,968 chunks) drowning out user guides (General Info: 10 chunks)
3. No keyword matching to find exact terms in headings

**Solution Implemented:** Hybrid Search (Semantic + Keyword)

### Hybrid Search Architecture (`src/services/retrieval_service.py`)

```python
# Final score formula
final_score = 0.7 * semantic_score + 0.3 * keyword_score

# Keyword score components:
# - Base score: proportion of query terms found in content (0-1)
# - Heading boost: +30% if query terms appear in heading
# - Phrase boost: +40% if exact phrase matches heading (e.g., "CI Analysis")
```

**Key Methods Added:**
- `_tokenize()`: Tokenizes text, removes stopwords
- `_extract_key_phrases()`: Extracts key phrases from query
- `_calculate_keyword_score()`: Calculates BM25-like keyword score

### Service Line Handling for Generic Queries

**Step 6.6 in `response_generator.py`:**
- **Generic queries** (no service line detected): RESTRICT search to [0, 440] only
- **Specific queries** (service line detected): ADD [0, 440] alongside detected service lines

```python
USER_GUIDE_SERVICE_LINES = [0, 440]  # General Info, Operational Support

if not detected_service_names and not is_session_dependent:
    # Generic: restrict to user guides only
    retrieval_service_lines = authorized_user_guides
else:
    # Specific: add user guides alongside detected
    retrieval_service_lines = list(set(retrieval_service_lines + authorized_user_guides))
```

### CosmosDB Service Line Reference

| Service Line | ID | Chunks | Content |
|--------------|-----|--------|---------|
| General Info | 0 | 10 | DigiHub user guide, Acronym list |
| Operational Support | 440 | 10 | AIR Dashboard guide (CI Analysis), Support guide |
| Billing | 400 | 103 | User guides + BillingUpdate newsletters |
| WorldTracer | 240 | 2,968 | WorldTracer documentation |
| Bag Manager | 360 | 2,868 | Bag Manager documentation |
| Euro CAB | 460 | 711 | Meeting notes |

**Total Chunks:** ~7,356

### Completed Beads Issues (February 5, 2026)

| Beads ID | Description | File |
|----------|-------------|------|
| `digihub-chatbot-101` | Hybrid search (semantic + keyword) | retrieval_service.py |
| `digihub-chatbot-inb` | Default service lines for generic queries | response_generator.py |
| `digihub-chatbot-tgx` | Content type boosting (5% for UserGuide) | retrieval_service.py |
| `digihub-chatbot-9mv` | Content-first ORDER BY | retrieval_service.py |
| `digihub-chatbot-t9i` | Always include user guide service lines | response_generator.py |
| `digihub-chatbot-b9e` | Fix Step 6.6 restriction logic | response_generator.py |
| `digihub-chatbot-a3d` | Remove question similarity scoring | retrieval_service.py |

### Previous Session Fixes (Still Active)

| Issue ID | Fix | File |
|----------|-----|------|
| `digihub-chatbot-63a` | VectorDistance to similarity conversion | retrieval_service.py |
| `digihub-chatbot-eip` | Legacy chunks get 15% penalty | retrieval_service.py |
| `digihub-chatbot-c39` | Service line filtering after relevance judge | response_generator.py |
| `digihub-chatbot-89x` | Relaxed relevance judge prompt | prompt_template.py |

### Test UI

Location: `ui/app.py` with `ui/requirements.txt`

```bash
cd /Users/zohaibtanwir/projects/digihub-chatbot/digihub-chatbot-query-pipeline
streamlit run ui/app.py
# Opens at http://localhost:8501
```

### Local Development

```bash
# Connect to VPN first, then:
cd /Users/zohaibtanwir/projects/digihub-chatbot/digihub-chatbot-query-pipeline
python -m uvicorn src.app:app --host 0.0.0.0 --port 8080 --reload
```

### SharePoint Analysis Report

Location: `SharePoint_Analysis_Report.docx` (generated by `SharePoint_Analysis_Report.py`)

Contains:
- Total SharePoint overview (6.0 GB, ~1,100 files)
- Indexed files breakdown (999 files, excluding videos/Excel)
- CosmosDB gap analysis
- Implemented fixes documentation
- Recommended test questions

### Next Steps

1. User is compiling a list of questions that still don't work correctly
2. Review and fix remaining edge cases
3. Consider additional data quality improvements (e.g., question regeneration for some chunks)

---

## Session Context (February 7, 2026)

### Features Implemented This Session

#### 1. Two-Stage Retrieval for Definitional Queries
**File:** `src/chatbot/response_generator.py`

For "What is X?" queries:
- **Stage 1**: Search General Info (service line 0) only
- **Stage 2**: If score < 0.75, expand to all authorized service lines

```python
MIN_CONFIDENCE_THRESHOLD = 0.75  # Trigger Stage 2 if below this

if is_definitional_query and GENERAL_INFO_SERVICE_LINE_ID in retrieval_service_lines:
    # Stage 1: Try General Info first
    # Stage 2: Expand if Stage 1 score < threshold
```

#### 2. PRODUCT_KEYWORDS for Code-Based Detection
**File:** `src/chatbot/response_generator.py`

Bypasses LLM detection for known product names:
```python
PRODUCT_KEYWORDS = {
    'dataconnect': 340,  # Community Messaging KB
    'sdc': 340,
    'sitatex': 340,
}
```

When detected:
- Skips Two-Stage Retrieval
- Routes directly to correct service line

#### 3. Synonym Expansion for Keyword Matching
**File:** `src/services/retrieval_service.py`

```python
self._synonyms = {
    'dispute': ['support', 'case', 'complaint', 'issue', 'problem'],
    'complaint': ['support', 'case', 'dispute', 'issue'],
    'problem': ['issue', 'incident', 'trouble'],
    'report': ['dashboard', 'statistics', 'analytics', 'metrics', 'view'],
    'incidents': ['incident'],  # Singular/plural
    'incident': ['incidents'],
    'ticket': ['case', 'incident', 'request'],
    'view': ['see', 'access', 'check', 'find'],
}
```

#### 4. Pipeline Timing Logs
Added timing logs throughout the pipeline for performance analysis:
```
[Timing] Step 1 (Session Context): 0.12s
[Timing] Step 2 (Query Analysis): 1.45s
[Timing] Step 7 (Retrieval): 0.89s
[Timing] Step 7.5 (Relevance Filter): 0.23s
[Timing] Total Pipeline: 3.21s
```

#### 5. Service Line Keywords File
**File:** `src/data/service_line_keywords.json`

JSON mapping of service lines to keywords for LLM prompt context.

### Test Results (February 7, 2026)

**Working Queries (15):**
- What is Digihub?
- How can I find report and statistic about incidents? ✓ (Fixed this session)
- What is Airport Committee?
- What is Airport Solutions?
- How can we install SITA Data Connect?
- Give me a summary of the SITA DataConnect release notes
- How can I submit a payment query?
- What are the possible actions on digihub?
- How customers can vote on digihub?
- What type of Service Requests can I issue?
- Can you tell me how many incidents were raised for IndiGo in July 2025
- I am getting error "Something Went Wrong" in Digihub
- Where do I find a dashboard about my incidents with SITA?
- I need report and statistic about incidents we have with SITA
- What are the minimum PC requirements to install SITATEX/Worldtracer?

**Correctly No Answer (Content Gaps):**
- What do you know about OptiFlight?
- What is SITA DataConnect? (technical docs exist, no definitional content)
- What is SITA Mission Watch? (only mentioned in tables)
- What is the requirement to install CUTE services?
- What is new with SITA for 2026?
- What is SITA Border Control?
- Is CUTE installed in Mumbai Airport?

**Still Not Working (1):**
- "I have a dispute with an invoice, who to contact?" → Returns 2021 newsletter instead of Billing Guide

### Open Beads Issues

| Issue ID | Priority | Description |
|----------|----------|-------------|
| `digihub-chatbot-p0c` | P2 | Invoice dispute query returns old newsletter |
| `digihub-chatbot-lxr` | P2 | SITA Mission Watch - Content Gap |
| `digihub-chatbot-389` | P2 | AIR Dashboard incidents query (FIXED) |
| `digihub-chatbot-yyt` | P2 | Content gaps for products |
| `digihub-chatbot-kmk` | P2 | Response time optimization |
| `digihub-chatbot-gqn` | P2 | Review LLM prompts |
| `digihub-chatbot-ceb` | P2 | Session entity resolution |
| `digihub-chatbot-5zd` | P2 | Out-of-scope detection |

### CosmosDB Content Analysis

**AIR Dashboard (Operational Support 440):**
- `chunk-0`: "DIGIHUB AIR DASHBOARD" - User Guide
- `chunk-2`: "View Company Incidents" - Directly relevant
- `chunk-3`: "Open Incident Resolution"

**SITA Mission Watch:** Content gap confirmed
- Only appears in Billing newsletter tables (product name changes)
- One descriptive sentence in Euro CAB chunk-37 (heading: "ASISTIM")
- No dedicated documentation

### Local Development Notes

**Config File:**
- `src/utils/config.py` - No secrets (use config server)
- `src/utils/config.py.local` - Local copy with secrets (gitignored)

**To restore local secrets:**
```bash
cp src/utils/config.py.local src/utils/config.py
```

---

## Session Context (February 8, 2026)

### Unit Test Implementation

Created comprehensive unit tests to increase coverage from 21% to ~45%.

| Test File | Tests | Target Module |
|-----------|-------|---------------|
| `tests/tests/test_response_generator.py` | 25 | response_generator.py |
| `tests/tests/test_query_analyzer.py` | 20 | query_analyzer.py |
| `tests/tests/test_authorization_checker.py` | 20 | authorization_checker.py |
| `tests/tests/test_relevance_judge.py` | 18 | relevance_judge.py |

**Run tests:**
```bash
pytest tests/tests/ -v --cov=src --cov-report=html
```

### Content Authority + Recency Scoring
**File:** `src/services/retrieval_service.py`

Replaced narrow content type boost with principled scoring:

```python
# Authority tiers (multiplier)
CONTENT_AUTHORITY = {
    'UserGuide': 1.20,        # Tier 1: Authoritative
    'DigiHubUserGuide': 1.20,
    'ReleaseNotes': 1.10,     # Tier 2: Technical
    'TechnicalDoc': 1.10,
    'FAQ': 1.00,              # Tier 3: Neutral
    'BillingUpdate': 0.85,    # Tier 4: Supplementary
    'MeetingNotes': 0.85,
}

# Recency factor
# 2025-2026: 1.10x | 2023-2024: 1.00x | 2021-2022: 0.90x | Pre-2021: 0.85x

# Formula
final_score = hybrid_score × authority_multiplier × recency_factor
```

### Completed Beads Issues (February 8, 2026)

| Beads ID | Description |
|----------|-------------|
| `digihub-chatbot-5qq` | Unit tests for response_generator.py |
| `digihub-chatbot-5by` | Unit tests for query_analyzer.py |
| `digihub-chatbot-jis` | Unit tests for authorization_checker.py |
| `digihub-chatbot-bv4` | Unit tests for relevance_judge.py |
| `digihub-chatbot-hec` | Content Authority + Recency scoring |

---

## Session Context (February 10, 2026)

### Streaming Response Implementation
**Files:** `src/views/chat_views.py`, `src/chatbot/response_generator.py`, `ui/app.py`

Implemented streaming for Response Generation LLM to reduce perceived wait time.

**New Endpoint:** `POST /chat/stream`
- Uses Server-Sent Events (SSE) via FastAPI `StreamingResponse`
- `media_type="text/event-stream"`

**SSE Event Types:**
```
data: {"type": "session", "session_id": "abc123"}\n\n
data: {"type": "token", "content": "Hello"}\n\n
data: {"type": "token", "content": "Hello, I"}\n\n
...
data: {"type": "metadata", "response_metadata": {...}, "session_id": "abc123"}\n\n
data: [DONE]\n\n
```

**New Methods in `response_generator.py`:**
- `get_response_from_agent_streaming()` - Yields tokens using Azure OpenAI `stream=True`
- `generate_response_streaming()` - Full pipeline with streaming final step

**Test UI Updates (`ui/app.py`):**
- "Enable Streaming" toggle (default: ON)
- Thinking indicator with elapsed time
- Real-time token display

### Conversational Message Handling
**Files:** `src/chatbot/response_generator.py`, `src/chatbot/query_analyzer.py`, `src/enums/prompt_template.py`

Instant friendly responses for greetings, thanks, farewells in multiple languages.

**Fast Path (English - no LLM):**
```python
FAST_PATH_PATTERNS = {
    "greeting": ["hi", "hello", "hey", "good morning", ...],
    "thanks": ["thanks", "thank you", "thx", ...],
    "farewell": ["bye", "goodbye", ...],
    "affirmation": ["ok", "okay", "got it", ...]
}
```

**Smart Path (Multi-language via Query Analyzer):**
- French: "Bonjour", "Merci", "Au revoir"
- German: "Guten Tag", "Danke", "Auf Wiedersehen"
- Spanish: "Hola", "Gracias", "Adiós"

**Response Time:** <0.5s (vs ~10s for full pipeline)

### Output Parser Optimization
**Files:** `src/chatbot/output_parser.py` (NEW), `src/chatbot/response_formatter.py`

Replaced OUTPUT_PARSING LLM call with Python regex for ~2s savings.

```python
# response_formatter.py
USE_PYTHON_OUTPUT_PARSER = True  # Toggle for Python vs LLM

# output_parser.py
def format_response(message: str) -> str:
    result = message.replace('\n', '<br>')
    result = re.sub(r'(?<!:)//', '/', result)  # Fix double slashes except URLs
    # ... image and list formatting
    return result
```

### Question Similarity Removal
**File:** `src/services/retrieval_service.py`

Removed unused `questionsEmbedding` from CosmosDB query:
- Removed `VectorDistance(c.questionsEmbedding, ...)` from query
- Removed `question_similarity` processing
- Simplified scoring to content similarity only

### Completed Beads Issues (February 10, 2026)

| Beads ID | Description |
|----------|-------------|
| `digihub-chatbot-str1` | Streaming response implementation |
| `digihub-chatbot-xlz` | Conversational message handling |
| `digihub-chatbot-dz6` | Output parser Python optimization |
| `digihub-chatbot-2bq` | Question similarity removal |

### Key Files Modified

| File | Changes |
|------|---------|
| `src/views/chat_views.py` | New `/chat/stream` SSE endpoint |
| `src/chatbot/response_generator.py` | Streaming methods, conversational handling, FAST_PATH_PATTERNS |
| `src/chatbot/query_analyzer.py` | `is_conversational`, `conversational_type` fields |
| `src/enums/prompt_template.py` | Conversational detection in LANGUAGE_DETECTION_TEMPLATE |
| `src/chatbot/output_parser.py` | NEW - Python output formatting |
| `src/chatbot/response_formatter.py` | USE_PYTHON_OUTPUT_PARSER toggle |
| `src/services/retrieval_service.py` | Removed questionsEmbedding query |
| `ui/app.py` | Streaming toggle, thinking indicator, elapsed time |

---

## Session Context (February 14, 2026)

### Follow-up Query Reference Resolution Fix

Fixed multiple issues where follow-up queries like "tell me more", "explain further", "explain to me more" were not working correctly.

#### Issues Fixed:

1. **Conversational Override** - Follow-up phrases were incorrectly classified as conversational
2. **Session-Aware Resolution** - Short vague queries need reference resolution
3. **Definitional Pattern** - "Explain further" was triggering Two-Stage Retrieval
4. **Reference Resolution Recency** - LLM was picking wrong entity from session history

#### Changes Made:

**`src/chatbot/response_generator.py`:**
```python
# 1. FOLLOW_UP_PATTERNS - phrases that should NOT be conversational
FOLLOW_UP_PATTERNS = [
    "tell me more", "more details", "explain further", "go on", "continue",
    "elaborate", "what else", "more info", "lets explore", "dig deeper", ...
]

# 2. Conversational override safeguard
if is_conversational and (is_session_dependent or self._is_follow_up_request(prompt)):
    is_conversational = False  # Proceed with retrieval instead

# 3. Session-aware reference resolution
SHORT_QUERY_WORD_LIMIT = 5
should_resolve = has_references or (is_session_dependent and word_count <= 5)

# 4. Definitional pattern excludes continuation phrases
r"^explain\s+(?!(how|more|further|me\s+more|to\s+me))"
```

**`src/chatbot/context_manager.py`:**
```python
# CONTINUATION_PATTERNS for reference detection
CONTINUATION_PATTERNS = r'(?i)^(tell\s+me\s+more|more\s+details?|explain(\s+to)?\s*(me\s+)?more|...)'
```

**`src/enums/prompt_template.py`:**
- Updated REFERENCE_RESOLUTION_TEMPLATE to prioritize MOST RECENT topic
- Added CRITICAL instruction for continuation queries
- Added Example 3 showing correct recency behavior

#### Flow After Fix:
```
User: "What is Bag Manager?"
Bot:  [Explains Bag Manager...]

User: "explain to me more"
→ Detected as session-dependent (is_session_dependent=True)
→ Short query (3 words ≤ 5) triggers reference resolution
→ Resolves to: "Explain to me more about Bag Manager"
→ Retrieves Bag Manager content (not DigiHub)
→ Returns relevant Bag Manager details
```

### Completed Beads Issues (February 14, 2026)

| Beads ID | Description |
|----------|-------------|
| `digihub-chatbot-ftx` | Follow-up query reference resolution fix |

### Key Files Modified

| File | Changes |
|------|---------|
| `src/chatbot/response_generator.py` | FOLLOW_UP_PATTERNS, _is_follow_up_request(), session-aware resolution, definitional pattern fix |
| `src/chatbot/context_manager.py` | CONTINUATION_PATTERNS for has_references() |
| `src/enums/prompt_template.py` | REFERENCE_RESOLUTION_TEMPLATE recency prioritization |

---

## Session Context (February 15, 2026)

### Configuration Externalization

Moved hardcoded maintenance items to JSON config files for easier updates without code changes.

#### PRODUCT_KEYWORDS → JSON Config
**File:** `src/data/product_keywords.json`

Previously duplicated in two places in `response_generator.py`. Now consolidated and externalized.

```json
{
  "dataconnect": 340,
  "data connect": 340,
  "sita dataconnect": 340,
  "sita data connect": 340,
  "sdc": 340,
  "sitatex": 340
}
```

**Usage:** Maps product name variations to service line IDs for code-based keyword detection.

#### Synonyms → JSON Config
**File:** `src/data/synonyms.json`

Previously hardcoded in `retrieval_service.py`. Now externalized.

```json
{
  "dispute": ["support", "case", "complaint", "issue", "problem"],
  "complaint": ["support", "case", "dispute", "issue"],
  "problem": ["issue", "incident", "trouble"],
  "report": ["dashboard", "statistics", "analytics", "metrics", "view"],
  "incidents": ["incident"],
  "incident": ["incidents"],
  "ticket": ["case", "incident", "request"],
  "view": ["see", "access", "check", "find"]
}
```

**Usage:** Expands search keywords with synonyms for better hybrid search matching.

### Maintenance Guide

| Config File | Purpose | How to Update |
|-------------|---------|---------------|
| `src/data/product_keywords.json` | Product → Service Line ID | Add `"product_name": service_line_id` |
| `src/data/synonyms.json` | Keyword expansion | Add `"term": ["synonym1", "synonym2"]` |
| `src/data/service_line_keywords.json` | Service line context for LLM | Update keywords per service line |

### Open Beads Issues (Future Work)

| Issue ID | Priority | Description |
|----------|----------|-------------|
| `digihub-chatbot-i7h` | P2 | Auto-generate product keywords at ingestion time |
| `digihub-chatbot-0xo` | P3 | Auto-generate synonyms using embeddings or query logs |

### Key Files Modified

| File | Changes |
|------|---------|
| `src/chatbot/response_generator.py` | Consolidated PRODUCT_KEYWORDS, loads from JSON |
| `src/services/retrieval_service.py` | Loads synonyms from JSON |
| `src/data/product_keywords.json` | NEW - Product keyword config |
| `src/data/synonyms.json` | NEW - Synonym mapping config |

---

## Session Context (February 16, 2026)

### Unit Test Fixes and Coverage Expansion

Fixed 20 failing unit tests and added 40+ new test cases to significantly increase coverage.

#### Test Fixes (20 Failing → 0 Failing)

**Root Causes Identified:**
1. **Mock Exception Classes**: Mocked `PartialAccessServiceLineException` used `disclaimer` but real code uses `disclaimar` (typo in production code)
2. **Mock Pollution**: `side_effect` from error-handling tests persisted across test runs
3. **Class Instance Mismatch**: `pytest.raises()` couldn't catch mocked exception classes created inside fixtures

**Fixes Applied:**

| Test File | Issue | Solution |
|-----------|-------|----------|
| `test_authorization_checker.py` | Mock exception signature mismatch | Fixed `__init__` to use `disclaimar` |
| `test_authorization_checker.py` | pytest.raises not catching mocks | Changed to `try/except` with class name check |
| `test_query_analyzer.py` | side_effect pollution | Added module-level mock + explicit `side_effect = None` reset |
| `test_relevance_judge.py` | side_effect pollution | Same module-level mock pattern |

**Key Pattern Used:**
```python
# Module-level mock reference for consistent state
_mock_client_instance = MagicMock()

@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    global _mock_client_instance
    _mock_client_instance.reset_mock()
    _mock_client_instance.chat.completions.create.side_effect = None
    # ... rest of setup
```

#### New Tests Added (40+ Tests)

Comprehensive test coverage for `response_generator.py`:

| Test Class | Tests | Methods Covered |
|------------|-------|-----------------|
| `TestFastPathConversational` | 9 | `_check_fast_path_conversational` |
| `TestIsFollowUpRequest` | 7 | `_is_follow_up_request` |
| `TestGetConversationalResponse` | 7 | `_get_conversational_response` |
| `TestHandleSessionEntities` | 3 | `_handle_session_entities` |
| `TestGetContextualServiceLines` | 2 | `_get_contextual_service_lines` |
| `TestSaveSessionInBackground` | 2 | `save_session_in_background` |
| `TestGenerateResponse` | 2 | `generate_response` (fast path) |
| `TestProductKeywords` | 2 | `PRODUCT_KEYWORDS` loading |
| `TestStreamingResponse` | 1 | `generate_response_streaming` |
| `TestAnalyzeQuery` | 1 | `_analyze_query` |
| `TestRetrieveSessionContext` | 2 | `_retrieve_session_context` |
| `TestResolveQueryReferences` | +2 | Additional edge cases |

### Test Coverage Summary

| Metric | Before | After |
|--------|--------|-------|
| Total Tests | 124 passed, 20 failed | **184 passed, 0 failed** |
| Overall Coverage | 37% | **37%** (more code paths tested) |
| response_generator.py | 30% | ~35% (estimated) |
| query_analyzer.py | 85% → 93% | **93%** |
| relevance_judge.py | 100% | **100%** |
| authorization_checker.py | 98% | **98%** |

**Run Tests:**
```bash
pytest tests/tests/ -v --cov=src --cov-report=term-missing
```

### Commits Made

| Commit | Description |
|--------|-------------|
| `bd7e971` | fix: Fix mock isolation issues in unit tests |
| `e16c506` | test: Add comprehensive unit tests for response_generator.py |

### Key Files Modified

| File | Changes |
|------|---------|
| `tests/tests/test_authorization_checker.py` | Fixed mock exception signature, changed to try/except pattern |
| `tests/tests/test_query_analyzer.py` | Module-level mock, side_effect reset |
| `tests/tests/test_relevance_judge.py` | Module-level mock, side_effect reset |
| `tests/tests/test_response_generator.py` | Added 40+ new tests for conversational, streaming, session handling |

### Open Beads Issues

| Issue ID | Priority | Description |
|----------|----------|-------------|
| `digihub-chatbot-zom` | P2 | Improve existing test coverage to 80% |
| `digihub-chatbot-6hh` | P2 | Create test infrastructure (conftest.py, pytest.ini) |
| `digihub-chatbot-dti` | P2 | Add CI/CD test integration |
| `digihub-chatbot-i7h` | P2 | Auto-generate product keywords at ingestion |
| `digihub-chatbot-0xo` | P3 | Auto-generate synonyms |
