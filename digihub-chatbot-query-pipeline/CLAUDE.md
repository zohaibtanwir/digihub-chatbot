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
- Vector similarity search using CosmosDB with Azure OpenAI embeddings
- Filters results by user's authorized service lines
- Returns top 7 chunks ordered by cosine similarity
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
- **Question-First Vector Retrieval**: Uses dual-embedding architecture (100% question-based ranking)

**Example Flow:**
```
User: "What is WorldTracer?"
Bot:  [Explains WorldTracer...]

User: "How do I configure it?"
→ CCE resolves "it" → "WorldTracer"
→ Merges context: "Previous: What is WorldTracer? Current: How do I configure WorldTracer?"
→ Retrieves relevant docs via question embeddings
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
│   │   ├── retrieval_service.py    # Vector search & retrieval (question-first)
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

## Session Context (February 4, 2026)

### Work Completed This Session

#### Critical Response Quality Fixes (8 issues - ALL COMPLETED)

| Issue ID | Fix | File |
|----------|-----|------|
| `digihub-chatbot-gre` | ORDER BY uses questionsEmbedding | retrieval_service.py |
| `digihub-chatbot-eip` | Legacy chunks get 15% penalty | retrieval_service.py |
| `digihub-chatbot-syj` | Reference resolution works regardless of is_session_dependent | response_generator.py |
| `digihub-chatbot-dzp` | Removed MIN_RELEVANCE_CHUNKS forcing | response_generator.py |
| `digihub-chatbot-cq5` | Consolidated out-of-scope detection signals | response_generator.py |
| `digihub-chatbot-apv` | Service line filtering uses ALL authorized lines | response_generator.py |
| `digihub-chatbot-0yz` | Contextual service lines don't override authorization | response_generator.py |
| `digihub-chatbot-ehd` | Simplified conflicting prompt instructions | prompt_template.py |

#### VectorDistance Bug Fix (CRITICAL - Commit 7959fd2)

**Problem:** CosmosDB VectorDistance with cosine returns [0, 2] range, but code assumed [0, 1].
- Distance 1.2 → Similarity -0.2 (NEGATIVE!) - chunks filtered out incorrectly

**Fix:** Added `distance_to_similarity()` helper in `retrieval_service.py`:
```python
def distance_to_similarity(distance: float) -> float:
    if distance is None:
        return 0.0
    similarity = 1 - (distance / 2)  # Normalize [0,2] to [0,1]
    return max(0.0, min(1.0, similarity))  # Clamp
```

### Beads Issue Tracker Status

**Total: 35 issues**
- Closed: 25
- In Progress: 1 (digihub-chatbot-srv - Split monolithic query analyzer prompt)
- Open: 9 (Unit testing tasks)

#### Open Testing Tasks (Priority Order)

| Priority | Issue ID | Task |
|----------|----------|------|
| P1 | `digihub-chatbot-5qq` | Create unit tests for response_generator.py |
| P1 | `digihub-chatbot-5by` | Create unit tests for query_analyzer.py |
| P1 | `digihub-chatbot-2u1` | Create unit tests for dataprocessor.py |
| P2 | `digihub-chatbot-6hh` | Create test infrastructure (conftest.py, pytest.ini) |
| P2 | `digihub-chatbot-bv4` | Create unit tests for relevance_judge.py |
| P2 | `digihub-chatbot-jis` | Create unit tests for authorization_checker.py |
| P2 | `digihub-chatbot-zom` | Improve existing test coverage to 80% |
| P2 | `digihub-chatbot-dti` | Add CI/CD test integration in Azure Pipelines |
| P3 | `digihub-chatbot-vyh` | Create integration tests for end-to-end flows |

### Documentation Created

- `DigiHub_Chatbot_Changes_Report.docx` - Comprehensive changes report with 9 tables
- `.env.example` - Environment configuration template

### Local Development Setup

#### Environment File Location
The `.env` file with credentials is in the root folder:
```
/Users/zohaibtanwir/projects/digihub-chatbot/.env
```

It has been copied to:
```
/Users/zohaibtanwir/projects/digihub-chatbot/digihub-chatbot-query-pipeline/.env
```

#### Required Environment Variables
```
CONFIG_URL=https://digihubdev.sita.aero/api/configserver/digihub-chatbot-query-pipeline/dev
KEY_VAULT_URL=https://kv-dev-westeurope-02.vault.azure.net/
AZURE_CLIENT_ID=<service-principal-id>
AZURE_TENANT_ID=<tenant-id>
AZURE_CLIENT_SECRET=<client-secret>
```

#### Running Locally

**IMPORTANT:** Config server requires SITA VPN connection.

```bash
# Connect to VPN first, then:
cd /Users/zohaibtanwir/projects/digihub-chatbot/digihub-chatbot-query-pipeline
python -m uvicorn src.app:app --host 0.0.0.0 --port 8080 --reload
```

### Python Version

Upgraded to **Python 3.13.11** (from 3.12.7)
- Path: `/opt/homebrew/opt/python@3.13/libexec/bin/python3`
- Updated `~/.zprofile` to use Python 3.13 as default

### Git Commits This Session

| Commit | Description |
|--------|-------------|
| `5f59a37` | chore: Sync Beads export state |
| `a9e3de7` | chore: Update Beads with VectorDistance bug fix tracking |
| `7959fd2` | **fix(retrieval): Fix VectorDistance to similarity conversion** |
| `7d23732` | chore: Add unit test implementation tasks to Beads |
| `711f23b` | fix(query-pipeline): Fix 8 critical issues impacting response quality |
| `62dfb14` | docs: Add comprehensive changes report |

### Next Steps When Resuming

1. **Connect to SITA VPN** to access config server
2. **Run the application** to test the VectorDistance fix:
   ```bash
   cd /Users/zohaibtanwir/projects/digihub-chatbot/digihub-chatbot-query-pipeline
   python -m uvicorn src.app:app --host 0.0.0.0 --port 8080 --reload
   ```
3. **Test with a sample query** to verify chunks are not filtered incorrectly
4. **Implement unit tests** starting with `digihub-chatbot-6hh` (test infrastructure)
