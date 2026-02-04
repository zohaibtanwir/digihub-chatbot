# Conversational Context Engine

## Overview
The **Conversational Context Engine** is the core system that enables the DigiHub chatbot to maintain conversation context and handle follow-up questions naturally. It combines session management, entity tracking, reference resolution, and intelligent query processing to deliver a seamless multi-turn conversation experience.

### Feature Name
**Official Name:** Conversational Context Engine (CCE)

### Core Components
1. **Session Management:** Persistent conversation storage with rich metadata
2. **Session Dependency Detection:** LLM-based analysis to detect follow-up questions
3. **Reference Resolution:** Pronoun-to-entity linking (e.g., "it" → "WorldTracer")
4. **Contextual Service Line Filtering:** Restricts retrieval to service lines from previous context
5. **Smart Query Merging:** Context-aware query expansion for improved retrieval
6. **Entity Tracking:** Automatic extraction and storage of services, topics, and technical terms
7. **Graduated Context Windows:** Different history depths optimized for each component
8. **Question-First Vector Retrieval:** Dual-embedding architecture with 100% question-based ranking

### Usage in Code and Documentation
- **Class/Module References:** `ConversationalContextEngine` or `context_engine`
- **Environment Variables:** `CCE_ENABLE_ENTITY_TRACKING`, `CCE_ENABLE_REFERENCE_RESOLUTION`
- **Logging Prefixes:** `[CCE]` for context-related log messages
- **API Documentation:** "Powered by Conversational Context Engine"
- **User-Facing:** "Our chatbot uses the Conversational Context Engine to remember your conversation"

---

## 1. Session Creation and Management

### Session ID Structure
Format: `{hashed_user_id}-{YYYY-MM-DD}-{increment}`

**Example:** `a7f3e9d2...1b-2026-01-26-1`

**Implementation:** [session_service.py:35-39](src/services/session_service.py#L35-L39)

**Process:**
1. User email ID is hashed using SHA-256 via `HashUtils.hash_user_id()`
2. Current date appended in `YYYY-MM-DD` format
3. Increment counter tracks multiple sessions per day (1, 2, 3...)
4. First conversation of the day gets `-1`, second gets `-2`, etc.

**Validation:** [chat_views.py:56-64](src/views/chat_views.py#L56-L64)
- System verifies session ID belongs to requesting user
- Extracts user hash from session ID and compares with current user
- Raises exception if mismatch detected

---

## 2. Data Stored in Session Context

### Session Object Structure
Each message (both user and assistant) is stored as a `Session` object:

**Location:** [session_object.py:6-20](src/dto/session_object.py#L6-L20)

```python
@dataclass
class Session:
    id: str                          # Unique document ID in CosmosDB
    messageId: str                   # Unique message identifier
    sessionId: str                   # Session identifier
    userId: str                      # Hashed user ID
    impersonated_user_id: str        # Original user if impersonated
    sender: str                      # "user" or "assistant"
    timestamp: int                   # Unix timestamp
    text: str                        # Message content
    citation: List[dict]             # Source documents (only for assistant)
    entities: List[str]              # Extracted entities (only for assistant)
    chunk_service_line: List[int]    # Service line IDs from retrieved chunks (only for assistant)
    score: Optional[float]           # Similarity score (only for assistant)
    confidence: Optional[float]      # LLM confidence (only for assistant)
    feedback_score: Optional[int]    # User feedback rating
    disclaimer: Optional[str]        # Warning message if applicable
```

### What Gets Saved Per Conversation Turn

**User Message:**
- Original query text (e.g., "How do I configure it?")
- Empty citation and entities arrays
- Timestamp of submission

**Assistant Message:**
- Generated response text
- Citations with file paths and sections
- Extracted entities:
  - **Services:** Product/service names (e.g., "WorldTracer", "BagManager")
  - **Topics:** Subject areas (e.g., "configuration", "troubleshooting")
  - **Technical Terms:** APIs, protocols, technical concepts
- Chunk service lines: List of service line IDs from retrieved chunks (e.g., [1, 0])
- Similarity score from vector search (0.0-1.0)
- LLM confidence score (0.0-1.0)
- Disclaimer if low confidence or out-of-scope

**Storage:** [session_service.py:42-64](src/services/session_service.py#L42-L64)
- Both messages saved separately to CosmosDB `Session` container
- Partitioned by `userId` for efficient retrieval
- Ordered by `timestamp` for chronological history

---

## 3. Relating Earlier Questions to New Questions

The system uses **three complementary mechanisms** to maintain context:

### Mechanism 1: Session Dependency Detection

**Purpose:** Determine if new query depends on previous context

**Implementation:** [query_analyzer.py:35-86](src/chatbot/query_analyzer.py#L35-L86)

**Process:**
1. Retrieve last 5 messages as context window
2. Pass to LLM with classification prompt
3. LLM analyzes if current query builds on previous questions

**Prompt Template:** [prompt_template.py:14-16](src/enums/prompt_template.py#L14-L16)

**Detection Criteria:**
- Query is continuation of previous question
- Uses pronouns referencing earlier content
- Builds upon or follows up on previous answer

**Examples:**
```
Session-Dependent (is_session_dependent=true):
Q1: "Tell me about Bag Manager?"
Q2: "How do I find a lost bag?"  ← Continues BagManager topic

Q1: "What errors occur in Mail Manager?"
Q2: "Why does that error happen?" ← References "that error"

Session-Independent (is_session_dependent=false):
Q1: "Tell me about Bag Manager?"
Q2: "Tell me about Community Messaging?" ← New topic
```

**Output JSON:**
```json
{
  "is_session_dependent": true,
  "translation": "How do I configure it?",
  "language": "english",
  "expanded_queries": ["Configure WorldTracer", "WorldTracer setup"],
  "prompt_vulnerability_level": 0.0,
  "is_prompt_vulnerable": false
}
```

### Mechanism 2: Reference Resolution (Pronoun Linking)

**Purpose:** Resolve pronouns like "it", "that", "them" to specific entities

**Implementation:** [context_manager.py:117-202](src/chatbot/context_manager.py#L117-L202)

**Process:**

**Step 1: Detect References**
```python
PRONOUN_PATTERNS = r'\b(it|that|them|those|this|the service|the product|...)\b'
has_references("How do I configure it?")  # Returns: True
```

**Step 2: Retrieve Entities from Recent Messages**
- Fetch entities from last 5 assistant messages
- Group into: services, topics, technical_terms
- Example output:
```python
{
  "services": ["WorldTracer", "BagManager"],
  "topics": ["configuration", "setup"],
  "technical_terms": ["API", "authentication"]
}
```

**Step 3: LLM-Based Resolution**
- Prompt: "Given query 'How do I configure it?' and entities [WorldTracer, ...], resolve pronouns"
- LLM returns: `{"resolved_query": "How do I configure WorldTracer?"}`

**Example Transformation:**
```
Original:  "How do I configure it?"
Resolved:  "How do I configure WorldTracer?"

Original:  "Why does that error occur?"
Resolved:  "Why does the Mail Manager subscription error occur?"

Original:  "Tell me more about them"
Resolved:  "Tell me more about lost baggage procedures"
```

### Mechanism 3: Contextual Service Line Filtering

**Purpose:** Limit retrieval to service lines from previous responses in contextual conversations

**Implementation:** [response_generator.py:404-448](src/chatbot/response_generator.py#L404-L448)

**Process:**

When a query is detected as session-dependent, the system narrows the retrieval scope to only the service lines that were relevant in the previous response, improving focus and reducing noise.

**Step 1: Save Service Lines from Retrieved Chunks**
```python
# During retrieval (Step 7 in generate_response)
chunk_service_line = list(set([doc.get('serviceNameid') for doc in top_docs]))
# Example: [1, 0] (WorldTracer + General Info)

# Save to session (Step 12)
SessionDBService().add_user_assistant_session(
    ...,
    chunk_service_line=chunk_service_line
)
```

**Step 2: Retrieve Previous Service Lines for Contextual Queries**
```python
# Only for session-dependent queries
if is_session_dependent:
    previous_service_lines = SessionDBService().retrieve_session_service_lines(
        user_id=user_id,
        session_id=session_id,
        limit=1  # Most recent assistant message
    )
    # Returns: [1, 0] from previous turn
```

**Step 3: Filter Service Lines**
```python
# Intersect with user's authorized service lines
filtered = [sl for sl in previous_service_lines if sl in final_id_list]

# Always include General Info (0)
retrieval_service_lines = list(set(filtered + [0]))

# Use filtered list for retrieval instead of all authorized service lines
```

**Example Scenario:**
```
User: "What is WorldTracer?"
→ Searches all authorized service lines: [0, 1, 2, 3, 4, 5, 6, 7]
→ Retrieves chunks from service lines: [1, 0] (WorldTracer + General)
→ Saves chunk_service_line=[1, 0] to session

User: "How do I configure it?" (session-dependent)
→ Retrieves previous chunk_service_line: [1, 0]
→ Filters retrieval to only service lines: [1, 0]
→ More focused results, less noise from irrelevant services
→ Saves new chunk_service_line to session
```

**Benefits:**
- **Reduced Noise:** Eliminates results from unrelated service lines
- **Improved Precision:** Focuses on the service line being discussed
- **Context Continuity:** Maintains service scope across conversation turns
- **Better Performance:** Smaller search space can improve retrieval speed
- **Graceful Fallback:** If no intersection with authorized lines, uses full list

**Storage:** [session_object.py:17](src/dto/session_object.py#L17)
```python
chunk_service_line: List[int] = field(default_factory=list)
```

**Retrieval Method:** [session_service.py:224-260](src/services/session_service.py#L224-L260)

### Mechanism 4: Smart Query Merging for Vector Search

**Purpose:** Enhance retrieval by combining context from recent queries

**Implementation:** [context_manager.py:204-285](src/chatbot/context_manager.py#L204-L285)

**Process:**

**Step 1: Extract Recent User Messages**
- Take only **last 3 user messages** (not all history)
- Exclude assistant responses to reduce noise

**Step 2: Build Weighted Query**
```python
query_parts = []

# Add previous messages with labels
for msg in recent_user_messages[:-1]:
    query_parts.append(f"Previous context: {msg['content']}")

# Add current query with emphasis
query_parts.append(f"Current query: {resolved_query}")

# Add explicit entity mentions (top 3)
if entities:
    top_entities = all_entities[:3]
    query_parts.append(f"Related to: {' '.join(top_entities)}")

merged_query = " ".join(query_parts)
```

**Example Output:**
```
Input Sequence:
Q1: "What is WorldTracer?"
Q2: "How does it work?"
Q3: "How do I configure it?"

Merged Query for Vector Search:
"Previous context: What is WorldTracer?
 Previous context: How does WorldTracer work?
 Current query: How do I configure WorldTracer?
 Related to: WorldTracer configuration setup"
```

**Benefits:**
- Improves semantic similarity search
- Weights current query highest (appears at end)
- Limits context to 3 messages (avoids overwhelming retrieval)
- Includes explicit entity mentions for better matching

---

## 4. Embedding Architecture and Vector Retrieval

The system uses a **dual-embedding architecture** for optimal question-answering performance.

### Document Embeddings (Pre-computed and Stored)

Each knowledge base chunk in CosmosDB contains **two separate embeddings**:

**Location:** [retrieval_service.py:300-314](src/services/retrieval_service.py#L300-L314)

#### A. Questions Embedding (`c.questionsEmbedding`)
- Embedding of questions associated with the chunk
- **Currently Active:** Used for primary ranking (100% weight)
- Optimized for matching user queries (which are also questions)

#### B. Content Embedding (`c.embedding`)
- Embedding of the actual chunk content
- **Currently Inactive:** Retrieved but not used in ranking
- Available for future hybrid scoring implementation

### Query Embeddings (Generated at Runtime)

**Location:** [retrieval_service.py:290](src/services/retrieval_service.py#L290)

When a user submits a query:
```python
query_embedding = self.embeddings.embed_query(query)
```

- Query is embedded using Azure OpenAI embeddings
- Generated fresh for each request
- Compared against pre-stored document embeddings

### Current Retrieval Strategy: Question-First Approach

**Implementation:** [retrieval_service.py:300-314](src/services/retrieval_service.py#L300-L314)

```sql
SELECT TOP 10
    c.id,
    c.serviceNameid,
    c.heading,
    c.serviceName,
    c.content,
    c.metadata.filepath as citation,
    c.questions,
    VectorDistance(c.questionsEmbedding, {query_embedding}) as question_score,  -- PRIMARY
    VectorDistance(c.embedding, {query_embedding}) as content_score              -- Retrieved but unused
FROM c
WHERE ARRAY_CONTAINS({service_line}, c.serviceNameid) AND c.validChunk = 'yes'
ORDER BY VectorDistance(c.questionsEmbedding, {query_embedding})                 -- 100% question-based ranking
```

**Current Behavior:**
1. Query embedded using Azure OpenAI
2. CosmosDB vector search compares query against `questionsEmbedding` field
3. Results ordered by question similarity score only
4. Content score is retrieved but **not used in ranking**
5. Top 10 chunks returned

**Why Question-First?**
- User queries are questions (e.g., "How do I configure WorldTracer?")
- Chunk questions are also questions (e.g., "How to configure WorldTracer?", "WorldTracer setup steps?")
- Direct question-to-question matching yields better semantic alignment
- Content embeddings contain broader context, can dilute specific question intent

### Planned Hybrid Scoring (Currently Disabled)

**Location:** [retrieval_service.py:323-358](src/services/retrieval_service.py#L323-L358) - *Commented out with TODO*

The codebase includes a commented-out hybrid scoring implementation:

```python
# PLANNED (not active):
hybrid_score = (
    0.7 * question_similarity +      # 70% weight on questions
    0.3 * content_similarity          # 30% weight on content
)
```

**When Enabled, This Would:**
- Combine both question and content similarity scores
- Weight question matching at 70% (primary signal)
- Weight content matching at 30% (secondary signal)
- Re-rank results by hybrid score
- Optionally filter by minimum question similarity threshold

**Current Status:** TODO assigned to "Nazeel" - implementation pending

### Embedding Flow in Conversational Context Engine

When CCE processes a follow-up question:

```
1. Smart Query Merging (CCE)
   ↓
   "Previous context: What is WorldTracer?
    Current query: How do I configure WorldTracer?
    Related to: WorldTracer configuration"

2. Embed Merged Query (Runtime)
   ↓
   query_embedding = embeddings.embed_query(merged_query)

3. Vector Search (Question-First)
   ↓
   VectorDistance(c.questionsEmbedding, query_embedding)

4. Return Top 10 Chunks
   ↓
   Ordered by question similarity score
```

### Performance Characteristics

**Embedding Generation:**
- Azure OpenAI API call: ~100-200ms
- Single query embedding per request
- Cached by CCE for multi-stage retrieval

**Vector Search:**
- CosmosDB native vector search
- Indexed on `questionsEmbedding` field
- Query time: ~50-150ms for TOP 10 results
- Filtered by service line authorization

**Total Retrieval Time:**
- Embedding: ~100-200ms
- Vector search: ~50-150ms
- **Total: ~150-350ms**

### Key Insight: Question Embedding Strategy

The dual-embedding architecture reflects a key insight:

**User queries ARE questions** → Best matched against **question embeddings** of chunks, not content embeddings.

Example:
- User: "How do I reset my password?"
- Chunk questions embedding: ["How to reset password?", "Password reset steps?"]
- Chunk content embedding: [Full text about password policies, reset procedures, security guidelines]
- **Question-to-question matching is more precise than question-to-content matching**

This is why the system currently uses 100% question-based retrieval rather than hybrid scoring.

---

## 5. Context Retrieval Limits

The system uses **graduated context windows** for different purposes:

| Purpose | Messages Retrieved | Implementation |
|---------|-------------------|----------------|
| **Full Session History** | Last 10 messages | [session_service.py:101-126](src/services/session_service.py#L101-L126) |
| **Query Classification Window** | Last 5 messages | [response_generator.py:333-342](src/chatbot/response_generator.py#L333-L342) |
| **Reference Resolution** | Last 3 messages | [context_manager.py:139-202](src/chatbot/context_manager.py#L139-L202) |
| **Smart Query Merging** | Last 3 user messages | [context_manager.py:204-285](src/chatbot/context_manager.py#L204-L285) |
| **Entity Retrieval** | Last 5 assistant messages | [session_service.py:151-214](src/services/session_service.py#L151-L214) |
| **Service Line Retrieval** | Last 1 assistant message | [session_service.py:224-260](src/services/session_service.py#L224-L260) |

**Rationale:**
- **10 messages** for full context provides complete conversation view to LLM
- **5 messages** balances context accuracy vs. token cost for classification
- **3 messages** for retrieval prevents query dilution while maintaining relevance
- **5 entity messages** captures recent topics without over-generalizing
- **1 service line message** maintains focused retrieval on the current service being discussed

---

## 6. Complete Flow: Follow-Up Question Processing

### Example Scenario
```
User: "What is WorldTracer?"
Bot:  [Explains WorldTracer is a baggage tracking system...]

User: "How do I configure it?" ← Follow-up question
```

### Step-by-Step Flow

**STEP 1: Receive Request** → [chat_views.py:28-108](src/views/chat_views.py#L28-L108)
- Extract user ID and session ID from request
- Validate session belongs to user
- Get user's authorized service lines

**STEP 2: Retrieve Session History** → [session_service.py:101-126](src/services/session_service.py#L101-L126)
```sql
SELECT c.text, c.sender FROM c
WHERE c.userId = @userId AND c.sessionId = @sessionId
ORDER BY c.timestamp ASC
OFFSET 0 LIMIT 10
```
Returns:
```python
[
  {"role": "user", "content": "What is WorldTracer?"},
  {"role": "assistant", "content": "WorldTracer is a baggage tracking system..."},
  {"role": "user", "content": "How do I configure it?"}
]
```

**STEP 3: Query Classification** → [query_analyzer.py:35-86](src/chatbot/query_analyzer.py#L35-L86)
- Take last 5 messages as context window
- Analyze: `query_classifer("How do I configure it?", session_context_window)`
- Returns:
```json
{
  "is_session_dependent": true,
  "translation": "How do I configure it?",
  "language": "english",
  "expanded_queries": ["Configure WorldTracer", "WorldTracer setup"]
}
```

**STEP 4A: Retrieve Session Entities** → [session_service.py:151-214](src/services/session_service.py#L151-L214)
```sql
SELECT c.entities FROM c
WHERE c.userId = @userId AND c.sessionId = @sessionId
ORDER BY c.timestamp DESC
OFFSET 0 LIMIT 5
```
Returns:
```python
{
  "services": ["WorldTracer"],
  "topics": ["baggage tracking", "configuration"],
  "technical_terms": ["API", "authentication"]
}
```

**STEP 4B: Reference Resolution** → [context_manager.py:139-202](src/chatbot/context_manager.py#L139-L202)
- Detect pronoun: "it" in query
- Resolve using entities and recent history
- Output: `"How do I configure WorldTracer?"`

**STEP 4C: Smart Query Building** → [context_manager.py:204-285](src/chatbot/context_manager.py#L204-L285)
- Extract last 3 user messages
- Build weighted query:
```
"Previous context: What is WorldTracer?
 Current query: How do I configure WorldTracer?
 Related to: WorldTracer configuration setup"
```

**STEP 4D: Contextual Service Line Filtering** → [response_generator.py:404-448](src/chatbot/response_generator.py#L404-L448)
- Since `is_session_dependent=true`, retrieve previous chunk_service_line
```sql
SELECT c.chunk_service_line FROM c
WHERE c.userId = @userId AND c.sessionId = @sessionId AND c.sender = 'assistant'
ORDER BY c.timestamp DESC
OFFSET 0 LIMIT 1
```
- Returns: `[1, 0]` (WorldTracer + General Info from previous response)
- Intersect with user's authorized service lines: `[1, 0]`
- Use filtered list for retrieval instead of all authorized service lines

**STEP 5: Vector Retrieval** → [retrieval_service.py](src/services/retrieval_service.py)
- Embed merged query using Azure OpenAI embeddings
- Search CosmosDB vector index
- Filter by contextual service lines `[1, 0]` (not all authorized service lines)
- Return top 7 chunks with highest cosine similarity
- Extract chunk_service_line from retrieved documents
- Deduplicate and organize by service line

**STEP 6: Prepare LLM Context** → [response_generator.py:519-533](src/chatbot/response_generator.py#L519-L533)
```python
populated_context_session = {
  "entities": {
    "services": ["WorldTracer"],
    "topics": ["configuration"],
    "technical_terms": ["API"]
  },
  "resolved_query": "How do I configure WorldTracer?",
  "is_session_dependent": true,
  "previous_service_lines": [5]  # WorldTracer service line ID
}
```

**STEP 7: LLM Response Generation** → [response_generator.py:100-292](src/chatbot/response_generator.py#L100-L292)
- Format prompt using RESPONSE_TEMPLATE
- Include:
  - User query (original): "How do I configure it?"
  - Retrieved context (7 chunks from vector search)
  - Session context (entities, resolved query)
  - Full conversation history (10 messages)
  - Current date for temporal context
- Call Azure OpenAI:
```python
messages = user_chat_history + [{"role": "user", "content": formatted_prompt}]
response = client.chat.completions.create(model=model, messages=messages)
```
- Parse JSON response:
```json
{
  "response": "To configure WorldTracer, follow these steps: 1. Access the admin panel...",
  "citation": [{"File": "docs/worldtracer-config.pdf", "Section": "Configuration"}],
  "confidence": 0.92,
  "score": 0.87,
  "disclaimer": null
}
```

**STEP 8: Entity Extraction** → [response_generator.py:616-660](src/chatbot/response_generator.py#L616-L660)
- Extract entities from Q&A pair using LLM
- Input: `Q: "How do I configure WorldTracer?" A: [response]`
- Output:
```python
{
  "services": ["WorldTracer"],
  "topics": ["configuration", "setup", "admin panel"],
  "technical_terms": ["API", "authentication", "credentials"]
}
```

**STEP 9: Save Session** → [session_service.py:42-64](src/services/session_service.py#L42-L64)
```python
# User message entry
{
  "sender": "user",
  "text": "How do I configure it?",
  "entities": [],
  "chunk_service_line": [],
  "citation": []
}

# Assistant message entry
{
  "sender": "assistant",
  "text": "To configure WorldTracer...",
  "entities": ["WorldTracer", "configuration", "setup", "API", "authentication"],
  "chunk_service_line": [1, 0],  # Service lines from retrieved chunks
  "citation": [{"File": "docs/worldtracer-config.pdf", "Section": "Configuration"}],
  "confidence": 0.92,
  "score": 0.87
}
```

**STEP 10: Return Response** → [chat_views.py:28-108](src/views/chat_views.py#L28-L108)
```json
{
  "session_id": "hash123-2026-01-26-1",
  "message_id": "uuid-of-this-turn",
  "response": "To configure WorldTracer...",
  "citation": [{"File": "docs/worldtracer-config.pdf", "Section": "Configuration"}],
  "confidence": 0.92,
  "score": 0.87,
  "disclaimer": null,
  "timestamp": "2026-01-26T10:30:45Z"
}
```

---

## 7. Key Architecture Decisions

### Why Last 3 Messages for Query Merging?
- **Too Few (1):** Loses valuable context
- **Too Many (5+):** Dilutes current query intent, increases noise
- **Just Right (3):** Balances relevance and specificity

### Why Extract Entities Per Message?
- Enables accurate pronoun resolution in future turns
- Avoids re-processing entire history for every query
- Provides explicit topic tracking across conversation

### Why Separate User and Assistant Messages in Storage?
- Allows independent retrieval (e.g., only user messages for query building)
- Different metadata for each (assistant has citations, entities)
- Enables differential analysis (user patterns vs. bot responses)

### Why Use LLM for Reference Resolution Instead of Simple Rules?
- Handles complex cases: "the error", "those features", "the product"
- Disambiguates when multiple entities present
- Context-aware: knows which service from conversation flow

### Why Graduated Context Windows?
- **Classification (5 messages):** Needs enough context to detect continuation
- **Retrieval (3 messages):** Focused recent context prevents query dilution
- **LLM Response (10 messages):** Full context for natural conversation
- Optimizes token usage while maintaining quality

### Why Question-First Retrieval (100% Question Embeddings)?
- **User queries are questions:** Natural language queries like "How do I configure it?" are questions
- **Chunk questions are questions:** Pre-generated questions for each chunk match user intent patterns
- **Better semantic alignment:** Question-to-question matching is more precise than question-to-content matching
- **Content embeddings too broad:** Content contains full context, which can dilute specific question intent
- **Hybrid scoring available:** Code exists for 70/30 weighting but currently disabled (TODO: Nazeel)
- **Performance:** Single vector search is faster than hybrid re-ranking

### Why Store Both Embeddings When Only One is Used?
- **Future flexibility:** Hybrid scoring can be enabled without re-embedding all documents
- **A/B testing:** Can easily test different weighting strategies (100% question vs. 70/30 hybrid)
- **Fallback mechanism:** Content embeddings available if question embeddings fail or underperform
- **Minimal storage cost:** Embeddings are vectors, storage cost is negligible compared to re-computation

### Why Use Contextual Service Line Filtering?
- **Reduced Noise:** When discussing WorldTracer, no need to search BagManager or other unrelated services
- **Improved Precision:** Retrieval stays focused on the service line being discussed
- **Context Continuity:** Multi-turn conversations maintain service scope automatically
- **Performance:** Smaller search space can improve retrieval speed
- **Authorization Respected:** Always intersects with user's authorized service lines
- **Graceful Fallback:** If no intersection, falls back to full authorized list
- **General Info Always Included:** Service line 0 (General Info) always added for flexibility

---

## 8. Feature Toggles

The system supports runtime configuration via environment variables:

**Location:** [context_manager.py:30-38](src/chatbot/context_manager.py#L30-L38)

```python
ENABLE_ENTITY_TRACKING=true         # Extract and store entities per message
ENABLE_REFERENCE_RESOLUTION=true    # Resolve pronouns to explicit entities
ENABLE_SMART_QUERY_MERGING=true    # Merge recent queries for retrieval
```

**Production Control:**
- Disable features during incidents without code changes
- A/B test context strategies
- Gradual rollout of new context mechanisms

---

## 9. Critical Files Reference

| Component | File Path | Lines |
|-----------|-----------|-------|
| Session ID Creation | [session_service.py](src/services/session_service.py) | 35-39 |
| Session Storage | [session_service.py](src/services/session_service.py) | 42-64 |
| History Retrieval | [session_service.py](src/services/session_service.py) | 101-126 |
| Entity Retrieval | [session_service.py](src/services/session_service.py) | 151-214 |
| Service Line Retrieval | [session_service.py](src/services/session_service.py) | 224-260 |
| Query Classification | [query_analyzer.py](src/chatbot/query_analyzer.py) | 35-86 |
| Reference Resolution | [context_manager.py](src/chatbot/context_manager.py) | 117-202 |
| Contextual Service Line Filtering | [response_generator.py](src/chatbot/response_generator.py) | 404-448 |
| Smart Query Merging | [context_manager.py](src/chatbot/context_manager.py) | 204-285 |
| Question-First Vector Retrieval | [retrieval_service.py](src/services/retrieval_service.py) | 254-378 |
| Query Embedding Generation | [retrieval_service.py](src/services/retrieval_service.py) | 290 |
| Main Orchestration | [response_generator.py](src/chatbot/response_generator.py) | 294-614 |
| Chat Endpoint | [chat_views.py](src/views/chat_views.py) | 28-108 |
| Session Object Schema | [session_object.py](src/dto/session_object.py) | 6-21 |

---

## 10. Performance Characteristics

**Session Retrieval:**
- Query: `O(log n)` with CosmosDB indexed query on `userId` + `sessionId`
- Limit: 10 messages (typically 5 conversation turns)
- Network: Single round-trip to CosmosDB

**Entity Retrieval:**
- Query: `O(log n)` with CosmosDB indexed query
- Limit: 5 assistant messages
- Cached per request (no redundant calls)

**Service Line Retrieval:**
- Query: `O(log n)` with CosmosDB indexed query
- Limit: 1 assistant message (most recent)
- Only triggered for session-dependent queries
- Latency: ~30-50ms

**Reference Resolution:**
- Only triggered when pronouns detected (regex check first)
- LLM call: ~500-1000 tokens
- Latency: ~200-500ms

**Smart Query Merging:**
- In-memory operation: `O(k)` where k=3 messages
- No external calls
- Latency: <1ms

**Vector Retrieval (Question-First):**
- Query embedding generation: ~100-200ms (Azure OpenAI API call)
- CosmosDB vector search: ~50-150ms (native vector index on questionsEmbedding)
- Total retrieval: ~150-350ms per query

**Total Continuation Overhead:**
- Session retrieval: ~50-100ms
- Entity retrieval: ~50-100ms (only if session-dependent)
- Service line retrieval: ~30-50ms (only if session-dependent)
- Reference resolution: ~200-500ms (only if pronouns present)
- Query merging: <1ms
- Vector retrieval: ~150-350ms
- **Total: ~480-1100ms** for follow-up questions (including vector search)

---

## Summary

The **Conversational Context Engine** maintains conversation context through a sophisticated multi-layer architecture:

### Key Components
1. **Session Management:** Persistent storage of all messages with rich metadata in CosmosDB
2. **Dependency Detection:** LLM-based analysis determines if queries build on previous context
3. **Reference Resolution:** Pronouns are resolved to explicit entities using conversation history
4. **Contextual Service Line Filtering:** Restricts retrieval to service lines from previous responses
5. **Smart Query Merging:** Recent context is intelligently combined for improved retrieval
6. **Graduated Context Windows:** Different amounts of history used for different purposes
7. **Entity Tracking:** Services, topics, and terms extracted and stored per turn
8. **Dual-Embedding Architecture:** Question and content embeddings stored, question-first retrieval active

### Design Philosophy
The Conversational Context Engine balances three critical objectives:
- **Conversation Quality:** Maintains natural dialogue flow and contextual understanding
- **Performance:** Uses limited history retrieval to optimize latency and token cost
- **Accuracy:** Employs explicit entity resolution to prevent ambiguity and misinterpretation

This engine enables users to ask follow-up questions like "How do I configure it?" or "Why does that error occur?" without repeating service names or context, creating a natural conversational experience similar to talking with a human expert.
