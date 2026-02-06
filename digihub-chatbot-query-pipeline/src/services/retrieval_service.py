import json
import math
import re
import time
from typing import Optional, Any, List, Dict, Tuple, Set
from src.services.cosmos_db_service import CosmosDBClientSingleton
from src.services.embedding_service import AzureEmbeddingService
from src.utils.config import (
    COSMOSDB_SERVICE_NAME_MAPPING_CONTAINER_NAME,
    SESSION_CONTAINER_NAME,
    KNOWLEDGE_BASE_CONTAINER,
    COSMOSDB_SHAREPOINT_DATA_CONTAINER_NAME,
    MIN_SIMILARITY_THRESHOLD,
    ENABLE_METADATA_FILTERING
)
from src.utils.logger import logger
from src.utils.request_utils import timing_decorator
from src.utils.metrics import RetrievalMetrics, LatencyTracker


class RetreivalService:

    def __init__(self):
        self.database = CosmosDBClientSingleton().get_database()
        self.container = self.database.get_container_client(KNOWLEDGE_BASE_CONTAINER)
        self.embeddings = AzureEmbeddingService().get_embeddings()
        # Common English stopwords to exclude from keyword matching
        self._stopwords = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
            'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
            'we', 'they', 'what', 'which', 'who', 'whom', 'how', 'when', 'where',
            'why', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
            'too', 'very', 'just', 'also', 'now', 'here', 'there', 'then', 'once'
        }

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into lowercase words, removing stopwords and short tokens.
        """
        if not text:
            return []
        # Convert to lowercase and extract alphanumeric tokens
        tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
        # Filter out stopwords and very short tokens (less than 2 chars)
        return [t for t in tokens if t not in self._stopwords and len(t) >= 2]

    def _extract_key_phrases(self, query: str) -> List[str]:
        """
        Extract potential key phrases from the query (multi-word terms).
        For example: "What is CI Analysis?" -> ["ci analysis"]
        """
        # Remove common question words and clean up
        cleaned = re.sub(r'\b(what|how|why|when|where|who|which|is|are|do|does|can|could|would|should|the|a|an)\b', '', query.lower())
        cleaned = re.sub(r'[^\w\s]', ' ', cleaned)  # Remove punctuation
        cleaned = ' '.join(cleaned.split())  # Normalize whitespace

        phrases = []
        if cleaned and len(cleaned) >= 2:
            phrases.append(cleaned.strip())
        return phrases

    def _calculate_keyword_score(self, query: str, doc: Dict) -> float:
        """
        Calculate keyword/BM25-like score for a document given a query.

        Uses a simplified BM25-inspired scoring:
        1. Tokenize query and document
        2. Calculate term frequency matches
        3. Boost for exact phrase matches in heading
        4. Normalize to 0-1 range

        Args:
            query: The search query
            doc: Document dict with 'content' and 'heading' fields

        Returns:
            Keyword score between 0.0 and 1.0
        """
        heading = doc.get('heading', '') or ''
        content = doc.get('content', '') or ''

        # Tokenize query and document
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return 0.0

        # Combine heading and content for matching, but weight heading higher
        heading_tokens = set(self._tokenize(heading))
        content_tokens = set(self._tokenize(content))

        # Calculate term matches
        query_token_set = set(query_tokens)
        heading_matches = query_token_set & heading_tokens
        content_matches = query_token_set & content_tokens
        all_matches = heading_matches | content_matches

        if not all_matches:
            return 0.0

        # Base score: proportion of query terms found
        base_score = len(all_matches) / len(query_token_set)

        # Heading boost: if query terms appear in heading, boost significantly
        # Heading is typically more relevant than body content
        heading_boost = 0.0
        if heading_matches:
            heading_match_ratio = len(heading_matches) / len(query_token_set)
            heading_boost = 0.3 * heading_match_ratio  # Up to 30% boost for heading matches

        # Exact phrase boost: check if key phrases appear in heading
        phrase_boost = 0.0
        key_phrases = self._extract_key_phrases(query)
        heading_lower = heading.lower()
        for phrase in key_phrases:
            if phrase in heading_lower:
                phrase_boost = 0.4  # 40% boost for exact phrase in heading
                break

        # Combine scores (cap at 1.0)
        final_score = min(1.0, base_score + heading_boost + phrase_boost)

        return final_score

    @timing_decorator
    def rag_retriever_agent(self,query: str, container_name: str, service_line: Optional[list[int]], top_k: int = 5, content_type_filter: Optional[str] = None):
        """
        Retrieve the most relevant context (chunks) from CosmosDB for the given query.

        Args:
            query (str): The user's query.
            container_name (str): The CosmosDB container name.
            service_line (str): The service line to filter documents.
            top_k (int): The number of top relevant documents to retrieve.
            content_type_filter (str): Optional content type to filter by (e.g., 'UserGuide' for generic queries).

        Returns:
            tuple: The concatenated context and metadata of the most relevant document.
        """

        # Retrieve the top relevant documents using content-only matching (0% question, 100% content)
        top_docs,query_embedding,citation = self.retrieve_with_question_matching(query, container_name, service_line, top_k=top_k, question_boost_weight=0.0, content_type_filter=content_type_filter)
        top_docs =  top_docs
        logger.info(f"#$#-top_docs {json.dumps(top_docs,indent=4)}")
        if not top_docs:
            top_doc = []
            return [], {}, {}, [], [], []
        else:
            top_doc = top_docs[0]


        metadata = {
            "file": top_docs[0].get("metadata", {}).get("filepath", "")
        }
        chunk_service_line = list(set([doc.get('serviceNameid') for doc in top_docs]))
        logger.info(f"Retrieved Metadata: {metadata} Top Doc:{top_doc}")
        return top_docs, metadata , top_doc , chunk_service_line,query_embedding , citation


    def retrieve_vectordb(self,query: str, container_name: str, service_line:  Optional[list[int]], top_k: int = 7):
        """
        Retrieve the most relevant documents from CosmosDB based on the query.
        """
        start_embedding =time.time()
        logger.info(f"Service Line {service_line}")
        query_embedding = self.embeddings.embed_query(query)
        container = self.database.get_container_client(container_name)
        # TODO: remove stop words from query
        query_word_object = json.dumps(query.split(" "))[1:-1]
        end_embedding = time.time()

        # Remove the filter and allow access to all subscriptions if user is impersonated
        start = time.time()
        query_filter = f"WHERE ARRAY_CONTAINS({service_line},c.serviceNameid)"
        query_stmt = f"""
            SELECT TOP 20 c.id,c.serviceNameid,c.heading,c.serviceName,c.content,c.metadata.filepath as citation,VectorDistance(c.embedding, {query_embedding} ) as cosine_score
                FROM c 
                {query_filter}
                ORDER BY 
                 VectorDistance(c.embedding, {query_embedding} )
            """

        # parameters = [{"name": "@folder", "value": service_line}]
        docs = list(container.query_items(
            query=query_stmt,
            # parameters=parameters,
            enable_cross_partition_query=True,
            populate_query_metrics=True
        ))
        logger.info(f"all docs {json.dumps(docs,indent=4)}")
        end = time.time()
        # ordered_list = sorted(
        #     docs,
        #     key=lambda x: (-x['cosine_score'], x['id'])
        # )[:3]
        # Removing the Chunks where the content of the chunk is having only heading
        ordered_list_filtered = sorted(
            [doc for doc in docs if not (len(doc.get('content', '').strip().splitlines()) == 1 and doc['content'].strip().startswith("## "))],
            key=lambda x: (-x['cosine_score'], x['id'])
        )[:7]
        logger.info(f"ordered_list_filtered{json.dumps(ordered_list_filtered,indent=4)}")
        query_metrics = container.client_connection.last_response_headers.get('x-ms-documentdb-query-metrics')
        logger.info(f"Embedding Time: {end_embedding-start_embedding} Query Response:{end-start} Query Metrics: {query_metrics}")
        citation = [{"File": doc.get('citation'), "Section": doc.get('heading')} for doc in ordered_list_filtered]
        return ordered_list_filtered,query_embedding,citation


    def retrieve_general_info_chunks(self,query: str, container_name: str, service_line:  list = [], top_k: int = 5,
                                     query_embedding:list = []
                                     ):
        """
        Retrieve the most relevant documents from CosmosDB based on the query.
        """
        try:
            start = time.time()
            logger.info(f"Service Line {service_line}")
            query_embedding = self.embeddings.embed_query(query)
            container = self.database.get_container_client(container_name)
            query_filter = f"WHERE ARRAY_CONTAINS({service_line},c.serviceNameid)"
            query_stmt = f"""
                SELECT TOP 20 c.id,c.partitionKey,c.heading,c.serviceNameid,c.serviceName,c.content,c.metadata.filepath as citation,VectorDistance(c.embedding, {query_embedding} ) as cosine_score
                    FROM c 
                    {query_filter}
                    ORDER BY
                    VectorDistance(c.embedding, {query_embedding} )
                """

            # parameters = [{"name": "@folder", "value": service_line}]
            top_docs = list(container.query_items(
                query=query_stmt,
                # parameters=parameters,
                enable_cross_partition_query=True
            ))
            if not top_docs:
                top_doc = []
                return [], {}, {}, [], [] 
            else:
                top_doc = top_docs[0]
            metadata = {
                "file": top_docs[0].get("metadata", {}).get("filepath", "")
            }
            chunk_service_line = list(set([doc.get('serviceNameid') for doc in top_docs]))
            top_docs = sorted(top_docs, key=lambda x: x['cosine_score'], reverse=True)[:3]
            citation = [{"File": doc.get('citation'), "Section": doc.get('heading')} for doc in top_docs]
            top_docs = self.retreive_neighbouring_chunks(chunks=top_docs,
                                                        container=container,
                                                        service_line=service_line
                                                        )

            # logger.info(f"[Latency] general_info_chunk: {en}")
            end = time.time()
            logger.info(f"[Latency] retreive_general_info.response: {end - start:.2f}s")
            return top_docs, metadata, top_doc, chunk_service_line , citation
        except Exception as e:
            logger.error(f"Error querying general info Cosmos DB: {str(e)}")
            return []


    def retreive_neighbouring_chunks(self, chunks: list, container: Any, service_line: list = [], top_k: int = 5):
        neighbouring_chunk_list = []
        for chunk in chunks:
            prefix, index = chunk.get('id').rsplit("-", 1)
            index = int(index)
            neighboring_ids = [f"{prefix}-{i}" for i in range(index - 1, index + 3)]
            id_list_sql = ", ".join(f'"{chunk_id}"' for chunk_id in neighboring_ids)
            query_stmt = f"""
                        SELECT c.id,c.partitionKey,c.serviceNameid,c.serviceName,c.content,c.metadata
                            FROM c    
                            WHERE c.partitionKey = "{chunk.get('partitionKey')}"
                                  AND c.id IN ({id_list_sql})
                        """
            top_docs = list(container.query_items(
                query=query_stmt,
                # parameters=parameters,
                enable_cross_partition_query=True
            ))
            neighbouring_chunk_list = neighbouring_chunk_list + top_docs

        return neighbouring_chunk_list

    def get_ranked_service_line_chunk(self, query: str, exclude_service_lines: Optional[List[int]] = None):

        logger.info(f"Service Line {query}")
        query_embedding = self.embeddings.embed_query(query)
        container = self.database.get_container_client(KNOWLEDGE_BASE_CONTAINER)
        
        query_stmt_base = f"""
            SELECT TOP 20 c.id,c.partitionKey,c.heading,c.serviceNameid,c.serviceName,c.content,c.metadata.filepath as citation,VectorDistance(c.embedding, @query_embedding) as cosine_score
            FROM c
        """

        parameters = [
            {"name": "@query_embedding", "value": query_embedding}
        ]

        where_clause = ""
        if exclude_service_lines:
            # Dynamically create placeholders for the IN clause
            in_clause_params = ','.join([f'@param{i}' for i in range(len(exclude_service_lines))])
            where_clause = f" WHERE c.serviceNameid NOT IN ({in_clause_params})"
            for i, value in enumerate(exclude_service_lines):
                parameters.append({"name": f"@param{i}", "value": value})

        query_stmt_suffix = " ORDER BY VectorDistance(c.embedding, @query_embedding)"

        final_query = query_stmt_base + where_clause + query_stmt_suffix

        docs = list(container.query_items(
            query=final_query,
            parameters=parameters,
            enable_cross_partition_query=True
        ))
        ordered_list_filtered = sorted(
            [doc for doc in docs if not (len(doc.get('content', '').strip().splitlines()) == 1 and doc['content'].strip().startswith("## "))],
            key=lambda x: (-x['cosine_score'], x['id'])
        )[:3]
        return ordered_list_filtered


    def get_all_service_line(self):
        container = self.database.get_container_client(COSMOSDB_SERVICE_NAME_MAPPING_CONTAINER_NAME)
        query_stmt = f"""             
                        SELECT DISTINCT c.service_id,c.name
                        FROM c
                            """
        # parameters = [{"name": "@folder", "value": service_line}]
        docs = list(container.query_items(
            query=query_stmt,
            # parameters=parameters,
            enable_cross_partition_query=True
        ))
        service_lines = [
        {"id": doc.get('service_id'), "name": doc.get('name')} 
        for doc in docs
        ]
        logger.info(f"Fetched all distinct service line : {service_lines}")
        return service_lines
    def _cosine_similarity(self, vec1: list, vec2: list) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score (0 to 1)
        """
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    @timing_decorator
    def retrieve_with_question_matching(
        self,
        query: str,
        container_name: str,
        service_line: Optional[list[int]],
        top_k: int = 7,
        question_boost_weight: float = 0.0,
        min_question_similarity: float = None,
        content_type_filter: str = None,
        year_filter: str = None
    ) -> Tuple[list, list, list]:
        """
        Retrieve chunks using hybrid search: semantic similarity + keyword matching.

        This method combines vector similarity with BM25-like keyword matching:
        1. Queries CosmosDB ordering by content embedding similarity (semantic)
        2. Calculates keyword scores based on query term matches in heading/content
        3. Combines scores: final = 0.7 * semantic + 0.3 * keyword
        4. Applies content type boost for user guides
        5. Filters by minimum similarity threshold

        The hybrid approach ensures:
        - Semantic similarity captures meaning/intent
        - Keyword matching captures exact term matches (e.g., "CI Analysis" in heading)
        - Queries like "What is CI Analysis?" find chunks with "CI Analysis" in heading
          even if semantic similarity alone would rank "What is X?" patterns higher

        Args:
            query: The user's query
            container_name: CosmosDB container name
            service_line: Service line IDs to filter by
            top_k: Number of results to return
            question_boost_weight: Deprecated - kept for backward compatibility, not used
            min_question_similarity: Minimum similarity score threshold (0.0 to 1.0)
                                    Default: MIN_SIMILARITY_THRESHOLD from config (0.35)
            content_type_filter: Optional content type to filter by (e.g., 'UserGuide')
            year_filter: Optional year to filter by (e.g., '2024')

        Returns:
            Tuple of (ordered_docs, query_embedding, citations)
        """
        start_time = time.time()
        logger.info(f"#Query before processing :{query}")

        # Use config default if min_question_similarity not provided
        effective_threshold = min_question_similarity if min_question_similarity is not None else (MIN_SIMILARITY_THRESHOLD or 0.35)
        logger.info(f"Using similarity threshold: {effective_threshold}")

        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)

        container = self.database.get_container_client(container_name)

        # Build query filter for service line with conditional validChunk check
        # Filter by validChunk='yes' when available, but include legacy chunks without this field
        base_filter = f"ARRAY_CONTAINS({service_line},c.serviceNameid)"
        filters = [base_filter, "(NOT IS_DEFINED(c.validChunk) OR c.validChunk = 'yes')"]

        # Add metadata filters if enabled and provided
        if ENABLE_METADATA_FILTERING:
            if content_type_filter:
                filters.append(f"c.metadata.contentType = '{content_type_filter}'")
                logger.info(f"Applying content type filter: {content_type_filter}")
            if year_filter:
                filters.append(f"c.metadata.year = '{year_filter}'")
                logger.info(f"Applying year filter: {year_filter}")

        query_filter = f"WHERE {' AND '.join(filters)}"

        # Query CosmosDB with CONTENT-FIRST ordering using content embedding
        # This ensures chunks with relevant content are retrieved even if generated questions don't match
        # Solves issues like "What is CI Analysis?" where no matching question was generated
        # Post-retrieval re-ranking applies hybrid scoring (30% questions, 70% content)
        query_stmt = f"""
            SELECT TOP 20
                c.id,
                c.serviceNameid,
                c.heading,
                c.serviceName,
                c.content,
                c.validChunk,
                c.metadata.filepath as citation,
                c.metadata.contentType as contentType,
                c.metadata.year as year,
                c.questions,
                VectorDistance(c.questionsEmbedding, {query_embedding}) as question_score,
                VectorDistance(c.embedding, {query_embedding}) as content_score
            FROM c
            {query_filter}
            ORDER BY VectorDistance(c.embedding, {query_embedding})
        """

        docs = list(container.query_items(
            query=query_stmt,
            enable_cross_partition_query=True,
            populate_query_metrics=True
        ))

        # Log retrieval statistics
        valid_chunk_count = sum(1 for d in docs if d.get('validChunk') == 'yes')
        legacy_chunk_count = sum(1 for d in docs if not d.get('validChunk'))
        logger.info(f"Retrieved {len(docs)} chunks: {valid_chunk_count} validated, {legacy_chunk_count} legacy (no validChunk field)")

        # Hybrid Search: Combine semantic similarity with keyword matching
        # This ensures both semantic meaning AND keyword matches are considered
        # Formula: final_score = SEMANTIC_WEIGHT * semantic_score + KEYWORD_WEIGHT * keyword_score
        SEMANTIC_WEIGHT = 0.7  # 70% weight for semantic/vector similarity
        KEYWORD_WEIGHT = 0.3   # 30% weight for keyword/BM25 matching
        LEGACY_CHUNK_PENALTY = 0.15  # Reduce legacy chunk scores by 15%

        # Content type boosting: User guides should rank higher than newsletters/meeting notes
        USER_GUIDE_CONTENT_TYPES = {'UserGuide', 'DigiHubUserGuide'}
        CONTENT_TYPE_BOOST = 0.05  # 5% boost for user guide content types

        def distance_to_similarity(distance: float) -> float:
            """
            Convert CosmosDB VectorDistance (cosine) to similarity score.

            CosmosDB VectorDistance with cosine returns values in range [0, 2]:
            - 0 = identical vectors (similarity 1.0)
            - 1 = orthogonal vectors (similarity 0.0)
            - 2 = opposite vectors (similarity -1.0, but we clamp to 0)

            Formula: similarity = 1 - (distance / 2), clamped to [0, 1]
            """
            if distance is None:
                return 0.0
            # Normalize from [0, 2] to [0, 1] and clamp
            similarity = 1 - (distance / 2)
            return max(0.0, min(1.0, similarity))

        for doc in docs:
            # Convert VectorDistance to similarity using proper normalization
            raw_content_score = doc.get('content_score', 2.0)  # Default to max distance if missing
            content_similarity = distance_to_similarity(raw_content_score)
            doc['content_similarity'] = content_similarity

            # Calculate keyword score for hybrid search
            keyword_score = self._calculate_keyword_score(query, doc)
            doc['keyword_score'] = keyword_score

            # Handle legacy chunks (no questionsEmbedding)
            raw_question_score = doc.get('question_score')
            if raw_question_score is None:
                doc['question_similarity'] = 0.0
                doc['is_legacy_chunk'] = True
                # Legacy: use semantic + keyword with penalty
                semantic_score = content_similarity * (1 - LEGACY_CHUNK_PENALTY)
            else:
                question_similarity = distance_to_similarity(raw_question_score)
                doc['question_similarity'] = question_similarity
                doc['is_legacy_chunk'] = False
                # Use content similarity as the semantic score (question similarity no longer used)
                semantic_score = content_similarity

            # Hybrid score: combine semantic and keyword scores
            doc['semantic_score'] = semantic_score
            doc['hybrid_score'] = (
                SEMANTIC_WEIGHT * semantic_score +
                KEYWORD_WEIGHT * keyword_score
            )

            logger.debug(
                f"Chunk {doc.get('id')}: content_sim={content_similarity:.4f}, "
                f"keyword={keyword_score:.4f}, semantic={semantic_score:.4f}, "
                f"hybrid={doc['hybrid_score']:.4f}, heading='{doc.get('heading', '')[:30]}'"
            )

            # Apply content type boost for user guides
            content_type = doc.get('contentType', '')
            if content_type in USER_GUIDE_CONTENT_TYPES:
                original_score = doc['hybrid_score']
                doc['hybrid_score'] = min(1.0, doc['hybrid_score'] * (1 + CONTENT_TYPE_BOOST))
                doc['content_type_boosted'] = True
                logger.debug(f"Content type boost applied: {content_type}, score {original_score:.4f} -> {doc['hybrid_score']:.4f}")
            else:
                doc['content_type_boosted'] = False

        # Log hybrid search summary
        keyword_boost_count = sum(1 for d in docs if d.get('keyword_score', 0) > 0.3)
        content_type_boost_count = sum(1 for d in docs if d.get('content_type_boosted', False))
        logger.info(
            f"Hybrid search scores: {len(docs)} chunks | "
            f"Keyword matches (>0.3): {keyword_boost_count} | "
            f"Content type boosted: {content_type_boost_count}"
        )

        # Filter out chunks with only headings
        filtered_docs = [
            doc for doc in docs
            if not (len(doc.get('content', '').strip().splitlines()) == 1
                   and doc['content'].strip().startswith("## "))
        ]

        # Sort by hybrid score (descending) - combines 70% semantic + 30% keyword
        ordered_list = sorted(
            filtered_docs,
            key=lambda x: (-x['hybrid_score'], -x.get('keyword_score', 0), -x.get('content_similarity', 0))
        )

        # Log top results for debugging retrieval quality
        if ordered_list:
            top_doc = ordered_list[0]
            logger.info(
                f"Top chunk: hybrid={top_doc.get('hybrid_score', 0):.4f}, "
                f"semantic={top_doc.get('semantic_score', 0):.4f}, "
                f"keyword={top_doc.get('keyword_score', 0):.4f}, "
                f"heading='{top_doc.get('heading', '')[:40]}', "
                f"content_type={top_doc.get('contentType', 'N/A')}"
            )

        # Log comprehensive retrieval quality metrics for all chunks
        RetrievalMetrics.log_chunk_scores(ordered_list, query=query, stage="hybrid_search")

        # Filter by minimum similarity threshold (configurable)
        original_count = len(ordered_list)
        if effective_threshold > 0:
            above_threshold = [d for d in ordered_list if d.get('hybrid_score', 0) >= effective_threshold]
            below_threshold = [d for d in ordered_list if d.get('hybrid_score', 0) < effective_threshold]
            # Prioritize chunks above threshold, but include below if needed to meet top_k
            ordered_list = (above_threshold + below_threshold)[:top_k]
            RetrievalMetrics.log_filtering_results(
                original_count=original_count,
                filtered_count=len(above_threshold),
                filter_type="similarity_threshold",
                threshold=effective_threshold
            )
        else:
            ordered_list = ordered_list[:top_k]

        # Build citations
        citations = [
            {
                "File": doc.get('citation'),
                "Section": doc.get('heading')
            }
            for doc in ordered_list
        ]

        query_metrics = container.client_connection.last_response_headers.get('x-ms-documentdb-query-metrics')
        end_time = time.time()

        logger.info(
            f"Question-first retrieval completed in {end_time - start_time:.2f}s. "
            f"Retrieved {len(ordered_list)} chunks. Query metrics: {query_metrics}"
        )

        return ordered_list, query_embedding, citations

    def get_acronym_definitions(self, acronyms: List[str]) -> List[Dict]:
        """
        Fetch acronym definitions from CosmosDB.

        Args:
            acronyms: List of acronym strings to look up (e.g., ["CSMA", "PAX"])

        Returns:
            List of dicts with acronym definitions:
            [{"acronym": "CSMA", "definition": "Carrier Sense Multiple Access"}, ...]
        """
        if not acronyms:
            return []

        container = self.database.get_container_client(KNOWLEDGE_BASE_CONTAINER)

        # Build parameterized query for the acronyms
        parameters = []
        param_names = []

        for i, acronym in enumerate(acronyms):
            param_name = f"@acronym{i}"
            param_names.append(param_name)
            parameters.append({"name": param_name, "value": acronym.upper()})

        param_names_str = ", ".join(param_names)
        query_stmt = f"""
            SELECT c.id, c["value"], c.metadata
            FROM c
            WHERE c.partitionKey = 'acronym'
              AND c.type = 'acronym'
              AND c.id IN ({param_names_str})
        """

        try:
            docs = list(container.query_items(
                query=query_stmt,
                parameters=parameters,
                enable_cross_partition_query=True
            ))

            # Map results to a clean format
            acronym_definitions = [
                {
                    "acronym": doc.get("id"), 
                    "definition": doc.get("value"),
                    "metadata": doc.get("metadata", {})  # Added metadata here
                }
                for doc in docs
            ]

            logger.info(f"Fetched {len(acronym_definitions)} acronym definitions with metadata for {acronyms}")
            return acronym_definitions
            
        except Exception as e:
            logger.error(f"Error fetching acronym definitions: {e}")
            return []
    def get_ids_from_file_paths(self, file_list):
        if not file_list:
            return []
            
        container = self.database.get_container_client(COSMOSDB_SHAREPOINT_DATA_CONTAINER_NAME)

        unique_file_list = list(set(file_list))

        parameters = []
        param_names = []
        
        for i, file_path in enumerate(unique_file_list):
            param_name = f"@f{i}"
            param_names.append(param_name)
            parameters.append({"name": param_name, "value": file_path})

        param_names_str = ", ".join(param_names)
        query_stmt = f"""             
                        SELECT c.id, c.pathwithfilename
                        FROM c
                        WHERE c.pathwithfilename IN ({param_names_str})
                     """
        
        try:
            # Fetch results from Cosmos DB
            docs = list(container.query_items(
                query=query_stmt,
                parameters=parameters,
                enable_cross_partition_query=True
            ))

            file_map = {doc['pathwithfilename']: doc['id'] for doc in docs}
            
            final_results = []

            # 4. Reconstruct the list based on the original file_list input
            # This ensures if you asked for the file twice, you get the ID twice.
            for file_path in file_list:
                if file_path in file_map:
                    final_results.append({
                        "id": file_map[file_path],
                        "pathwithfilename": file_path
                    })
                else:
                    # Optional: Handle files that weren't found in DB
                    logger.warning(f"File not found in DB: {file_path}")
                    final_results.append({
                        "id": None, 
                        "pathwithfilename": file_path,
                        "status": "Not Found"
                    })

            logger.info(f"Query executed. Mapped {len(final_results)} results for {len(file_list)} input files.")
            
            return final_results

        except Exception as e:
            logger.error(f"Error querying Cosmos DB: {str(e)}")
            return []
