import json
import time
from typing import Optional, Any, List, Dict, Tuple
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

    @timing_decorator
    def rag_retriever_agent(self,query: str, container_name: str, service_line: Optional[list[int]], top_k: int = 5):
        """
        Retrieve the most relevant context (chunks) from CosmosDB for the given query.

        Args:
            query (str): The user's query.
            container_name (str): The CosmosDB container name.
            service_line (str): The service line to filter documents.
            top_k (int): The number of top relevant documents to retrieve.
            similarity_threshold (float): The minimum similarity score to consider a document relevant.

        Returns:
            tuple: The concatenated context and metadata of the most relevant document.
        """

        # Retrieve the top relevant documents using question-based hybrid matching (70% question, 30% content)
        top_docs,query_embedding,citation = self.retrieve_with_question_matching(query, container_name, service_line, top_k=top_k, question_boost_weight=0.7)
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
        question_boost_weight: float = 0.7,
        min_question_similarity: float = None,
        content_type_filter: str = None,
        year_filter: str = None
    ) -> Tuple[list, list, list]:
        """
        Retrieve chunks using question-first hybrid approach with native CosmosDB vector search.

        This method prioritizes question matching over content matching:
        1. Queries CosmosDB ordering by questionsEmbedding similarity (question-first)
        2. Fetches both question and content scores from the database
        3. Re-ranks using weighted hybrid score (default: 70% questions, 30% content)
        4. Filters by minimum similarity threshold (configurable via MIN_SIMILARITY_THRESHOLD)
        5. Optionally filters by metadata (content type, year)

        Args:
            query: The user's query
            container_name: CosmosDB container name
            service_line: Service line IDs to filter by
            top_k: Number of results to return
            question_boost_weight: Weight for question matching (0.0 to 1.0)
                                  0.7 = 70% questions, 30% content (default)
                                  1.0 = only question matching
            min_question_similarity: Minimum question similarity score to include a chunk (0.0 to 1.0)
                                    Chunks below this threshold are deprioritized
                                    Default: MIN_SIMILARITY_THRESHOLD from config (0.35)
            content_type_filter: Optional content type to filter by (e.g., 'UserGuide', 'APIDocs')
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

        # Query CosmosDB with QUESTION-FIRST ordering using questionsEmbedding
        # Chunks with questionsEmbedding are ordered by question similarity (prioritized)
        # Legacy chunks without questionsEmbedding will have NULL question_score and sort to end
        # Post-retrieval re-ranking handles proper scoring for all chunks
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
            ORDER BY VectorDistance(c.questionsEmbedding, {query_embedding})
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

        # Calculate hybrid scores using both question and content similarity
        # Legacy chunks without questionsEmbedding are penalized since we can't verify question relevance
        LEGACY_CHUNK_PENALTY = 0.15  # Reduce legacy chunk scores by 15%

        for doc in docs:
            # Convert VectorDistance to similarity: smaller distance = higher similarity
            # Handle legacy chunks that may not have questionsEmbedding
            raw_question_score = doc.get('question_score')
            raw_content_score = doc.get('content_score', 1.0)

            # If no question score (legacy chunk), use only content similarity with penalty
            if raw_question_score is None:
                content_similarity = 1 - raw_content_score
                # Legacy chunks only have content similarity, no question relevance verification
                # Apply penalty and use content-only scoring
                doc['question_similarity'] = 0.0  # No question embedding to compare
                doc['content_similarity'] = content_similarity
                doc['is_legacy_chunk'] = True
                # Legacy score: content similarity with penalty (not hybrid weighted)
                doc['hybrid_score'] = content_similarity * (1 - LEGACY_CHUNK_PENALTY)
                logger.debug(f"Legacy chunk {doc.get('id')}: content_sim={content_similarity:.4f}, penalized_score={doc['hybrid_score']:.4f}")
            else:
                question_similarity = 1 - raw_question_score
                content_similarity = 1 - raw_content_score
                doc['question_similarity'] = question_similarity
                doc['content_similarity'] = content_similarity
                doc['is_legacy_chunk'] = False
                # Proper hybrid score with question-first weighting
                doc['hybrid_score'] = (
                    question_boost_weight * question_similarity +
                    (1 - question_boost_weight) * content_similarity
                )

        # Filter out chunks with only headings
        filtered_docs = [
            doc for doc in docs
            if not (len(doc.get('content', '').strip().splitlines()) == 1
                   and doc['content'].strip().startswith("## "))
        ]

        # Sort by hybrid score (descending) - question similarity is the dominant factor
        ordered_list = sorted(
            filtered_docs,
            key=lambda x: (-x['hybrid_score'], -x.get('question_similarity', 0))
        )

        # Log top results for debugging retrieval quality
        if ordered_list:
            top_doc = ordered_list[0]
            logger.info(
                f"Top chunk: hybrid_score={top_doc.get('hybrid_score', 0):.4f}, "
                f"question_sim={top_doc.get('question_similarity', 0):.4f}, "
                f"content_sim={top_doc.get('content_similarity', 0):.4f}, "
                f"legacy={top_doc.get('is_legacy_chunk', False)}"
            )

        # Log comprehensive retrieval quality metrics for all chunks
        RetrievalMetrics.log_chunk_scores(ordered_list, query=query, stage="question_matching")

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
