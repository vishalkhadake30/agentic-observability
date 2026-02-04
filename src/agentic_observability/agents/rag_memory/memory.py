"""
RAG Memory Agent

Retrieves similar historical incidents from vector database for context.

WHY RAG (Retrieval-Augmented Generation):
- Anomalies without context are meaningless
- "CPU at 90%" could be normal during deployment or critical if sustained
- Historical incidents provide patterns: "Last time this happened, it was X"

This agent:
1. Converts current anomaly to vector embedding
2. Searches vector database for similar past incidents  
3. Returns top-K most relevant incidents with context
4. Enables the Reasoning Agent to learn from history
"""

from typing import Any, Optional
from dataclasses import dataclass, field
import structlog
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, SearchParams
from sentence_transformers import SentenceTransformer

from ..base import BaseAgent

logger = structlog.get_logger()


@dataclass
class IncidentContext:
    """Retrieved historical incident with similarity score"""
    incident_id: str
    timestamp: str
    description: str
    root_cause: str
    resolution: str
    similarity_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGResult:
    """Result from RAG memory retrieval"""
    similar_incidents: list[IncidentContext]
    query_embedding: list[float]
    total_found: int
    search_time_ms: float


class RAGMemoryAgent(BaseAgent):
    """
    RAG Memory Agent for retrieving similar historical incidents.
    
    Uses:
    - Sentence Transformers for text → vector embeddings
    - Qdrant for vector similarity search
    - Cosine similarity for finding similar incidents
    
    WHY THIS ARCHITECTURE:
    - Embeddings capture semantic meaning (not just keyword matching)
    - Vector search is fast (HNSW algorithm, O(log n))
    - Qdrant handles billions of vectors with <10ms latency
    
    Example:
        agent = RAGMemoryAgent(
            name="rag-memory",
            vector_db_url="http://localhost:6333",
            collection_name="incidents",
            embedding_model="all-MiniLM-L6-v2"
        )
        await agent.initialize()
        
        result = await agent.execute({
            "query": "High CPU usage on web server",
            "anomaly_type": "spike",
            "metric_name": "cpu_percent",
            "top_k": 5
        })
        
        for incident in result["similar_incidents"]:
            print(f"Similar: {incident.description} (score: {incident.similarity_score})")
    """
    
    def __init__(
        self,
        name: str = "rag-memory",
        vector_db_url: str = "http://localhost:6333",
        collection_name: str = "observability_incidents",
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        **kwargs
    ):
        """
        Initialize RAG Memory Agent.
        
        Args:
            name: Agent identifier
            vector_db_url: Qdrant server URL
            collection_name: Vector collection name
            embedding_model: SentenceTransformer model name
            embedding_dim: Embedding vector dimension (384 for all-MiniLM-L6-v2)
            **kwargs: Passed to BaseAgent
        """
        super().__init__(name=name, **kwargs)
        
        self.vector_db_url = vector_db_url
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.embedding_dim = embedding_dim
        
        # Will be initialized in initialize()
        self._qdrant_client: Optional[QdrantClient] = None
        self._embedding_model: Optional[SentenceTransformer] = None
        
        self.logger.info(
            "rag_agent_configured",
            vector_db_url=vector_db_url,
            collection=collection_name,
            embedding_model=embedding_model
        )
    
    async def initialize(self) -> None:
        """
        Initialize Qdrant client and embedding model.
        
        Creates collection if it doesn't exist.
        """
        await super().initialize()

        # Allow dependency injection (tests/local-dev) by skipping re-init if already provided.
        if self._qdrant_client is None:
            self.logger.info("initializing_qdrant_client", url=self.vector_db_url)
            self._qdrant_client = QdrantClient(url=self.vector_db_url)

        if self._embedding_model is None:
            self.logger.info("loading_embedding_model", model=self.embedding_model_name)
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Create collection if it doesn't exist
        try:
            collections = self._qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                self.logger.info("creating_collection", name=self.collection_name)
                self._qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                self.logger.info("collection_created", name=self.collection_name)
            else:
                self.logger.info("collection_exists", name=self.collection_name)
        
        except Exception as e:
            self.logger.warning(
                "collection_creation_check_failed",
                error=str(e),
                msg="Will attempt to use collection anyway"
            )
        
        self.logger.info("rag_agent_initialized")
    
    async def _execute_impl(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Execute RAG memory retrieval.
        
        Args:
            input_data: Must contain:
                - query: Text description of current anomaly
                - top_k: Number of similar incidents to retrieve (default: 5)
                - Optional filters: anomaly_type, metric_name, severity, etc.
        
        Returns:
            Dict with:
                - similar_incidents: List of IncidentContext objects
                - query_embedding: Vector representation of query
                - total_found: Number of matches
                - search_time_ms: Search latency
        """
        import time
        
        # Extract parameters
        query_text = input_data.get("query", "")
        top_k = input_data.get("top_k", 5)
        filter_conditions = input_data.get("filters", {})
        
        if not query_text:
            raise ValueError("Query text is required for RAG memory retrieval")
        
        self.logger.info(
            "executing_rag_search",
            query_length=len(query_text),
            top_k=top_k,
            filters=filter_conditions
        )
        
        start_time = time.time()
        
        if self._embedding_model is None:
            raise RuntimeError("Embedding model not initialized")

        # 1. Generate embedding for query
        embedding_obj = self._embedding_model.encode(query_text)
        if hasattr(embedding_obj, "tolist"):
            query_embedding = embedding_obj.tolist()
        else:
            # Some tests may stub encode() with a plain list.
            query_embedding = embedding_obj

        # If encode returns a batch (list[list[float]]), take first vector.
        if query_embedding and isinstance(query_embedding, list) and isinstance(query_embedding[0], list):
            query_embedding = query_embedding[0]
        
        embedding_time = time.time() - start_time
        self.logger.debug(
            "query_embedding_generated",
            embedding_dim=len(query_embedding),
            time_ms=embedding_time * 1000
        )
        
        # 2. Build Qdrant filter if provided
        qdrant_filter = None
        if filter_conditions:
            # TODO: Implement filter building
            pass
        
        # 3. Search vector database
        search_start = time.time()
        
        if self._qdrant_client is None:
            raise RuntimeError("Qdrant client not initialized")

        search_results = self._qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        
        search_time = time.time() - search_start
        
        self.logger.info(
            "vector_search_completed",
            results_found=len(search_results),
            search_time_ms=search_time * 1000
        )
        
        # 4. Convert results to IncidentContext objects
        similar_incidents = []
        
        for result in search_results:
            payload = result.payload or {}
            
            incident = IncidentContext(
                incident_id=payload.get("incident_id", str(result.id)),
                timestamp=payload.get("timestamp", ""),
                description=payload.get("description", ""),
                root_cause=payload.get("root_cause", "Unknown"),
                resolution=payload.get("resolution", ""),
                similarity_score=result.score,
                metadata=payload.get("metadata", {})
            )
            
            similar_incidents.append(incident)
            
            self.logger.debug(
                "incident_retrieved",
                incident_id=incident.incident_id,
                similarity=round(incident.similarity_score, 3),
                description_preview=incident.description[:100]
            )
        
        total_time = time.time() - start_time
        
        self.logger.info(
            "rag_retrieval_completed",
            total_incidents=len(similar_incidents),
            total_time_ms=total_time * 1000
        )
        
        return {
            "similar_incidents": [
                {
                    "incident_id": inc.incident_id,
                    "timestamp": inc.timestamp,
                    "description": inc.description,
                    "root_cause": inc.root_cause,
                    "resolution": inc.resolution,
                    "similarity_score": inc.similarity_score,
                    "metadata": inc.metadata
                }
                for inc in similar_incidents
            ],
            "query_embedding": query_embedding,
            "total_found": len(similar_incidents),
            "search_time_ms": total_time * 1000
        }
    
    async def store_incident(
        self,
        incident_id: str,
        description: str,
        root_cause: str,
        resolution: str,
        timestamp: str,
        metadata: Optional[dict[str, Any]] = None
    ) -> str:
        """
        Store a new incident in the vector database.
        
        WHY SEPARATE METHOD: Storing incidents is a different operation
        from retrieving them. This keeps execute() focused on retrieval.
        
        Args:
            incident_id: Unique incident identifier
            description: Full text description of the incident
            root_cause: Identified root cause
            resolution: How it was resolved
            timestamp: When it occurred (ISO format)
            metadata: Additional context (severity, affected services, etc.)
        
        Returns:
            incident_id of stored incident
        """
        self.logger.info(
            "storing_incident",
            incident_id=incident_id,
            description_length=len(description)
        )
        
        # Generate embedding for incident description
        embedding = self._embedding_model.encode(description).tolist()
        
        # Prepare payload
        payload = {
            "incident_id": incident_id,
            "description": description,
            "root_cause": root_cause,
            "resolution": resolution,
            "timestamp": timestamp,
            "metadata": metadata or {}
        }
        
        # Store in Qdrant
        self._qdrant_client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=incident_id,
                    vector=embedding,
                    payload=payload
                )
            ]
        )
        
        self.logger.info("incident_stored", incident_id=incident_id)
        
        return incident_id
    
    async def cleanup(self) -> None:
        """Cleanup Qdrant client"""
        await super().cleanup()
        
        if self._qdrant_client:
            self._qdrant_client.close()
            self.logger.info("qdrant_client_closed")
