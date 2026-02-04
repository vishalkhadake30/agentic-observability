"""
Unit Tests for RAG Memory Agent

Tests cover:
- Initialization and cleanup
- Embedding generation
- Vector search
- Incident storage
- Error handling
- Integration with BaseAgent patterns (circuit breaker, retry, metrics)
"""

import pytest
import asyncio
from typing import Any
from unittest.mock import Mock, MagicMock, patch, AsyncMock

from src.agentic_observability.agents.rag_memory.memory import RAGMemoryAgent, IncidentContext
from src.agentic_observability.agents.base import AgentState


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
async def mock_qdrant():
    """Mock Qdrant client"""
    with patch("src.agentic_observability.agents.rag_memory.memory.QdrantClient") as mock_client:
        # Configure mock
        instance = mock_client.return_value
        instance.get_collections.return_value = Mock(collections=[])
        instance.create_collection.return_value = None
        instance.search.return_value = []
        instance.upsert.return_value = None
        
        yield mock_client


@pytest.fixture
async def mock_sentence_transformer():
    """Mock SentenceTransformer model"""
    with patch("src.agentic_observability.agents.rag_memory.memory.SentenceTransformer") as mock_model:
        # Configure mock to return fake embeddings
        instance = mock_model.return_value
        instance.encode.return_value = Mock(tolist=lambda: [0.1] * 384)
        
        yield mock_model


@pytest.fixture
async def rag_agent(mock_qdrant, mock_sentence_transformer):
    """Initialized RAG Memory Agent with mocked dependencies"""
    agent = RAGMemoryAgent(
        name="test-rag",
        vector_db_url="http://localhost:6333",
        collection_name="test_incidents",
        embedding_model="all-MiniLM-L6-v2",
        circuit_breaker_threshold=5,
        max_retries=2,
        base_retry_delay=0.01
    )
    
    await agent.initialize()
    yield agent
    await agent.cleanup()


# ============================================================================
# Test Initialization and Configuration
# ============================================================================

@pytest.mark.asyncio
class TestRAGAgentInitialization:
    """Test agent initialization and setup"""
    
    async def test_initialization(self, mock_qdrant, mock_sentence_transformer):
        """
        Test: Agent initializes correctly with all dependencies
        
        VERIFY:
        - Qdrant client created
        - Embedding model loaded
        - Collection created
        - State is IDLE
        """
        agent = RAGMemoryAgent(
            name="init-test",
            vector_db_url="http://test:6333",
            collection_name="test_collection"
        )
        
        await agent.initialize()
        
        # Verify Qdrant client initialized
        mock_qdrant.assert_called_once_with(url="http://test:6333")
        
        # Verify embedding model loaded
        mock_sentence_transformer.assert_called_once_with("all-MiniLM-L6-v2")
        
        # Verify agent state
        assert agent._state == AgentState.IDLE
        assert agent._initialized is True
        
        await agent.cleanup()
    
    async def test_collection_creation_when_not_exists(self, mock_qdrant, mock_sentence_transformer):
        """
        Test: Creates collection if it doesn't exist
        
        VERIFY:
        - Collection creation called with correct parameters
        """
        # Mock: Collection doesn't exist
        mock_instance = mock_qdrant.return_value
        mock_instance.get_collections.return_value = Mock(collections=[])
        
        agent = RAGMemoryAgent(name="create-test", collection_name="new_collection")
        await agent.initialize()
        
        # Verify collection creation
        mock_instance.create_collection.assert_called_once()
        call_args = mock_instance.create_collection.call_args
        assert call_args.kwargs["collection_name"] == "new_collection"
        
        await agent.cleanup()
    
    async def test_collection_not_created_when_exists(self, mock_qdrant, mock_sentence_transformer):
        """
        Test: Skips collection creation if it already exists
        
        VERIFY:
        - Collection creation not called
        """
        # Reset mock to clear any previous calls
        mock_qdrant.reset_mock()
        
        # Mock: Collection exists with matching name
        existing_collection = Mock()
        existing_collection.name = "observability_incidents"
        
        mock_instance = mock_qdrant.return_value
        mock_instance.get_collections.return_value = Mock(collections=[existing_collection])
        
        agent = RAGMemoryAgent(name="exists-test")
        await agent.initialize()
        
        # Verify collection creation NOT called
        mock_instance.create_collection.assert_not_called()
        
        await agent.cleanup()


# ============================================================================
# Test RAG Search and Retrieval
# ============================================================================

@pytest.mark.asyncio
class TestRAGSearch:
    """Test vector search and incident retrieval"""
    
    async def test_successful_search(self, rag_agent, mock_qdrant):
        """
        Test: Successful RAG search returns similar incidents
        
        VERIFY:
        - Embedding generated for query
        - Vector search executed
        - Results parsed correctly
        - Similar incidents returned
        """
        # Mock search results
        mock_instance = mock_qdrant.return_value
        mock_instance.search.return_value = [
            Mock(
                id="incident-1",
                score=0.95,
                payload={
                    "incident_id": "incident-1",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "description": "High CPU usage on web server",
                    "root_cause": "Memory leak in application",
                    "resolution": "Restarted service, fixed memory leak",
                    "metadata": {"severity": "high"}
                }
            ),
            Mock(
                id="incident-2",
                score=0.87,
                payload={
                    "incident_id": "incident-2",
                    "timestamp": "2026-01-02T00:00:00Z",
                    "description": "CPU spike during deployment",
                    "root_cause": "Normal behavior",
                    "resolution": "No action needed",
                    "metadata": {"severity": "low"}
                }
            )
        ]
        
        # Execute search
        result = await rag_agent.execute({
            "query": "High CPU usage detected",
            "top_k": 5
        })
        
        # Verify results
        assert "similar_incidents" in result
        assert len(result["similar_incidents"]) == 2
        assert result["total_found"] == 2
        
        # Verify first incident
        incident1 = result["similar_incidents"][0]
        assert incident1["incident_id"] == "incident-1"
        assert incident1["similarity_score"] == 0.95
        assert "High CPU usage" in incident1["description"]
        assert incident1["root_cause"] == "Memory leak in application"
        
        # Verify search was called correctly
        mock_instance.search.assert_called_once()
        call_args = mock_instance.search.call_args
        assert call_args.kwargs["collection_name"] == "test_incidents"
        assert call_args.kwargs["limit"] == 5
    
    async def test_search_with_top_k(self, rag_agent, mock_qdrant):
        """
        Test: top_k parameter limits number of results
        
        VERIFY:
        - Search called with correct limit
        """
        mock_instance = mock_qdrant.return_value
        mock_instance.search.return_value = []
        
        await rag_agent.execute({
            "query": "Test query",
            "top_k": 3
        })
        
        call_args = mock_instance.search.call_args
        assert call_args.kwargs["limit"] == 3
    
    async def test_search_without_query_fails(self, rag_agent):
        """
        Test: Search fails if query is missing
        
        VERIFY:
        - ValueError raised
        - Error message mentions query
        """
        with pytest.raises(ValueError, match="Query text is required"):
            await rag_agent.execute({"top_k": 5})
    
    async def test_empty_search_results(self, rag_agent, mock_qdrant):
        """
        Test: Handles empty search results gracefully
        
        VERIFY:
        - Returns empty list of incidents
        - No errors raised
        """
        mock_instance = mock_qdrant.return_value
        mock_instance.search.return_value = []
        
        result = await rag_agent.execute({
            "query": "Nonexistent incident",
            "top_k": 5
        })
        
        assert result["total_found"] == 0
        assert result["similar_incidents"] == []


# ============================================================================
# Test Incident Storage
# ============================================================================

@pytest.mark.asyncio
class TestIncidentStorage:
    """Test storing incidents in vector database"""
    
    async def test_store_incident(self, rag_agent, mock_qdrant):
        """
        Test: Store incident creates embedding and saves to database
        
        VERIFY:
        - Embedding generated
        - Incident stored with correct payload
        - Returns incident_id
        """
        mock_instance = mock_qdrant.return_value
        
        incident_id = await rag_agent.store_incident(
            incident_id="test-123",
            description="Database connection timeout",
            root_cause="Network congestion",
            resolution="Increased connection pool size",
            timestamp="2026-02-01T12:00:00Z",
            metadata={"severity": "medium", "service": "api"}
        )
        
        # Verify returned incident_id
        assert incident_id == "test-123"
        
        # Verify upsert was called
        mock_instance.upsert.assert_called_once()
        call_args = mock_instance.upsert.call_args
        
        # Verify payload
        points = call_args.kwargs["points"]
        assert len(points) == 1
        point = points[0]
        
        assert point.id == "test-123"
        assert point.payload["description"] == "Database connection timeout"
        assert point.payload["root_cause"] == "Network congestion"
        assert point.payload["metadata"]["severity"] == "medium"


# ============================================================================
# Test Agent Resilience Patterns
# ============================================================================

@pytest.mark.asyncio
class TestRAGAgentResilience:
    """Test circuit breaker, retry, and metrics"""
    
    async def test_circuit_breaker_inherited(self, rag_agent):
        """
        Test: RAG agent inherits circuit breaker from BaseAgent
        
        VERIFY:
        - Circuit breaker exists
        - State machine exists
        - Metrics tracked
        """
        assert rag_agent._circuit_breaker is not None
        assert rag_agent._state == AgentState.IDLE
        assert rag_agent._metrics is not None
    
    async def test_metrics_recorded_on_success(self, rag_agent, mock_qdrant):
        """
        Test: Successful execution records metrics
        
        VERIFY:
        - Execution count incremented
        - Success count incremented
        - Latency recorded
        """
        mock_instance = mock_qdrant.return_value
        mock_instance.search.return_value = []
        
        initial_executions = rag_agent._metrics.total_executions
        initial_successes = rag_agent._metrics.total_successes
        
        await rag_agent.execute({"query": "Test", "top_k": 1})
        
        assert rag_agent._metrics.total_executions == initial_executions + 1
        assert rag_agent._metrics.total_successes == initial_successes + 1
        assert len(rag_agent._metrics.latencies) > 0
    
    async def test_health_check(self, rag_agent):
        """
        Test: Health check reflects agent state
        
        VERIFY:
        - is_healthy() returns True when initialized
        - get_health_status() shows detailed info
        """
        assert rag_agent.is_healthy() is True
        
        health = rag_agent.get_health_status()
        assert health["healthy"] is True
        assert health["agent"] == "test-rag"
        assert health["circuit_breaker"]["state"] == "closed"


# ============================================================================
# Test Error Handling
# ============================================================================

@pytest.mark.asyncio
class TestErrorHandling:
    """Test error scenarios"""
    
    async def test_qdrant_connection_error_retries(self, mock_qdrant, mock_sentence_transformer):
        """
        Test: Connection errors trigger retry logic
        
        VERIFY:
        - Retries attempted
        - Circuit breaker may open after failures
        """
        # Mock: Qdrant connection fails
        mock_instance = mock_qdrant.return_value
        mock_instance.search.side_effect = Exception("Connection refused")
        
        agent = RAGMemoryAgent(
            name="error-test",
            max_retries=2,
            base_retry_delay=0.01
        )
        await agent.initialize()
        
        # Execute should fail after retries
        with pytest.raises(Exception, match="Connection refused"):
            await agent.execute({"query": "Test", "top_k": 1})
        
        # Verify retries occurred
        assert agent._metrics.total_retries > 0
        
        await agent.cleanup()
    
    async def test_embedding_generation_error(self, mock_qdrant, mock_sentence_transformer):
        """
        Test: Embedding generation error handled
        
        VERIFY:
        - Error propagates correctly
        """
        # Mock: Embedding model fails
        mock_model_instance = mock_sentence_transformer.return_value
        mock_model_instance.encode.side_effect = Exception("Model failed")
        
        agent = RAGMemoryAgent(name="embed-error-test", max_retries=0)
        await agent.initialize()
        
        with pytest.raises(Exception, match="Model failed"):
            await agent.execute({"query": "Test", "top_k": 1})
        
        await agent.cleanup()


# ============================================================================
# Test Cleanup
# ============================================================================

@pytest.mark.asyncio
class TestCleanup:
    """Test resource cleanup"""
    
    async def test_cleanup_closes_qdrant_client(self, mock_qdrant, mock_sentence_transformer):
        """
        Test: Cleanup properly closes Qdrant client
        
        VERIFY:
        - close() called on Qdrant client
        - State reset
        """
        agent = RAGMemoryAgent(name="cleanup-test")
        await agent.initialize()
        
        mock_instance = mock_qdrant.return_value
        
        await agent.cleanup()
        
        # Verify close was called
        mock_instance.close.assert_called_once()
        
        # Verify state
        assert agent._initialized is False
        assert agent._state == AgentState.IDLE


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
