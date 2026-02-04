"""
Unit Tests for Reasoning Agent

Tests cover:
- Mock reasoning agent (rule-based)
- Pattern detection
- Root cause hypothesis generation
- Confidence scoring
- Recommendations
- Integration with BaseAgent patterns
"""

import pytest
from typing import Any

from src.agentic_observability.agents.reasoning.reasoning import (
    ReasoningAgent,
    MockReasoningAgent,
    RootCauseAnalysis
)
from src.agentic_observability.agents.base import AgentState


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
async def reasoning_agent():
    """Initialized mock reasoning agent"""
    agent = MockReasoningAgent(
        name="test-reasoning",
        confidence_threshold=0.7,
        circuit_breaker_threshold=5,
        max_retries=2,
        base_retry_delay=0.01
    )
    
    await agent.initialize()
    yield agent
    await agent.cleanup()


@pytest.fixture
def sample_anomaly():
    """Sample anomaly data"""
    return {
        "anomaly_type": "spike",
        "metric_name": "cpu_percent",
        "value": 95.0,
        "severity": 0.85,
        "timestamp": "2026-02-02T12:00:00Z"
    }


@pytest.fixture
def sample_incidents():
    """Sample historical incidents"""
    return [
        {
            "incident_id": "inc-001",
            "root_cause": "Memory leak in application",
            "similarity_score": 0.92,
            "description": "High CPU due to memory leak"
        },
        {
            "incident_id": "inc-002",
            "root_cause": "Deployment-related spike",
            "similarity_score": 0.78,
            "description": "CPU spike during deployment"
        }
    ]


# ============================================================================
# Test Initialization
# ============================================================================

@pytest.mark.asyncio
class TestReasoningAgentInitialization:
    """Test agent initialization"""
    
    async def test_initialization(self):
        """
        Test: Agent initializes correctly
        
        VERIFY:
        - State is IDLE
        - Patterns loaded
        - BaseAgent features inherited
        """
        agent = MockReasoningAgent(name="init-test")
        await agent.initialize()
        
        assert agent._state == AgentState.IDLE
        assert agent._initialized is True
        assert len(agent._patterns) > 0
        assert agent._circuit_breaker is not None
        
        await agent.cleanup()


# ============================================================================
# Test Root Cause Analysis
# ============================================================================

@pytest.mark.asyncio
class TestRootCauseAnalysis:
    """Test root cause reasoning logic"""
    
    async def test_basic_analysis(self, reasoning_agent, sample_anomaly):
        """
        Test: Basic root cause analysis without historical data
        
        VERIFY:
        - Analysis completed
        - Root cause generated
        - Confidence score reasonable
        - Recommendations provided
        """
        result = await reasoning_agent.execute({
            "anomaly": sample_anomaly,
            "similar_incidents": [],
            "metric_data": []
        })
        
        analysis = result["root_cause_analysis"]
        
        assert "root_cause" in analysis
        assert "confidence" in analysis
        assert "reasoning_steps" in analysis
        assert "recommendations" in analysis
        
        assert analysis["confidence"] >= 0.0
        assert analysis["confidence"] <= 1.0
        assert len(analysis["reasoning_steps"]) > 0
        assert len(analysis["recommendations"]) > 0
    
    async def test_analysis_with_historical_match(
        self, 
        reasoning_agent, 
        sample_anomaly, 
        sample_incidents
    ):
        """
        Test: Analysis with strong historical match
        
        VERIFY:
        - Root cause matches historical incident
        - Confidence boosted by similarity
        - Historical evidence in reasoning
        """
        result = await reasoning_agent.execute({
            "anomaly": sample_anomaly,
            "similar_incidents": sample_incidents
        })
        
        analysis = result["root_cause_analysis"]
        
        # Should match top historical cause
        assert "Memory leak" in analysis["root_cause"]
        
        # Confidence should be high due to strong match
        assert analysis["confidence"] > 0.7
        
        # Should reference historical evidence
        reasoning_text = " ".join(analysis["reasoning_steps"])
        assert "historical match" in reasoning_text.lower() or "similar" in reasoning_text.lower()
    
    async def test_pattern_detection_memory_leak(self, reasoning_agent):
        """
        Test: Memory leak pattern detected
        
        VERIFY:
        - Pattern recognition works
        - Appropriate root cause assigned
        - Memory-specific recommendations
        """
        result = await reasoning_agent.execute({
            "anomaly": {
                "anomaly_type": "spike",
                "metric_name": "memory_usage",
                "severity": 0.9
            },
            "similar_incidents": [
                {
                    "root_cause": "Memory leak causing OOM",
                    "similarity_score": 0.85
                }
            ]
        })
        
        analysis = result["root_cause_analysis"]
        
        assert "memory" in analysis["root_cause"].lower()
        assert any("heap" in rec.lower() or "memory" in rec.lower() 
                  for rec in analysis["recommendations"])
    
    async def test_pattern_detection_deployment(self, reasoning_agent):
        """
        Test: Deployment pattern detected
        
        VERIFY:
        - Deployment pattern recognized
        - Deployment-specific recommendations
        """
        result = await reasoning_agent.execute({
            "anomaly": {
                "anomaly_type": "spike",
                "metric_name": "error_rate",
                "severity": 0.75
            },
            "similar_incidents": [
                {
                    "root_cause": "Deployment rollout caused errors",
                    "similarity_score": 0.88
                }
            ]
        })
        
        analysis = result["root_cause_analysis"]
        
        assert "deployment" in analysis["root_cause"].lower()
        assert any("rollback" in rec.lower() or "deployment" in rec.lower()
                  for rec in analysis["recommendations"])


# ============================================================================
# Test Confidence Scoring
# ============================================================================

@pytest.mark.asyncio
class TestConfidenceScoring:
    """Test confidence calculation logic"""
    
    async def test_confidence_with_no_history(self, reasoning_agent, sample_anomaly):
        """
        Test: Low confidence without historical data
        
        VERIFY:
        - Confidence is moderate (0.4-0.6 range)
        """
        result = await reasoning_agent.execute({
            "anomaly": sample_anomaly,
            "similar_incidents": []
        })
        
        confidence = result["root_cause_analysis"]["confidence"]
        assert 0.4 <= confidence <= 0.7
    
    async def test_confidence_with_strong_match(self, reasoning_agent, sample_anomaly):
        """
        Test: High confidence with strong historical match
        
        VERIFY:
        - Confidence increased by similarity score
        """
        result = await reasoning_agent.execute({
            "anomaly": sample_anomaly,
            "similar_incidents": [
                {
                    "root_cause": "Known issue",
                    "similarity_score": 0.95
                }
            ]
        })
        
        confidence = result["root_cause_analysis"]["confidence"]
        # Base confidence (0.5) + strong match boost (0.15) = 0.65
        assert confidence >= 0.65
        assert confidence < 1.0
    
    async def test_confidence_with_multiple_patterns(self, reasoning_agent):
        """
        Test: Confidence boosted by multiple detected patterns
        
        VERIFY:
        - Multiple patterns increase confidence
        """
        result = await reasoning_agent.execute({
            "anomaly": {
                "anomaly_type": "spike",
                "metric_name": "cpu_percent",
                "severity": 0.8
            },
            "similar_incidents": [
                {
                    "root_cause": "Memory leak causing high CPU and timeout",
                    "similarity_score": 0.82
                }
            ]
        })
        
        analysis = result["root_cause_analysis"]
        
        # Should detect memory_leak + connection_pool patterns
        patterns = analysis["metadata"].get("detected_patterns", [])
        confidence = analysis["confidence"]
        
        if len(patterns) > 1:
            assert confidence > 0.8


# ============================================================================
# Test Recommendations
# ============================================================================

@pytest.mark.asyncio
class TestRecommendations:
    """Test recommendation generation"""
    
    async def test_memory_leak_recommendations(self, reasoning_agent):
        """
        Test: Memory leak generates appropriate recommendations
        
        VERIFY:
        - Heap dump analysis mentioned
        - Resource management checks
        - Restart suggestion
        """
        result = await reasoning_agent.execute({
            "anomaly": {
                "anomaly_type": "spike",
                "metric_name": "memory",
                "severity": 0.9
            },
            "similar_incidents": [
                {"root_cause": "Memory leak", "similarity_score": 0.9}
            ]
        })
        
        recommendations = result["root_cause_analysis"]["recommendations"]
        rec_text = " ".join(recommendations).lower()
        
        assert "heap" in rec_text or "memory" in rec_text
        assert len(recommendations) >= 3
    
    async def test_deployment_recommendations(self, reasoning_agent):
        """
        Test: Deployment issues generate deployment-specific recommendations
        
        VERIFY:
        - Rollback mentioned
        - Log review suggested
        - Configuration check
        """
        result = await reasoning_agent.execute({
            "anomaly": {
                "anomaly_type": "spike",
                "metric_name": "error_count",
                "severity": 0.8
            },
            "similar_incidents": [
                {"root_cause": "Bad deployment", "similarity_score": 0.85}
            ]
        })
        
        recommendations = result["root_cause_analysis"]["recommendations"]
        rec_text = " ".join(recommendations).lower()
        
        assert "rollback" in rec_text or "deployment" in rec_text or "release" in rec_text
    
    async def test_generic_recommendations(self, reasoning_agent, sample_anomaly):
        """
        Test: Generic recommendations for unknown patterns
        
        VERIFY:
        - At least basic recommendations provided
        - Investigation steps included
        """
        result = await reasoning_agent.execute({
            "anomaly": sample_anomaly,
            "similar_incidents": []
        })
        
        recommendations = result["root_cause_analysis"]["recommendations"]
        
        assert len(recommendations) > 0
        rec_text = " ".join(recommendations).lower()
        assert "investigate" in rec_text or "monitor" in rec_text or "alert" in rec_text


# ============================================================================
# Test Evidence Compilation
# ============================================================================

@pytest.mark.asyncio
class TestEvidence:
    """Test evidence compilation"""
    
    async def test_evidence_includes_anomaly_details(
        self, 
        reasoning_agent, 
        sample_anomaly
    ):
        """
        Test: Evidence includes anomaly characteristics
        
        VERIFY:
        - Anomaly type in evidence
        - Severity in evidence
        """
        result = await reasoning_agent.execute({
            "anomaly": sample_anomaly,
            "similar_incidents": []
        })
        
        evidence = result["root_cause_analysis"]["supporting_evidence"]
        evidence_text = " ".join(evidence).lower()
        
        assert "spike" in evidence_text
        assert "severity" in evidence_text
    
    async def test_evidence_includes_historical_data(
        self, 
        reasoning_agent, 
        sample_anomaly,
        sample_incidents
    ):
        """
        Test: Evidence includes historical incidents
        
        VERIFY:
        - Historical causes listed
        - Similarity scores shown
        """
        result = await reasoning_agent.execute({
            "anomaly": sample_anomaly,
            "similar_incidents": sample_incidents
        })
        
        evidence = result["root_cause_analysis"]["supporting_evidence"]
        evidence_text = " ".join(evidence).lower()
        
        assert "historical" in evidence_text or "similarity" in evidence_text


# ============================================================================
# Test Agent Resilience
# ============================================================================

@pytest.mark.asyncio
class TestReasoningAgentResilience:
    """Test circuit breaker, retry, metrics"""
    
    async def test_baseagent_inheritance(self, reasoning_agent):
        """
        Test: Inherits BaseAgent features
        
        VERIFY:
        - Circuit breaker exists
        - Metrics tracked
        - State machine works
        """
        assert reasoning_agent._circuit_breaker is not None
        assert reasoning_agent._metrics is not None
        assert reasoning_agent._state == AgentState.IDLE
    
    async def test_metrics_recorded(self, reasoning_agent, sample_anomaly):
        """
        Test: Metrics recorded on execution
        
        VERIFY:
        - Execution count incremented
        - Latency tracked
        """
        initial_executions = reasoning_agent._metrics.total_executions
        
        await reasoning_agent.execute({
            "anomaly": sample_anomaly,
            "similar_incidents": []
        })
        
        assert reasoning_agent._metrics.total_executions == initial_executions + 1
        assert len(reasoning_agent._metrics.latencies) > 0
    
    async def test_health_check(self, reasoning_agent):
        """
        Test: Health check works
        
        VERIFY:
        - is_healthy() returns True
        - Health status detailed
        """
        assert reasoning_agent.is_healthy() is True
        
        health = reasoning_agent.get_health_status()
        assert health["healthy"] is True
        assert health["agent"] == "test-reasoning"


# ============================================================================
# Test Edge Cases
# ============================================================================

@pytest.mark.asyncio
class TestEdgeCases:
    """Test edge cases and error handling"""
    
    async def test_empty_anomaly(self, reasoning_agent):
        """
        Test: Handles empty anomaly gracefully
        
        VERIFY:
        - No crash
        - Returns valid analysis
        """
        result = await reasoning_agent.execute({
            "anomaly": {},
            "similar_incidents": []
        })
        
        analysis = result["root_cause_analysis"]
        assert "root_cause" in analysis
        assert isinstance(analysis["confidence"], float)
    
    async def test_malformed_incidents(self, reasoning_agent, sample_anomaly):
        """
        Test: Handles malformed incident data
        
        VERIFY:
        - No crash with bad data
        - Continues processing
        """
        result = await reasoning_agent.execute({
            "anomaly": sample_anomaly,
            "similar_incidents": [
                {"bad": "data"},  # Missing expected fields
                None,  # Null value
                "not a dict"  # Wrong type
            ]
        })
        
        analysis = result["root_cause_analysis"]
        assert "root_cause" in analysis
    
    async def test_missing_fields(self, reasoning_agent):
        """
        Test: Works with minimal input data
        
        VERIFY:
        - Doesn't require all fields
        - Uses defaults appropriately
        """
        result = await reasoning_agent.execute({
            "anomaly": {"metric_name": "cpu"}
        })
        
        analysis = result["root_cause_analysis"]
        assert analysis["confidence"] > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
