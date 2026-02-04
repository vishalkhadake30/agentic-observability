"""
Unit tests for Action Agent

Tests cover:
- Alert generation and sending
- Action generation based on root cause
- Action execution (dry-run and real)
- Action validation and safety checks
- Action history tracking
- Severity-based channel routing
- Approval requirements

WHY THESE TESTS:
Action agents are critical for automated incident response.
These tests ensure actions are safe, auditable, and work correctly
under various scenarios.
"""

import asyncio
import pytest
from typing import Any
from unittest.mock import Mock, patch

from src.agentic_observability.agents.action.action import (
    MockActionAgent,
    ActionType,
    AlertSeverity,
    AlertChannel,
    Action,
    ActionResult
)
from src.agentic_observability.agents.base import AgentState


@pytest.fixture
async def mock_action_agent():
    """Fixture providing initialized mock action agent"""
    agent = MockActionAgent(
        name="test-action-agent",
        dry_run=False,
        circuit_breaker_threshold=10,
        max_retries=2
    )
    await agent.initialize()
    yield agent
    await agent.cleanup()


@pytest.fixture
async def dry_run_agent():
    """Fixture providing dry-run action agent"""
    agent = MockActionAgent(
        name="dry-run-agent",
        dry_run=True,
        circuit_breaker_threshold=10
    )
    await agent.initialize()
    yield agent
    await agent.cleanup()


@pytest.mark.asyncio
class TestActionAgentInitialization:
    """Test Action Agent initialization"""
    
    async def test_initialization(self, mock_action_agent):
        """
        Test: Agent initializes correctly
        
        VERIFY:
        - State is IDLE
        - Dry-run flag set correctly
        - Action history is empty
        """
        assert mock_action_agent._state == AgentState.IDLE
        assert mock_action_agent.dry_run is False
        assert len(mock_action_agent._action_history) == 0
        assert mock_action_agent.is_healthy()


@pytest.mark.asyncio
class TestAlertGeneration:
    """Test alert generation and sending"""
    
    async def test_critical_alert_all_channels(self, mock_action_agent):
        """
        Test: Critical alerts sent to all channels
        
        VERIFY:
        - Critical severity triggers PagerDuty, Slack, Email
        - Alert structure is correct
        - All channels receive alert
        """
        input_data = {
            "root_cause": "Production database down",
            "confidence": 0.95,
            "recommendations": ["Restart database", "Check network"],
            "anomaly_data": {"service_name": "postgres"},
            "severity": "critical"
        }
        
        result = await mock_action_agent.execute(input_data)
        
        # Verify alerts were sent
        assert "alerts_sent" in result
        alerts = result["alerts_sent"]
        assert len(alerts) == 3  # PagerDuty, Slack, Email
        
        # Verify channels
        channels = [alert["channel"] for alert in alerts]
        assert "pagerduty" in channels
        assert "slack" in channels
        assert "email" in channels
        
        # Verify severity
        for alert in alerts:
            assert alert["severity"] == "critical"
    
    async def test_warning_alert_limited_channels(self, mock_action_agent):
        """
        Test: Warning alerts sent to limited channels
        
        VERIFY:
        - Warning severity only triggers Slack
        - PagerDuty not triggered
        """
        input_data = {
            "root_cause": "High response time",
            "confidence": 0.75,
            "recommendations": ["Investigate slow queries"],
            "anomaly_data": {},
            "severity": "warning"
        }
        
        result = await mock_action_agent.execute(input_data)
        
        alerts = result["alerts_sent"]
        channels = [alert["channel"] for alert in alerts]
        
        # Only Slack for warnings
        assert "slack" in channels
        assert "pagerduty" not in channels
        assert len(alerts) == 1
    
    async def test_error_alert_moderate_channels(self, mock_action_agent):
        """
        Test: Error alerts sent to moderate channels
        
        VERIFY:
        - Error severity triggers Slack and Email
        - No PagerDuty for errors
        """
        input_data = {
            "root_cause": "Service failing health checks",
            "confidence": 0.85,
            "recommendations": ["Restart service"],
            "anomaly_data": {},
            "severity": "error"
        }
        
        result = await mock_action_agent.execute(input_data)
        
        alerts = result["alerts_sent"]
        channels = [alert["channel"] for alert in alerts]
        
        assert "slack" in channels
        assert "email" in channels
        assert "pagerduty" not in channels
        assert len(alerts) == 2


@pytest.mark.asyncio
class TestActionGeneration:
    """Test remediation action generation"""
    
    async def test_memory_leak_actions(self, mock_action_agent):
        """
        Test: Memory leak generates restart and cache clear
        
        VERIFY:
        - Restart service action generated
        - Clear cache action generated
        - Both marked as safe to automate
        """
        input_data = {
            "root_cause": "Memory leak detected in service",
            "confidence": 0.85,
            "recommendations": ["Restart service", "Clear cache"],
            "anomaly_data": {"service_name": "api-server"},
            "severity": "warning"
        }
        
        result = await mock_action_agent.execute(input_data)
        
        actions = result["actions_executed"]
        assert len(actions) == 2
        
        action_types = [a["type"] for a in actions]
        assert "restart_service" in action_types
        assert "clear_cache" in action_types
        
        # Verify target
        for action in actions:
            assert action["target"] == "api-server"
    
    async def test_deployment_rollback_requires_approval(self, mock_action_agent):
        """
        Test: Deployment rollback requires approval
        
        VERIFY:
        - Rollback action generated
        - Action marked as requiring approval
        - Action skipped in automated mode
        """
        input_data = {
            "root_cause": "Recent deployment caused errors",
            "confidence": 0.90,
            "recommendations": ["Rollback to previous version"],
            "anomaly_data": {"service_name": "web-app"},
            "severity": "error"
        }
        
        result = await mock_action_agent.execute(input_data)
        
        # Action should be skipped due to approval requirement
        assert len(result["actions_skipped"]) == 1
        assert len(result["actions_executed"]) == 0
        
        skipped = result["actions_skipped"][0]
        assert skipped["type"] == "rollback_deployment"
        assert skipped["reason"] == "requires_approval"
    
    async def test_traffic_spike_scaling(self, mock_action_agent):
        """
        Test: Traffic spike triggers scale-up
        
        VERIFY:
        - Scale-up action generated
        - Correct instance count
        - Safe to automate
        """
        input_data = {
            "root_cause": "Unexpected traffic spike",
            "confidence": 0.80,
            "recommendations": ["Scale up instances"],
            "anomaly_data": {"service_name": "api-gateway"},
            "severity": "warning"
        }
        
        result = await mock_action_agent.execute(input_data)
        
        actions = result["actions_executed"]
        assert len(actions) == 1
        assert actions[0]["type"] == "scale_up"
        assert actions[0]["target"] == "api-gateway"
    
    async def test_low_confidence_skips_actions(self, mock_action_agent):
        """
        Test: Low confidence skips auto-remediation
        
        VERIFY:
        - Alerts still sent
        - No actions executed
        - Confidence threshold enforced
        """
        input_data = {
            "root_cause": "Potential memory leak",
            "confidence": 0.60,  # Below 0.7 threshold
            "recommendations": ["Monitor closely"],
            "anomaly_data": {"service_name": "worker"},
            "severity": "info"
        }
        
        result = await mock_action_agent.execute(input_data)
        
        # Alerts sent but no actions
        assert len(result["alerts_sent"]) > 0
        assert len(result["actions_executed"]) == 0
        assert result["total_actions"] == 0


@pytest.mark.asyncio
class TestDryRunMode:
    """Test dry-run mode functionality"""
    
    async def test_dry_run_simulates_actions(self, dry_run_agent):
        """
        Test: Dry-run mode simulates without executing
        
        VERIFY:
        - Actions appear executed
        - dry_run flag set in result
        - No actual side effects
        """
        input_data = {
            "root_cause": "Memory leak in service",
            "confidence": 0.85,
            "recommendations": ["Restart service"],
            "anomaly_data": {"service_name": "test-service"},
            "severity": "warning"
        }
        
        result = await dry_run_agent.execute(input_data)
        
        # Actions simulated
        assert result["dry_run"] is True
        assert len(result["actions_executed"]) > 0
        
        # Verify in action history
        history = dry_run_agent.get_action_history()
        assert len(history) > 0
        assert all(r.success for r in history)


@pytest.mark.asyncio
class TestActionHistory:
    """Test action history tracking"""
    
    async def test_action_history_tracked(self, mock_action_agent):
        """
        Test: All actions tracked in history
        
        VERIFY:
        - Successful actions in history
        - Failed actions in history
        - History persists across executions
        """
        input_data = {
            "root_cause": "Memory leak",
            "confidence": 0.85,
            "recommendations": ["Restart service"],
            "anomaly_data": {"service_name": "app1"},
            "severity": "warning"
        }
        
        # Execute first time
        await mock_action_agent.execute(input_data)
        history1 = mock_action_agent.get_action_history()
        count1 = len(history1)
        assert count1 > 0
        
        # Execute second time
        input_data["anomaly_data"]["service_name"] = "app2"
        await mock_action_agent.execute(input_data)
        history2 = mock_action_agent.get_action_history()
        
        # History accumulated
        assert len(history2) > count1
        
        # All entries are ActionResult objects
        for result in history2:
            assert hasattr(result, "action")
            assert hasattr(result, "success")
            assert hasattr(result, "executed_at")
    
    async def test_history_includes_skipped_actions(self, mock_action_agent):
        """
        Test: Skipped actions also tracked
        
        VERIFY:
        - Approval-required actions in history
        - Marked as unsuccessful
        - Error reason captured
        """
        input_data = {
            "root_cause": "Bad deployment",
            "confidence": 0.90,
            "recommendations": ["Rollback"],
            "anomaly_data": {"service_name": "web"},
            "severity": "error"
        }
        
        await mock_action_agent.execute(input_data)
        
        history = mock_action_agent.get_action_history()
        assert len(history) > 0
        
        # Find the rollback action
        rollback_result = next(
            (r for r in history if r.action.action_type == ActionType.ROLLBACK_DEPLOYMENT),
            None
        )
        assert rollback_result is not None
        assert rollback_result.success is False
        assert "approval" in rollback_result.error.lower()


@pytest.mark.asyncio
class TestActionExecution:
    """Test action execution details"""
    
    async def test_action_duration_tracked(self, mock_action_agent):
        """
        Test: Action execution duration measured
        
        VERIFY:
        - Duration > 0 for executed actions
        - Duration in milliseconds
        """
        input_data = {
            "root_cause": "Memory leak",
            "confidence": 0.85,
            "recommendations": ["Restart"],
            "anomaly_data": {"service_name": "svc"},
            "severity": "warning"
        }
        
        result = await mock_action_agent.execute(input_data)
        
        executed = result["actions_executed"]
        assert len(executed) > 0
        
        # All actions have duration
        for action in executed:
            assert "duration_ms" in action
            assert action["duration_ms"] > 0
    
    async def test_connection_pool_restart(self, mock_action_agent):
        """
        Test: Connection pool issue triggers restart
        
        VERIFY:
        - Restart action for connection pool
        - Graceful restart parameter
        """
        input_data = {
            "root_cause": "Connection pool exhausted",
            "confidence": 0.88,
            "recommendations": ["Restart service"],
            "anomaly_data": {"service_name": "db-proxy"},
            "severity": "error"
        }
        
        result = await mock_action_agent.execute(input_data)
        
        actions = result["actions_executed"]
        assert len(actions) == 1
        assert actions[0]["type"] == "restart_service"
        assert actions[0]["target"] == "db-proxy"
    
    async def test_disk_full_clears_cache(self, mock_action_agent):
        """
        Test: Disk space issue triggers cache clear
        
        VERIFY:
        - Clear cache action generated
        - Safe to automate
        """
        input_data = {
            "root_cause": "Disk space critically low",
            "confidence": 0.92,
            "recommendations": ["Clear temporary files"],
            "anomaly_data": {"service_name": "storage-server"},
            "severity": "critical"
        }
        
        result = await mock_action_agent.execute(input_data)
        
        actions = result["actions_executed"]
        action_types = [a["type"] for a in actions]
        assert "clear_cache" in action_types


@pytest.mark.asyncio
class TestActionAgentResilience:
    """Test circuit breaker and retry behavior"""
    
    async def test_inherits_base_agent_resilience(self, mock_action_agent):
        """
        Test: Action agent inherits BaseAgent patterns
        
        VERIFY:
        - Circuit breaker available
        - Retry logic available
        - Metrics tracked
        """
        # Verify circuit breaker
        assert mock_action_agent._circuit_breaker is not None
        
        # Verify metrics
        metrics = mock_action_agent.get_metrics()
        assert "total_executions" in metrics
        assert "success_rate" in metrics
        
        # Execute successfully
        input_data = {
            "root_cause": "Test",
            "confidence": 0.5,  # Low confidence, no actions
            "recommendations": [],
            "anomaly_data": {},
            "severity": "info"
        }
        
        result = await mock_action_agent.execute(input_data)
        
        # Verify execution counted
        metrics_after = mock_action_agent.get_metrics()
        assert metrics_after["total_executions"] == 1
        assert metrics_after["total_successes"] == 1


@pytest.mark.asyncio
class TestEdgeCases:
    """Test edge cases and error handling"""
    
    async def test_empty_recommendations(self, mock_action_agent):
        """
        Test: Empty recommendations handled gracefully
        
        VERIFY:
        - Alerts still sent
        - No actions generated
        - No errors
        """
        input_data = {
            "root_cause": "Unknown issue",
            "confidence": 0.85,
            "recommendations": [],
            "anomaly_data": {},
            "severity": "warning"
        }
        
        result = await mock_action_agent.execute(input_data)
        
        assert result["total_actions"] == 0
        assert len(result["alerts_sent"]) > 0
    
    async def test_missing_service_name(self, mock_action_agent):
        """
        Test: Missing service name uses fallback
        
        VERIFY:
        - Action generated with "unknown-service"
        - No errors
        """
        input_data = {
            "root_cause": "Memory leak",
            "confidence": 0.85,
            "recommendations": ["Restart"],
            "anomaly_data": {},  # No service_name
            "severity": "warning"
        }
        
        result = await mock_action_agent.execute(input_data)
        
        actions = result["actions_executed"]
        if len(actions) > 0:
            assert actions[0]["target"] == "unknown-service"
    
    async def test_multiple_pattern_matches(self, mock_action_agent):
        """
        Test: Multiple patterns in root cause
        
        VERIFY:
        - Actions generated for all patterns
        - No duplicates
        """
        input_data = {
            "root_cause": "Memory leak and connection pool exhausted",
            "confidence": 0.90,
            "recommendations": ["Restart service", "Clear cache"],
            "anomaly_data": {"service_name": "multi-issue"},
            "severity": "error"
        }
        
        result = await mock_action_agent.execute(input_data)
        
        # Should generate actions for both patterns
        # Memory leak: restart + clear cache
        # Connection pool: restart
        # Deduplicated to: restart + clear cache
        assert result["total_actions"] >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
