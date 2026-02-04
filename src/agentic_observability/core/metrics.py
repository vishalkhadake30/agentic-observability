"""
Metrics Collection

OpenTelemetry and Prometheus metrics for agent performance,
latency, error rates, and system health.
"""

from typing import Dict, Optional
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from prometheus_client import start_http_server
import structlog

logger = structlog.get_logger()


class MetricsCollector:
    """
    Centralized metrics collection using OpenTelemetry.
    
    Tracks:
    - Agent execution latency
    - Success/failure rates
    - Circuit breaker state changes
    - API request counts
    """
    
    def __init__(self, service_name: str = "agentic-observability"):
        self.service_name = service_name
        
        # Setup Prometheus exporter
        prometheus_reader = PrometheusMetricReader()
        provider = MeterProvider(metric_readers=[prometheus_reader])
        metrics.set_meter_provider(provider)
        
        self.meter = metrics.get_meter(service_name)
        
        # Define metrics
        self.agent_latency = self.meter.create_histogram(
            name="agent.execution.latency",
            description="Agent execution latency in seconds",
            unit="s"
        )
        
        self.agent_requests = self.meter.create_counter(
            name="agent.requests.total",
            description="Total agent requests"
        )
        
        self.agent_errors = self.meter.create_counter(
            name="agent.errors.total",
            description="Total agent errors"
        )
        
        self.circuit_breaker_state = self.meter.create_up_down_counter(
            name="circuit_breaker.state",
            description="Circuit breaker state (0=closed, 1=half_open, 2=open)"
        )
        
        self.api_requests = self.meter.create_counter(
            name="api.requests.total",
            description="Total API requests"
        )
        
        logger.info("metrics_collector_initialized", service=service_name)
    
    def record_agent_latency(
        self,
        agent_name: str,
        latency: float,
        attributes: Optional[Dict[str, str]] = None
    ) -> None:
        """Record agent execution latency"""
        attrs = {"agent": agent_name}
        if attributes:
            attrs.update(attributes)
        self.agent_latency.record(latency, attributes=attrs)
    
    def increment_agent_requests(
        self,
        agent_name: str,
        status: str = "success"
    ) -> None:
        """Increment agent request counter"""
        self.agent_requests.add(
            1,
            attributes={"agent": agent_name, "status": status}
        )
    
    def increment_agent_errors(
        self,
        agent_name: str,
        error_type: str
    ) -> None:
        """Increment agent error counter"""
        self.agent_errors.add(
            1,
            attributes={"agent": agent_name, "error_type": error_type}
        )
    
    def record_circuit_breaker_state(
        self,
        breaker_name: str,
        state: str
    ) -> None:
        """Record circuit breaker state change"""
        state_value = {"closed": 0, "half_open": 1, "open": 2}.get(state, 0)
        self.circuit_breaker_state.add(
            state_value,
            attributes={"circuit_breaker": breaker_name, "state": state}
        )
    
    def increment_api_requests(
        self,
        endpoint: str,
        method: str,
        status_code: int
    ) -> None:
        """Increment API request counter"""
        self.api_requests.add(
            1,
            attributes={
                "endpoint": endpoint,
                "method": method,
                "status_code": str(status_code)
            }
        )


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create global metrics collector"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def start_metrics_server(port: int = 9090) -> None:
    """Start Prometheus metrics HTTP server"""
    start_http_server(port)
    logger.info("prometheus_server_started", port=port)
