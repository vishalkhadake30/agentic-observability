"""
Reasoning Agent - Root Cause Analysis

Analyzes anomalies and historical incidents to determine root causes.

WHY TWO IMPLEMENTATIONS:
1. MockReasoningAgent (rule-based): Works immediately, no API needed
2. LLMReasoningAgent (future): Claude/GPT for advanced reasoning

DESIGN PATTERN: Strategy Pattern
- Both implement same interface
- Coordinator doesn't care which is used
- Swap implementations without code changes

Example Usage:
    # Use mock (no LLM needed)
    agent = MockReasoningAgent(name="reasoning")
    
    # Later, swap with LLM (when credits available)
    # agent = LLMReasoningAgent(name="reasoning", api_key="...")
    
    result = await agent.execute({
        "anomaly": {...},
        "similar_incidents": [...],
        "metric_data": [...]
    })
"""

from typing import Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import structlog

from ..base import BaseAgent

logger = structlog.get_logger()


@dataclass
class RootCauseAnalysis:
    """Result of root cause reasoning"""
    root_cause: str
    confidence: float  # 0.0 to 1.0
    reasoning_steps: list[str]
    supporting_evidence: list[str]
    recommendations: list[str]
    similar_incident_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ReasoningAgent(BaseAgent):
    """
    Base Reasoning Agent (abstract interface).
    
    Subclasses implement different reasoning strategies:
    - MockReasoningAgent: Rule-based pattern matching
    - LLMReasoningAgent: Claude/GPT-powered reasoning (TODO)
    """
    
    def __init__(self, name: str = "reasoning", **kwargs):
        super().__init__(name=name, **kwargs)
    
    async def _execute_impl(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Subclasses override this to implement reasoning logic.
        
        Args:
            input_data: Must contain:
                - anomaly: Anomaly detection result
                - similar_incidents: Historical incidents from RAG
                - metric_data: Optional time series data
        
        Returns:
            Dict with root_cause_analysis
        """
        raise NotImplementedError("Subclasses must implement reasoning logic")


class MockReasoningAgent(ReasoningAgent):
    """
    Rule-based reasoning agent (no LLM required).
    
    WHY MOCK IMPLEMENTATION:
    - Works immediately without API credits
    - Demonstrates architecture and testing
    - Production-quality pattern matching
    - Easy to swap with LLM later
    
    REASONING STRATEGY:
    1. Analyze anomaly type (spike, drop, drift)
    2. Check similar incidents for patterns
    3. Apply rule-based heuristics
    4. Generate structured root cause analysis
    
    PATTERN MATCHING RULES:
    - "cpu" + "memory leak" in history → Memory leak likely
    - "spike" + "deployment" in history → Deployment related
    - "database" + "timeout" → Connection pool exhaustion
    - "gradual drift" + no history → New baseline pattern
    
    Example:
        agent = MockReasoningAgent()
        await agent.initialize()
        
        result = await agent.execute({
            "anomaly": {
                "anomaly_type": "spike",
                "metric_name": "cpu_percent",
                "severity": 0.8
            },
            "similar_incidents": [
                {"root_cause": "Memory leak", "similarity_score": 0.9}
            ]
        })
        
        print(result["root_cause_analysis"]["root_cause"])
        # Output: "Memory leak causing CPU spike"
    """
    
    def __init__(
        self,
        name: str = "mock-reasoning",
        confidence_threshold: float = 0.7,
        **kwargs
    ):
        """
        Initialize mock reasoning agent.
        
        Args:
            name: Agent identifier
            confidence_threshold: Minimum confidence for high-certainty causes
            **kwargs: Passed to BaseAgent
        """
        super().__init__(name=name, **kwargs)
        self.confidence_threshold = confidence_threshold
        
        # Pattern matching rules (expandable)
        self._patterns = {
            "memory_leak": ["memory leak", "heap", "oom", "out of memory"],
            "deployment": ["deployment", "deploy", "release", "rollout"],
            "connection_pool": ["timeout", "connection", "pool", "database"],
            "traffic_spike": ["traffic", "requests", "load", "ddos"],
            "disk_full": ["disk", "storage", "inode", "filesystem"],
            "network_issue": ["network", "latency", "packet", "dns"]
        }
        
        self.logger.info(
            "mock_reasoning_agent_initialized",
            patterns=len(self._patterns),
            confidence_threshold=confidence_threshold
        )
    
    async def _execute_impl(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Execute rule-based root cause analysis.
        
        Args:
            input_data: Contains anomaly, similar_incidents, metric_data
        
        Returns:
            Dict with root_cause_analysis
        """
        # Accept both legacy keys and coordinator keys.
        anomaly = input_data.get("anomaly") or input_data.get("anomaly_data") or {}
        similar_incidents = (
            input_data.get("similar_incidents")
            or input_data.get("historical_incidents")
            or []
        )
        metric_data = input_data.get("metric_data") or input_data.get("metrics") or []
        
        self.logger.info(
            "analyzing_root_cause",
            anomaly_type=anomaly.get("anomaly_type"),
            metric_name=anomaly.get("metric_name"),
            similar_incidents_count=len(similar_incidents)
        )
        
        # Step 1: Analyze anomaly characteristics
        anomaly_type = anomaly.get("anomaly_type", "unknown")
        metric_name = anomaly.get("metric_name", "unknown")
        severity = anomaly.get("severity", 0.5)
        
        # Step 2: Extract patterns from similar incidents
        historical_causes = []
        for incident in similar_incidents:
            if isinstance(incident, dict):
                cause = incident.get("root_cause", "")
                similarity = incident.get("similarity_score", 0.0)
                if cause and similarity > 0.5:
                    historical_causes.append((cause, similarity))
        
        # Step 3: Apply pattern matching
        detected_patterns = self._detect_patterns(anomaly, historical_causes)
        
        # Step 4: Generate root cause hypothesis
        root_cause, confidence, reasoning = self._generate_hypothesis(
            anomaly_type,
            metric_name,
            severity,
            detected_patterns,
            historical_causes
        )
        
        # Step 5: Generate recommendations
        recommendations = self._generate_recommendations(
            root_cause,
            anomaly_type,
            metric_name
        )
        
        # Step 6: Compile evidence
        evidence = self._compile_evidence(
            anomaly,
            detected_patterns,
            historical_causes
        )
        
        analysis = RootCauseAnalysis(
            root_cause=root_cause,
            confidence=confidence,
            reasoning_steps=reasoning,
            supporting_evidence=evidence,
            recommendations=recommendations,
            similar_incident_ids=[
                inc.get("incident_id", "") 
                for inc in similar_incidents 
                if isinstance(inc, dict)
            ],
            metadata={
                "anomaly_type": anomaly_type,
                "metric_name": metric_name,
                "severity": severity,
                "detected_patterns": detected_patterns,
                "reasoning_method": "rule_based"
            }
        )
        
        self.logger.info(
            "root_cause_analysis_completed",
            root_cause=root_cause,
            confidence=round(confidence, 2),
            patterns_detected=len(detected_patterns)
        )
        
        # Coordinator expects top-level keys; keep the full structured analysis too.
        return {
            "root_cause": analysis.root_cause,
            "root_cause_hypothesis": analysis.root_cause,
            "confidence": analysis.confidence,
            "recommendations": analysis.recommendations,
            "supporting_evidence": analysis.supporting_evidence,
            "reasoning_steps": analysis.reasoning_steps,
            "root_cause_analysis": {
                "root_cause": analysis.root_cause,
                "confidence": analysis.confidence,
                "reasoning_steps": analysis.reasoning_steps,
                "supporting_evidence": analysis.supporting_evidence,
                "recommendations": analysis.recommendations,
                "similar_incident_ids": analysis.similar_incident_ids,
                "metadata": analysis.metadata,
            },
        }
    
    def _detect_patterns(
        self,
        anomaly: dict[str, Any],
        historical_causes: list[tuple[str, float]]
    ) -> list[str]:
        """
        Detect patterns in anomaly and historical data.
        
        Returns:
            List of detected pattern names (e.g., ["memory_leak", "deployment"])
        """
        detected = []
        
        # Check anomaly description/metadata
        anomaly_text = str(anomaly).lower()
        
        # Check historical causes
        historical_text = " ".join([cause.lower() for cause, _ in historical_causes])
        
        combined_text = anomaly_text + " " + historical_text
        
        for pattern_name, keywords in self._patterns.items():
            if any(keyword in combined_text for keyword in keywords):
                detected.append(pattern_name)
        
        return detected
    
    def _generate_hypothesis(
        self,
        anomaly_type: str,
        metric_name: str,
        severity: float,
        detected_patterns: list[str],
        historical_causes: list[tuple[str, float]]
    ) -> tuple[str, float, list[str]]:
        """
        Generate root cause hypothesis with confidence and reasoning steps.
        
        Returns:
            (root_cause, confidence, reasoning_steps)
        """
        reasoning = []
        
        # Default hypothesis
        root_cause = f"{anomaly_type.title()} in {metric_name}"
        confidence = 0.5
        
        reasoning.append(f"Detected {anomaly_type} anomaly in {metric_name}")
        reasoning.append(f"Severity: {severity:.2f}")
        
        # Adjust based on patterns
        if detected_patterns:
            primary_pattern = detected_patterns[0]
            root_cause = self._pattern_to_cause(primary_pattern, anomaly_type, metric_name)
            confidence = min(0.75 + (len(detected_patterns) * 0.05), 0.95)
            
            reasoning.append(f"Detected pattern: {primary_pattern}")
            reasoning.append(f"Pattern matches {len(detected_patterns)} known issue(s)")
        
        # Boost confidence with historical evidence
        if historical_causes:
            top_cause, top_similarity = historical_causes[0]
            if top_similarity > 0.8:
                confidence = min(confidence + 0.15, 0.98)
                root_cause = top_cause
                reasoning.append(f"Strong historical match: '{top_cause}' (similarity: {top_similarity:.2f})")
            elif top_similarity > 0.5:
                confidence = min(confidence + 0.1, 0.95)
                root_cause = top_cause
                reasoning.append(f"Historical match: '{top_cause}' (similarity: {top_similarity:.2f})")
        
        # Adjust for severity
        if severity > 0.8:
            reasoning.append("High severity suggests critical issue requiring immediate attention")
        
        return root_cause, confidence, reasoning
    
    def _pattern_to_cause(
        self,
        pattern: str,
        anomaly_type: str,
        metric_name: str
    ) -> str:
        """Map detected pattern to root cause description"""
        pattern_causes = {
            "memory_leak": "Memory leak causing resource exhaustion",
            "deployment": "Deployment-related configuration or code issue",
            "connection_pool": "Database connection pool exhaustion",
            "traffic_spike": "Sudden traffic increase or DDoS attack",
            "disk_full": "Disk space exhaustion",
            "network_issue": "Network connectivity or latency problem"
        }
        
        base_cause = pattern_causes.get(pattern, f"{pattern.replace('_', ' ').title()} detected")
        
        # Contextualize with metric
        if "cpu" in metric_name.lower():
            return f"{base_cause} affecting CPU usage"
        elif "memory" in metric_name.lower():
            return f"{base_cause} affecting memory usage"
        elif "disk" in metric_name.lower():
            return f"{base_cause} affecting disk I/O"
        else:
            return base_cause
    
    def _generate_recommendations(
        self,
        root_cause: str,
        anomaly_type: str,
        metric_name: str
    ) -> list[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        root_cause_lower = root_cause.lower()
        
        # Pattern-specific recommendations
        if "memory leak" in root_cause_lower:
            recommendations.extend([
                "Analyze heap dump to identify leaking objects",
                "Check for unclosed resources (connections, file handles)",
                "Review recent code changes for memory management issues",
                "Consider restarting affected services as temporary mitigation"
            ])
        
        elif "deployment" in root_cause_lower:
            recommendations.extend([
                "Review recent deployment logs for errors",
                "Compare configuration changes in latest release",
                "Consider rollback if issue persists",
                "Enable canary deployment for future releases"
            ])
        
        elif "connection pool" in root_cause_lower or "timeout" in root_cause_lower:
            recommendations.extend([
                "Increase database connection pool size",
                "Review slow queries and optimize",
                "Check database server health and capacity",
                "Implement connection retry logic"
            ])
        
        elif "traffic" in root_cause_lower or "ddos" in root_cause_lower:
            recommendations.extend([
                "Enable rate limiting on affected endpoints",
                "Scale up infrastructure to handle load",
                "Review traffic patterns for DDoS indicators",
                "Implement caching to reduce backend load"
            ])
        
        elif "disk" in root_cause_lower:
            recommendations.extend([
                "Clean up old logs and temporary files",
                "Expand disk capacity",
                "Implement log rotation policies",
                "Archive old data to cold storage"
            ])
        
        else:
            # Generic recommendations
            recommendations.extend([
                f"Investigate {metric_name} patterns over longer time window",
                "Compare with similar incidents in history",
                "Monitor related metrics for correlation",
                "Set up alerts for recurrence"
            ])
        
        return recommendations
    
    def _compile_evidence(
        self,
        anomaly: dict[str, Any],
        detected_patterns: list[str],
        historical_causes: list[tuple[str, float]]
    ) -> list[str]:
        """Compile supporting evidence"""
        evidence = []
        
        # Anomaly evidence
        anomaly_type = anomaly.get("anomaly_type", "unknown")
        severity = anomaly.get("severity", 0)
        evidence.append(f"Anomaly type: {anomaly_type}")
        evidence.append(f"Severity score: {severity:.2f}")
        
        if anomaly.get("value"):
            evidence.append(f"Anomalous value: {anomaly['value']}")
        
        # Pattern evidence
        if detected_patterns:
            evidence.append(f"Matching patterns: {', '.join(detected_patterns)}")
        
        # Historical evidence
        if historical_causes:
            top_3 = historical_causes[:3]
            for cause, similarity in top_3:
                evidence.append(f"Historical: '{cause}' (similarity: {similarity:.2f})")
        
        return evidence


class HuggingFaceReasoningAgent(ReasoningAgent):
    """
    LLM-powered reasoning agent using HuggingFace Inference API.
    
    WHY HUGGINGFACE:
    - Free tier with generous limits (~1000 calls/day)
    - Open-source models (Llama 3, Mistral, etc.)
    - Cost-effective at scale vs proprietary APIs
    - Data privacy - can deploy on-prem later
    
    PRODUCTION CONSIDERATIONS:
    At enterprise scale, we'd:
    - Fine-tune Llama 3 on historical incident data
    - Deploy model on-prem for data privacy
    - Use TorchServe/vLLM for low-latency serving
    - Implement model monitoring and drift detection
    
    Example:
        # With HuggingFace token (free)
        agent = HuggingFaceReasoningAgent(
            name="hf-reasoning",
            hf_token="hf_...",
            model="meta-llama/Meta-Llama-3-8B-Instruct"
        )
        
        # Without token (falls back to mock)
        agent = HuggingFaceReasoningAgent(name="reasoning")
    """
    
    def __init__(
        self,
        name: str = "hf-reasoning",
        hf_token: Optional[str] = None,
        model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        max_tokens: int = 500,
        temperature: float = 0.7,
        use_mock_fallback: bool = True,
        **kwargs
    ):
        """
        Initialize HuggingFace reasoning agent.
        
        Args:
            name: Agent identifier
            hf_token: HuggingFace API token (get from https://huggingface.co/settings/tokens)
            model: HF model to use (default: Llama 3 8B Instruct)
            max_tokens: Maximum tokens in response
            temperature: LLM temperature (0=deterministic, 1=creative)
            use_mock_fallback: If True, fallback to mock reasoning on API errors
            **kwargs: Passed to BaseAgent
        """
        super().__init__(name=name, **kwargs)
        
        self.hf_token = hf_token
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_mock_fallback = use_mock_fallback
        
        # Initialize HuggingFace client if token available
        self._hf_client: Optional[Any] = None
        self._use_hf = False
        
        if self.hf_token:
            try:
                from huggingface_hub import InferenceClient
                self._hf_client = InferenceClient(token=self.hf_token)
                self._use_hf = True
                self.logger.info(
                    "huggingface_reasoning_enabled",
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            except ImportError:
                self.logger.warning(
                    "huggingface_hub_not_installed",
                    info="Install with: pip install huggingface-hub"
                )
                self._use_hf = False
            except Exception as e:
                self.logger.error(
                    "huggingface_client_init_failed",
                    error=str(e),
                    exc_info=True
                )
                self._use_hf = False
        else:
            self.logger.info(
                "huggingface_token_not_provided",
                fallback="mock_reasoning",
                info="Set HUGGINGFACE_TOKEN env var to enable LLM reasoning"
            )
        
        # Initialize mock reasoning agent as fallback
        if self.use_mock_fallback:
            self._mock_agent = MockReasoningAgent(
                name=f"{name}-mock-fallback",
                **kwargs
            )
    
    async def initialize(self) -> None:
        """Initialize agent and fallback"""
        await super().initialize()
        if self.use_mock_fallback and self._mock_agent:
            await self._mock_agent.initialize()
    
    async def cleanup(self) -> None:
        """Cleanup agent and fallback"""
        if self.use_mock_fallback and self._mock_agent:
            await self._mock_agent.cleanup()
        await super().cleanup()
    
    async def _execute_impl(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Execute LLM-powered root cause analysis with fallback.
        
        Args:
            input_data: Contains anomaly, similar_incidents, metric_data
        
        Returns:
            Dict with root_cause_analysis
        """
        # If HuggingFace not available, use mock fallback
        if not self._use_hf or not self._hf_client:
            if self.use_mock_fallback and self._mock_agent:
                self.logger.info("using_mock_fallback", reason="hf_not_available")
                return await self._mock_agent._execute_impl(input_data)
            else:
                raise RuntimeError("HuggingFace client not initialized and no fallback")
        
        # Try HuggingFace LLM reasoning
        try:
            return await self._hf_reasoning(input_data)
        except Exception as e:
            self.logger.error(
                "hf_reasoning_failed",
                error=str(e),
                exc_info=True
            )
            
            # Fallback to mock if enabled
            if self.use_mock_fallback and self._mock_agent:
                self.logger.info("using_mock_fallback", reason="hf_error")
                return await self._mock_agent._execute_impl(input_data)
            else:
                raise
    
    async def _hf_reasoning(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Execute HuggingFace LLM reasoning"""
        # Accept both legacy keys and coordinator keys
        anomaly = input_data.get("anomaly") or input_data.get("anomaly_data") or {}
        similar_incidents = (
            input_data.get("similar_incidents")
            or input_data.get("historical_incidents")
            or []
        )
        metric_data = input_data.get("metric_data") or input_data.get("metrics") or []
        
        self.logger.info(
            "executing_hf_reasoning",
            model=self.model,
            anomaly_type=anomaly.get("anomaly_type"),
            similar_incidents_count=len(similar_incidents)
        )
        
        # Build prompt for LLM
        prompt = self._build_prompt(anomaly, similar_incidents, metric_data)
        
        # Call HuggingFace Inference API using chat completion
        try:
            messages = [
                {"role": "system", "content": "You are an expert Site Reliability Engineer analyzing production incidents."},
                {"role": "user", "content": prompt}
            ]
            
            response = await asyncio.to_thread(
                self._hf_client.chat_completion,
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # Extract text from chat completion response
            response_text = response.choices[0].message.content if response.choices else ""
            
            self.logger.info(
                "hf_response_received",
                response_length=len(response_text) if response_text else 0
            )
            
        except Exception as e:
            self.logger.error(
                "hf_api_call_failed",
                error=str(e),
                model=self.model,
                exc_info=True
            )
            raise
        
        # Parse LLM response
        analysis = self._parse_llm_response(response_text, anomaly, similar_incidents)
        
        self.logger.info(
            "hf_reasoning_completed",
            root_cause=analysis.root_cause,
            confidence=round(analysis.confidence, 2),
            reasoning_method="huggingface_llm"
        )
        
        # Return in coordinator-expected format
        return {
            "root_cause": analysis.root_cause,
            "root_cause_hypothesis": analysis.root_cause,
            "confidence": analysis.confidence,
            "recommendations": analysis.recommendations,
            "supporting_evidence": analysis.supporting_evidence,
            "reasoning_steps": analysis.reasoning_steps,
            "root_cause_analysis": {
                "root_cause": analysis.root_cause,
                "confidence": analysis.confidence,
                "reasoning_steps": analysis.reasoning_steps,
                "supporting_evidence": analysis.supporting_evidence,
                "recommendations": analysis.recommendations,
                "similar_incident_ids": analysis.similar_incident_ids,
                "metadata": analysis.metadata,
            },
        }
    
    def _build_prompt(
        self,
        anomaly: dict[str, Any],
        similar_incidents: list[dict[str, Any]],
        metric_data: list[dict[str, Any]]
    ) -> str:
        """
        Build structured prompt for LLM.
        
        PROMPT ENGINEERING:
        - Clear role definition (you are an SRE expert)
        - Structured input format
        - Explicit output format
        - Few-shot examples would improve this
        """
        prompt = f"""You are an expert Site Reliability Engineer analyzing a production incident.

ANOMALY DETECTED:
- Type: {anomaly.get('anomaly_type', 'unknown')}
- Metric: {anomaly.get('metric_name', 'unknown')}
- Severity: {anomaly.get('severity', 0.0):.2f}
- Value: {anomaly.get('value', 'N/A')}
- Description: {anomaly.get('description', 'No description')}

"""
        
        if similar_incidents:
            prompt += "SIMILAR PAST INCIDENTS:\n"
            for i, incident in enumerate(similar_incidents[:5], 1):
                if isinstance(incident, dict):
                    prompt += f"{i}. Root Cause: {incident.get('root_cause', 'Unknown')}\n"
                    prompt += f"   Similarity: {incident.get('similarity_score', 0.0):.2f}\n"
                    if incident.get('resolution'):
                        prompt += f"   Resolution: {incident.get('resolution')}\n"
            prompt += "\n"
        
        if metric_data:
            prompt += f"METRIC DATA: {len(metric_data)} data points available\n\n"
        
        prompt += """YOUR TASK:
Analyze this incident and provide:
1. ROOT CAUSE: A clear, specific root cause (1 sentence)
2. CONFIDENCE: Your confidence level (0.0 to 1.0)
3. REASONING: Step-by-step analysis (3-5 steps)
4. EVIDENCE: Supporting evidence (2-3 points)
5. RECOMMENDATIONS: Actionable fixes (3-4 recommendations)

Format your response as follows:
ROOT CAUSE: [your analysis]
CONFIDENCE: [0.0-1.0]
REASONING:
- Step 1: [first reasoning step]
- Step 2: [second reasoning step]
- Step 3: [third reasoning step]
EVIDENCE:
- [evidence point 1]
- [evidence point 2]
RECOMMENDATIONS:
- [recommendation 1]
- [recommendation 2]
- [recommendation 3]

Begin your analysis:"""
        
        return prompt
    
    def _parse_llm_response(
        self,
        response: str,
        anomaly: dict[str, Any],
        similar_incidents: list[dict[str, Any]]
    ) -> RootCauseAnalysis:
        """
        Parse LLM response into structured RootCauseAnalysis.
        
        Handles various response formats gracefully.
        """
        # Default values
        root_cause = "Unknown root cause"
        confidence = 0.5
        reasoning_steps = []
        evidence = []
        recommendations = []
        
        try:
            lines = response.strip().split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Parse sections
                if line.startswith("ROOT CAUSE:"):
                    root_cause = line.replace("ROOT CAUSE:", "").strip()
                    current_section = None
                
                elif line.startswith("CONFIDENCE:"):
                    try:
                        conf_str = line.replace("CONFIDENCE:", "").strip()
                        confidence = float(conf_str)
                        confidence = max(0.0, min(1.0, confidence))  # Clamp 0-1
                    except ValueError:
                        confidence = 0.7  # Default if parsing fails
                    current_section = None
                
                elif line.startswith("REASONING:"):
                    current_section = "reasoning"
                
                elif line.startswith("EVIDENCE:"):
                    current_section = "evidence"
                
                elif line.startswith("RECOMMENDATIONS:"):
                    current_section = "recommendations"
                
                elif line.startswith("-") or line.startswith("•"):
                    # List item
                    item = line.lstrip("-•").strip()
                    if current_section == "reasoning":
                        reasoning_steps.append(item)
                    elif current_section == "evidence":
                        evidence.append(item)
                    elif current_section == "recommendations":
                        recommendations.append(item)
            
            # If parsing failed, use the raw response
            if not root_cause or root_cause == "Unknown root cause":
                # Try to extract first sentence as root cause
                sentences = response.split('.')
                if sentences:
                    root_cause = sentences[0].strip()
            
            # Ensure we have at least some content
            if not reasoning_steps:
                reasoning_steps = ["LLM analysis completed"]
            if not evidence:
                evidence = [f"Anomaly detected in {anomaly.get('metric_name', 'metric')}"]
            if not recommendations:
                recommendations = ["Investigate the root cause further", "Monitor related metrics"]
            
        except Exception as e:
            self.logger.error(
                "llm_response_parsing_failed",
                error=str(e),
                response_preview=response[:200] if response else None
            )
            # Return default analysis on parsing failure
            root_cause = "Analysis completed - see raw LLM response"
            reasoning_steps = [response[:500]]  # Include truncated response
        
        return RootCauseAnalysis(
            root_cause=root_cause,
            confidence=confidence,
            reasoning_steps=reasoning_steps,
            supporting_evidence=evidence,
            recommendations=recommendations,
            similar_incident_ids=[
                inc.get("incident_id", "")
                for inc in similar_incidents
                if isinstance(inc, dict)
            ],
            metadata={
                "anomaly_type": anomaly.get("anomaly_type"),
                "metric_name": anomaly.get("metric_name"),
                "severity": anomaly.get("severity"),
                "reasoning_method": "huggingface_llm",
                "model": self.model,
                "raw_llm_response": response[:1000]  # Store for debugging
            }
        )
