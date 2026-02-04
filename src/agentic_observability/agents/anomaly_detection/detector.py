"""
Anomaly Detection Agent

Statistical and ML-based anomaly detection for time-series data.

WHY THIS APPROACH:
- Z-score detection catches sudden spikes/drops (fast, interpretable)
- IQR (Interquartile Range) is robust to outliers
- Moving average detects drift and trend changes
- Combination of methods reduces false positives

PRODUCTION PATTERNS:
- Async implementation for high-throughput
- Configurable sensitivity thresholds
- Multiple detection algorithms for robustness
"""

from typing import Any, Optional
from dataclasses import dataclass, field
import asyncio
from enum import Enum

import structlog
from scipy import stats
import numpy as np

from ..base import BaseAgent

logger = structlog.get_logger()


class AnomalyType(Enum):
    """Types of anomalies detected"""
    SPIKE = "spike"           # Sudden increase
    DROP = "drop"             # Sudden decrease
    DRIFT = "drift"           # Gradual trend change
    OUTLIER = "outlier"       # Statistical outlier
    NORMAL = "normal"         # No anomaly detected


@dataclass
class AnomalyResult:
    """
    Result from anomaly detection.
    
    WHY: Structured output enables downstream agents to reason about
    the type and severity of anomalies.
    """
    is_anomaly: bool
    anomaly_type: AnomalyType
    confidence: float  # 0.0 to 1.0
    severity: float    # 0.0 to 1.0
    methods_triggered: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    explanation: str = ""


class AnomalyDetectionAgent(BaseAgent):
    """
    Detects anomalies using statistical methods.
    
    Implements multiple detection algorithms:
    - Z-score (standard deviations from mean)
    - IQR (Interquartile Range)
    - Moving average deviation
    
    USAGE:
        agent = AnomalyDetectionAgent(
            name="anomaly-detector",
            z_score_threshold=3.0,
            iqr_multiplier=1.5
        )
        await agent.initialize()
        result = await agent.execute({
            "values": [1.0, 1.1, 1.2, 10.0, 1.1],
            "timestamps": [...]
        })
    """
    
    def __init__(
        self,
        name: str = "anomaly-detection-agent",
        z_score_threshold: float = 3.0,
        robust_z_score_threshold: float = 3.5,
        iqr_multiplier: float = 1.5,
        window_size: int = 50,
        **kwargs
    ):
        """
        Initialize anomaly detection agent.
        
        Args:
            name: Agent identifier
            z_score_threshold: Number of std devs for Z-score (typically 2-3)
            iqr_multiplier: Multiplier for IQR method (typically 1.5)
            window_size: Size of moving window for trend detection
            **kwargs: Passed to BaseAgent
        """
        super().__init__(name=name, **kwargs)
        self.z_score_threshold = z_score_threshold
        self.robust_z_score_threshold = robust_z_score_threshold
        self.iqr_multiplier = iqr_multiplier
        self.window_size = window_size
        
        self.logger.info(
            "anomaly_detection_configured",
            z_threshold=z_score_threshold,
            iqr_multiplier=iqr_multiplier,
            window_size=window_size
        )
    
    async def _execute_impl(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Execute anomaly detection on input data.
        
        Args:
            input_data: Must contain 'values' (list of floats)
                       Optional: 'timestamps' (list of timestamps)
        
        Returns:
            Dict with anomaly detection results
        """
        # Accept both legacy key ("values") and API/coordinator key ("metrics")
        raw_values = input_data.get("values")
        if raw_values is None:
            raw_values = input_data.get("metrics")

        values = raw_values or []
        metric_name = str(input_data.get("metric_name", "unknown"))
        service_name = str(input_data.get("service_name", "unknown"))
        
        if not values or len(values) < 2:
            return {
                "is_anomaly": False,
                "anomaly_type": AnomalyType.NORMAL.value,
                "confidence": 0.0,
                "severity": 0.0,
                "anomaly_score": 0.0,
                "metric_name": metric_name,
                "service_name": service_name,
                "anomaly_description": "Insufficient data points",
                "reason": "Insufficient data points"
            }
        
        # Convert to numpy array for efficient computation
        data = np.array(values, dtype=float)
        
        # Run detection methods
        z_score_result = await self._detect_zscore(data)
        robust_z_score_result = await self._detect_robust_zscore(data)
        iqr_result = await self._detect_iqr(data)
        moving_avg_result = await self._detect_moving_average(data)
        
        # Combine results
        result = self._combine_results(
            z_score_result,
            robust_z_score_result,
            iqr_result,
            moving_avg_result,
            data
        )
        
        self.logger.info(
            "anomaly_detection_completed",
            is_anomaly=result.is_anomaly,
            type=result.anomaly_type.value,
            confidence=result.confidence,
            methods=result.methods_triggered
        )
        
        anomaly_score = max(result.confidence, result.severity)
        anomaly_description = (
            f"{service_name}:{metric_name} {result.anomaly_type.value} - {result.explanation}"
        )

        return {
            "is_anomaly": result.is_anomaly,
            "anomaly_type": result.anomaly_type.value,
            "confidence": result.confidence,
            "severity": result.severity,
            # Coordinator expects these fields for downstream stages
            "anomaly_score": anomaly_score,
            "anomaly_description": anomaly_description,
            "metric_name": metric_name,
            "service_name": service_name,
            "methods_triggered": result.methods_triggered,
            "metrics": result.metrics,
            "explanation": result.explanation,
            "data_points_analyzed": len(data)
        }
    
    async def _detect_zscore(self, data: np.ndarray) -> Optional[AnomalyResult]:
        """
        Z-score detection: measures how many standard deviations
        a point is from the mean.
        
        WHY: Fast, interpretable, good for detecting sudden spikes/drops
        """
        if len(data) < 3:
            return None
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return None
        
        z_scores = np.abs((data - mean) / std)
        max_z_score = np.max(z_scores)
        max_idx = np.argmax(z_scores)
        
        if max_z_score > self.z_score_threshold:
            # Determine if spike or drop
            anomaly_value = data[max_idx]
            anomaly_type = AnomalyType.SPIKE if anomaly_value > mean else AnomalyType.DROP
            
            return AnomalyResult(
                is_anomaly=True,
                anomaly_type=anomaly_type,
                confidence=min(max_z_score / (self.z_score_threshold * 2), 1.0),
                severity=min(max_z_score / 5.0, 1.0),
                methods_triggered=["z-score"],
                metrics={"z_score": float(max_z_score), "mean": float(mean), "std": float(std)},
                explanation=f"Z-score of {max_z_score:.2f} exceeds threshold {self.z_score_threshold}"
            )
        
        return None

    async def _detect_robust_zscore(self, data: np.ndarray) -> Optional[AnomalyResult]:
        """
        Robust Z-score detection using Median Absolute Deviation (MAD).

        WHY: Standard Z-score can miss anomalies in small samples or when a few
        extreme values inflate the mean/std (classic masking effect). MAD is
        robust and reliably flags spikes like [10, 11, 10, 500, 600].
        """
        if len(data) < 4:
            return None

        median = float(np.median(data))
        abs_deviation = np.abs(data - median)
        mad = float(np.median(abs_deviation))

        if mad == 0.0:
            return None

        # Consistent estimator for normal distributions.
        robust_z = 0.6745 * (data - median) / mad
        abs_robust_z = np.abs(robust_z)
        max_score = float(np.max(abs_robust_z))
        max_idx = int(np.argmax(abs_robust_z))

        if max_score > self.robust_z_score_threshold:
            anomaly_value = float(data[max_idx])
            anomaly_type = AnomalyType.SPIKE if anomaly_value > median else AnomalyType.DROP

            return AnomalyResult(
                is_anomaly=True,
                anomaly_type=anomaly_type,
                confidence=min(max_score / (self.robust_z_score_threshold * 2), 1.0),
                severity=min(max_score / 10.0, 1.0),
                methods_triggered=["robust-z-score"],
                metrics={
                    "robust_z_score": max_score,
                    "median": median,
                    "mad": mad,
                },
                explanation=(
                    f"Robust Z-score of {max_score:.2f} exceeds threshold "
                    f"{self.robust_z_score_threshold}"
                ),
            )

        return None
    
    async def _detect_iqr(self, data: np.ndarray) -> Optional[AnomalyResult]:
        """
        IQR (Interquartile Range) detection: robust to outliers.
        
        WHY: Less sensitive to extreme values than Z-score,
        better for skewed distributions
        """
        if len(data) < 4:
            return None
        
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        if iqr == 0:
            return None
        
        lower_bound = q1 - (self.iqr_multiplier * iqr)
        upper_bound = q3 + (self.iqr_multiplier * iqr)
        
        outliers = (data < lower_bound) | (data > upper_bound)
        
        if np.any(outliers):
            outlier_indices = np.where(outliers)[0]
            max_outlier_idx = outlier_indices[0]
            outlier_value = data[max_outlier_idx]
            
            # Calculate severity based on distance from bounds
            if outlier_value < lower_bound:
                distance = lower_bound - outlier_value
                anomaly_type = AnomalyType.DROP
            else:
                distance = outlier_value - upper_bound
                anomaly_type = AnomalyType.SPIKE
            
            severity = min(distance / (iqr * 2), 1.0)
            
            return AnomalyResult(
                is_anomaly=True,
                anomaly_type=AnomalyType.OUTLIER,
                confidence=0.8,
                severity=severity,
                methods_triggered=["iqr"],
                metrics={
                    "q1": float(q1),
                    "q3": float(q3),
                    "iqr": float(iqr),
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound)
                },
                explanation=f"Value {outlier_value:.2f} outside IQR bounds [{lower_bound:.2f}, {upper_bound:.2f}]"
            )
        
        return None
    
    async def _detect_moving_average(self, data: np.ndarray) -> Optional[AnomalyResult]:
        """
        Moving average deviation detection: identifies drift and trend changes.
        
        WHY: Catches gradual changes that Z-score and IQR might miss
        """
        if len(data) < self.window_size:
            return None
        
        # Calculate moving average
        moving_avg = np.convolve(data, np.ones(self.window_size) / self.window_size, mode='valid')
        
        # Calculate deviation from moving average
        recent_data = data[-len(moving_avg):]
        deviations = np.abs(recent_data - moving_avg)
        max_deviation = np.max(deviations)
        
        # Threshold: 2x the std of deviations
        threshold = 2 * np.std(deviations)
        
        if max_deviation > threshold and threshold > 0:
            return AnomalyResult(
                is_anomaly=True,
                anomaly_type=AnomalyType.DRIFT,
                confidence=0.7,
                severity=min(max_deviation / (threshold * 2), 1.0),
                methods_triggered=["moving-average"],
                metrics={
                    "max_deviation": float(max_deviation),
                    "threshold": float(threshold)
                },
                explanation=f"Deviation {max_deviation:.2f} from moving average exceeds threshold {threshold:.2f}"
            )
        
        return None
    
    def _combine_results(
        self,
        z_score_result: Optional[AnomalyResult],
        robust_z_score_result: Optional[AnomalyResult],
        iqr_result: Optional[AnomalyResult],
        moving_avg_result: Optional[AnomalyResult],
        data: np.ndarray
    ) -> AnomalyResult:
        """
        Combine results from multiple detection methods.
        
        WHY: Multiple methods reduce false positives and increase confidence
        """
        results = [
            r
            for r in [z_score_result, robust_z_score_result, iqr_result, moving_avg_result]
            if r is not None
        ]
        
        if not results:
            return AnomalyResult(
                is_anomaly=False,
                anomaly_type=AnomalyType.NORMAL,
                confidence=0.0,
                severity=0.0,
                explanation="No anomalies detected by any method"
            )
        
        # Aggregate results
        all_methods = []
        all_metrics = {}
        max_confidence = 0.0
        max_severity = 0.0
        primary_type = AnomalyType.NORMAL
        
        for result in results:
            all_methods.extend(result.methods_triggered)
            all_metrics.update(result.metrics)
            if result.confidence > max_confidence:
                max_confidence = result.confidence
                primary_type = result.anomaly_type
            max_severity = max(max_severity, result.severity)
        
        # Multiple methods agreeing increases confidence
        confidence_boost = len(results) * 0.1
        final_confidence = min(max_confidence + confidence_boost, 1.0)
        
        explanations = [r.explanation for r in results]
        combined_explanation = "; ".join(explanations)
        
        return AnomalyResult(
            is_anomaly=True,
            anomaly_type=primary_type,
            confidence=final_confidence,
            severity=max_severity,
            methods_triggered=all_methods,
            metrics=all_metrics,
            explanation=f"{len(results)} method(s) detected anomaly: {combined_explanation}"
        )
