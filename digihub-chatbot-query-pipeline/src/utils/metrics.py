"""
Metrics and Logging Utilities for the Query Pipeline.

Provides centralized utilities for:
- Retrieval quality metrics logging
- Pipeline latency tracking
- Performance monitoring
"""

import time
from typing import List, Dict, Any, Optional
from functools import wraps
from src.utils.logger import logger


class RetrievalMetrics:
    """Utility class for logging retrieval quality metrics."""

    @staticmethod
    def log_chunk_scores(chunks: List[Dict], query: str = "", stage: str = "retrieval") -> Dict[str, Any]:
        """
        Log comprehensive metrics for all retrieved chunks.

        Args:
            chunks: List of retrieved chunk dictionaries
            query: The user query (for context in logs)
            stage: Pipeline stage identifier

        Returns:
            Dict containing computed metrics
        """
        if not chunks:
            logger.info(f"[Metrics:{stage}] No chunks retrieved")
            return {"chunk_count": 0}

        # Extract scores
        hybrid_scores = [c.get('hybrid_score', 0) for c in chunks]
        question_sims = [c.get('question_similarity', 0) for c in chunks]
        content_sims = [c.get('content_similarity', 0) for c in chunks]
        legacy_count = sum(1 for c in chunks if c.get('is_legacy_chunk', False))

        # Compute statistics
        metrics = {
            "chunk_count": len(chunks),
            "legacy_chunk_count": legacy_count,
            "hybrid_score": {
                "max": max(hybrid_scores),
                "min": min(hybrid_scores),
                "avg": sum(hybrid_scores) / len(hybrid_scores),
                "scores": [f"{s:.4f}" for s in hybrid_scores]
            },
            "question_similarity": {
                "max": max(question_sims) if question_sims else 0,
                "min": min(question_sims) if question_sims else 0,
                "avg": sum(question_sims) / len(question_sims) if question_sims else 0
            },
            "content_similarity": {
                "max": max(content_sims) if content_sims else 0,
                "min": min(content_sims) if content_sims else 0,
                "avg": sum(content_sims) / len(content_sims) if content_sims else 0
            }
        }

        # Log summary
        logger.info(
            f"[Metrics:{stage}] Retrieved {len(chunks)} chunks | "
            f"Hybrid: max={metrics['hybrid_score']['max']:.4f}, avg={metrics['hybrid_score']['avg']:.4f} | "
            f"Question sim: avg={metrics['question_similarity']['avg']:.4f} | "
            f"Legacy chunks: {legacy_count}"
        )

        # Log individual chunk scores for detailed analysis
        logger.debug(f"[Metrics:{stage}] All hybrid scores: {metrics['hybrid_score']['scores']}")

        return metrics

    @staticmethod
    def log_filtering_results(
        original_count: int,
        filtered_count: int,
        filter_type: str,
        threshold: Optional[float] = None
    ):
        """
        Log results of a filtering operation.

        Args:
            original_count: Number of chunks before filtering
            filtered_count: Number of chunks after filtering
            filter_type: Type of filter applied (e.g., 'relevance', 'threshold', 'dedup')
            threshold: Optional threshold value used
        """
        removed = original_count - filtered_count
        retention_rate = (filtered_count / original_count * 100) if original_count > 0 else 0

        threshold_info = f" (threshold={threshold})" if threshold is not None else ""
        logger.info(
            f"[Metrics:filter:{filter_type}] {original_count} -> {filtered_count} chunks "
            f"({removed} removed, {retention_rate:.1f}% retained){threshold_info}"
        )


class LatencyTracker:
    """
    Context manager and utilities for tracking pipeline latency.

    Usage:
        with LatencyTracker("stage_name") as tracker:
            # do work
        # automatically logs latency on exit

        # Or manual tracking:
        tracker = LatencyTracker.start("stage_name")
        # do work
        tracker.stop()
    """

    def __init__(self, stage_name: str, log_on_exit: bool = True):
        """
        Initialize latency tracker.

        Args:
            stage_name: Name of the pipeline stage being tracked
            log_on_exit: Whether to automatically log on context exit
        """
        self.stage_name = stage_name
        self.log_on_exit = log_on_exit
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.latency_ms: Optional[float] = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.latency_ms = (self.end_time - self.start_time) * 1000

        if self.log_on_exit:
            self.log()

        return False  # Don't suppress exceptions

    @classmethod
    def start(cls, stage_name: str) -> 'LatencyTracker':
        """Start a new latency tracker."""
        tracker = cls(stage_name, log_on_exit=False)
        tracker.start_time = time.time()
        return tracker

    def stop(self) -> float:
        """Stop the tracker and return latency in milliseconds."""
        self.end_time = time.time()
        self.latency_ms = (self.end_time - self.start_time) * 1000
        self.log()
        return self.latency_ms

    def log(self):
        """Log the latency measurement."""
        if self.latency_ms is not None:
            logger.info(f"[Latency] {self.stage_name}: {self.latency_ms:.2f}ms")


def track_latency(stage_name: str):
    """
    Decorator for tracking function latency.

    Usage:
        @track_latency("my_function")
        def my_function():
            pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with LatencyTracker(stage_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


class PipelineMetrics:
    """
    Aggregate metrics for the entire query pipeline.

    Collects metrics from multiple stages and provides summary logging.
    """

    def __init__(self, trace_id: str = ""):
        self.trace_id = trace_id
        self.stage_latencies: Dict[str, float] = {}
        self.retrieval_metrics: Dict[str, Any] = {}
        self.start_time = time.time()

    def record_latency(self, stage: str, latency_ms: float):
        """Record latency for a pipeline stage."""
        self.stage_latencies[stage] = latency_ms

    def record_retrieval_metrics(self, metrics: Dict[str, Any]):
        """Record retrieval quality metrics."""
        self.retrieval_metrics = metrics

    def log_summary(self):
        """Log a summary of all collected metrics."""
        total_time = (time.time() - self.start_time) * 1000

        latency_parts = [f"{stage}={ms:.0f}ms" for stage, ms in self.stage_latencies.items()]
        latency_str = ", ".join(latency_parts) if latency_parts else "no stages recorded"

        chunk_count = self.retrieval_metrics.get("chunk_count", "N/A")
        avg_score = self.retrieval_metrics.get("hybrid_score", {}).get("avg", "N/A")
        if isinstance(avg_score, float):
            avg_score = f"{avg_score:.4f}"

        logger.info(
            f"[Pipeline Summary] Total: {total_time:.0f}ms | "
            f"Stages: {latency_str} | "
            f"Chunks: {chunk_count} | "
            f"Avg hybrid score: {avg_score}"
        )
