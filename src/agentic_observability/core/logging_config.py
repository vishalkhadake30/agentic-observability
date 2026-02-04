"""
Logging Configuration

Configures structured logging with file rotation and timestamp-based log files.

WHY FILE LOGGING:
- Production systems need persistent logs for audit trails
- Rotating logs prevent disk space exhaustion
- Timestamped files make log analysis easier
- Enables correlation of issues across service restarts
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Any
import structlog
from structlog.types import FilteringBoundLogger


def configure_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    service_name: str = "agentic-observability",
    enable_file_logging: bool = True
) -> FilteringBoundLogger:
    """
    Configure structured logging with console and file output.
    
    WHY THIS APPROACH:
    - structlog provides structured logging (JSON-friendly, machine-parseable)
    - File rotation prevents logs from consuming all disk space
    - Timestamped files make it easy to find logs from specific time periods
    - Both console and file output for development and production needs
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files
        service_name: Service name for log identification
        enable_file_logging: Enable file-based logging (True for production)
        
    Returns:
        Configured structlog logger
    """
    # Create logs directory if it doesn't exist
    if enable_file_logging:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamped log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"{service_name}_{timestamp}.log"
        
        # Also create a symlink to latest log file for easy access
        latest_log = log_path / f"{service_name}_latest.log"
        
        # Configure file handler with rotation
        # WHY ROTATION: Prevents logs from growing unbounded
        from logging.handlers import RotatingFileHandler
        
        file_handler = RotatingFileHandler(
            filename=str(log_file),
            maxBytes=100 * 1024 * 1024,  # 100MB per file
            backupCount=10,  # Keep 10 backup files
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Create symlink to latest log (remove old one if exists)
        if latest_log.exists() or latest_log.is_symlink():
            latest_log.unlink()
        latest_log.symlink_to(log_file.name)
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper()),
        handlers=[
            logging.StreamHandler(sys.stdout),
            file_handler if enable_file_logging else logging.NullHandler()
        ]
    )
    
    # Configure structlog processors
    # WHY THESE PROCESSORS:
    # - add_log_level: Makes filtering by level easier
    # - TimeStamper: Critical for production debugging (when did this happen?)
    # - StackInfoRenderer: Shows stack traces for errors
    # - format_exc_info: Formats exceptions nicely
    # - JSONRenderer/ConsoleRenderer: Machine-readable vs human-readable output
    
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),  # ISO 8601 timestamps
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    # Use JSON renderer for file logs (machine-parseable)
    # Use console renderer for stdout (human-readable)
    if enable_file_logging:
        # For production: JSON format in files for log aggregation tools
        processors.append(structlog.processors.JSONRenderer())
    else:
        # For development: colored console output
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    logger = structlog.get_logger(service_name)
    
    if enable_file_logging:
        logger.info(
            "logging_configured",
            log_level=log_level,
            log_file=str(log_file),
            latest_log=str(latest_log),
            max_file_size_mb=100,
            backup_count=10
        )
    
    return logger


def get_logger(name: str = __name__) -> FilteringBoundLogger:
    """
    Get a logger instance.
    
    WHY SEPARATE FUNCTION:
    - Provides consistent logger interface across the application
    - Allows binding context (e.g., request IDs, user IDs)
    
    Args:
        name: Logger name (typically __name__ from calling module)
        
    Returns:
        Configured structlog logger with context binding
    """
    return structlog.get_logger(name)


def add_context(**kwargs: Any) -> None:
    """
    Add context to all subsequent log messages in this execution context.
    
    WHY CONTEXT:
    - Enables tracing requests across the system (correlation IDs)
    - Makes debugging easier (which user? which request?)
    - Essential for distributed systems
    
    Example:
        add_context(correlation_id="abc-123", user_id="user-456")
        logger.info("processing_request")  # Will include correlation_id and user_id
    
    Args:
        **kwargs: Key-value pairs to add to logging context
    """
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """
    Clear all context variables.
    
    WHY:
    - Prevents context bleeding between requests (especially in async code)
    - Clean slate for each new request/task
    """
    structlog.contextvars.clear_contextvars()
