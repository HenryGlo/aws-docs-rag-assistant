"""Structured logging configuration using structlog.

Why structured logging:
- Logs are key-value pairs (machine-parseable)
- Easy to filter and aggregate in production
- Better than f-strings for observability tools (Datadog, CloudWatch)
"""
import logging
import sys

import structlog

from src.config import settings


def setup_logging() -> structlog.BoundLogger:
  """Configure structured logging for the application."""
  logging.basicConfig(
      format="%(message)s",
      stream=sys.stdout,
      level=settings.log_level,
  )

  structlog.configure(
      processors=[
          structlog.contextvars.merge_contextvars,
          structlog.processors.add_log_level,
          structlog.processors.TimeStamper(fmt="iso"),
          structlog.dev.ConsoleRenderer(),
      ],
      wrapper_class=structlog.make_filtering_bound_logger(
          getattr(logging, settings.log_level.upper())
      ),
  )

  return structlog.get_logger()


logger = setup_logging()
