"""
logging_setup.py
----------------
Production rotating-file + console logging for the HyperloopBert pipeline.

Usage:
    from common.logging_setup import setup_logging
    logger = setup_logging(__name__)
"""

import logging
import os
from logging.handlers import RotatingFileHandler

_LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

# Track loggers that have already been configured so we do not add duplicate handlers.
_configured_loggers: set = set()


def setup_logging(
    name: str,
    log_dir: str = "logs",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create and return a logger with a RotatingFileHandler and a StreamHandler.

    Parameters
    ----------
    name : str
        Logger name (pass __name__ from the calling module).
    log_dir : str
        Directory where log files are written. Created automatically if absent.
    level : int
        Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Notes
    -----
    - Log file path : <log_dir>/<name>.log
    - Rotating file : 10 MB per file, 5 backup copies kept.
    - Format        : %(asctime)s | %(name)s | %(levelname)s | %(message)s
    - Idempotent    : calling this function twice with the same name does not
                      add duplicate handlers.
    - No emoji anywhere in log output.
    """
    logger = logging.getLogger(name)

    if name in _configured_loggers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Create log directory if it does not exist.
    os.makedirs(log_dir, exist_ok=True)

    # Derive a safe filename from the logger name.
    safe_name = name.replace("/", ".").replace("\\", ".")
    log_file = os.path.join(log_dir, f"{safe_name}.log")

    # Rotating file handler: 10 MB per file, 5 backups.
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # Console (stream) handler.
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # Prevent log records from propagating to the root logger to avoid
    # duplicate output when the root logger also has handlers attached.
    logger.propagate = False

    _configured_loggers.add(name)
    return logger
