import logging
import sys
from pathlib import Path
from src.config.settings import LOG_LEVEL, LOG_FILE_PATH, LOG_CONSOLE_OUTPUT, LOG_FORMAT

def setup_logger(name: str = "POWERGRID") -> logging.Logger:
    """
    Sets up a standardized logger for the project.
    Outputs logs to both sys.stdout (console) and a rolling log file at logs/pipeline.log.
    """
    logger = logging.getLogger(name)
    # If logger is already configured, don't duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    formatter = logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    # Console Stream Handler
    if LOG_CONSOLE_OUTPUT:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File Handler
    log_file = Path(LOG_FILE_PATH)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
