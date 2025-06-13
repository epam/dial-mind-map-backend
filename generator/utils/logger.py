import logging
import os
from datetime import datetime

from .constants import LOGS_DIR
from .context import cur_run_id


class RunIDFilter(logging.Filter):
    """Adds run_id to log records using the context variable."""

    def filter(self, record):
        record.run_id = cur_run_id.get()
        return True


def setup_logging():
    """Configure logging to both console and timestamped file."""
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.getcwd(), LOGS_DIR)
    os.makedirs(log_dir, exist_ok=True)

    # Generate log file name with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"{timestamp}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(run_id)s - %(levelname)s - %(message)s",
    )

    # Create file handler and set level to INFO
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Create console handler and set level to INFO
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    run_id_filter = RunIDFilter()
    file_handler.addFilter(run_id_filter)
    console_handler.addFilter(run_id_filter)

    # Get the root logger
    logger = logging.getLogger()

    # Clear existing handlers to avoid duplication
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    # Add handlers to the root logger
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
