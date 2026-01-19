import logging
import os
import re
import sys
from datetime import datetime
from logging import Filter, LogRecord

from dotenv import load_dotenv
from uvicorn.logging import DefaultFormatter

from .context import cur_run_id
from .misc import env_to_bool

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", logging.INFO)


class HealthCheckFilter(Filter):
    def filter(self, record: LogRecord):
        return not re.search(r"(\s+)/health(\s+)", record.getMessage())


class RunIDFilter(Filter):
    def filter(self, record: LogRecord):
        record.run_id = cur_run_id.get()
        return True


def configure_loggers():
    # run_id corresponds to an instance of a geneartor created by
    # user's generation request
    run_id_filter = RunIDFilter()

    # Making the uvicorn logger delegate logging to the root logger
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.handlers = []
    uvicorn_logger.propagate = True

    # Setting up log levels
    for name in ["mindmap", "uvicorn"]:
        logging.getLogger(name).setLevel(LOG_LEVEL)

    # Configuring the root logger
    root = logging.getLogger()

    to_remove = []
    for handler in root.handlers:
        if (
            isinstance(handler, logging.StreamHandler)
            and handler.stream == sys.stderr
        ):
            to_remove.append(handler)

    for handler in to_remove:
        logging.getLogger().removeHandler(handler)

    # If stderr handler is already set, then no need to add another one
    formatter = DefaultFormatter(
        fmt=(
            "%(levelprefix)s | %(asctime)s | [%(run_id)s] | %(name)s "
            "| %(process)d | %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
        use_colors=True,
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    handler.addFilter(run_id_filter)

    root.addHandler(handler)

    should_log_to_file = env_to_bool("LOG_TO_FILE")
    if should_log_to_file:
        # Determine path
        log_dir = os.path.join(os.getcwd(), os.getenv("LOGS_DIR", "logs"))
        os.makedirs(log_dir, exist_ok=True)

        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filepath = os.path.join(log_dir, f"{timestamp}.log")

        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - [%(run_id)s] - %(levelname)s - %(message)s"
        )

        file_handler = logging.FileHandler(log_filepath)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        file_handler.addFilter(run_id_filter)

        root.addHandler(file_handler)


logger = logging.getLogger("mindmap")
