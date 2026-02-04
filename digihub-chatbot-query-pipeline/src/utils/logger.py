import logging
from logging.handlers import RotatingFileHandler
from contextvars import ContextVar
from uuid import uuid4

# Adding context variable to get trace id from request - using default uuid for first few messages after server startup
log_context: ContextVar[str] = ContextVar("trace_id", default=f"aaaaaaaa-aaaa-4aaa-aaaa-aaaaaaaaaaaa")

# Log file configuration
LOG_FILE = "QueryPipeline.log"
LOG_FORMAT = "%(asctime)s - %(trace_id)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = logging.INFO

# Adding context filter so that it can retrieve dynamically trace id from context variable
class TraceIDContextFilter(logging.Filter):
    def filter(self, record):
        record.trace_id = log_context.get()
        return True

def setup_logger(name: str, log_file: str = LOG_FILE, level: int = LOG_LEVEL) -> logging.Logger:
    """
    Sets up a logger with both file and console handlers.

    Args:
        name (str): The name of the logger.
        log_file (str): The file where logs will be written.
        level (int): The logging level (e.g., logging.INFO, logging.DEBUG, logging.ERROR).

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate log entries
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler with log rotation
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=5 * 1024 * 1024, # The log file will be rotated when it reaches 5 MB.
        backupCount=3 # Up to three backup files will be kept. When a new log file is created, the oldest backup will be removed if there are already three backups.
        )
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Add context filter for all handlers
    for handler in logger.handlers:
        handler.addFilter(TraceIDContextFilter())

    return logger

# Initialize the logger for the application
logger = setup_logger("DigiHubChatbot:QueryPipeline")
