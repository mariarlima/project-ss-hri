# log_config.py
import sys
import logging

# Create a common logger
logger = logging.getLogger(__name__)

# Ensure we don't add handlers multiple times (if the module is imported more than once)
if not logger.hasHandlers():
    logger.setLevel(logging.DEBUG)

    # Create a formatter for both handlers
    formatter = logging.Formatter('%(asctime)s %(levelname)s [%(module)s:%(lineno)d]: %(message)s')

    # File handler: logs messages to a file
    file_handler = logging.FileHandler("robot.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Console handler: logs messages to stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# Log a startup message
logger.info("Starting...")