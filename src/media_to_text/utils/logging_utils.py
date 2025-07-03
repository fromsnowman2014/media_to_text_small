"""
Logging utilities for the media-to-text converter.
"""

import os
import logging
from typing import Optional, Literal

LogLevel = Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

class ColoredFormatter(logging.Formatter):
    """Custom formatter adding colors to log messages"""
    
    COLORS = {
        'DEBUG': '\033[94m',      # Blue
        'INFO': '\033[92m',       # Green
        'WARNING': '\033[93m',    # Yellow
        'ERROR': '\033[91m',      # Red
        'CRITICAL': '\033[91m\033[1m',  # Bold Red
        'RESET': '\033[0m',       # Reset
    }
    
    def format(self, record):
        log_message = super().format(record)
        if record.levelname in self.COLORS:
            return f"{self.COLORS[record.levelname]}{log_message}{self.COLORS['RESET']}"
        return log_message


def setup_logger(name: str, log_file: Optional[str] = None, 
                 level: LogLevel = 'INFO', 
                 console_output: bool = True) -> logging.Logger:
    """
    Set up a logger with optional file and console output.
    
    Args:
        name: Logger name
        log_file: Path to the log file. If None, no file logging.
        level: Logging level
        console_output: Whether to output logs to console
        
    Returns:
        Configured logger
    """
    # Convert string level to logging level
    numeric_level = getattr(logging, level)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Format for logs
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File handler
    if log_file:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler()
        colored_formatter = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(colored_formatter)
        logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str, debug_mode: bool = False) -> logging.Logger:
    """
    Get a logger with standard configuration.
    
    Args:
        name: Logger name
        debug_mode: If True, set log level to DEBUG
        
    Returns:
        Configured logger
    """
    level = 'DEBUG' if debug_mode else 'INFO'
    return setup_logger(name, level=level)
