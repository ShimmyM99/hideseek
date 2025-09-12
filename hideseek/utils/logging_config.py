import logging
import sys
from pathlib import Path
from typing import Optional
import colorama
from colorama import Fore, Back, Style


class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to log levels"""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Back.YELLOW,
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, '')
        reset = Style.RESET_ALL
        
        # Add color to the level name
        record.levelname = f"{log_color}{record.levelname}{reset}"
        
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration for HideSeek
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file name
        log_dir: Optional log directory path
        
    Returns:
        Configured logger instance
    """
    # Initialize colorama for Windows compatibility
    colorama.init(autoreset=True)
    
    # Create logger
    logger = logging.getLogger('hideseek')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    # Create colored formatter for console
    console_format = '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
    colored_formatter = ColoredFormatter(console_format, datefmt='%H:%M:%S')
    console_handler.setFormatter(colored_formatter)
    
    logger.addHandler(console_handler)
    
    # Create file handler if log file is specified
    if log_file:
        if log_dir:
            log_path = Path(log_dir) / log_file
            log_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            log_path = Path(log_file)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        
        # Create plain formatter for file (no colors)
        file_format = '%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s'
        file_formatter = logging.Formatter(file_format, datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name"""
    return logging.getLogger(f'hideseek.{name}')


# Create default logger for the application
setup_logging()