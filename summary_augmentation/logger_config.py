import logging
import sys
import os
from pathlib import Path


def setup_logger(name: str, 
                 log_level: str = "INFO", 
                 log_file: str = None,
                 console_level: str = None,
                 file_level: str = None,
                 console_output: bool = True):
    """
    Setup logger with consistent formatting across the project
    
    Args:
        name: Usually __name__ from the calling module
        log_level: DEBUG, INFO, WARNING, ERROR, CRITICAL (backward compatibility)
        log_file: Optional file path for logging to file
        console_level: Log level for console output (if None, uses log_level)
        file_level: Log level for file output (if None, uses log_level)  
        console_output: Whether to enable console output
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if logger already configured
    if logger.handlers:
        return logger
    
    # Determine effective log levels
    effective_console_level = (console_level or log_level).upper()
    effective_file_level = (file_level or log_level).upper()
    
    # Set logger to the most verbose level needed
    min_level = min(
        getattr(logging, effective_console_level),
        getattr(logging, effective_file_level) if log_file else getattr(logging, effective_console_level)
    )
    logger.setLevel(min_level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, effective_console_level))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        # Create directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, effective_file_level))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class LoggerFactory:
    """Centralized logger management for consistent logging across the project"""
    
    _loggers = {}
    _global_config = {
        'console_level': 'INFO',
        'file_level': 'DEBUG',
        'log_file': None,
        'console_output': True
    }
    
    @classmethod
    def get_logger(cls, name: str, **kwargs):
        """
        Get or create a standardized logger
        
        Args:
            name: Logger name (usually __name__)
            **kwargs: Arguments passed to setup_logger
        
        Returns:
            Configured logger instance
        """
        if name not in cls._loggers:
            # Merge global config with provided kwargs
            config = cls._global_config.copy()
            config.update(kwargs)
            cls._loggers[name] = setup_logger(name, **config)
        return cls._loggers[name]
    
    @classmethod
    def set_global_config(cls, **config):
        """
        Set global configuration for all future loggers
        
        Args:
            console_level: Log level for console output
            file_level: Log level for file output
            log_file: Global log file path
            console_output: Enable/disable console output
        """
        cls._global_config.update(config)
    
    @classmethod
    def enable_file_logging_for_all(cls, log_file: str, file_level: str = "DEBUG"):
        """
        Add file logging to all existing loggers
        
        Args:
            log_file: Path to log file
            file_level: Log level for file output
        """
        # Create directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s'
        )
        
        for logger_name, logger in cls._loggers.items():
            # Check if file handler already exists
            has_file_handler = any(isinstance(h, logging.FileHandler) for h in logger.handlers)
            
            if not has_file_handler:
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setLevel(getattr(logging, file_level.upper()))
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
                
                # Adjust logger level if needed
                current_level = logger.level
                file_level_num = getattr(logging, file_level.upper())
                if file_level_num < current_level:
                    logger.setLevel(file_level_num)
        
        # Update global config for future loggers
        cls._global_config['log_file'] = log_file
        cls._global_config['file_level'] = file_level
    
    @classmethod
    def set_console_level_for_all(cls, console_level: str):
        """
        Change console log level for all existing loggers
        
        Args:
            console_level: New console log level
        """
        level_num = getattr(logging, console_level.upper())
        
        for logger in cls._loggers.values():
            for handler in logger.handlers:
                if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                    handler.setLevel(level_num)
        
        cls._global_config['console_level'] = console_level
    
    @classmethod
    def get_logger_info(cls):
        """
        Get information about all managed loggers
        
        Returns:
            Dict with logger information
        """
        info = {
            'global_config': cls._global_config.copy(),
            'active_loggers': {},
            'total_loggers': len(cls._loggers)
        }
        
        for name, logger in cls._loggers.items():
            handlers_info = []
            for handler in logger.handlers:
                handler_info = {
                    'type': type(handler).__name__,
                    'level': logging.getLevelName(handler.level)
                }
                if isinstance(handler, logging.FileHandler):
                    handler_info['file'] = handler.baseFilename
                handlers_info.append(handler_info)
            
            info['active_loggers'][name] = {
                'level': logging.getLevelName(logger.level),
                'handlers': handlers_info
            }
        
        return info


def get_global_log_level():
    """Get log level from environment variable or default"""
    return os.getenv('HEBREW_SUMMARY_LOG_LEVEL', 'INFO')


def get_global_log_file():
    """Get log file from environment variable if set"""
    return os.getenv('HEBREW_SUMMARY_LOG_FILE', None)


# Convenience function for quick setup
def quick_setup(console_level: str = "INFO", file_level: str = "DEBUG", log_file: str = None):
    """
    Quick setup for global logging configuration
    
    Args:
        console_level: Log level for console output
        file_level: Log level for file output  
        log_file: Optional log file path
    """
    LoggerFactory.set_global_config(
        console_level=console_level,
        file_level=file_level,
        log_file=log_file
    )