import logging
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    """
    Creates and returns a logger object.

    Args:
        name: Name of the logger.

    Returns:
        Configured logger.
    """

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_dir / "app.log")

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger