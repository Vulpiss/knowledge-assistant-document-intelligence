import sys
from loguru import logger

from app.core.config import config


def setup_logger() -> None:
    config.ensure_directories()

    logger.remove()

    if sys.stdout is not None:
        logger.add(
            sys.stdout,
            level=config.log_level,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:"
                "<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            ),
        )

    logger.add(
        config.log_dir / "app.log",
        level=config.log_level,
        rotation="1 MB",
        retention="7 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    )
