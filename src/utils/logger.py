"""Lightweight logging setup used across the pipeline."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured = False


def setup_logging(level: int = logging.INFO, log_file: str | Path | None = None) -> None:
    global _configured
    if _configured:
        return
    root = logging.getLogger()
    if not root.handlers:
        # basicConfig is a no-op once a host app (uvicorn/streamlit/pytest)
        # has configured the root logger - only call it in that case
        handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
        if log_file is not None:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_file))
        logging.basicConfig(level=level, format=_FORMAT, handlers=handlers)
    elif log_file is not None:
        # host configured logging already: still attach our file handler
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(fh)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
