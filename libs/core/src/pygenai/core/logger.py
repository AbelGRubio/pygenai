"""Custom logging utilities built on top of Python's logging module."""

import functools
import logging
import os
import sys
import time
from datetime import UTC, datetime
from logging import CRITICAL, DEBUG, ERROR, INFO, WARNING
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any, ClassVar

import orjson

try:
    from rich.console import Console
    from rich.default_styles import DEFAULT_STYLES
    from rich.logging import RichHandler
    from rich.table import Table
    from rich.theme import Theme

    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    Console = Any
    RichHandler = Any
    Table = Any
    Theme = Any


detail_level = logging.DEBUG + 5

logging.addLevelName(detail_level, "DETAIL")


def default_handler(obj: Any) -> str:
    """Default handler for logging."""
    return f"<<Non-serializable object {type(obj).__name__}>>"


class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for log records."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as JSON."""
        log_data = record.__dict__
        log_data["timestamp"] = int(datetime.fromtimestamp(record.created, tz=UTC).timestamp() * 1e6)

        return orjson.dumps(log_data, option=orjson.OPT_NON_STR_KEYS, default=default_handler).decode("utf-8")


class UvicornFilter(logging.Filter):
    """Filter out Uvicorn access logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter by name."""
        return not record.name.startswith("uvicorn.access")


class DictNormalizerFilter(logging.Filter):
    """Convierte diccionarios de logs (como los de structlog/LangGraph) en strings bonitos."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter by name."""
        if isinstance(record.msg, dict):
            event = record.msg.get("event", "")
            details = {k: v for k, v in record.msg.items() if k != "event"}
            record.msg = f"{event} | {details}"
        return True


class LoggerApi(logging.Logger):
    """Application logger with Rich, rotating files, and optional OTEL export."""

    _timers: ClassVar[dict[str, float]] = {}
    _timers_it: ClassVar[dict[str, float]] = {}
    _console: ClassVar[Console | None] = None

    def __init__(self, name: str | None = None, level: int = logging.NOTSET) -> None:
        """Initialize logger handlers and logging metadata."""
        if not name:
            name = "api"

        super().__init__(name, level)

        self._titles_level = {
            DEBUG: f"DEBUG {name.upper()}",
            INFO: f"INFO {name.upper()}",
            WARNING: f"WARNING {name.upper()}",
            ERROR: f"ERROR {name.upper()}",
            detail_level: f"DETAIL {name.upper()}",
            CRITICAL: f"CRITICAL {name.upper()}",
        }
        self.propagate = True
        self.start_global_logger()

    @property
    def console(self) -> Console | None:
        """Instance and get the console."""
        if LoggerApi._console is None:
            LoggerApi._console = self.start_console()
        return LoggerApi._console

    @staticmethod
    def start_console() -> Console | None:
        """Start console."""
        if not HAS_RICH:
            return None

        custom_theme = Theme({
            **DEFAULT_STYLES,
            "logging.level.detail": "magenta",
            "logging.level.debug": "cyan",
            "logging.level.info": "green",
            "logging.level.warning": "yellow",
            "logging.level.error": "bold red",
            "logging.level.critical": "bold white on red",
        })

        return Console(
            theme=custom_theme,
            soft_wrap=True,
            stderr=False,
            force_terminal=True,
            color_system="truecolor",
            width=200,
        )

    def start_global_logger(self) -> None:
        """Configure console and file logging handlers."""
        root = logging.getLogger()

        if len(root.handlers) > 0:
            return

        json_logs = os.getenv("JSON_LOGS", "false") == "true"

        if json_logs or not HAS_RICH:
            console_handler = logging.StreamHandler(sys.stderr)
        else:
            # Rich console handler.
            console_handler = RichHandler(
                console=self.console,
                omit_repeated_times=True,
                rich_tracebacks=True,
                show_time=True,
                show_level=True,
                show_path=False,
                markup=True,
            )

        console_handler.setLevel(logging.DEBUG)
        console_handler.addFilter(DictNormalizerFilter())
        console_handler.addFilter(UvicornFilter())

        if json_logs:
            formatter = JsonFormatter()
        else:
            base_format = "%(name)s\t%(threadName)s\t%(message)s"
            format_ = f"%(asctime)s\t{base_format}" if not HAS_RICH else base_format

            formatter = logging.Formatter(format_, datefmt="[%Y-%m-%d %H:%M:%S]")

        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)
        # Rotating file handler.
        Path(".logs").mkdir(exist_ok=True)
        file_handler = TimedRotatingFileHandler(".logs/app.log", when="midnight", interval=1, backupCount=7)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s\t%(levelname)s\t%(name)s\t%(threadName)s\t%(message)s",
                "[%Y-%m-%d %H:%M:%S]",
            )
        )
        root.addHandler(file_handler)
        root.setLevel(0)

    def _get_title(self, level: int) -> str:
        """Return the display title associated with a log level."""
        return self._titles_level.get(level, self.name.upper())

    def detail(self, msg: str, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        """Log a message at the custom DETAIL level."""
        if self.isEnabledFor(detail_level):
            super().log(detail_level, msg, *args, **kwargs)

    def start_time(self, name: str | None = None, level: int | None = DEBUG) -> None:
        """Start a timer only if a name is provided."""
        if not name:
            name = "no-timer-name"

        self._timers[name] = time.perf_counter()

        if level:
            self.log(level, f"[cyan]⏱️ Timer started: {name}")

    def time_it(self, name: str | None = None, msg: str | None = None, level: int | None = DEBUG) -> float:
        """Stop timer.

        If name is provided, uses stored start time.
        If not, measures elapsed from a fresh start (no storage).
        """
        now = time.perf_counter()

        if not name:
            name = "no-timer-name"

        if name and hasattr(self, "_timers") and name in self._timers:
            start = self._timers.pop(name)
            elapsed = now - start
        else:
            # fallback: no stored timer
            elapsed = 0.0

        self._timers_it[name] = elapsed

        final_msg = msg or (f"[cyan]⏱️ {name}: {elapsed:.4f}s" if name else "⏱️ Timer finished")

        if level:
            self.log(level, final_msg)

        return elapsed

    def dec_time_it(self, name: str | None = None, level: int = DEBUG) -> Any:
        """Return a decorator that measures function execution time."""

        def decorator(func: Any) -> Any:
            """Wrap a callable with timing instrumentation."""

            @functools.wraps(func)
            def wrapper(*args: tuple, **kwargs: dict) -> Any:
                """Execute the wrapped callable and record elapsed time."""
                timer_name = name or func.__name__

                self.start_time(timer_name)

                try:
                    return func(*args, **kwargs)
                finally:
                    self.time_it(timer_name, level=level)

            return wrapper

        return decorator

    def print_timers(self) -> None:
        """Print all recorded timers in a table."""
        if not hasattr(self, "_timers_it") or not self._timers_it:
            self.info("No timers recorded")
            return

        table = Table(title="⏱️ Timers")

        table.add_column("Name", justify="left")
        table.add_column("Time (s)", justify="right")

        total = 0.0

        for name, elapsed in self._timers_it.items():
            table.add_row(name, f"{elapsed:.6f}")
            total += elapsed

        table.add_row("[bold]TOTAL[/bold]", f"[bold]{total:.6f}[/bold]")

        if self.console:
            self.console.print(table)
        self.info("Timers: " + ", ".join(f"{k}={v:.4f}s" for k, v in self._timers_it.items()))


logging.setLoggerClass(LoggerApi)


def get_logger(name: str) -> LoggerApi:
    """Create and return a configured `LoggerApi` instance."""
    return logging.getLogger(name)


def propague_loggers(no_propagate_prefixes: list | None = None) -> None:
    """Configure logger logger."""
    if no_propagate_prefixes is None:
        no_propagate_prefixes = ["boto3", "urllib3", "botocore", "s3transfer", "bcdocs"]

    loggers = logging.Logger.manager.loggerDict.keys()

    for n in loggers:
        logger_ = logging.getLogger(n)

        if any(n.startswith(prefix) for prefix in no_propagate_prefixes):
            logger_.handlers = []
            logger_.propagate = False
        else:
            logger_.handlers = []
            logger_.propagate = True
            logger_.setLevel(logging.DEBUG)
