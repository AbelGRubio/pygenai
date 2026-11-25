"""Tests for custom logging utilities."""

import logging

import orjson

from pygenai.core import logger as logger_module


def test_default_handler_and_json_formatter() -> None:
    record = logging.LogRecord("test", logging.INFO, __file__, 1, "message", (), None)

    assert logger_module.default_handler(object()).startswith("<<Non-serializable object")
    formatted = logger_module.JsonFormatter().format(record)
    payload = orjson.loads(formatted)

    assert payload["name"] == "test"
    assert "timestamp" in payload


def test_filters() -> None:
    uvicorn_record = logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1, "ok", (), None)
    dict_record = logging.LogRecord("test", logging.INFO, __file__, 1, {"event": "loaded", "count": 2}, (), None)

    assert logger_module.UvicornFilter().filter(uvicorn_record) is False
    assert logger_module.DictNormalizerFilter().filter(dict_record) is True
    assert dict_record.msg == "loaded | {'count': 2}"


def test_logger_titles_detail_and_timers(monkeypatch) -> None:
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])

    api_logger = logger_module.LoggerApi("test-api", logging.DEBUG)

    assert api_logger._get_title(logging.INFO) == "INFO TEST-API"
    assert api_logger._get_title(999) == "TEST-API"

    api_logger.start_time("test", level=None)
    elapsed = api_logger.time_it("test", msg="done", level=None)

    assert elapsed >= 0
    assert api_logger._timers_it["test"] == elapsed


def test_detail_method(monkeypatch) -> None:
    api_logger = logger_module.LoggerApi("detail-test")
    api_logger.setLevel(logger_module.detail_level)

    api_logger.detail("detail message")

    assert api_logger.isEnabledFor(logger_module.detail_level)


def test_decorator_timer() -> None:
    api_logger = logger_module.LoggerApi("decorator-test")

    @api_logger.dec_time_it("decorated", level=None)
    def add(a: int, b: int) -> int:
        return a + b

    assert add(1, 2) == 3
    assert "decorated" in api_logger._timers_it


def test_print_timers_empty(monkeypatch) -> None:
    api_logger = logger_module.LoggerApi("empty-timers")
    api_logger._timers_it = {}

    monkeypatch.setattr(api_logger, "info", lambda message: None)

    api_logger.print_timers()


def test_print_timers_with_values(monkeypatch) -> None:
    api_logger = logger_module.LoggerApi("timer-table")
    api_logger._timers_it = {"slow": 1.25}
    api_logger._console = None

    monkeypatch.setattr(logger_module.LoggerApi, "start_console", lambda self: None)
    monkeypatch.setattr(api_logger, "info", lambda message: None)

    api_logger.print_timers()


def test_start_console(monkeypatch) -> None:
    monkeypatch.setattr(logger_module, "HAS_RICH", True)

    console = logger_module.LoggerApi.start_console()

    assert console is not None


def test_start_global_logger_json(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("JSON_LOGS", "true")
    monkeypatch.chdir(tmp_path)
    root = logging.getLogger()
    root.handlers = []

    api_logger = logger_module.LoggerApi("json-test")

    assert root.handlers
    assert root.level == 0


def test_propague_loggers(monkeypatch) -> None:
    logger_a = logging.getLogger("boto3.test")
    logger_b = logging.getLogger("application.test")

    logger_module.propague_loggers()

    assert logger_a.propagate is False
    assert logger_a.handlers == []
    assert logger_b.propagate is True
    assert logger_b.level == logging.DEBUG


def test_get_logger() -> None:
    logger = logger_module.get_logger("test.logger")

    assert isinstance(logger, logging.Logger)
