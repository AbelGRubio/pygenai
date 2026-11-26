"""Tests for the local filesystem Unit of Work.

====================================================================================================
Name:         libs/rag/tests/files/test_local.py
Description:  Covers local file operations and exception paths.
Project:      Pygenai
Date:         2026-11-25
Status:       Development

Copyright ©2026 All rights reserved.
====================================================================================================
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pygenai.rag.files.local import LocalFileSystemUnitOfWork


class TestLocalFileSystemUnitOfWork:
    """Tests for local filesystem operations."""

    def test_basic_operations(self, tmp_path: Path) -> None:
        """Create, read, upload, move, delete, and change routes."""
        base = tmp_path / "base"
        uow = LocalFileSystemUnitOfWork(str(base))

        assert uow.create_text_file("hello", "a.txt")
        assert (base / "a.txt").exists()
        assert uow.get_file("a.txt") == b"hello"
        assert "a.txt" in uow.list_folder("")

        src = tmp_path / "src.bin"
        src.write_bytes(b"binary")
        assert uow.upload_file(str(src), name="copied.bin")
        assert uow.get_file("copied.bin") == b"binary"

        assert uow.move_file("a.txt", "b.txt")
        assert not (base / "a.txt").exists()
        assert uow.get_file("b.txt") == b"hello"

        assert uow.delete_file("b.txt")
        assert not (base / "b.txt").exists()

        new_route = tmp_path / "other"
        assert Path(uow.move_route(str(new_route))).exists()

    def test_missing_paths_and_delete(self, tmp_path: Path) -> None:
        """Return false or empty results for missing files and folders."""
        uow = LocalFileSystemUnitOfWork(str(tmp_path / "base"))

        assert uow.list_folder("missing") == []
        assert uow.delete_file("missing.txt") is False
        assert uow.upload_file(str(tmp_path / "missing.bin")) is False

        with pytest.raises(FileNotFoundError):
            uow.get_file("missing.txt")

    def test_move_with_new_route_and_name_defaults(self, tmp_path: Path) -> None:
        """Move into a relative destination route and use the source basename."""
        uow = LocalFileSystemUnitOfWork(str(tmp_path / "base"))
        source = tmp_path / "source.txt"
        source.write_text("content", encoding="utf-8")

        assert uow.upload_file(str(source)) is True
        assert uow.move_file("source.txt", "moved.txt", "nested") is True
        assert uow.get_file("nested/moved.txt") == b"content"

    def test_create_move_upload_and_delete_error_paths(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Convert filesystem exceptions into false return values."""
        uow = LocalFileSystemUnitOfWork(str(tmp_path / "base"))

        def raise_os_error(*args: object, **kwargs: object) -> None:
            raise OSError("filesystem failure")

        monkeypatch.setattr(Path, "resolve", raise_os_error)

        assert uow.create_text_file("text", "a.txt") is False
        assert uow.delete_file("a.txt") is False
        assert uow.move_file("a.txt", "b.txt") is False
        assert uow.upload_file("source.txt") is False
