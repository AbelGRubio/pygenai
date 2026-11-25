"""Tests for the abstract filesystem Unit of Work contract.

====================================================================================================
Name:         libs/rag/tests/files/test_base.py
Description:  Verifies initialization and abstract method contracts.
Project:      Pygenai
Date:         2026-11-25
Status:       Development

Copyright ©2026 All rights reserved.
====================================================================================================
"""

from __future__ import annotations

import pytest

from pygenai.rag.files._base import AbstractFileSystemUnitOfWork


class TestAbstractFileSystemUnitOfWork:
    """Tests for abstract filesystem behavior."""

    def test_cannot_instantiate_abstract_class(self) -> None:
        """Reject direct construction of the abstract contract."""
        with pytest.raises(TypeError):
            AbstractFileSystemUnitOfWork("root")  # pyrefly: ignore

    def test_all_unimplemented_methods_raise(self) -> None:
        """Each abstract method exists and raises until a subclass implements it."""
        class IncompleteUnitOfWork(AbstractFileSystemUnitOfWork):
            def create_text_file(self, text: str, file: str, encoding: str = "utf-8") -> bool:
                return False

            def delete_file(self, file: str) -> bool:
                return False

            def get_file(self, file: str) -> bytes:
                return b""

            def list_folder(self, prefix: str) -> list:
                return []

            def move_file(self, old_file: str, new_file: str, new_route: str | None = None) -> bool:
                return False

            def move_route(self, route: str) -> str:
                return route

            def upload_file(self, file_route: str, name: str | None = None) -> bool:
                return False

        uow = IncompleteUnitOfWork("root")
        assert uow._route == "root"

        with pytest.raises(NotImplementedError):
            AbstractFileSystemUnitOfWork.create_text_file(uow, "text", "file.txt")
        with pytest.raises(NotImplementedError):
            AbstractFileSystemUnitOfWork.delete_file(uow, "file.txt")
        with pytest.raises(NotImplementedError):
            AbstractFileSystemUnitOfWork.get_file(uow, "file.txt")
        with pytest.raises(NotImplementedError):
            AbstractFileSystemUnitOfWork.list_folder(uow, "folder")
        with pytest.raises(NotImplementedError):
            AbstractFileSystemUnitOfWork.move_file(uow, "old.txt", "new.txt")
        with pytest.raises(NotImplementedError):
            AbstractFileSystemUnitOfWork.move_route(uow, "other")
        with pytest.raises(NotImplementedError):
            AbstractFileSystemUnitOfWork.upload_file(uow, "local.txt")
