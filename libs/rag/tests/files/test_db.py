"""Tests for the MongoDB/GridFS filesystem Unit of Work.

====================================================================================================
Name:         libs/rag/tests/files/test_db.py
Description:  Covers GridFS operations, filtering, and exception paths.
Project:      Pygenai
Date:         2026-11-25
Status:       Development

Copyright ©2026 All rights reserved.
====================================================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import pygenai.rag.files.db as db_mod
from pygenai.rag.files import MongoFileSystemUnitOfWork


class FakeFileObj:
    """Minimal GridFS file object."""

    def __init__(
        self,
        _id: int,
        filename: str,
        metadata: dict[str, Any],
        uploadDate: int,
        content: bytes,
    ) -> None:
        self._id = _id
        self.filename = filename
        self.metadata = metadata
        self.uploadDate = uploadDate
        self._content = content

    def read(self) -> bytes:
        return self._content


class FakeGridFS:
    """In-memory GridFS implementation."""

    def __init__(self) -> None:
        self._storage = []
        self._next_id = 1

    def put(self, data: bytes, filename: str | None = None, metadata: dict[str, Any] | None = None) -> int:
        doc = {
            "_id": self._next_id,
            "filename": filename,
            "metadata": metadata or {},
            "uploadDate": self._next_id,
            "content": data,
        }
        self._storage.append(doc)
        self._next_id += 1
        return doc["_id"]

    def get(self, _id: int) -> FakeFileObj:
        for d in self._storage:
            if d["_id"] == _id:
                return FakeFileObj(d["_id"], d["filename"], d["metadata"], d["uploadDate"], d["content"])
        raise Exception("NotFound")

    def delete(self, _id: int) -> None:
        self._storage = [d for d in self._storage if d["_id"] != _id]


class FakeCursor:
    """Cursor implementing sort, limit, and iteration."""

    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self._docs = docs
        self._limit = None

    def sort(self, sort_spec: list[tuple[str, int]]) -> FakeCursor:
        # sort_spec like [("uploadDate", -1)]
        key, direction = sort_spec[0]
        reverse = direction < 0
        self._docs.sort(key=lambda d: d.get(key), reverse=reverse)
        return self

    def limit(self, n: int) -> FakeCursor:
        self._limit = n
        return self

    def __iter__(self) -> Any:
        docs = self._docs
        if self._limit is not None:
            docs = docs[: self._limit]
        for d in docs:
            yield d


class FakeFilesCollection:
    """Queryable GridFS files collection."""

    def __init__(self, gridfs: FakeGridFS) -> None:
        self._gridfs = gridfs

    def find(self, query: dict[str, Any]) -> FakeCursor:
        def matches(d: dict[str, Any]) -> bool:
            if "filename" in query:
                q = query["filename"]
                if isinstance(q, dict) and "$regex" in q:
                    import re

                    pattern = q["$regex"]
                    return re.match(pattern, d["filename"]) is not None
                else:
                    if d["filename"] != q:
                        return False
            if "metadata.route" in query:
                if d.get("metadata", {}).get("route") != query["metadata.route"]:
                    return False
            return True

        docs = [d for d in self._gridfs._storage if matches(d)]
        return FakeCursor(docs)


class FakeFS:
    """Filesystem namespace containing the files collection."""

    def __init__(self, gridfs: FakeGridFS) -> None:
        self.files = FakeFilesCollection(gridfs)


class FakeDB:
    """Mongo database stand-in."""

    def __init__(self, gridfs: FakeGridFS) -> None:
        self.fs = FakeFS(gridfs)


class FakeMongoClient:
    """Mongo client stand-in exposing one fake database."""

    def __init__(self) -> None:
        self._gridfs = FakeGridFS()

    def __getitem__(self, name: str) -> FakeDB:
        # return a DB that exposes fs.files.find
        return FakeDB(self._gridfs)


@pytest.fixture(autouse=True)
def fake_mongo_and_gridfs(monkeypatch: pytest.MonkeyPatch) -> FakeMongoClient:
    fake_client = FakeMongoClient()
    # monkeypatch MongoClient to return our fake client
    monkeypatch.setattr(db_mod, "MongoClient", lambda uri=None: fake_client)
    # monkeypatch gridfs.GridFS to return a wrapper around the fake gridfs

    def fake_gridfs(db: Any) -> FakeGridFS:
        # ignore db, return the internal FakeGridFS instance
        return fake_client._gridfs

    monkeypatch.setattr(db_mod.gridfs, "GridFS", fake_gridfs)
    return fake_client


def test_db_basic_operations(tmp_path: Path, fake_mongo_and_gridfs: FakeMongoClient) -> None:
    uow = MongoFileSystemUnitOfWork(route="root", mongo_uri="fake://", database="db")

    # create text
    assert uow.create_text_file("hello db", "a.txt")
    data = uow.get_file("a.txt")
    assert data == b"hello db"

    # list
    items = uow.list_folder("")
    assert "a.txt" in items

    # upload local file
    src = tmp_path / "f.bin"
    src.write_bytes(b"blobdb")
    assert uow.upload_file(str(src), name="blob.bin")
    assert uow.get_file("blob.bin") == b"blobdb"

    # move
    assert uow.move_file("a.txt", "a2.txt")
    assert uow.get_file("a2.txt") == b"hello db"

    # delete
    assert uow.delete_file("a2.txt")
    with pytest.raises(FileNotFoundError):
        uow.get_file("a2.txt")


class TestMongoFailurePaths:
    """Tests for Mongo/GridFS edge and failure paths."""

    def test_normalize_move_route_and_nested_listing(self) -> None:
        """Normalize prefixes and handle nested route listings."""
        uow = MongoFileSystemUnitOfWork(route="/root/", mongo_uri="fake://", database="db")

        assert uow._normalize_prefix(None) == ""
        assert uow._normalize_prefix("//folder//") == "folder"
        assert uow.move_route("/next/") == "next"
        assert uow.list_folder("") == []

    def test_move_missing_and_upload_missing(self) -> None:
        """Return false for missing move sources and local uploads."""
        uow = MongoFileSystemUnitOfWork(route="root", mongo_uri="fake://", database="db")

        assert uow.move_file("missing.txt", "target.txt") is False
        assert uow.upload_file("/missing/source.txt") is False

    def test_create_delete_and_move_failures(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Return false when GridFS mutating operations raise."""
        uow = MongoFileSystemUnitOfWork(route="root", mongo_uri="fake://", database="db")

        def raise_error(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("gridfs failure")

        monkeypatch.setattr(uow._fs, "put", raise_error)
        assert uow.create_text_file("text", "a.txt") is False

        monkeypatch.setattr(uow._fs, "delete", raise_error)
        monkeypatch.setattr(uow._fs, "put", raise_error)

        def raise_query_error(query: dict[str, Any]) -> Any:
            del query
            raise RuntimeError("query failure")

        monkeypatch.setattr(uow._db.fs.files, "find", raise_query_error)
        assert uow.delete_file("a.txt") is False
        assert uow.move_file("a.txt", "b.txt") is False

    def test_get_missing_object(self) -> None:
        """Raise FileNotFoundError for an object absent from GridFS."""
        uow = MongoFileSystemUnitOfWork(route="root", mongo_uri="fake://", database="db")

        with pytest.raises(FileNotFoundError, match="not found"):
            uow.get_file("missing.txt")

    def test_list_with_folder_prefix(self) -> None:
        """List top-level files under a nested route."""
        uow = MongoFileSystemUnitOfWork(route="root", mongo_uri="fake://", database="db")

        assert uow.create_text_file("one", "folder/one.txt") is True
        assert uow.create_text_file("two", "folder/nested/two.txt") is True

        assert uow.list_folder("folder") == ["one.txt"]
