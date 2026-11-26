"""Tests for the S3 filesystem Unit of Work.

====================================================================================================
Name:         libs/rag/tests/files/test_s3.py
Description:  Covers S3 route resolution, operations, and exception handling.
Project:      Pygenai
Date:         2026-11-25
Status:       Development

Copyright ©2026 All rights reserved.
====================================================================================================
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest

import pygenai.rag.files.s3 as s3_mod
from pygenai.rag.files.s3 import S3FileSystemUnitOfWork


class FakeS3Client:
    """In-memory stand-in for the boto3 S3 client."""

    def __init__(self) -> None:
        self.storage: dict[tuple[str, str], bytes] = {}

    def upload_fileobj(self, fileobj: Any, bucket: str, key: str) -> None:
        fileobj.seek(0)
        self.storage[(bucket, key)] = fileobj.read()

    def delete_object(self, Bucket: str, Key: str) -> None:
        self.storage.pop((Bucket, Key), None)

    def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:
        if (Bucket, Key) not in self.storage:
            raise Exception("NoSuchKey")
        return {"Body": io.BytesIO(self.storage[(Bucket, Key)])}

    def list_objects_v2(self, Bucket: str, Prefix: str) -> dict[str, Any]:
        keys = [k for (b, k) in self.storage.keys() if b == Bucket and k.startswith(Prefix)]
        contents = [{"Key": k} for k in keys]
        return {"Contents": contents} if contents else {}

    def copy_object(self, Bucket: str, CopySource: dict[str, str], Key: str) -> None:
        src_bucket = CopySource["Bucket"]
        src_key = CopySource["Key"]
        data = self.storage.get((src_bucket, src_key))
        if data is None:
            raise Exception("SourceNotFound")
        self.storage[(Bucket, Key)] = data

    def upload_file(self, file_route: str, Bucket: str, Key: str) -> None:
        with open(file_route, "rb") as f:
            self.storage[(Bucket, Key)] = f.read()


@pytest.fixture(autouse=True)
def fake_boto3_client(monkeypatch: pytest.MonkeyPatch) -> FakeS3Client:
    fake = FakeS3Client()

    def fake_client(*args: Any, **kwargs: Any) -> FakeS3Client:
        return fake

    monkeypatch.setattr(s3_mod, "boto3", type("B", (), {"client": staticmethod(fake_client)}))
    return fake


def test_s3_basic_operations(tmp_path: Path, fake_boto3_client: FakeS3Client) -> None:
    bucket = "test-bucket"
    uow = S3FileSystemUnitOfWork(route="", bucket=bucket)

    # create text file
    assert uow.create_text_file("hello s3", "a.txt")
    assert fake_boto3_client.storage[(bucket, "/a.txt")] == b"hello s3"

    # get file
    assert uow.get_file("a.txt") == b"hello s3"

    # list folder
    items = uow.list_folder("")
    assert "a.txt" in items

    # upload local file
    src = tmp_path / "f.bin"
    src.write_bytes(b"blob")
    assert uow.upload_file(str(src), name="blob.bin")
    assert fake_boto3_client.storage[(bucket, "/blob.bin")] == b"blob"

    # move file
    assert uow.move_file("a.txt", "a2.txt")
    assert (bucket, "/a.txt") not in fake_boto3_client.storage
    assert (bucket, "/a2.txt") in fake_boto3_client.storage

    # delete file
    assert uow.delete_file("a2.txt")
    assert (bucket, "/a2.txt") not in fake_boto3_client.storage


class TestS3RouteResolution:
    """Tests for S3 route parsing and key normalization."""

    def test_normalize_prefix_and_join_key(self) -> None:
        """Normalize slashes and join route/key components."""
        assert S3FileSystemUnitOfWork._normalize_prefix(None) == ""
        assert S3FileSystemUnitOfWork._normalize_prefix("//folder//") == "folder"
        assert S3FileSystemUnitOfWork._join_key("", "a.txt") == "/a.txt"
        assert S3FileSystemUnitOfWork._join_key("root", "folder", "a.txt") == "root/folder/a.txt"

    def test_resolve_s3_uri_explicit_bucket_and_legacy_route(self) -> None:
        """Resolve URI, explicit bucket, and legacy bucket/prefix formats."""
        assert S3FileSystemUnitOfWork._resolve_bucket_and_route(None, "s3://bucket/prefix") == ("bucket", "prefix")
        assert S3FileSystemUnitOfWork._resolve_bucket_and_route("bucket", "prefix") == ("bucket", "prefix")
        assert S3FileSystemUnitOfWork._resolve_bucket_and_route(None, "bucket/prefix") == ("bucket", "prefix")

    def test_resolve_invalid_routes(self) -> None:
        """Reject routes without a resolvable bucket."""
        with pytest.raises(ValueError, match="bucket is required"):
            S3FileSystemUnitOfWork._resolve_bucket_and_route(None, "/prefix")
        with pytest.raises(ValueError, match="bucket is required"):
            S3FileSystemUnitOfWork._resolve_bucket_and_route(None, "")


class TestS3FailurePaths:
    """Tests for successful and exceptional S3 operations."""

    def test_create_delete_move_and_upload_failures(self, fake_boto3_client: FakeS3Client) -> None:
        """Return False when mutating S3 operations raise."""
        uow = S3FileSystemUnitOfWork(route="s3://bucket/root")

        def raise_error(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("s3 failure")

        uow.s3_client.upload_fileobj = raise_error
        uow.s3_client.delete_object = raise_error
        uow.s3_client.copy_object = raise_error
        uow.s3_client.upload_file = raise_error

        assert uow.create_text_file("text", "a.txt") is False
        assert uow.delete_file("a.txt") is False
        assert uow.move_file("a.txt", "b.txt") is False
        assert uow.upload_file("/missing/file.txt") is False

    def test_get_and_list_exceptions_are_reraised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Propagate retrieval and listing errors after logging."""
        uow = S3FileSystemUnitOfWork(route="s3://bucket/root")

        def raise_error(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("read failure")

        uow.s3_client.get_object = raise_error
        uow.s3_client.list_objects_v2 = raise_error

        with pytest.raises(RuntimeError, match="read failure"):
            uow.get_file("missing.txt")
        with pytest.raises(RuntimeError, match="read failure"):
            uow.list_folder("folder")

    def test_move_route_s3_and_bucket_routes(self) -> None:
        """Switch bucket and prefix for both supported route formats."""
        uow = S3FileSystemUnitOfWork(route="bucket/root")

        assert uow.move_route("s3://other/prefix") == "s3://other/prefix"
        assert uow._bucket == "other"
        assert uow._route == "prefix"

        assert uow.move_route("bucket-two/new-root") == "bucket-two/new-root"
        assert uow._bucket == "other"
        assert uow._route == "bucket-two/new-root"
