"""MongoDB/GridFS implementation of the filesystem Unit of Work.

Provides MongoFileSystemUnitOfWork, an adapter that implements the
AbstractFileSystemUnitOfWork interface using MongoDB's GridFS for storing
binary blobs and simple metadata to scope objects under a "route" prefix.

The adapter exposes the same methods as other filesystem adapters and aims
for similar return values and exception handling patterns.

Note: This module requires the `pymongo` package and will raise an
ImportError at import time if it's not available.
"""

from __future__ import annotations

import logging
import os

import gridfs
from pymongo import MongoClient

from ._base import AbstractFileSystemUnitOfWork

logger = logging.getLogger(__name__)


class MongoFileSystemUnitOfWork(AbstractFileSystemUnitOfWork):
    """GridFS-based implementation of AbstractFileSystemUnitOfWork.

    Files are stored in GridFS with filenames representing object keys. Each
    stored file receives metadata containing the configured route so that
    objects can be scoped under a prefix (similar to a folder). Methods try
    to follow the same semantics as the local and S3 adapters: boolean
    returns for mutating ops and exceptions logged and propagated where
    appropriate for retrieval and listing operations.

    Args:
        route: Base prefix used to scope keys inside the database.
        mongo_uri: MongoDB connection URI (e.g. "mongodb://localhost:27017").
        database: Name of the MongoDB database to use for GridFS.
    """

    def __init__(self, route: str, mongo_uri: str = "mongodb://localhost:27017", database: str = "files") -> None:
        """Initialize the MongoFileSystemUnitOfWork with a route and MongoDB connection."""
        super().__init__(route=route)
        self._route = self._normalize_prefix(route)
        self._client = MongoClient(mongo_uri)
        self._db = self._client[database]
        # GridFS instance (uses fs.files and fs.chunks collections by default)
        self._fs = gridfs.GridFS(self._db)

    @staticmethod
    def _normalize_prefix(prefix: str | None) -> str:
        """Normalize a prefix by stripping surrounding slashes and defaulting to "".

        Ensures consistent key construction when combining route and object keys.
        """
        if not prefix:
            return ""
        return prefix.strip("/")

    def _object_key(self, key: str | None) -> str:
        """Combine configured route and a key into a single object key string."""
        parts = [p.strip("/") for p in (self._route, key) if p and p.strip("/")]
        return "/".join(parts)

    def create_text_file(self, text: str, file: str, encoding: str = "utf-8") -> bool:
        """Store text as a GridFS object under the scoped key.

        The text is encoded using the provided encoding and saved to GridFS with
        metadata that records the configured route. Returns True on success,
        False on failure.
        """
        try:
            key = self._object_key(file)
            data = text.encode(encoding)
            self._fs.put(data, filename=key, metadata={"route": self._route})
            return True
        except Exception:
            logger.exception("Create text file '%s' failed with an exception.", file)
            return False

    def delete_file(self, file: str) -> bool:
        """Delete objects matching the key under the current route.

        GridFS may store multiple revisions; this removes any matching files.
        Returns True if at least one file was deleted, otherwise False.
        """
        key = self._object_key(file)
        try:
            deleted = False
            # Query files collection for matching filename and route metadata
            for f in self._db.fs.files.find({"filename": key, "metadata.route": self._route}):
                try:
                    self._fs.delete(f["_id"])
                    deleted = True
                except Exception:
                    logger.exception("Failed to delete GridFS file id=%s", f.get("_id"))
            return deleted
        except Exception:
            logger.exception("Delete file '%s' failed with an exception.", file)
            return False

    def get_file(self, file: str) -> bytes:
        """Retrieve the latest version of a file's bytes from GridFS.

        Raises an exception if the object is not found or another error occurs.
        """
        key = self._object_key(file)
        try:
            # Find the most recent file matching filename and route metadata
            cursor = (
                self._db.fs.files
                .find({"filename": key, "metadata.route": self._route})
                .sort([("uploadDate", -1)])
                .limit(1)
            )
            docs = list(cursor)
            doc = docs[0] if docs else None
            if not doc:
                raise FileNotFoundError(f"Object '{key}' not found in GridFS")
            file_obj = self._fs.get(doc["_id"])
            return file_obj.read()
        except Exception:
            logger.exception("Exception getting object '%s' from GridFS.", key)
            raise

    def list_folder(self, prefix: str) -> list:
        """List object keys directly under a prefix (non-recursive).

        The returned names are relative to the provided prefix. If the prefix
        is empty, lists objects under the configured route.
        """
        folder_prefix = self._normalize_prefix(prefix)
        # Build the full prefix to search for (route + folder_prefix)
        if self._route and folder_prefix:
            search_prefix = f"{self._route}/{folder_prefix}".strip("/")
        elif self._route:
            search_prefix = self._route
        else:
            search_prefix = folder_prefix

        match_prefix = search_prefix + "/" if search_prefix else ""

        try:
            results = []
            query = {"filename": {"$regex": f"^{match_prefix}"}} if match_prefix else {}
            # also ensure metadata.route equals configured route for strict scoping
            if self._route:
                query["metadata.route"] = self._route

            for doc in self._db.fs.files.find(query):
                filename = doc.get("filename", "")
                # strip the folder prefix and leading slashes
                rel = filename[len(match_prefix) :].lstrip("/") if match_prefix else filename

                # only include top-level entries (no nested '/'), mimic non-recursive
                if rel and "/" not in rel:
                    results.append(rel)
            return results
        except Exception:
            logger.exception("Exception listing objects with prefix '%s' in GridFS.", prefix)
            raise

    def move_file(self, old_file: str, new_file: str, new_route: str | None = None) -> bool:
        """Copy a file to a new key (and optional new route) then remove the old one.

        This implements move semantics by reading the source, writing a new
        GridFS object with the destination key and metadata, and deleting the
        original file(s) matching the source key.
        """
        old_key = self._object_key(old_file)
        target_route = self._normalize_prefix(new_route) if new_route is not None else self._route
        new_key_parts = [p for p in (target_route, new_file) if p]
        new_key = "/".join([p.strip("/") for p in new_key_parts])
        try:
            # Read source (take most recent)
            cursor = (
                self._db.fs.files
                .find({"filename": old_key, "metadata.route": self._route})
                .sort([("uploadDate", -1)])
                .limit(1)
            )
            docs = list(cursor)
            doc = docs[0] if docs else None
            if not doc:
                return False
            src_file = self._fs.get(doc["_id"])
            content = src_file.read()
            # Put new file with new metadata
            self._fs.put(content, filename=new_key, metadata={"route": target_route})

            for f in self._db.fs.files.find({"filename": old_key, "metadata.route": self._route}):
                try:
                    self._fs.delete(f["_id"])
                except Exception:
                    logger.exception("Failed to delete GridFS file id=%s", f.get("_id"))
            return True
        except Exception:
            logger.exception("Move file '%s' to '%s' failed with an exception.", old_file, new_file)
            return False

    def move_route(self, route: str) -> str:
        """Change the base route used to scope stored objects.

        This only updates the local configuration; it does not migrate objects
        between routes in the database.
        """
        self._route = self._normalize_prefix(route)
        return self._route

    def upload_file(self, file_route: str, name: str | None = None) -> bool:
        """Read a local file and store it in GridFS under the configured route.

        If 'name' is not provided, the source file's basename is used.
        """
        try:
            if not os.path.exists(file_route):
                return False
            basename = name or os.path.basename(file_route)
            key = self._object_key(basename)
            with open(file_route, "rb") as f:
                data = f.read()
            self._fs.put(data, filename=key, metadata={"route": self._route})
            return True
        except Exception:
            logger.exception("Upload file '%s' failed with an exception.", file_route)
            return False
