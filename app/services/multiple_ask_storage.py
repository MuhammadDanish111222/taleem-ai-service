"""Small server-only Supabase Storage boundary for temporary Multiple Ask inputs."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.core.config import get_settings


class TemporaryStorageError(RuntimeError):
    pass


class TemporaryStorageObjectNotFound(TemporaryStorageError):
    pass


class TemporaryStorageObjectTooLarge(TemporaryStorageError):
    pass


@dataclass(frozen=True)
class ObjectMetadata:
    content_type: str | None
    size_bytes: int | None


@dataclass(frozen=True)
class SignedUploadCapability:
    upload_url: str
    method: str
    required_headers: dict[str, str]


class TemporaryUploadStorage:
    """Issues one-object upload URLs and verifies objects without exposing keys.

    This intentionally uses the Storage REST API from the service process only;
    the web BFF and browser never receive a service-role credential.
    """

    def __init__(self, *, base_url: str | None = None, service_key: str | None = None):
        settings = get_settings()
        self._base_url = (base_url or settings.SUPABASE_URL).rstrip("/")
        self._service_key = service_key or settings.SUPABASE_SERVICE_ROLE_KEY

    def _headers(self, *, json_body: bool = True) -> dict[str, str]:
        if not self._base_url or not self._service_key:
            raise TemporaryStorageError("TEMPORARY_STORAGE_UNAVAILABLE")
        headers = {
            "apikey": self._service_key,
            "Authorization": f"Bearer {self._service_key}",
        }
        if json_body:
            headers["Content-Type"] = "application/json"
        return headers

    def _request_json(
        self, method: str, path: str, payload: dict | None = None
    ) -> dict:
        request = Request(
            f"{self._base_url}/storage/v1{path}",
            data=json.dumps(payload).encode("utf-8") if payload is not None else None,
            headers=self._headers(),
            method=method,
        )
        try:
            with urlopen(request, timeout=10) as response:  # nosec B310: configured service URL
                raw = response.read().decode("utf-8")
        except (HTTPError, URLError, TimeoutError) as exc:
            raise TemporaryStorageError("TEMPORARY_STORAGE_UNAVAILABLE") from exc
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TemporaryStorageError("TEMPORARY_STORAGE_INVALID_RESPONSE") from exc
        if not isinstance(parsed, dict):
            raise TemporaryStorageError("TEMPORARY_STORAGE_INVALID_RESPONSE")
        return parsed

    async def create_signed_upload_url(
        self, *, bucket: str, object_key: str, content_type: str
    ) -> SignedUploadCapability:
        response = await asyncio.to_thread(
            self._request_json,
            "POST",
            f"/object/upload/sign/{bucket}/{object_key}",
            {"upsert": False},
        )
        signed_path = response.get("url") or response.get("signedUrl")
        if not isinstance(signed_path, str) or not signed_path.startswith("/"):
            raise TemporaryStorageError("TEMPORARY_STORAGE_INVALID_RESPONSE")
        # Supabase returns a relative path. Reject an absolute URL so an upstream
        # response can never turn this endpoint into an open redirect. Supabase's
        # supported uploadToSignedUrl flow uploads the raw body with HTTP PUT.
        return SignedUploadCapability(
            upload_url=f"{self._base_url}/storage/v1{signed_path}",
            method="PUT",
            # The signed-upload creation request controls overwrite behavior.
            # The upload itself only needs to preserve the declared content type.
            required_headers={"Content-Type": content_type},
        )

    def _read_object_limited_sync(
        self, *, bucket: str, object_key: str, max_bytes: int, prefix_bytes: int | None
    ) -> bytes:
        headers = self._headers(json_body=False)
        if prefix_bytes is not None:
            headers["Range"] = f"bytes=0-{prefix_bytes - 1}"
        request = Request(
            f"{self._base_url}/storage/v1/object/{bucket}/{object_key}",
            headers=headers,
            method="GET",
        )
        try:
            with urlopen(request, timeout=15) as response:  # nosec B310: configured service URL
                content_length = response.headers.get("Content-Length")
                if content_length is not None and int(content_length) > max_bytes:
                    raise TemporaryStorageObjectTooLarge(
                        "TEMPORARY_STORAGE_OBJECT_TOO_LARGE"
                    )
                result = bytearray()
                remaining = prefix_bytes if prefix_bytes is not None else max_bytes + 1
                while remaining > 0:
                    chunk = response.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    result.extend(chunk)
                    remaining -= len(chunk)
                if prefix_bytes is None and len(result) > max_bytes:
                    raise TemporaryStorageObjectTooLarge(
                        "TEMPORARY_STORAGE_OBJECT_TOO_LARGE"
                    )
                return bytes(result)
        except HTTPError as exc:
            if exc.code == 404:
                raise TemporaryStorageObjectNotFound(
                    "TEMPORARY_STORAGE_OBJECT_NOT_FOUND"
                ) from exc
            raise TemporaryStorageError("TEMPORARY_STORAGE_UNAVAILABLE") from exc
        except (URLError, TimeoutError, ValueError) as exc:
            raise TemporaryStorageError("TEMPORARY_STORAGE_UNAVAILABLE") from exc

    async def read_object_prefix(
        self, *, bucket: str, object_key: str, max_bytes: int
    ) -> bytes:
        return await asyncio.to_thread(
            self._read_object_limited_sync,
            bucket=bucket,
            object_key=object_key,
            max_bytes=max_bytes,
            prefix_bytes=max_bytes,
        )

    async def read_object_limited(
        self, *, bucket: str, object_key: str, max_bytes: int
    ) -> bytes:
        return await asyncio.to_thread(
            self._read_object_limited_sync,
            bucket=bucket,
            object_key=object_key,
            max_bytes=max_bytes,
            prefix_bytes=None,
        )

    async def object_metadata(self, *, bucket: str, object_key: str) -> ObjectMetadata:
        response = await asyncio.to_thread(
            self._request_json,
            "GET",
            f"/object/info/{bucket}/{object_key}",
        )
        metadata = response.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        content_type = response.get("contentType") or metadata.get("mimetype")
        raw_size = metadata.get("size") or response.get("size")
        try:
            size_bytes = int(raw_size) if raw_size is not None else None
        except (TypeError, ValueError):
            size_bytes = None
        return ObjectMetadata(
            content_type=content_type if isinstance(content_type, str) else None,
            size_bytes=size_bytes,
        )

    async def delete_object(self, *, bucket: str, object_key: str) -> None:
        await asyncio.to_thread(
            self._request_json,
            "DELETE",
            f"/object/{bucket}",
            {"prefixes": [object_key]},
        )
