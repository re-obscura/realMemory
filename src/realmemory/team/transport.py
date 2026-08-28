"""Сетевой транспорт клиента координатора: urllib, без зависимостей.

Ошибки нормализованы в иерархию CoordinatorError, чтобы CLI/TUI/агент могли
показывать человеку понятную причину (недоступен / токен / несовпадение
эмбеддеров), а код — различать их программно.
"""
from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request


class CoordinatorError(Exception):
    """Базовая ошибка взаимодействия с координатором."""


class TransportError(CoordinatorError):
    """Координатор недоступен (сеть/порт/таймаут)."""


class AuthError(CoordinatorError):
    """Токен не задан или отклонён (401)."""


class EmbedderMismatch(CoordinatorError):
    """В кэше записи других эмбеддеров: косинус несравним."""

    def __init__(self, known: list[str], requested: str) -> None:
        super().__init__(
            f"в командном кэше записи эмбеддера {known}, локально "
            f"запрошен {requested!r} — сравнение бессмысленно; "
            "синхронизируйте версии модели в команде")
        self.known = known
        self.requested = requested


def encode_vector(vec) -> str:
    import numpy as np

    return base64.b64encode(np.asarray(vec, dtype=np.float32).tobytes()).decode()


class CoordinatorClient:
    def __init__(self, base_url: str, token: str | None = None,
                 timeout_s: float = 4.0) -> None:
        self.base = base_url.rstrip("/")
        self.token = token or None
        self.timeout = float(timeout_s)

    def _call(self, method: str, path: str, payload: dict | None = None) -> dict:
        url = f"{self.base}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8")[:300]
                parsed = json.loads(detail)
            except (ValueError, UnicodeDecodeError):
                parsed = {}
            if exc.code == 401:
                raise AuthError("координатор отклонил токен (401)") from exc
            if exc.code == 409 and parsed.get("error") == "embedder_mismatch":
                raise EmbedderMismatch(parsed.get("known_embedders", []),
                                       parsed.get("requested", "")) from exc
            raise CoordinatorError(
                f"координатор ответил {exc.code}: {detail or exc.reason}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise TransportError(f"координатор недоступен ({url}): {exc}") from exc

    # -- типизированные обёртки -------------------------------------------------

    def health(self) -> dict:
        return self._call("GET", "/health")

    def heartbeat(self, identity: str, address: str = "",
                  projects: list[str] | None = None) -> dict:
        return self._call("POST", "/heartbeat",
                          {"identity": identity, "address": address,
                           "projects": projects})

    def presence(self) -> list[dict]:
        return self._call("GET", "/presence").get("presence", [])

    def publish_batch(self, items: list[dict]) -> int:
        out = self._call("POST", "/publish", {"items": items})
        return int(out.get("accepted", 0))

    def retract_batch(self, tombstones: list[dict]) -> int:
        out = self._call("POST", "/retract", {"tombstones": tombstones})
        return int(out.get("retracted", 0))

    def search(self, query_vec, k: int = 5, embedder: str = "",
               author: str | None = None, project: str | None = None) -> list[dict]:
        out = self._call("POST", "/search",
                         {"query_embedding_b64": encode_vector(query_vec),
                          "k": k, "embedder": embedder,
                          "author": author, "project": project})
        return out.get("hits", [])

    def cache_dump(self) -> dict:
        return self._call("GET", "/cache/dump")

    def raw_post(self, path: str, payload: dict) -> dict:
        """Прямой вызов произвольного маршрута (peer-endpoint /recall)."""
        return self._call("POST", path, payload)
