"""Общий каркас HTTP-демонов командного слоя (координатор и peer).

Только стандартная библиотека + numpy (базовая зависимость пакета). Здесь
живут скучные, но критичные вещи, одинаковые для обоих демонов: лимит тела
запроса, строгий JSON (без NaN/Infinity и не-объектов), аутентификация общим
токеном с постоянным временем сравнения, потолок одновременных обработчиков,
декодирование векторов и косинус-скоринг. Битый вход от сети обязан давать
чистый 400/413, а не 500 или падение соединения.
"""
from __future__ import annotations

import base64
import binascii
import hmac
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

MAX_BODY_BYTES = 32 * 1024 * 1024
MAX_HTTP_HANDLERS = 32
# сколько байт сверхлимитного тела осушаем перед ответом 413: close с
# непрочитанными данными в буфере даёт клиенту RST вместо нашего ответа
_DRAIN_ON_TOO_LARGE = 1024 * 1024


class BadRequest(Exception):
    """Некорректный запрос клиента: ответ 400."""


class PayloadTooLarge(BadRequest):
    """Тело запроса больше лимита: ответ 413."""


def _reject_constant(name: str):
    raise ValueError(f"недопустимый JSON-литерал {name}")


def decode_vector_b64(s) -> np.ndarray:
    """float32-вектор из base64; мусор в кодировке или размере — BadRequest."""
    try:
        raw = base64.b64decode(str(s or ""), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise BadRequest(f"битый base64: {exc}") from exc
    if len(raw) % 4:
        raise BadRequest("блоб вектора не кратен sizeof(float32)")
    return np.frombuffer(raw, dtype=np.float32)


def finite_float(value, what: str) -> float:
    """float без NaN/Infinity: не-конечные ломают NOT NULL-схему и сортировки."""
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise BadRequest(f"{what}: не число ({value!r})") from exc
    if not math.isfinite(out):
        raise BadRequest(f"{what}: недопустимое не-конечное значение")
    return out


def cosine_scores(q: np.ndarray, vecs: list[np.ndarray]) -> np.ndarray:
    """Косинусы запроса к строкам; нулевые нормы дают 0. Сравнимые размерности
    гарантируются вызывающей стороной (несравнимые строки фильтруются до)."""
    if not vecs:
        return np.empty(0, dtype=np.float32)
    mat = np.stack([np.asarray(v, dtype=np.float32) for v in vecs])
    qn = float(np.linalg.norm(q))
    if qn <= 0.0:
        return np.zeros(len(vecs), dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1)
    out = np.zeros(len(vecs), dtype=np.float32)
    ok = norms > 0.0
    out[ok] = (mat[ok] @ q) / (norms[ok] * qn)
    return out


class BoundedThreadingHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer с потолком одновременных обработчиков: флуд из LAN
    не должен плодить неограниченное число потоков. Сверх потолка соединение
    закрывается без ответа — клиент видит обрыв и повторяет по своей логике."""

    max_handlers = MAX_HTTP_HANDLERS

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._slots = threading.BoundedSemaphore(self.max_handlers)

    def process_request(self, request, client_address) -> None:
        if not self._slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._slots.release()
            raise

    def process_request_thread(self, request, client_address) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


class TeamHandler(BaseHTTPRequestHandler):
    """Общие части координатора и peer: токен, JSON, ответы, тишина в лог."""

    max_body_bytes = MAX_BODY_BYTES

    @property
    def state(self):
        return self.server.state  # type: ignore[attr-defined]

    @property
    def expected_token(self) -> str | None:
        return getattr(self.server, "token", None)  # type: ignore[attr-defined]

    def log_message(self, fmt, *args) -> None:  # тихие демоны
        return

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        want = self.expected_token
        if not want:
            return True
        got = self.headers.get("Authorization", "")
        # сравнение с постоянным временем: токен — общий секрет команды
        return hmac.compare_digest(got, f"Bearer {want}")

    def _read_json(self) -> dict:
        """Тело запроса как JSON-объект. Пустое тело — {}; битый JSON, не-объект
        или NaN/Infinity внутри — BadRequest (400); слишком большое —
        PayloadTooLarge (413)."""
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError as exc:
            raise BadRequest("битый Content-Length") from exc
        if length <= 0:
            return {}
        if length > self.max_body_bytes:
            try:
                self.rfile.read(min(length, _DRAIN_ON_TOO_LARGE))
            except OSError:
                pass
            raise PayloadTooLarge(
                f"тело {length} байт больше лимита {self.max_body_bytes}")
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"),
                              parse_constant=_reject_constant)
        except (ValueError, UnicodeDecodeError) as exc:
            raise BadRequest(f"битый JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise BadRequest("ожидается JSON-объект")
        return data
