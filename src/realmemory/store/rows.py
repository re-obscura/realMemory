"""Мэппинг строк БД и текстовые утилиты keyword-канала.

Кодирование/декодирование записей (блобы эмбеддинга и SDR), колоночные
списки и токенизация запросов для FTS5. Точка декодирования одна —
_row_to_record; дублирования маппинга в других модулях нет и быть не должно.
"""
from __future__ import annotations

import json
import re

import numpy as np

from ..types import MemoryRecord

_INSERT_SQL = (
    "INSERT INTO memories(text,kind,status,meta,embedding,sdr,"
    "created_at,updated_at,reinforced_count,last_reinforced_at,base_strength,"
    "valid_from,valid_to,superseded_by,scope,author) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
)

_COLS = (
    "id,text,kind,status,meta,embedding,sdr,created_at,updated_at,"
    "reinforced_count,last_reinforced_at,base_strength,valid_from,valid_to,superseded_by,scope,author"
)

# колоночный «лёгкий» срез для проходов «сна»: динамика следа без блобов
_COLS_LIGHT = "id,kind,created_at,last_reinforced_at,base_strength,reinforced_count"

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# ключ канонического сравнения скоупов: регистр и разновид разделителей
# («CPM.Backend», «cpm_backend», «cpm backend») не должны расщеплять проект —
# на живых данных догфудинга один проект жил под тремя именами, и recall
# между ними не пересекался
_SCOPE_KEY_RE = re.compile(r"[._\- ]+")


def scope_key(name: str) -> str:
    return _SCOPE_KEY_RE.sub(".", str(name).casefold())


def tokenize(text: str) -> list[str]:
    """Токены для keyword-канала: слова в нижнем регистре (unicode)."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def search_tokens(text: str) -> set[str]:
    """Токены, значимые для сопоставления запроса и следа: 1-символьные
    отбрасываются синхронно с build_fts_query, иначе каналы определяют
    «токен» по-разному и full_match ведёт себя несогласованно."""
    return {t for t in tokenize(text) if len(t) > 1}


def build_fts_query(text: str, max_terms: int = 12) -> str | None:
    """Выражение MATCH из токенов запроса: "tok1" OR "tok2" ..."""
    seen: dict[str, None] = {}
    for t in tokenize(text):
        if len(t) > 1:
            seen.setdefault(t)
        if len(seen) >= max_terms:
            break
    if not seen:
        return None
    return " OR ".join(f'"{t}"' for t in seen)


def _pack_f32(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def _pack_i32(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.int32).tobytes()


def _row_to_record(row: tuple, dim: int) -> MemoryRecord:
    (rid, text, kind, status, meta_json, emb_b, sdr_b, created, updated,
     rcount, rlast, base, vfrom, vto, sby, scope, author) = row
    return MemoryRecord(
        id=int(rid),
        text=text,
        kind=kind,
        status=status,
        meta=json.loads(meta_json),
        embedding=np.frombuffer(emb_b, dtype=np.float32, count=dim).copy(),
        sdr=np.frombuffer(sdr_b, dtype=np.int32).copy(),
        created_at=float(created),
        updated_at=float(updated),
        reinforced_count=int(rcount),
        last_reinforced_at=float(rlast),
        base_strength=float(base),
        valid_from=float(vfrom),
        valid_to=None if vto is None else float(vto),
        superseded_by=None if sby is None else int(sby),
        scope=str(scope),
        author=str(author or ""),
    )
