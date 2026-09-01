"""Упрочнение v0.8.3: сетевые демоны отвечают чистыми 4xx на битый вход
(лимит тела, не-объектный JSON, NaN/Infinity, битые векторы), общий
HTTP-каркас с потолком потоков, честный выбор эмбеддера в командном recall,
семантика нулевых значений в политике шеринга."""
import base64
import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from realmemory import Hippocampus, MemoryConfig
from realmemory.team._http import cosine_scores
from realmemory.team.coordinator import make_server as make_coordinator
from realmemory.team.peer import make_peer_server
from realmemory.team.policy import ProjectRule, load_policy
from realmemory.team.recall_team import embed_query_text


def _cfg(**over) -> MemoryConfig:
    fields = {
        "dim": 256, "n_units": 512, "k_sparse": 48, "sdr_seed": 5,
        "bucket_cap": 32,
        "tau_episodic": 60 * 86400.0,
        "tau_semantic": 600 * 86400.0,
        "gc_grace_below_floor_s": 5 * 86400.0,
    }
    fields.update(over)
    cfg = MemoryConfig(**fields)
    cfg.validate()
    return cfg


def _vec(seed: int, dim: int = 384) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


def _post_raw(url, body: bytes, token: str | None = "sekret"):
    """POST без конвертов: статус и разобранный ответ даже при 4xx/5xx."""
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        return exc.code, json.loads(raw) if raw else {}


def _serve(srv):
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_address[1]}"


# -- битый вход: чистые 400/413 вместо 500 и обрывов ---------------------------------

def test_peer_rejects_oversized_body(tmp_path):
    srv = make_peer_server(tmp_path / "brain", host="127.0.0.1", port=0)
    url = _serve(srv)
    handler_cls = srv.RequestHandlerClass  # класс обработчика этого сервера
    saved_limit = handler_cls.max_body_bytes
    handler_cls.max_body_bytes = 64  # низкий потолок: гигантское тело не слать
    try:
        big = json.dumps({"embedder": "x", "k": 5, "pad": "x" * 4096}).encode()
        status, payload = _post_raw(f"{url}/recall", big)
        assert status == 413
        assert "лимита" in payload["error"]
    finally:
        handler_cls.max_body_bytes = saved_limit
        srv.shutdown()
        srv.server_close()


def test_peer_rejects_non_object_json(tmp_path):
    srv = make_peer_server(tmp_path / "brain", host="127.0.0.1", port=0)
    url = _serve(srv)
    try:
        status, payload = _post_raw(f"{url}/recall", b"[1,2,3]")
        assert status == 400
        assert "JSON-объект" in payload["error"]
    finally:
        srv.shutdown()
        srv.server_close()


def test_coordinator_rejects_nan_and_bad_vector(tmp_path):
    srv = make_coordinator(tmp_path / "coord", host="127.0.0.1", port=0,
                           token="sekret")
    url = _serve(srv)
    try:
        # NaN-литерал в published_at: в базу попадал бы NULL/мусор
        item = {"publication_id": "p1", "project": "proj", "author": "a",
                "text": "текст",
                "embedding_b64": base64.b64encode(_vec(1).tobytes()).decode(),
                "published_at": float("nan"), "content_hash": "h",
                "embedder": "fastembed:t"}
        raw = json.dumps({"items": [item]}).encode()  # python-json пишет NaN литералом
        status, payload = _post_raw(f"{url}/publish", raw)
        assert status == 400
        assert "NaN" in payload["error"]

        # битый base64 вектора запроса
        status, _ = _post_raw(f"{url}/search", json.dumps(
            {"query_embedding_b64": "!!!не-base64!!!", "k": 3,
             "embedder": "fastembed:t"}).encode())
        assert status == 400
    finally:
        srv.shutdown()
        srv.server_close()


def test_coordinator_search_skips_wrong_dim_rows(tmp_path):
    """Строки с другой размерностью несравнимы: поиск отвечает пусто (200),
    а не падает 500-й от np.dot."""
    srv = make_coordinator(tmp_path / "coord", host="127.0.0.1", port=0,
                           token="sekret")
    url = _serve(srv)
    try:
        item = {"publication_id": "p1", "project": "proj", "author": "a",
                "text": "решение о кэше rk401",
                "embedding_b64": base64.b64encode(_vec(2).tobytes()).decode(),
                "published_at": 1.0, "content_hash": "h", "embedder": "fastembed:t"}
        assert _post_raw(f"{url}/publish", json.dumps({"items": [item]}).encode())[0] == 200
        status, payload = _post_raw(f"{url}/search", json.dumps({
            "query_embedding_b64": base64.b64encode(_vec(3, dim=4).tobytes()).decode(),
            "k": 3, "embedder": "fastembed:t"}).encode())
        assert status == 200
        assert payload["hits"] == []
    finally:
        srv.shutdown()
        srv.server_close()


def test_cosine_scores_zero_norms_are_zero():
    q = np.asarray([1.0, 0.0], dtype=np.float32)
    scores = cosine_scores(q, [np.asarray([2.0, 0.0], dtype=np.float32),
                               np.zeros(2, dtype=np.float32),
                               np.asarray([0.0, 5.0], dtype=np.float32)])
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == 0.0
    assert scores[2] == 0.0
    assert cosine_scores(q, []).size == 0


# -- политика: 0 и пустой список — легитимные значения, а не «наследовать» ------------

def test_policy_zero_min_reinforcements_and_empty_kinds(tmp_path):
    path = tmp_path / "team.yaml"
    path.write_text(
        "default_kinds: [semantic]\n"
        "min_reinforcements: 2\n"
        "projects:\n"
        "  - name: proj\n"
        "    kinds: []\n"
        "    min_reinforcements: 0\n",
        encoding="utf-8")
    policy = load_policy(path)
    rule = policy.project_rule("proj")
    assert policy.min_reinforcements_for(rule) == 0
    assert policy.kinds_for(rule) == []
    # None в правиле — унаследовать глобальные
    other = ProjectRule(name="other")
    assert policy.min_reinforcements_for(other) == 2
    assert policy.kinds_for(other) == ["semantic"]


# -- командный recall: эмбеддер выбирается честно -------------------------------------

def test_embed_query_text_unknown_embedder_raises(tmp_path):
    class Weird:
        dim = 64
        name = "weird:v1"

        def embed(self, text):
            return np.zeros(64, dtype=np.float32)

    root = tmp_path / "rm"
    h = Hippocampus.open(root, config=_cfg(dim=64, n_units=256, k_sparse=32),
                         embedder=Weird())
    try:
        h.remember("факт для маркера эмбеддера wk303")
    finally:
        h.close()
    with pytest.raises(RuntimeError, match="weird"):
        embed_query_text(root, "запрос")


def test_embed_query_text_hashing_brain_roundtrip(tmp_path):
    root = tmp_path / "rm"
    from realmemory.encoding.embedder import HashingEmbedder

    h = Hippocampus.open(root, config=_cfg(dim=64, n_units=256, k_sparse=32),
                         embedder=HashingEmbedder(dim=64))
    try:
        h.remember("факт для раундтрипа hk304")
    finally:
        h.close()
    _vec_out, name = embed_query_text(root, "запрос")
    assert name == HashingEmbedder(dim=64).name


def test_embed_query_text_empty_brain_defaults_to_hashing(tmp_path):
    _vec_out, name = embed_query_text(tmp_path / "нет-мозга", "вопрос")
    assert name.startswith("hashing(")
