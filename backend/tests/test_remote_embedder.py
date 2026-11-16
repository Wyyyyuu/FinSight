import pytest
import requests
from backend.rag.embedder import EmbeddingService


def payload(count=1, **overrides):
    return {"model": "bge-m3", "dim": 1024, "revision": "pinned-maxlen512-cpu",
            "dense": [[1.] + [0.] * 1023 for _ in range(count)],
            "sparse": [{"42": .5} for _ in range(count)], **overrides}


class Response:
    def __init__(self, data):
        self.data = data
    def raise_for_status(self):
        pass
    def json(self):
        return self.data


def test_remote_batches_and_retains_profile(monkeypatch):
    calls = []
    def post(url, json, timeout):
        calls.append(json["texts"])
        return Response(payload(len(json["texts"])))
    monkeypatch.setattr(requests, "post", post)
    svc = EmbeddingService(force_backend="bge-m3-http")
    result = svc.encode(["中文 revenue"] * 35)
    assert list(map(len, calls)) == [32, 3]
    assert len(result.dense) == len(result.sparse) == 35
    assert result.profile == "bge-m3:1024:pinned-maxlen512-cpu"
    assert not svc.is_hash_fallback


@pytest.mark.parametrize("override", [dict(model="hash"), dict(dim=96), dict(revision=""),
                                       dict(dense=[[float("nan")] * 1024]), dict(sparse=[]),
                                       dict(sparse=[{"42": float("inf")}])])
def test_remote_rejects_invalid_output(monkeypatch, override):
    monkeypatch.setattr(requests, "post", lambda *a, **kw: Response(payload(**override)))
    with pytest.raises(ValueError):
        EmbeddingService(force_backend="bge-m3-http").encode(["test"])


def test_remote_failure_never_falls_back(monkeypatch):
    def fail(*args, **kwargs):
        raise requests.ConnectionError("offline")
    monkeypatch.setattr(requests, "post", fail)
    with pytest.raises(requests.ConnectionError):
        EmbeddingService(force_backend="bge-m3-http").encode(["test"])


def test_revision_change_mid_batch_is_rejected(monkeypatch):
    outputs = iter([payload(32), payload(1, revision="other-model")])
    monkeypatch.setattr(requests, "post", lambda *a, **kw: Response(next(outputs)))
    with pytest.raises(ValueError, match="profile changed"):
        EmbeddingService(force_backend="bge-m3-http").encode(["test"] * 33)
