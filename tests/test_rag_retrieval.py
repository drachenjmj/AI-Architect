"""test_rag_retrieval.py - Offline unit tests for the RAG web-fallback path.

Covers the retrieve_chunks() decision boundary (KB hit vs. fallback trigger)
and the explicit fallback statuses introduced to end the silent-failure
problem: an empty, failed, or disabled fallback must never be logged as a
successful "web_fallback".

Everything is faked — no Chroma index, no API key, no network. The
Google-grounding client is replaced by a recording fake, which is also how
these tests prove the grounding request uses WEB_SEARCH_MODEL, not MODEL_NAME.
"""

from __future__ import annotations

from types import SimpleNamespace

import architect

CHUNK_KEYS = {"content", "source", "page", "box", "distance"}

# Several existing test modules permanently stub architect.retrieve_chunks
# with a direct assignment (no monkeypatch), so the module attribute cannot be
# trusted once the full suite is running. pytest imports every test module
# before executing any test, so the reference captured HERE is the real
# function; it still resolves get_vectorstore/_rag_logger/web_search_fallback
# through the architect module dict at call time, where our fakes live.
_retrieve_chunks = architect.retrieve_chunks
_web_search_fallback = architect.web_search_fallback


class FakeDoc:
    """LangChain-style document stand-in."""

    def __init__(self, content, source="kb_doc.pdf", page=1, box=1):
        self.page_content = content
        self.metadata = {"source": source, "page": page, "box": box}


class FakeVectorstore:
    """Chroma stand-in returning canned (doc, distance) rows."""

    def __init__(self, results):
        self.results = results
        self.queries = []

    def similarity_search_with_score(self, query, k=3):
        self.queries.append(query)
        return self.results[:k]


class FakeLogger:
    """RagLogger stand-in that records calls instead of touching the disk."""

    def __init__(self):
        self.calls = []

    def log_query(self, **kwargs):
        self.calls.append(kwargs)

    @property
    def statuses(self):
        return [c["status"] for c in self.calls]


class FakeWebFallback:
    """web_search_fallback stand-in: returns chunks or raises."""

    def __init__(self, chunks=None, error=None):
        self.chunks = chunks if chunks is not None else _web_chunks()
        self.error = error
        self.calls = []

    def __call__(self, query):
        self.calls.append(query)
        if self.error is not None:
            raise self.error
        return [dict(c) for c in self.chunks]


def _web_chunks(n=2):
    return [
        {
            "content": f"web finding {i}",
            "source": f"https://example.com/{i}",
            "page": 0,
            "box": 3,
            "distance": None,
        }
        for i in range(n)
    ]


def _setup(monkeypatch, results, fallback):
    """Wire fakes into architect and return (vectorstore, logger, fallback)."""
    vs = FakeVectorstore(results)
    logger = FakeLogger()
    monkeypatch.setattr(architect, "get_vectorstore", lambda: vs)
    monkeypatch.setattr(architect, "_rag_logger", logger)
    monkeypatch.setattr(architect, "web_search_fallback", fallback)
    monkeypatch.setattr(architect, "WEB_FALLBACK_ENABLED", True)
    return vs, logger, fallback


# ── trigger boundary ────────────────────────────────────────────────────────

def test_kb_hit_under_threshold_returns_kb_and_skips_fallback(monkeypatch):
    fallback = FakeWebFallback()
    _vs, logger, _fb = _setup(
        monkeypatch, [(FakeDoc("event-driven content"), 0.30)], fallback
    )

    chunks, origin = _retrieve_chunks("event-driven architecture")

    assert origin == "kb"
    assert fallback.calls == []
    assert chunks[0]["content"] == "event-driven content"
    assert chunks[0]["distance"] == 0.30
    for c in chunks:
        assert set(c) == CHUNK_KEYS
    assert logger.statuses == ["success"]


def test_no_chroma_results_triggers_fallback(monkeypatch):
    fallback = FakeWebFallback()
    _setup(monkeypatch, [], fallback)

    chunks, origin = _retrieve_chunks("quantum clustering")

    assert fallback.calls == ["quantum clustering"]
    assert origin == "web"
    assert chunks


def test_all_hits_above_threshold_trigger_fallback(monkeypatch):
    fallback = FakeWebFallback()
    far = FakeDoc("weak match", source="far.pdf")
    assert architect.DISTANCE_THRESHOLD is not None
    _setup(monkeypatch, [(far, architect.DISTANCE_THRESHOLD + 0.01)], fallback)

    chunks, origin = _retrieve_chunks("unrelated topic")

    assert fallback.calls == ["unrelated topic"]
    assert origin == "web"
    assert chunks


# ── fallback outcome statuses ───────────────────────────────────────────────

def test_successful_fallback_returns_web_and_logs_success(monkeypatch):
    fallback = FakeWebFallback(chunks=_web_chunks(2))
    _setup(monkeypatch, [], fallback)

    chunks, origin = _retrieve_chunks("gap topic")

    assert origin == "web"
    for c in chunks:
        assert set(c) == CHUNK_KEYS
    assert architect._rag_logger.statuses == ["no_results", "web_fallback"]
    last = architect._rag_logger.calls[-1]
    assert last["chunks_returned"] == 2
    assert last["source_files"] == [
        "https://example.com/0", "https://example.com/1",
    ]


def test_empty_fallback_returns_none_and_logs_empty_status(monkeypatch):
    fallback = FakeWebFallback(chunks=[])
    _setup(monkeypatch, [], fallback)

    chunks, origin = _retrieve_chunks("gap topic")

    assert origin == "none"
    assert chunks == []
    statuses = architect._rag_logger.statuses
    assert statuses[-1] == "web_fallback_empty"
    assert "web_fallback" not in statuses  # never logged as a success


def test_fallback_exception_returns_none_and_logs_error_status(monkeypatch):
    fallback = FakeWebFallback(error=RuntimeError("api down"))
    _setup(monkeypatch, [], fallback)

    chunks, origin = _retrieve_chunks("gap topic")

    assert origin == "none"
    assert chunks == []
    statuses = architect._rag_logger.statuses
    assert statuses[-1] == "web_fallback_error"
    assert "web_fallback" not in statuses
    err = architect._rag_logger.calls[-1]
    assert err["chunks_returned"] == 0 and err["source_files"] == []


def test_disabled_fallback_makes_no_external_call(monkeypatch):
    fallback = FakeWebFallback()
    _setup(monkeypatch, [], fallback)
    monkeypatch.setattr(architect, "WEB_FALLBACK_ENABLED", False)

    chunks, origin = _retrieve_chunks("gap topic")

    assert origin == "none"
    assert chunks == []
    assert fallback.calls == []
    assert architect._rag_logger.statuses[-1] == "web_fallback_disabled"


# ── grounding model routing & grounding-evidence requirement ────────────────
#
# Fakes mirror the google-genai response shape consumed by
# _extract_grounding_entries(): candidates[0].grounding_metadata with
# grounding_chunks[{web:{uri,title}}] and grounding_supports[{segment:{text},
# grounding_chunk_indices}]. Test URIs deliberately avoid "vertexaisearch" so
# _resolved() returns them directly and no URL resolution/network happens.

class FakeGroundingResponse:
    """Configurable grounding response; never touches the network."""

    def __init__(self, text="", chunks=(), supports=(), with_candidate=True):
        self.text = text
        gm = None
        if with_candidate:
            gm = SimpleNamespace(
                grounding_chunks=list(chunks),
                grounding_supports=list(supports),
            )
        self.candidates = [SimpleNamespace(grounding_metadata=gm)]


def _gchunk(uri, title=None):
    return SimpleNamespace(web=SimpleNamespace(uri=uri, title=title))


def _gsupport(text, indices):
    return SimpleNamespace(
        segment=SimpleNamespace(text=text),
        grounding_chunk_indices=list(indices),
    )


class FakeGroundingClient:
    """Records the generate_content kwargs; returns a canned response."""

    def __init__(self, response):
        self.response = response
        self.models = self
        self.requests = []

    def generate_content(self, *, model, contents, config):
        self.requests.append({"model": model, "contents": contents})
        return self.response


_GROUNDED_URI = "https://cloud.architecture.best/patterns/event-driven"


def _grounded_response():
    return FakeGroundingResponse(
        text="Event-driven services decouple components.",
        chunks=[_gchunk(_GROUNDED_URI, "Event-Driven Patterns")],
        supports=[_gsupport("Event-driven services decouple components.", [0])],
    )


def test_grounding_request_uses_web_search_model_not_main(monkeypatch):
    client = FakeGroundingClient(_grounded_response())
    monkeypatch.setattr(architect, "_get_web_client", lambda: client)

    chunks = _web_search_fallback("strangler fig pattern")

    assert len(client.requests) == 1
    assert client.requests[0]["model"] == architect.WEB_SEARCH_MODEL
    assert client.requests[0]["model"] != architect.MODEL_NAME
    # The main pipeline model stays exactly as before this refactor.
    assert architect.MODEL_NAME == "gemini-3.1-flash-lite"
    assert architect.WEB_SEARCH_MODEL == "gemini-2.5-flash"
    assert chunks and set(chunks[0]) == CHUNK_KEYS
    assert _GROUNDED_URI in chunks[0]["source"]


# ── grounding evidence is required for web_fallback success ─────────────────

def test_plain_ungrounded_text_yields_no_chunks():
    response = FakeGroundingResponse(
        text="A confident-sounding answer with no sources.",
        with_candidate=False,  # no candidates -> no grounding metadata
    )

    assert architect._extract_grounding_entries(response) == []


def test_ungrounded_text_with_empty_grounding_metadata_yields_no_chunks():
    response = FakeGroundingResponse(
        text="Still no sources.",  # candidate exists, but no grounding chunks
        chunks=[],
        supports=[],
    )

    assert architect._extract_grounding_entries(response) == []


def _setup_live_fallback(monkeypatch, client):
    """Wire a fake grounding client but leave the REAL web_search_fallback.

    Unlike _setup(), web_search_fallback is not stubbed, so the call path
    runs end-to-end (retrieve_chunks -> _apply_web_fallback ->
    web_search_fallback -> fake client) with no network.
    """
    monkeypatch.setattr(architect, "get_vectorstore", lambda: FakeVectorstore([]))
    monkeypatch.setattr(architect, "_rag_logger", FakeLogger())
    monkeypatch.setattr(architect, "WEB_FALLBACK_ENABLED", True)
    monkeypatch.setattr(architect, "_get_web_client", lambda: client)


def test_ungrounded_response_end_to_end_logs_empty_status(monkeypatch):
    client = FakeGroundingClient(
        FakeGroundingResponse(text="Answer text but zero grounding evidence.")
    )
    _setup_live_fallback(monkeypatch, client)  # KB miss triggers fallback

    chunks, origin = _retrieve_chunks("gap topic")

    assert origin == "none"
    assert chunks == []
    assert client.requests  # the fallback DID execute against the API...
    statuses = architect._rag_logger.statuses
    assert statuses[-1] == "web_fallback_empty"  # ...but is not a success
    assert "web_fallback" not in statuses


def test_grounded_response_end_to_end_logs_successful_web_fallback(monkeypatch):
    client = FakeGroundingClient(_grounded_response())
    _setup_live_fallback(monkeypatch, client)

    chunks, origin = _retrieve_chunks("event-driven architecture")

    assert origin == "web"
    assert chunks and set(chunks[0]) == CHUNK_KEYS
    assert _GROUNDED_URI in chunks[0]["source"]
    assert chunks[0]["source"] != "web_search"
    assert architect._rag_logger.statuses[-1] == "web_fallback"


def test_grounding_uri_without_supports_uses_full_text_with_real_source():
    """Allowed middle case: URI(s) exist, no per-segment supports."""
    response = FakeGroundingResponse(
        text="Summary grounded in real sources.",
        chunks=[_gchunk(_GROUNDED_URI, "Event-Driven Patterns")],
        supports=[],
    )

    entries = architect._extract_grounding_entries(response)

    assert len(entries) == 1
    assert _GROUNDED_URI in entries[0]["source"]
    assert entries[0]["content"].startswith("Summary grounded")
