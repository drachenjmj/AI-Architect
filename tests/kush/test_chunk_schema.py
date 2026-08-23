"""test_chunk_schema.py - Contract tests for chunk_schema.json.

The TOP-LEVEL schema validates one flat runtime Chunk
({content, source, page, box, distance}); RetrievalResponse is defined as the
JSON-equivalent two-item array of the Python (chunks, origin) tuple returned
by architect.retrieve_chunks(). Every reject case below uses
pytest.raises(jsonschema.ValidationError), so a schema that silently accepted
everything would fail loudly.

All validation goes through EXPLICIT Draft7Validator instances: plain
jsonschema.validate() would fall back to the 2020-12 draft for subschemas
without their own $schema, and 2020-12 rejects Draft-07's positional
array-form `items` used by RetrievalResponse.

jsonschema (4.x) is already in the venv as a chromadb dependency; runtime-
alignment tests reuse the offline fakes from test_rag_retrieval.py, so no
network, no Chroma store, no API key.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

import architect
import test_rag_retrieval as rag_fakes

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "chunk_schema.json"  # repo root
SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

# Draft-07 validators for the two contracts in the file.
CHUNK_VALIDATOR = jsonschema.Draft7Validator(SCHEMA)
RR_VALIDATOR = jsonschema.Draft7Validator({
    "$schema": "http://json-schema.org/draft-07/schema#",
    "$ref": "#/definitions/RetrievalResponse",
    "definitions": SCHEMA["definitions"],
})

RUNTIME_KEYS = {"content", "source", "page", "box", "distance"}


def _kb_chunk():
    return {
        "content": "Event-driven microservices support independent scaling.",
        "source": "microservices-on-aws.pdf",
        "page": 4,
        "box": 1,
        "distance": 0.42,
    }


def _web_chunk():
    return {
        "content": "Current best practice: ...",
        "source": "https://example.com/patterns",
        "page": 0,
        "box": 3,
        "distance": None,
    }


# ── schema is a valid Draft-07 schema ───────────────────────────────────────

def test_schema_passes_draft7_check_schema():
    jsonschema.Draft7Validator.check_schema(SCHEMA)


def test_top_level_targets_chunk_definition():
    assert SCHEMA["$ref"] == "#/definitions/Chunk"
    assert set(SCHEMA["definitions"]["Chunk"]["required"]) == RUNTIME_KEYS


# ── top-level validation: accepts what it must ──────────────────────────────

def test_valid_kb_chunk_with_numeric_distance_validates():
    CHUNK_VALIDATOR.validate(_kb_chunk())


def test_valid_box3_web_chunk_with_null_distance_validates():
    CHUNK_VALIDATOR.validate(_web_chunk())


# ── top-level validation: rejects what it must ──────────────────────────────

def test_missing_distance_is_rejected():
    bad = _kb_chunk()
    del bad["distance"]
    with pytest.raises(jsonschema.ValidationError):
        CHUNK_VALIDATOR.validate(bad)


def test_arbitrary_object_is_rejected():
    with pytest.raises(jsonschema.ValidationError):
        CHUNK_VALIDATOR.validate({"foo": "bar"})


def test_extra_field_is_rejected():
    bad = _kb_chunk()
    bad["id"] = "CHUNK-001"
    with pytest.raises(jsonschema.ValidationError):
        CHUNK_VALIDATOR.validate(bad)


def test_obsolete_nested_chunk_distance_shape_is_rejected():
    nested = {
        "chunk": {
            "content": "x", "source": "y", "page": 1, "box": 1,
        },
        "distance": 0.1,
    }
    with pytest.raises(jsonschema.ValidationError):
        CHUNK_VALIDATOR.validate(nested)


@pytest.mark.parametrize("bad_box", [0, 4, "2", None])
def test_box_outside_1_to_3_is_rejected(bad_box):
    bad = _kb_chunk()
    bad["box"] = bad_box
    with pytest.raises(jsonschema.ValidationError):
        CHUNK_VALIDATOR.validate(bad)


# ── runtime output stays schema-aligned (fakes, no network) ─────────────────

def test_runtime_kb_output_validates_top_level(monkeypatch):
    doc = rag_fakes.FakeDoc("event-driven content", source="kb.pdf", page=2)
    monkeypatch.setattr(
        architect, "get_vectorstore",
        lambda: rag_fakes.FakeVectorstore([(doc, 0.30)]),
    )
    monkeypatch.setattr(architect, "_rag_logger", rag_fakes.FakeLogger())

    chunks, origin = rag_fakes._retrieve_chunks("event-driven architecture")

    assert origin == "kb"
    for c in chunks:
        assert set(c) == RUNTIME_KEYS
        CHUNK_VALIDATOR.validate(c)  # each real KB chunk passes top level
    RR_VALIDATOR.validate([chunks, origin])


def test_runtime_web_output_validates_top_level(monkeypatch):
    monkeypatch.setattr(
        architect, "web_search_fallback",
        rag_fakes.FakeWebFallback(rag_fakes._web_chunks(1)),
    )
    monkeypatch.setattr(
        architect, "get_vectorstore", lambda: rag_fakes.FakeVectorstore([]),
    )
    monkeypatch.setattr(architect, "_rag_logger", rag_fakes.FakeLogger())
    monkeypatch.setattr(architect, "WEB_FALLBACK_ENABLED", True)

    chunks, origin = rag_fakes._retrieve_chunks("gap topic")

    assert origin == "web"
    for c in chunks:
        assert c["box"] == 3 and c["distance"] is None
        CHUNK_VALIDATOR.validate(c)  # each real web chunk passes top level
    RR_VALIDATOR.validate([chunks, origin])


# ── RetrievalResponse = JSON-equivalent of the Python tuple ─────────────────

def test_retrieval_response_accepts_tuple_equivalent():
    RR_VALIDATOR.validate([[_kb_chunk(), _web_chunk()], "web"])
    RR_VALIDATOR.validate([[], "none"])


def test_retrieval_response_rejects_invented_object_form():
    with pytest.raises(jsonschema.ValidationError):
        RR_VALIDATOR.validate({"chunks": [_kb_chunk()], "origin": "kb"})


def test_retrieval_response_rejects_invalid_origin():
    with pytest.raises(jsonschema.ValidationError):
        RR_VALIDATOR.validate([[], "INVALID"])
