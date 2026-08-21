"""test_ui_context_record.py - Focused regression test for the Context Record
version TypeError fix.

Scope: the RENDERING boundary only. A checkpointed run can carry
ContextRecord.version as a numeric string ("1") — the context gate's edit path
stores text values without schema validation — and `render_context_record`
used to crash on `record.version <= 1` with
`TypeError: '<=' not supported between instances of 'str' and 'int'`.

The version normalization helper is pure, so its cases run without Streamlit.
The full `render_context_record` smoke test runs inside Streamlit's bare-mode
proxy (no server, no browser): st calls become no-op widgets, which is enough
to prove the DONE screen no longer throws on the checkpointed run's shape.
"""

from __future__ import annotations

from pipeline.state import ArchitectState, ContextRecord, Stage, new_run
import ui_sections


def _record(version) -> ContextRecord:
    record = ContextRecord(project_name="E-Commerce Platform")
    # The bug reproduces only WITHOUT validation: the gate's edit path
    # (apply_user_edits) setattr's plain strings, and Pydantic v2 does not
    # validate assignments. object.__setattr__ is not needed — plain set works
    # the same way here.
    record.version = version
    return record


def _state(record: ContextRecord) -> ArchitectState:
    state = new_run("Modernize an e-commerce platform.")
    state.stage = Stage.DONE
    state.context_record = record
    return state


# ── the pure normalizer ─────────────────────────────────────────────────────

def test_integer_version_is_preserved():
    for v in (1, 2, 7):
        assert ui_sections._version_number(_record(v)) == v


def test_numeric_string_version_behaves_like_int():
    assert ui_sections._version_number(_record("1")) == 1
    assert ui_sections._version_number(_record("2")) == 2


def test_malformed_version_falls_back_to_one():
    # Smallest safe fallback consistent with the v1 UI branch: never crash
    # the DONE screen over a display label.
    for bad in ("", "one", None, [], {}):
        assert ui_sections._version_number(_record(bad)) == 1


# ── the rendering path no longer raises ─────────────────────────────────────

def test_render_with_integer_version():
    ui_sections.render_context_record(_state(_record(1)))      # v1 branch
    ui_sections.render_context_record(_state(_record(3)))      # later-version branch


def test_render_with_numeric_string_version():
    # The exact checkpointed shape from run 20260821T103837Z-a7ae0a73:
    # version "1" as a string used to raise TypeError on the DONE screen.
    ui_sections.render_context_record(_state(_record("1")))
    ui_sections.render_context_record(_state(_record("2")))


def test_render_with_malformed_version_does_not_crash():
    ui_sections.render_context_record(_state(_record("garbage")))
