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


# ── the optional-context round (Clarifier UX cleanup) ───────────────────────
#
# `render_optional_context` and `render_context_approval` both use `st.form`,
# and calling a function that opens an `st.form` OUTSIDE a real Streamlit
# script-run (bare mode, as the version tests above do for the form-free
# `render_context_record`) leaks Streamlit's form-id tracking into every
# later AppTest in the process — exactly the hazard test_ui_workspace.py's
# docstring documents for importing ui.py itself. So only the DEFENSIVE
# early-return paths (no record locked yet; nothing left to ask) are smoke
# tested here, since neither ever reaches `st.form`. The FIELD-MAPPING
# behaviour these panels exist to make safe is pinned at the state level in
# test_context_gate.py, where a real `ContextEdits` is constructed and
# applied without needing a live browser to click a button.

def test_render_optional_context_with_nothing_to_ask_returns_skip():
    """Every `OPTIONAL_CONTEXT_FIELDS` entry already has a value -> a
    defensive no-op skip, never an empty form (so this path never opens
    `st.form`). `optional_slots` offers a field whenever it is empty AND not
    currently required (see `clarifier.optional_slots`) -- it does NOT
    require prompt signal, so leaving these fields unset here would make
    every one of them show up rather than none."""
    state = new_run("Build an internal admin tool.")  # no NFR/cloud/etc. signal
    state.context_record = ContextRecord(
        project_name="Admin Tool",
        business_goal="g",
        problem_statement="p",
        functional_requirements=["Manage internal records"],
        non_functional_requirements=["Single small internal team; no scale target"],
        cloud_provider="On-prem, no preference",
        budget="No specific budget constraint",
        compliance_requirements=["None applicable"],
    )
    assert ui_sections.render_optional_context(state) == ("skip", None)


def test_render_optional_context_without_a_record_returns_none():
    state = new_run("Build a system to sell sneakers online.")
    assert ui_sections.render_optional_context(state) is None


def test_render_context_approval_without_a_record_returns_none():
    state = new_run("Build a system to sell sneakers online.")
    assert ui_sections.render_context_approval(state) is None
