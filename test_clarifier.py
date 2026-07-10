"""test_clarifier.py — offline unit tests for the Clarifier (Kati).

No API key or network needed: the LLM is mocked by replacing `llm_call` with a
function that returns a canned `ClarificationResult`. This isolates the part we
own — the deterministic gate and the pause/resume routing — from the model.

Run either way:
    python -m pytest test_clarifier.py
    python test_clarifier.py
"""
from __future__ import annotations

from pipeline.state import new_run, Stage, ClarificationResult, ClarifyingQuestion, ContextField, ReviewResult
from pipeline.agents import clarifier as clar
from pipeline.agents import reviewer as rev
from pipeline import orchestrator

PROMPT = "Build me a system to sell sneakers online."


# ── canned LLM judgments ────────────────────────────────────────────────
def _missing(state, prompt, *, system="", model="", response_schema=None):
    """Something architecture-critical is unknown."""
    return ClarificationResult(
        captured_context=[ContextField(key="domain", value="e-commerce")],
        assumptions=["Assume REST API (low-stakes)."],
        questions=[
            ClarifyingQuestion(question="Expected peak users?", why_needed="Sets scale."),
            ClarifyingQuestion(question="GDPR in scope?", why_needed="Compliance shapes design."),
        ],
        missing_critical=["expected scale", "compliance"],
    )


def _complete(state, prompt, *, system="", model="", response_schema=None):
    """Everything critical is known."""
    return ClarificationResult(
        captured_context=[
            ContextField(key="domain", value="e-commerce"),
            ContextField(key="scale", value="50k peak"),
            ContextField(key="cloud", value="AWS"),
        ],
        assumptions=["Assume English-only UI (low-stakes)."],
        questions=[],
        missing_critical=[],
    )


# ── tests ────────────────────────────────────────────────────────────────
def test_pauses_when_critical_missing():
    clar.llm_call = _missing
    out = clar.clarifier_node(new_run(PROMPT))
    assert out["stage"] is Stage.AWAITING_INPUT
    assert out["clarifying_questions"] == ["Expected peak users?", "GDPR in scope?"]
    assert "context_record" not in out  # must NOT lock context while paused


def test_advances_and_locks_context_when_complete():
    clar.llm_call = _complete
    out = clar.clarifier_node(new_run(PROMPT))
    assert out["stage"] is Stage.CLARIFYING
    assert out["context_record"] is not None
    assert "scale: 50k peak" in out["context_record"].summary
    assert "Assumptions" in out["context_record"].summary
    assert out["clarifying_questions"] == []


def test_full_pause_then_resume():
    """First pass pauses; after answers are supplied, resume runs to the end.

    Exercises the orchestrator's entry-router (resume re-enters the clarifier).
    The real Reviewer (mocked LLM) correctly fails the architect STUB's
    placeholder design, so the run now ends in REFINING, not DONE — the refine
    loop itself is wired in W3.
    """
    def _stateful(state, prompt, *, system="", model="", response_schema=None):
        return _complete(state, prompt) if state.clarification_answers else _missing(state, prompt)
    clar.llm_call = _stateful
    rev.llm_call = lambda state, prompt, **kw: ReviewResult()  # canned neutral judgment

    s = new_run(PROMPT)
    s = orchestrator.run_pipeline(s)
    assert s.stage is Stage.AWAITING_INPUT
    assert s.clarifying_questions
    assert s.context_record is None

    s.clarification_answers = {q: "some answer" for q in s.clarifying_questions}
    s = orchestrator.run_pipeline(s)
    assert s.stage is Stage.REFINING  # reviewer rightly rejects the stub design
    assert s.review is not None and s.review.requires_refinement
    assert s.context_record is not None
    agents_run = [h.agent for h in s.history]
    assert "researcher" in agents_run and "reviewer" in agents_run


if __name__ == "__main__":
    test_pauses_when_critical_missing()
    print("PASS  pauses when critical missing")
    test_advances_and_locks_context_when_complete()
    print("PASS  advances + locks ContextRecord when complete")
    test_full_pause_then_resume()
    print("PASS  full pause -> resume -> DONE")
    print("\nALL CLARIFIER TESTS PASSED")
