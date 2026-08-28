"""test_token_accounting.py — regression tests for run-cost accounting (Kati).

THE BUG THESE EXIST TO CATCH
----------------------------
`llm_call` used to record usage by mutating `state.input_tokens` /
`state.output_tokens` in place, and no node ever RETURNED those fields.
LangGraph persists only what a node returns, so every count was discarded: an
11-call run ended with `input_tokens=0, output_tokens=0`. The knock-on effect
was that `refine_gate.evaluate_caps` compared 0 against MAX_TOTAL_TOKENS on
every visit, so the run-cost guardrail could never trip and only the iteration
cap did any work.

Nothing in the old suite noticed, because every test asserted on artifacts and
routing and none on cost. These tests close that gap at the level the bug lived
at — the FULL graph run — rather than at the unit level, since `llm_call` and
the nodes were each individually "fine" and only the wiring between them was
broken.

No API key or network: all four LLM-calling agents and the RAG retrieval are
stubbed. Each stubbed call reports a fixed, asymmetric usage
(`test_clarifier.fake_usage`), so expected totals are just calls x per-call.
"""
from __future__ import annotations

import pytest

import test_clarifier as tc
from pipeline import orchestrator, refine_gate
from pipeline.agents import architect as arch
from pipeline.agents import clarifier as clar
from pipeline.agents import repo_ingestor as ingest
from pipeline.agents import reviewer as rev
from pipeline.llm import MODELS, LLMError, LLMUsage, attach_usage, estimate_cost_usd
from pipeline.refine_gate import MAX_REFINE_ITERATIONS, evaluate_caps
from pipeline.state import ContextRecord, Stage, new_run

PROMPT = "Fix our monolithic online shop so it survives seasonal peak load."

IN = tc.FAKE_INPUT_TOKENS      # 1000 per stubbed call
OUT = tc.FAKE_OUTPUT_TOKENS    # 500 per stubbed call


# ── stubbing helpers ─────────────────────────────────────────────────────
def _install_stubs(monkeypatch) -> list[str]:
    """Stub every LLM call and the RAG lookup; return a live call log.

    `monkeypatch` (rather than the bare module assignment other test files
    use) so these stubs are undone afterwards and cannot leak into another
    test file's expectations.

    The returned list grows by one entry per LLM call, which is what lets a
    test say "the totals must equal what the stubs handed out" without
    hard-coding a call count that would rot the moment the graph changes.
    """
    calls: list[str] = []

    def _log(label, build):
        def stub(state, prompt, *, system="", model="flash-lite", response_schema=None, thinking_level=None):
            calls.append(label)
            value = build(state, prompt, response_schema)
            return value, tc.fake_usage()

        return stub

    def _clarifier(state, prompt, response_schema):
        # Missing critical facts until answers exist, then complete — which is
        # what makes a pause-then-resume run possible without a human.
        canned = tc._complete if state.clarification_answers else tc._missing
        return canned(state, prompt)[0]  # canned responses return (value, usage)

    def _architect(state, prompt, response_schema):
        return tc._architect_response(state, prompt, response_schema=response_schema)[0]

    monkeypatch.setattr(clar, "llm_call", _log("clarifier", _clarifier))
    monkeypatch.setattr(arch, "llm_call", _log("architect", _architect))
    monkeypatch.setattr(
        rev, "llm_call", _log("reviewer", lambda s, p, rs: rev.LLMJudgments())
    )
    monkeypatch.setattr(
        ingest, "llm_call", _log("repo_ingestor", lambda s, p, rs: None)
    )

    import architect as legacy_architect

    monkeypatch.setattr(
        legacy_architect,
        "retrieve_chunks",
        lambda query, k=3: (
            [{"content": "Use asynchronous processing to absorb peak traffic.",
              "source": "offline-test-kb", "page": 1, "box": 1, "distance": 0.1}],
            "offline-test",
        ),
    )
    return calls


def _seed_designable_state(**overrides):
    """A state parked at RESEARCHING with a locked context, so the Architect
    runs first without needing the clarifier or the researcher.
    """
    state = new_run(PROMPT)
    state.context_record = ContextRecord(
        project_name="Seasonal Shop",
        summary="cloud: AWS; scale: 50k peak users; compliance: GDPR",
    )
    state.stage = Stage.RESEARCHING
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


# ── 1. THE BUG: usage must survive a full graph run ──────────────────────
def test_usage_survives_a_full_graph_run(monkeypatch):
    """Final state totals == what the stubs handed out.

    This is the actual regression. Before the fix it failed with
    `0 == 16500`: the nodes never returned their usage, so LangGraph dropped
    every count on the floor.
    """
    calls = _install_stubs(monkeypatch)

    state = orchestrator.run_pipeline(_seed_designable_state())

    assert calls, "no LLM calls were made — the stubs are not wired up"
    assert state.input_tokens == len(calls) * IN
    assert state.output_tokens == len(calls) * OUT
    assert state.input_tokens + state.output_tokens == len(calls) * (IN + OUT)


# ── 2. the token cap must genuinely trip, for the TOKEN reason ───────────
def test_token_cap_stops_the_refine_loop(monkeypatch):
    """The run-cost guardrail stops the loop on the BUDGET, not on iterations.

    Distinguishing the two matters: while accounting was broken the iteration
    cap silently did all the work, and a test that only asserted
    `stopped_on_cap` would have passed against the bug.
    """
    _install_stubs(monkeypatch)

    # Reaching the first gate costs architect (2 calls) + reviewer (1) = 4500
    # tokens, so a 4000 ceiling is already breached on the gate's FIRST visit —
    # before a single refine iteration has been spent.
    monkeypatch.setattr(refine_gate, "MAX_TOTAL_TOKENS", 4000)

    state = orchestrator.run_pipeline(_seed_designable_state())

    assert state.stage is Stage.DONE          # graceful, not FAILED
    assert state.stopped_on_cap is True
    # THE point of this test: stopped on the budget, and demonstrably not on
    # the iteration cap, which never got the chance to count past zero.
    assert "token_budget" in state.history[-1].note
    assert "max_iterations" not in state.history[-1].note
    assert state.refine_iterations == 0
    assert state.refine_iterations < MAX_REFINE_ITERATIONS


def test_evaluate_caps_reads_real_totals(monkeypatch):
    """`evaluate_caps` needed no change — it simply starts seeing real numbers."""
    monkeypatch.setattr(refine_gate, "MAX_TOTAL_TOKENS", 4000)

    fresh = new_run(PROMPT)
    assert evaluate_caps(fresh) == (False, "")

    over = new_run(PROMPT)
    over.input_tokens = 3000
    over.output_tokens = 1500
    stop, reason = evaluate_caps(over)
    assert stop and "token_budget" in reason


# ── 3. per-agent attribution must reconcile with the run totals ──────────
def test_per_agent_totals_sum_to_state_totals(monkeypatch):
    """`usage_by_agent()` is a groupby over `history` — it must reconcile.

    If a node ever stamped one number on its StepLog and returned another, the
    report's per-agent breakdown would quietly disagree with its headline
    total. This pins them together.
    """
    calls = _install_stubs(monkeypatch)

    state = orchestrator.run_pipeline(_seed_designable_state())
    per_agent = state.usage_by_agent()

    assert sum(u.input_tokens for u in per_agent.values()) == state.input_tokens
    assert sum(u.output_tokens for u in per_agent.values()) == state.output_tokens
    assert per_agent["architect"].total_tokens > 0
    assert per_agent["reviewer"].total_tokens > 0
    # Attribution is per agent, not per call — but the two no longer sit in a
    # fixed ratio. The Architect calls TWICE on the initial design and ONCE per
    # refine pass, because phase 1 is skipped when refining so that feature IDs
    # survive the loop (see agents/architect.py); the Reviewer calls once per
    # visit throughout. Deriving both from the live call log rather than from a
    # ratio is what keeps this test honest the next time either one changes.
    assert calls.count("architect") == 2 + MAX_REFINE_ITERATIONS
    assert per_agent["architect"].input_tokens == calls.count("architect") * IN
    assert per_agent["reviewer"].input_tokens == calls.count("reviewer") * IN
    # An agent that calls no LLM stays honestly at zero.
    assert per_agent["refine_gate"].total_tokens == 0
    assert sum(u.cost_usd for u in per_agent.values()) == pytest.approx(
        state.total_cost_usd()
    )
    assert len(calls) * (IN + OUT) == sum(u.total_tokens for u in per_agent.values())


# ── 4. cost must actually be computed for a known model ──────────────────
def test_cost_is_computed_for_a_known_model():
    """A real model ID gets a real, non-zero, direction-aware price."""
    model_id = MODELS["flash-lite"]

    cost = estimate_cost_usd(model_id, 1_000_000, 1_000_000)
    assert cost is not None
    assert cost > 0

    # Output is priced higher than input for both models we ship, so the two
    # directions must not be collapsed into one blended rate.
    input_only = estimate_cost_usd(model_id, 1_000_000, 0)
    output_only = estimate_cost_usd(model_id, 0, 1_000_000)
    assert output_only > input_only > 0
    assert cost == pytest.approx(input_only + output_only)

    # Zero tokens cost nothing; an unverified model reports UNKNOWN, not free.
    assert estimate_cost_usd(model_id, 0, 0) == 0
    assert estimate_cost_usd("not-a-real-model", 1_000, 1_000) is None


def test_run_reports_non_zero_cost(monkeypatch):
    """The end-to-end figure that goes into the report is non-zero and adds up."""
    _install_stubs(monkeypatch)

    state = orchestrator.run_pipeline(_seed_designable_state())

    assert state.total_cost_usd() > 0
    assert state.total_cost_usd() == pytest.approx(
        sum(step.cost_usd for step in state.history)
    )
    # Every step that spent tokens records which model spent them.
    for step in state.history:
        if step.input_tokens or step.output_tokens:
            assert step.model
            assert step.cost_usd > 0


# ── 5. resume must accumulate, not double-count ──────────────────────────
def test_resume_accumulates_without_double_counting(monkeypatch):
    """Two successive `run_pipeline` calls add up exactly once.

    The UI resumes a paused run by re-invoking the graph with the SAME state
    object, so the reducers see a non-zero starting value. A reducer that
    re-applied the seed — or a node that returned a running total instead of
    its own delta — would double-count here.
    """
    calls = _install_stubs(monkeypatch)

    state = new_run(PROMPT)
    state = orchestrator.run_pipeline(state)      # pass 1 — pauses for answers
    assert state.stage is Stage.AWAITING_HUMAN

    after_first = len(calls)
    assert after_first > 0
    assert state.input_tokens == after_first * IN
    assert state.output_tokens == after_first * OUT

    state.clarification_answers = {q: "some answer" for q in state.clarifying_questions}
    state = orchestrator.run_pipeline(state)      # pass 2 — resumes to DONE
    assert state.stage is Stage.DONE

    # Totals cover BOTH passes, counted exactly once each.
    assert len(calls) > after_first, "the resume pass made no calls"
    assert state.input_tokens == len(calls) * IN
    assert state.output_tokens == len(calls) * OUT
    # The per-agent view spans both passes too, and still reconciles.
    per_agent = state.usage_by_agent()
    assert sum(u.input_tokens for u in per_agent.values()) == state.input_tokens


# ── 6. the @node error path: keep partial usage, count it once ───────────
def test_failed_node_keeps_already_billed_tokens(monkeypatch):
    """An agent that raises AFTER a successful call must not lose that call.

    The Architect validates its phase-1 result and can raise before phase 2
    ever runs. Those tokens were really spent, so the FAILED update has to
    carry them — exactly once.
    """
    _install_stubs(monkeypatch)

    def _fail_on_second_call(state, prompt, *, system="", model="", response_schema=None, thinking_level=None):
        if response_schema is arch.FeatureDesign:
            return tc._architect_response(
                state, prompt, response_schema=response_schema
            )[0], tc.fake_usage()
        raise RuntimeError("phase 2 exploded")

    monkeypatch.setattr(arch, "llm_call", _fail_on_second_call)

    out = arch.architect_node(_seed_designable_state())

    assert out["stage"] is Stage.FAILED
    # Phase 1 was billed and is kept; phase 2 never happened and is not invented.
    assert out["input_tokens"] == IN
    assert out["output_tokens"] == OUT
    assert out["history"][0].input_tokens == IN
    assert out["history"][0].output_tokens == OUT


def test_failed_node_does_not_double_count_carried_usage(monkeypatch):
    """Usage attached by `llm_call` itself merges with the node's — never twice.

    A call can be billed and still unusable (the model returns nothing
    schema-valid). `llm_call` attaches that usage to the error it raises, and
    the node attaches its own tally on the way out; the merge has to add them,
    not stack the same call twice.
    """
    _install_stubs(monkeypatch)

    def _billed_but_unusable(state, prompt, *, system="", model="", response_schema=None, thinking_level=None):
        if response_schema is arch.FeatureDesign:
            return tc._architect_response(
                state, prompt, response_schema=response_schema
            )[0], tc.fake_usage()
        err = LLMError("no schema-valid JSON")
        attach_usage(err, tc.fake_usage())  # phase 2 was billed anyway
        raise err

    monkeypatch.setattr(arch, "llm_call", _billed_but_unusable)

    out = arch.architect_node(_seed_designable_state())

    assert out["stage"] is Stage.FAILED
    # TWO billed calls: phase 1 (returned) + phase 2 (raised). Not one, not three.
    assert out["input_tokens"] == 2 * IN
    assert out["output_tokens"] == 2 * OUT


def test_sum_usage_reports_unknown_cost_honestly():
    """A total that includes an unpriced call is None, not a silent undercount."""
    from pipeline.llm import sum_usage

    priced = tc.fake_usage()
    unpriced = LLMUsage(model="not-a-real-model", input_tokens=10, output_tokens=5)

    assert sum_usage([priced, priced]).cost_usd == pytest.approx(2 * priced.cost_usd)
    assert sum_usage([priced, unpriced]).cost_usd is None
    assert sum_usage([priced, unpriced]).input_tokens == priced.input_tokens + 10
