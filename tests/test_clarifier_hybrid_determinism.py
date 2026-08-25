"""test_clarifier_hybrid_determinism.py — offline tests for the Clarifier's
deterministic decision boundary: which critical fields are actually relevant,
what happens to the Clarifier's own free-form `assumptions` text, what an
ask-cap gap is and is not allowed to do to a ContextRecord field, what an
explicit human "you recommend" request is and is not allowed to produce,
which cloud-relevance signal is a false positive, why the safe-assumption
check has to look at a WHOLE line rather than a fragment of one, why striking
a safe-field recommendation has to clear the field too, and what happens when
a requested recommendation comes back with nothing to propose.

Eight sections below:

  A. RELEVANCE POLICY — only `business_goal`, `problem_statement` and
     `functional_requirements` are unconditionally critical.
     `non_functional_requirements`, `cloud_provider`, `budget`,
     `compliance_requirements` and `existing_systems` are each gated by a
     small fixed vocabulary checked against the raw prompt PLUS any
     clarification answers gathered so far (`clarifier._slot_is_relevant`,
     `clarifier._signal_text`). A generic request with none of that wording
     is asked only the three always-critical questions — not cloud, budget
     and NFR on every project regardless of whether they apply.

  B. SAFE ASSUMPTIONS — A TINY POSITIVE ALLOWLIST — a model-authored
     `assumptions` line survives `_freeze_context_record` only if it
     positively classifies as one of a small fixed set of SAFE,
     non-architectural categories (`clarifier._SAFE_ASSUMPTION_PATTERNS`):
     display/UI-language metadata, non-technical project naming, and
     stakeholder-label normalization. A negative keyword blacklist can never
     enumerate every future architecture decision (as proven by "Use
     PostgreSQL", "Use Django", "Expose REST APIs", "Deploy in a single
     region", "Use OAuth/OIDC" — none of which the OLD blacklist caught), and
     dropping every line unconditionally (the first fix attempted) also threw
     away genuinely harmless ones. The allowlist is the actual fix: an
     assumption that cannot be positively classified as safe is dropped,
     exactly like before.

  C. ABSORBED CRITICAL GAPS STAY UNKNOWN — an ask-cap gap forces its matching
     `captured` field back to empty in the frozen record REGARDLESS of what
     the model proposed for it, for every critical field (not just the
     technology-shaped ones the old code special-cased). "Ask cap stops the
     loop; it must not turn unknown facts into fake facts."

  D. RECOMMENDATION SEMANTICS — `apply_user_edits(..., recommend=[field])`
     means the human explicitly delegated that field to the Clarifier.
     `_freeze_context_record` (via its `recommend_requested` parameter) must
     tell that apart from an ordinary unanswered, cap-absorbed gap: a
     recommend request for a genuinely safe, non-architectural field
     (`project_name`, `users` — the only two `EDITABLE_RECORD_FIELDS` outside
     `CRITICAL_RECORD_FIELDS`) MAY get a real, labelled, vetoable
     recommendation; a recommend request for an architecture-critical field
     (e.g. `cloud_provider`) never gets a fabricated value, no matter how
     explicitly it was asked for — it still comes back as an explicit
     "not safe to auto-recommend" marker.

  E. CLOUD RELEVANCE SIGNAL — `_CLOUD_SIGNAL_TERMS` no longer contains the
     generic word "provider", which made unrelated phrases like "third-party
     payment provider" or "identity provider" falsely mark `cloud_provider`
     as relevant. Matching is also phrase-aware (`clarifier._term_present`)
     so a short token like "aws" cannot fire on an unrelated word.

  F. THE SAFE-ASSUMPTION BOUNDARY IS STRUCTURAL, NOT "CONTAINS A MARKER" — a
     line like "Project name: Sneaker Hub; use CockroachDB for storage."
     contains the safe marker "project name" but also smuggles a database
     choice past a substring check. `_is_safe_low_stakes_assumption` requires
     the ENTIRE line to match one of a tiny set of full-line shapes with a
     short/plain payload, so a second clause has nowhere to attach.

  G. VETOING A SAFE-FIELD RECOMMENDATION CLEARS THE FIELD TOO — striking a
     `[recommended]` assumption used to remove the label but leave
     `record.project_name` (or `.users`) holding the rejected value.
     `apply_user_edits` now recovers the field deterministically from the
     struck text's own fixed shape (`_parse_safe_recommendation_text`) and
     clears it — but only when the field still holds exactly the value that
     was recommended, so an unrelated struck assumption never touches it.

  H. AN EMPTY SAFE-FIELD RECOMMENDATION STAYS VISIBLE — if the human asks the
     clarifier to recommend `project_name` or `users` and the model proposes
     nothing, the request must not just vanish: a deterministic unresolved
     marker and open question are written instead, with no fabricated value
     and no extra clarification round.

All offline. No API key, no network, no live LLM call.
"""

from __future__ import annotations

import test_clarifier as tc
from pipeline.agents import clarifier as clar
from pipeline.state import (
    CapturedContext,
    ClarificationResult,
    ContextEdits,
    ContextRecord,
    PendingDecision,
    Stage,
    new_run,
)

# Deliberately signal-free: no scale/cloud/cost/compliance/brownfield wording
# anywhere. This is the "generic request" section 1 of the required tests is
# about — only the three unconditionally-critical fields should ever be asked
# for a prompt like this.
GENERIC_PROMPT = "Build me a system to sell sneakers online."

# Carries a signal term for every conditional field at once, so a single state
# can stand in for "everything is relevant" without needing eight prompts.
EVERYTHING_RELEVANT_PROMPT = (
    "Modernize our legacy monolith: build a new system to sell sneakers "
    "online, at massive scale, hosted on AWS, within a strict budget, "
    "handling patient health data for our clinic customers."
)


def _filled_captured(**overrides) -> CapturedContext:
    """Every unconditionally-critical field filled, plus the two conditional
    fields most tests below turn on and off (`non_functional_requirements`,
    `cloud_provider`, `budget`), so nothing is missing unless a test
    deliberately empties one via `overrides`."""
    base = dict(
        business_goal="Sell sneakers online",
        problem_statement="The monolith falls over on peak sale days",
        functional_requirements=["Browse and buy sneakers online"],
        non_functional_requirements=["50k peak users"],
        cloud_provider="AWS",
        budget="medium",
    )
    base.update(overrides)
    return CapturedContext(**base)


# ══════════════════════════════════════════════════════════════════════════
# A. Relevance policy — 9 tests
# ══════════════════════════════════════════════════════════════════════════
def test_generic_request_asks_only_the_always_critical_slots():
    """No scale/cloud/cost/compliance/brownfield signal anywhere -> only the
    three fields that are critical on EVERY request are gaps."""
    state = new_run(GENERIC_PROMPT)
    captured = CapturedContext()  # nothing grounded at all

    assert clar.missing_critical_slots(state, captured) == [
        "business_goal",
        "problem_statement",
        "functional_requirements",
    ]


def test_scale_wording_makes_non_functional_requirements_relevant():
    state = new_run("Build a system that must handle massive scale and peak load.")
    captured = _filled_captured(non_functional_requirements=[])

    assert clar.missing_critical_slots(state, captured) == ["non_functional_requirements"]


def test_cloud_wording_makes_cloud_provider_relevant():
    state = new_run("Build a system; it needs an AWS-hosted deployment.")
    captured = _filled_captured(cloud_provider="")

    assert clar.missing_critical_slots(state, captured) == ["cloud_provider"]


def test_budget_wording_makes_budget_relevant():
    state = new_run("Build a system; keep it within a tight budget.")
    captured = _filled_captured(budget="")

    assert clar.missing_critical_slots(state, captured) == ["budget"]


def test_compliance_signal_still_makes_compliance_relevant():
    state = new_run("Build a healthcare system that stores patient records.")
    captured = _filled_captured(compliance_requirements=[])

    assert clar.missing_critical_slots(state, captured) == ["compliance_requirements"]


def test_irrelevant_compliance_stays_empty_and_unasked():
    """The negative case for the test above: no regulated-domain wording ->
    an empty `compliance_requirements` is not a gap at all."""
    state = new_run(GENERIC_PROMPT)
    captured = _filled_captured()  # compliance_requirements left empty
    assert captured.compliance_requirements == []

    assert clar.missing_critical_slots(state, captured) == []


def test_brownfield_wording_without_a_repo_url_makes_existing_systems_relevant():
    state = new_run("Modernize our legacy monolith into a new architecture.")
    assert state.initial_request.repo_url == ""
    captured = _filled_captured(existing_systems=[])

    assert clar.missing_critical_slots(state, captured) == ["existing_systems"]


def test_repo_url_still_makes_existing_systems_relevant():
    state = new_run(GENERIC_PROMPT, repo_url="https://github.com/example/sneaker-shop")
    captured = _filled_captured(existing_systems=[])

    assert clar.missing_critical_slots(state, captured) == ["existing_systems"]


def test_a_clarification_answer_can_make_a_conditional_slot_relevant():
    """A dimension the raw prompt never raised can become relevant once the
    human's own answer raises it — `_signal_text` reads answers too."""
    state = new_run(GENERIC_PROMPT)
    captured = _filled_captured(cloud_provider="")

    # Before any answer: no cloud signal anywhere -> not a gap.
    assert clar.missing_critical_slots(state, captured) == []

    state.clarification_answers["What scale do you need?"] = (
        "We want everything running in a specific AWS region."
    )

    # Same captured facts, new signal -> now a gap.
    assert clar.missing_critical_slots(state, captured) == ["cloud_provider"]


def test_missing_critical_slots_is_stable_and_ordered_across_repeated_calls():
    """Same `(state, captured)` -> same ordered gap list, every time — the
    property that makes `missing_critical_slots` a deterministic replacement
    for the model's own (unreliable) `missing_critical`."""
    state = new_run(EVERYTHING_RELEVANT_PROMPT)
    captured = CapturedContext()  # nothing grounded -> every relevant field gaps

    results = [tuple(clar.missing_critical_slots(state, captured)) for _ in range(25)]
    assert len(set(results)) == 1
    # Every field is relevant for this prompt, so the gap list is the full,
    # stable `CRITICAL_RECORD_FIELDS` order.
    assert results[0] == clar.CRITICAL_RECORD_FIELDS


# ══════════════════════════════════════════════════════════════════════════
# B. Safe assumptions — a tiny positive allowlist — 9 tests
# ══════════════════════════════════════════════════════════════════════════
# Every "dropped" line below is a MODEL-AUTHORED free-form `assumptions` entry
# that would have needed its own blacklist term under the old design — none of
# them positively classify as one of the tiny SAFE categories
# (`clarifier._SAFE_ASSUMPTION_PATTERNS`), so `_is_safe_low_stakes_assumption`
# rejects them and `_freeze_context_record` drops them, regardless of whether
# they happen to mention an architecture decision by name.
def _dropped(text: str) -> None:
    result = ClarificationResult(captured=_filled_captured(), assumptions=[text])
    record = clar._freeze_context_record(result)
    assert record.assumptions == [], f"expected {text!r} to be dropped"


def test_a_positively_allowed_low_stakes_assumption_survives():
    """The positive case: display/UI-language metadata is one of the tiny SAFE
    categories, so it survives, labelled — unlike everything in this section."""
    result = ClarificationResult(
        captured=_filled_captured(),
        assumptions=["Assume English-only UI (low-stakes)."],
    )
    record = clar._freeze_context_record(result)
    assert record.assumptions == [
        f"{clar.CLARIFIER_LABEL} Assume English-only UI (low-stakes)."
    ]


def test_postgresql_assumption_does_not_survive():
    _dropped("Use PostgreSQL as the primary database.")


def test_python_django_assumption_does_not_survive():
    _dropped("Use Python/Django for the new service.")


def test_rest_api_assumption_does_not_survive():
    _dropped("Expose REST APIs between services.")


def test_single_region_assumption_does_not_survive():
    _dropped("Deploy in a single region.")


def test_oauth_oidc_assumption_does_not_survive():
    _dropped("Use OAuth/OIDC for authentication.")


def test_unclassified_free_text_assumption_does_not_survive():
    """Harmless-sounding, but names none of the tiny SAFE categories — an
    assumption that cannot be positively classified is dropped, the same as
    an unambiguously architectural one. The allowlist proves safety; silence
    on the unsafe list is never enough."""
    _dropped("The team prefers a quarterly release cadence.")


def test_previously_known_forbidden_examples_still_do_not_survive():
    """The examples the OLD blacklist was written to catch still do not
    survive — not because a blacklist term matches them, but because none of
    them names a SAFE category either."""
    result = ClarificationResult(
        captured=_filled_captured(),
        assumptions=[
            "Apply the Strangler pattern for a no-downtime migration.",
            "Use AWS RDS for the primary database.",
            "Run background jobs on AWS Lambda.",
            "Deploy the services on ECS behind an API Gateway.",
            "Adopt an event-driven architecture with eventual consistency.",
        ],
    )
    record = clar._freeze_context_record(result)
    assert record.assumptions == []


def test_historical_already_persisted_assumptions_remain_loadable_and_untouched():
    """Backward compatibility: a ContextRecord written before this policy
    existed is never reached back into and re-filtered. `_freeze_context_record`
    only ever WRITES a fresh record's `assumptions`; loading and round-tripping
    an old one must preserve its free-form text exactly as it was."""
    old_style = ContextRecord(
        business_goal="Sell sneakers online",
        assumptions=[
            "Assume English-only UI (low-stakes).",
            "Use PostgreSQL as the primary database.",  # would be dropped today
        ],
    )
    restored = ContextRecord.model_validate_json(old_style.model_dump_json())
    assert restored.assumptions == old_style.assumptions


# ══════════════════════════════════════════════════════════════════════════
# C. Absorbed critical gaps stay unknown — 9 tests
# ══════════════════════════════════════════════════════════════════════════
def test_absorbed_budget_gap_is_forced_empty_even_when_the_model_proposed_one():
    result = ClarificationResult(
        captured=_filled_captured(budget="Roughly $500k — a plausible mid-market default.")
    )
    record = clar._freeze_context_record(result, absorbed_gaps=["budget"])
    assert record.budget == ""


def test_absorbed_non_functional_requirements_gap_is_forced_empty():
    result = ClarificationResult(
        captured=_filled_captured(non_functional_requirements=["99.99% availability, 10k RPS"])
    )
    record = clar._freeze_context_record(
        result, absorbed_gaps=["non_functional_requirements"]
    )
    assert record.non_functional_requirements == []


def test_absorbed_compliance_requirements_gap_is_forced_empty():
    result = ClarificationResult(
        captured=_filled_captured(compliance_requirements=["GDPR (assumed likely)"])
    )
    record = clar._freeze_context_record(result, absorbed_gaps=["compliance_requirements"])
    assert record.compliance_requirements == []


def test_absorbed_existing_systems_gap_is_forced_empty():
    result = ClarificationResult(
        captured=_filled_captured(existing_systems=["Presumably a Django monolith"])
    )
    record = clar._freeze_context_record(result, absorbed_gaps=["existing_systems"])
    assert record.existing_systems == []


def test_absorbed_cloud_provider_gap_is_forced_empty():
    """Existing behavior, generalized: this is no longer a special case for
    `cloud_provider` alone — see the four tests above for the other fields."""
    result = ClarificationResult(
        captured=_filled_captured(cloud_provider="AWS (best professional default)")
    )
    record = clar._freeze_context_record(result, absorbed_gaps=["cloud_provider"])
    assert record.cloud_provider == ""


def test_every_absorbed_gap_gets_a_labelled_unresolved_assumption_and_open_question():
    absorbed = ["non_functional_requirements", "cloud_provider", "budget"]
    result = ClarificationResult(
        captured=_filled_captured(
            non_functional_requirements=[], cloud_provider="", budget=""
        )
    )
    record = clar._freeze_context_record(result, absorbed_gaps=absorbed)

    assert len(record.assumptions) == len(absorbed)
    assert len(record.open_questions) == len(absorbed)
    for gap in absorbed:
        assert any(
            gap in a and a.startswith(clar.CLARIFIER_LABEL) and "unresolved" in a
            for a in record.assumptions
        ), gap
        assert any(gap in q for q in record.open_questions), gap


def test_repeated_freeze_of_the_same_result_is_byte_identical():
    result = ClarificationResult(
        captured=_filled_captured(non_functional_requirements=[], cloud_provider="")
    )
    absorbed = ["non_functional_requirements", "cloud_provider"]

    dumps = {
        clar._freeze_context_record(result, absorbed_gaps=absorbed).model_dump_json()
        for _ in range(10)
    }
    assert len(dumps) == 1


def test_clarification_round_cap_is_unchanged():
    assert clar.MAX_ASK_ROUNDS == 3
    from pipeline.run import MAX_CLARIFICATION_ROUNDS

    assert MAX_CLARIFICATION_ROUNDS >= clar.MAX_ASK_ROUNDS


# ══════════════════════════════════════════════════════════════════════════
# D. Recommendation semantics — 6 tests
# ══════════════════════════════════════════════════════════════════════════
def test_cap_absorbed_budget_gap_remains_empty_and_unresolved():
    """The NEGATIVE control: budget was simply never answered — NOT a
    recommend request. It stays unresolved with the ordinary cap-absorbed
    wording, exactly as before recommend semantics existed."""
    result = ClarificationResult(captured=_filled_captured(budget=""))
    record = clar._freeze_context_record(result, absorbed_gaps=["budget"])

    assert record.budget == ""
    assert any(
        "budget" in a
        and a.startswith(clar.CLARIFIER_LABEL)
        and "unresolved after the clarification cap" in a
        for a in record.assumptions
    )


def test_explicit_recommendation_for_a_safe_field_produces_a_labelled_recommendation():
    """The POSITIVE case: `project_name` is not architecture-critical, so an
    explicit recommend request MAY get a real, labelled, vetoable value."""
    result = ClarificationResult(captured=_filled_captured(project_name="Sneaker Hub"))
    record = clar._freeze_context_record(result, recommend_requested=["project_name"])

    assert record.project_name == "Sneaker Hub"
    assert record.assumptions == [
        f"{clar.CLARIFIER_LABEL} [recommended] project name: Sneaker Hub — "
        "proposed by the clarifier at your request; not confirmed until you "
        "accept or edit it."
    ]


def test_explicit_recommendation_for_cloud_provider_does_not_invent_a_value():
    """`cloud_provider` is architecture-critical: naming AWS/Azure/GCP is
    exactly the kind of architecture decision the Clarifier may never make,
    so an explicit recommend request still comes back unresolved — not
    silently dropped, but an explicit "not safe to auto-recommend" marker."""
    result = ClarificationResult(captured=_filled_captured(cloud_provider="AWS"))
    record = clar._freeze_context_record(result, recommend_requested=["cloud_provider"])

    assert record.cloud_provider == ""
    joined = " ".join(record.assumptions)
    assert "AWS" not in joined and "Azure" not in joined and "GCP" not in joined
    assert any(
        a.startswith(clar.CLARIFIER_LABEL) and "not safe to auto-recommend" in a
        for a in record.assumptions
    )


def test_explicit_recommendation_text_is_distinguishable_from_a_stated_fact():
    """The recommended value is framed as a proposal, not folded silently
    into the record looking like something the user actually said."""
    result = ClarificationResult(captured=_filled_captured(users=["Retail shoppers"]))
    record = clar._freeze_context_record(result, recommend_requested=["users"])

    assert record.users == ["Retail shoppers"]
    assert len(record.assumptions) == 1
    assumption = record.assumptions[0]
    assert assumption.startswith(clar.CLARIFIER_LABEL)
    assert "[recommended]" in assumption
    assert "not confirmed" in assumption


def test_vetoing_a_safe_field_recommendation_prevents_it_reappearing():
    result = ClarificationResult(captured=_filled_captured(project_name="Sneaker Hub"))
    record = clar._freeze_context_record(result, recommend_requested=["project_name"])
    recommendation_text = record.assumptions[0]

    updated = clar.apply_user_edits(
        record, ContextEdits(struck_assumptions=[recommendation_text])
    )
    assert updated.assumptions == []

    # And the veto ledger stops it from being re-proposed on a later freeze.
    record_again = clar._freeze_context_record(
        result,
        recommend_requested=["project_name"],
        vetoed_assumptions=[recommendation_text],
    )
    assert record_again.assumptions == []


def test_recommend_flow_still_costs_exactly_one_llm_call(monkeypatch):
    calls = {"n": 0}

    def _counting_complete(state, prompt, **kwargs):
        calls["n"] += 1
        return tc._complete(state, prompt, **kwargs)

    monkeypatch.setattr(clar, "llm_call", _counting_complete)

    out = clar.clarifier_node(new_run(GENERIC_PROMPT, require_context_approval=True))
    assert calls["n"] == 1

    state = new_run(GENERIC_PROMPT, require_context_approval=True)
    state.context_record = out["context_record"]
    state.stage = Stage.AWAITING_HUMAN
    state.pending_decision = PendingDecision.CONTEXT_LOCK

    clar.submit_context_edits(state, ContextEdits(recommend=["project_name"]))
    clar.open_for_rejudge(state)

    clar.clarifier_node(state)
    assert calls["n"] == 2  # exactly one MORE call for the rejudge, not zero or two


# ══════════════════════════════════════════════════════════════════════════
# E. Cloud relevance signal — 5 tests
# ══════════════════════════════════════════════════════════════════════════
def test_payment_provider_does_not_make_cloud_provider_relevant():
    state = new_run("We need to integrate a third-party payment provider.")
    captured = _filled_captured(cloud_provider="")
    assert clar.missing_critical_slots(state, captured) == []


def test_identity_provider_does_not_make_cloud_provider_relevant():
    state = new_run("Users should log in via an external identity provider.")
    captured = _filled_captured(cloud_provider="")
    assert clar.missing_critical_slots(state, captured) == []


def test_cloud_provider_phrase_makes_cloud_provider_relevant():
    state = new_run("We have no cloud provider preference yet.")
    captured = _filled_captured(cloud_provider="")
    assert clar.missing_critical_slots(state, captured) == ["cloud_provider"]


def test_aws_azure_gcp_onprem_hosting_still_make_cloud_provider_relevant():
    for phrase in (
        "Deploy on AWS.",
        "We use Azure today.",
        "Target GCP for hosting.",
        "This must run on-prem.",
        "Needs managed hosting.",
        "It should be hosted for us.",
    ):
        state = new_run(f"Build a system. {phrase}")
        captured = _filled_captured(cloud_provider="")
        assert clar.missing_critical_slots(state, captured) == ["cloud_provider"], phrase


def test_cloud_relevance_signal_is_stable_across_repeated_calls():
    state = new_run("Build a system; it needs an AWS-hosted deployment.")
    captured = _filled_captured(cloud_provider="")
    results = {tuple(clar.missing_critical_slots(state, captured)) for _ in range(25)}
    assert results == {("cloud_provider",)}


# ══════════════════════════════════════════════════════════════════════════
# F. Structural safe-assumption boundary — 6 tests
# ══════════════════════════════════════════════════════════════════════════
def test_valid_display_language_assumption_survives():
    result = ClarificationResult(
        captured=_filled_captured(),
        assumptions=["Assume the default display language is Spanish."],
    )
    record = clar._freeze_context_record(result)
    assert record.assumptions == [
        f"{clar.CLARIFIER_LABEL} Assume the default display language is Spanish."
    ]


def test_valid_project_name_assumption_survives():
    result = ClarificationResult(
        captured=_filled_captured(),
        assumptions=["Assume the project name is Sneaker Hub."],
    )
    record = clar._freeze_context_record(result)
    assert record.assumptions == [
        f"{clar.CLARIFIER_LABEL} Assume the project name is Sneaker Hub."
    ]


def test_valid_stakeholder_label_assumption_survives():
    result = ClarificationResult(
        captured=_filled_captured(),
        assumptions=["Assume the stakeholder label is Retail Shoppers."],
    )
    record = clar._freeze_context_record(result)
    assert record.assumptions == [
        f"{clar.CLARIFIER_LABEL} Assume the stakeholder label is Retail Shoppers."
    ]


def test_safe_marker_plus_cockroachdb_clause_is_dropped():
    """The exact smuggling example: a safe marker ("project name") sharing a
    line with an architecture choice. The full-line shape does not match, so
    it is dropped — not because "cockroachdb" sits on some blacklist (it
    never needs to), but because the line is not ONE plain clause."""
    _dropped("Project name: Sneaker Hub; use CockroachDB for storage.")


def test_safe_marker_plus_arbitrary_second_technical_clause_is_dropped():
    _dropped(
        "Assume the project name is Sneaker Hub, deployed as a single "
        "monolithic service."
    )


def test_postgresql_django_rest_single_region_oauth_still_dropped():
    """The pre-existing forbidden examples (see the dedicated tests in
    section B for each individually), re-confirmed as a group under the NEW
    structural full-line check."""
    for text in (
        "Use PostgreSQL as the primary database.",
        "Use Python/Django for the new service.",
        "Expose REST APIs between services.",
        "Deploy in a single region.",
        "Use OAuth/OIDC for authentication.",
    ):
        _dropped(text)


# ══════════════════════════════════════════════════════════════════════════
# G. Vetoing a safe-field recommendation clears the field too — 4 tests
# ══════════════════════════════════════════════════════════════════════════
def test_striking_a_project_name_recommendation_clears_the_field():
    result = ClarificationResult(captured=_filled_captured(project_name="Sneaker Hub"))
    record = clar._freeze_context_record(result, recommend_requested=["project_name"])
    recommendation_text = record.assumptions[0]

    updated = clar.apply_user_edits(
        record, ContextEdits(struck_assumptions=[recommendation_text])
    )
    assert updated.assumptions == []
    assert updated.project_name == ""


def test_striking_a_users_recommendation_clears_the_field():
    result = ClarificationResult(captured=_filled_captured(users=["Retail shoppers"]))
    record = clar._freeze_context_record(result, recommend_requested=["users"])
    recommendation_text = record.assumptions[0]

    updated = clar.apply_user_edits(
        record, ContextEdits(struck_assumptions=[recommendation_text])
    )
    assert updated.assumptions == []
    assert updated.users == []


def test_striking_an_unrelated_assumption_does_not_clear_recommended_fields():
    result = ClarificationResult(
        captured=_filled_captured(project_name="Sneaker Hub", users=["Retail shoppers"]),
        assumptions=["Assume the default display language is Spanish."],
    )
    record = clar._freeze_context_record(
        result, recommend_requested=["project_name", "users"]
    )
    assert len(record.assumptions) == 3  # 1 safe assumption + 2 recommendations

    unrelated = next(a for a in record.assumptions if "display language" in a)
    updated = clar.apply_user_edits(record, ContextEdits(struck_assumptions=[unrelated]))

    assert updated.project_name == "Sneaker Hub"
    assert updated.users == ["Retail shoppers"]
    assert unrelated not in updated.assumptions
    assert len(updated.assumptions) == 2


def test_field_already_changed_since_the_recommendation_is_not_clobbered():
    """A struck recommendation only clears the field if it STILL holds the
    recommended value — a fresh explicit edit in the SAME pass must win, not
    be silently undone by an unrelated strike of the old proposal."""
    result = ClarificationResult(captured=_filled_captured(project_name="Sneaker Hub"))
    record = clar._freeze_context_record(result, recommend_requested=["project_name"])
    recommendation_text = record.assumptions[0]

    updated = clar.apply_user_edits(
        record,
        ContextEdits(
            fields={"project_name": "Kicks Direct"},
            struck_assumptions=[recommendation_text],
        ),
    )
    assert updated.project_name == "Kicks Direct"


# ══════════════════════════════════════════════════════════════════════════
# G2. Veto persistence — a struck value must not re-enter a later freeze —
#     4 tests. The strike LEDGER (state.vetoed_assumptions) outlives the
#     record it was cast against; these pin that it is enforced on the field
#     VALUES, deterministically, via `_parse_safe_recommendation_text`.
# ══════════════════════════════════════════════════════════════════════════
def test_vetoed_project_name_cannot_silently_reenter_a_later_freeze():
    """THE regression: the model proposes "Sneaker Hub", the human strikes
    it, and a later re-judge returns the SAME value. The value must stay
    off the new record — previously it re-entered as an unlabeled plain
    fact, wearing no strikeable recommendation label at all."""
    result = ClarificationResult(captured=_filled_captured(project_name="Sneaker Hub"))
    v1 = clar._freeze_context_record(result, recommend_requested=["project_name"])
    recommendation_text = v1.assumptions[0]
    post_veto = clar.apply_user_edits(
        v1, ContextEdits(struck_assumptions=[recommendation_text])
    )
    assert post_veto.project_name == ""  # the immediate strike (section G)

    v2 = clar._freeze_context_record(
        result,  # the re-judge returns the same captured value again
        recommend_requested=["project_name"],
        vetoed_assumptions=[recommendation_text],
        previous=post_veto,
    )

    assert v2.project_name == ""
    assert recommendation_text not in v2.assumptions
    assert v2.assumptions == []  # nothing resurrected under any label


def test_vetoed_users_recommendation_cannot_silently_reenter_either():
    """The shared implementation covers the other safe field: a struck
    `users` recommendation is enforced on the list VALUE, with the list
    rendered exactly as `_build_safe_recommendation_text` renders it."""
    result = ClarificationResult(captured=_filled_captured(users=["Retail shoppers"]))
    v1 = clar._freeze_context_record(result, recommend_requested=["users"])
    recommendation_text = v1.assumptions[0]
    post_veto = clar.apply_user_edits(
        v1, ContextEdits(struck_assumptions=[recommendation_text])
    )
    assert post_veto.users == []

    v2 = clar._freeze_context_record(
        result,
        recommend_requested=["users"],
        vetoed_assumptions=[recommendation_text],
        previous=post_veto,
    )

    assert v2.users == []
    assert v2.assumptions == []


def test_later_explicit_human_edit_wins_over_the_old_veto():
    """The veto blocks the CLARIFIER's proposal, not the HUMAN's choice. If
    the superseded record already carries a value the human set themselves
    after the strike (even the same string), the re-freeze must keep it —
    a ledger that could erase a human's explicit edit would be a bug of its
    own."""
    result = ClarificationResult(captured=_filled_captured(project_name="Sneaker Hub"))
    v1 = clar._freeze_context_record(result, recommend_requested=["project_name"])
    recommendation_text = v1.assumptions[0]
    post_veto = clar.apply_user_edits(
        v1, ContextEdits(struck_assumptions=[recommendation_text])
    )
    human_set = post_veto.model_copy(deep=True)
    human_set.project_name = "Sneaker Hub"  # the human chose it themselves

    v2 = clar._freeze_context_record(
        result,
        recommend_requested=["project_name"],
        vetoed_assumptions=[recommendation_text],
        previous=human_set,
    )

    assert v2.project_name == "Sneaker Hub"  # the human's edit survives


def test_veto_persistence_does_not_mutate_the_incoming_result():
    """The re-judge's `ClarificationResult` is read, never written: the
    blanking happens on the record's own field values only. The result
    object stays exactly what the model produced."""
    result = ClarificationResult(captured=_filled_captured(project_name="Sneaker Hub"))
    v1 = clar._freeze_context_record(result, recommend_requested=["project_name"])
    recommendation_text = v1.assumptions[0]
    post_veto = clar.apply_user_edits(
        v1, ContextEdits(struck_assumptions=[recommendation_text])
    )

    clar._freeze_context_record(
        result,
        recommend_requested=["project_name"],
        vetoed_assumptions=[recommendation_text],
        previous=post_veto,
    )

    assert result.captured.project_name == "Sneaker Hub"  # untouched
    assert result.assumptions == []


# ══════════════════════════════════════════════════════════════════════════
# H. An empty safe-field recommendation stays visible — 1 test
# ══════════════════════════════════════════════════════════════════════════
def test_empty_safe_field_recommendation_leaves_a_visible_unresolved_marker():
    """The model proposed nothing for a requested `project_name`
    recommendation. The OLD code silently `continue`d and the request just
    vanished. It must now leave a deterministic marker — no fabricated value,
    no fresh ask-round."""
    result = ClarificationResult(captured=_filled_captured())  # project_name still ""
    record = clar._freeze_context_record(result, recommend_requested=["project_name"])

    assert record.project_name == ""
    assert len(record.assumptions) == 1
    assumption = record.assumptions[0]
    assert assumption.startswith(clar.CLARIFIER_LABEL)
    assert "[recommendation requested]" in assumption
    assert "no proposal" in assumption
    assert any(
        "project name" in q and "nothing to propose" in q for q in record.open_questions
    )


def test_context_veto_edit_and_rejudge_still_work(monkeypatch):
    """The context-lock gate's edit/accept flow, unaffected by any of the
    above: filling a value is still free (no model call), and the record
    still locks for approval when required. The full pause/absorb/re-judge
    round trip is pinned end-to-end in test_context_gate.py; this is the
    smoke test that the wiring this file's changes touch — freeze,
    `EDITABLE_RECORD_FIELDS` — did not come apart."""
    monkeypatch.setattr(clar, "llm_call", tc._complete)
    out = clar.clarifier_node(new_run(GENERIC_PROMPT, require_context_approval=True))
    assert out["pending_decision"] is PendingDecision.CONTEXT_LOCK

    state = new_run(GENERIC_PROMPT, require_context_approval=True)
    state.context_record = out["context_record"]
    state.stage = Stage.AWAITING_HUMAN
    state.pending_decision = PendingDecision.CONTEXT_LOCK

    reasons = clar.submit_context_edits(state, ContextEdits(fields={"budget": "large"}))
    assert reasons == []
    assert state.context_record.budget == "large"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
