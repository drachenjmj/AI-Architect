"""test_demo_consistency.py — the offline demo states vs the CURRENT
deterministic review checks.

Scope: Part C of the Overview-polish pass. The manual review suspected the
`--demo` fixture predated the cross-artifact TARGET-SERVICE OWNERSHIP check
(a data flow names a "notification service" that has no Component
Description). These tests prove the actual state of affairs with the real
check, and pin it:

  * `run_deterministic_checks(...).unowned_target_services` is empty for
    BOTH demo variants — no dangling target-service reference exists;
  * every service reference the scanner extracts resolves to a Component
    Description (verified through the check's own matching rule);
  * the Reviewer pipeline module itself is untouched — these tests only
    CALL it.

Why the screenshot was misleading is documented in
`unpinned`-style detail below: the ownership rule counts only TitleCase
"X Service"/"X services" names as target references; the demo's
"notification service" appears only in lowercase prose (a flow
description) and as a lowercase dependency of the Order Worker — i.e. an
external system the design USES, not a service it claims to own. SQS and
PostgreSQL are the same kind of participant.
"""

from __future__ import annotations

from pipeline.review_checks import (
    _component_key,
    _target_service_references,
    run_deterministic_checks,
)
from ui_demo import DEMO_VARIANTS, build_demo_state


def test_demo_has_no_unowned_target_services():
    for variant in DEMO_VARIANTS:
        checks = run_deterministic_checks(build_demo_state(variant))
        assert checks.unowned_target_services == {}, variant


def test_every_referenced_target_service_resolves_to_a_component():
    for variant in DEMO_VARIANTS:
        state = build_demo_state(variant)
        references = _target_service_references(state)
        assert references, variant  # the scanner does see the references

        component_keys = [
            _component_key(component.name)
            for component in state.components
            if _component_key(component.name)
        ]
        for display in references:
            key = _component_key(display)
            assert any(existing[-len(key):] == key for existing in component_keys), (
                f"{variant}: {display!r} dangles"
            )


def test_notification_service_is_prose_not_a_target_service_reference():
    """The reference that looked inconsistent in review: the flow says
    "Order Worker → notification service". Under the ownership rule — the
    rule a real run is judged by — a lowercase bare noun is NOT a
    target-service reference, and the demo's TitleCase references
    ("Checkout Service") all resolve. This is the regression pin for
    exactly that fact."""
    state = build_demo_state("pass")
    references = _target_service_references(state)

    assert "Checkout Service" in references            # owned and described
    assert "Notification Service" not in references    # prose, not a claim
    assert references  # and everything the scanner DOES see, resolves (above)


def test_demo_variants_covered():
    assert DEMO_VARIANTS == ("pass", "capped")
