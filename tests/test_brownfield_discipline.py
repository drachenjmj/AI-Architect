"""test_brownfield_discipline.py — brownfield design-discipline contracts.

Scope: the three discipline improvements from the benchmark findings
(service boundaries, technology conservatism, structured migration), proven
OFFLINE at the contract level:

  * the Architect's generation instructions carry the decision rules and
    request a structured migration sequence;
  * the Blueprint schema carries `migration_steps` backward-compatibly
    (old checkpoints deserialize unchanged, order round-trips);
  * the Reviewer's EXISTING semantic judgments now cover decomposition
    quality, technology-change justification and migration coherence —
    without new scoring fields and without mandating any decomposition.

The comparative `ecommerce-microservice` repository was a benchmark used to
EXPOSE general weaknesses, not ground truth: nothing here may hard-code its
service names, technologies or decomposition, and the prompt tests pin
that. No LLM calls anywhere.
"""

from __future__ import annotations

import json
import re

from pipeline.agents import architect as arch
from pipeline.agents import reviewer as rev
from pipeline.persistence import load_state, save_state
from pipeline.state import (
    ArchitectState,
    Blueprint,
    MigrationStep,
    Stage,
    new_run,
)


def _flat(text: str) -> str:
    """Collapse all whitespace, so prompt assertions are robust against the
    prompts' line wrapping (a wrapped phrase is still the same phrase)."""
    return " ".join(text.split())


_ARCH_PROMPT = _flat(arch.ARCHITECTURE_SYSTEM_PROMPT)
_REV_PROMPT = _flat(rev.REVIEWER_SYSTEM)

# Names from the comparative reference repository that must NEVER appear in
# generation or review instructions (no overfitting to the benchmark).
_BENCHMARK_NAMES = (
    "Cart Service",
    "Ordering Aggregator",
    "ecommerce-microservice",
    "harsh020",
    "ecommerce_monolith",  # the source repo slug, beyond repo evidence tags
)


# ── 1. Architect: service-boundary discipline ─────────────────────────────


def test_architect_prompt_rejects_module_to_service_mapping():
    prompt = _ARCH_PROMPT
    assert "Never map one" in prompt and "microservice" in prompt
    assert "not automatic service boundaries" in prompt
    assert "CAPABILITIES" in prompt


def test_architect_prompt_requires_boundary_drivers():
    prompt = _ARCH_PROMPT
    assert "standalone service only when" in prompt
    assert "at least one driver" in prompt
    for driver in (
        "independent scaling",
        "data-consistency boundary",
        "change cadence",
        "fault isolation",
        "data ownership",
    ):
        assert driver in prompt, driver


def test_architect_prompt_prefers_cohesion_and_does_not_force_microservices():
    prompt = _ARCH_PROMPT
    assert "cohesive bounded context" in prompt
    assert "SMALLEST decomposition" in prompt
    assert "operational complexity is a real" in prompt
    assert "not the assumed target" in prompt          # microservices optional
    assert "modular monolith" in prompt                # retention stays valid


# ── 2. Architect: technology conservatism ─────────────────────────────────


def test_architect_prompt_makes_existing_stack_the_default():
    prompt = _ARCH_PROMPT
    assert "Existing technologies remain the DEFAULT" in prompt


def test_architect_prompt_demands_justified_technology_changes():
    prompt = _ARCH_PROMPT
    assert "NOT valid reasons" in prompt
    assert "novelty" in prompt and "generic best" in prompt
    assert "justification" in prompt
    # Polyglot is allowed when justified — not banned.
    assert "Polyglot architecture is allowed" in prompt


# ── 3. Architect: structured migration sequence ────────────────────────────


def test_architect_prompt_requests_structured_migration_steps():
    prompt = _ARCH_PROMPT
    assert "migration_steps" in prompt
    assert "brownfield modernization" in prompt
    for concept in (
        "coexistence", "data transition", "first", "validation",
    ):
        assert concept in prompt, concept
    # Greenfield must be allowed to leave it empty.
    assert "leave" in prompt and "empty" in prompt
    # And fabrication is forbidden.
    assert "Do not fabricate" in prompt


# ── 4. No benchmark overfitting in either prompt ──────────────────────────


def test_neither_prompt_hardcodes_benchmark_names():
    for prompt in (_ARCH_PROMPT, _REV_PROMPT):
        for name in _BENCHMARK_NAMES:
            assert name not in prompt, name


# ── 5. Schema: MigrationStep on Blueprint, backward-compatibly ────────────


def _step(title: str, objective: str = "") -> MigrationStep:
    return MigrationStep(
        title=title,
        objective=objective or f"Objective of {title}.",
        changes=[f"Change made by {title}"],
        coexistence_or_data_strategy=f"Coexistence during {title}.",
        exit_condition=f"{title} validated.",
    )


def test_blueprint_with_migration_steps_validates_and_roundtrips():
    blueprint = Blueprint(
        project_name="Generic Modernization",
        selected_pattern="Incremental extraction",
        stakeholder_view="The shop keeps selling during migration.",
        technical_view="Seams first, then extraction.",
        migration_steps=[_step("Seam A"), _step("Extract B"), _step("Cut over C")],
    )

    restored = Blueprint.model_validate_json(blueprint.model_dump_json())

    assert [s.title for s in restored.migration_steps] == [
        "Seam A", "Extract B", "Cut over C",
    ]  # list order preserved
    assert restored.migration_steps[1].exit_condition == "Extract B validated."
    assert restored.migration_steps[0].coexistence_or_data_strategy


def test_old_blueprint_json_without_migration_steps_still_validates():
    # A checkpoint written before the field existed: no `migration_steps` key.
    old_json = json.dumps(
        {
            "project_name": "Old Run",
            "stakeholder_view": "x",
            "technical_view": "y",
        }
    )
    blueprint = Blueprint.model_validate_json(old_json)
    assert blueprint.migration_steps == []          # the default, on load


def test_empty_migration_plan_is_valid():
    blueprint = Blueprint(
        stakeholder_view="x", technical_view="y", migration_steps=[]
    )
    assert Blueprint.model_validate_json(
        blueprint.model_dump_json()
    ).migration_steps == []


def test_old_style_checkpoint_round_trips_through_persistence(tmp_path, monkeypatch):
    """A checkpoint file with NO migration field loads via the real
    persistence path — old saved History runs keep working, and their files
    are never rewritten."""
    monkeypatch.setenv("AI_ARCHITECT_RUNS_DIR", str(tmp_path / "runs"))

    state = new_run("Modernize a shop.")
    state.blueprint = Blueprint(
        project_name="Old Shop",
        stakeholder_view="s", technical_view="t",
    )
    state.stage = Stage.DONE
    save_state(state)
    # Rewrite the checkpoint exactly as an OLD writer would have: strip the
    # field the old schema did not have.
    checkpoint = next((tmp_path / "runs" / state.run_id).glob("*.json"))
    data = json.loads(checkpoint.read_text(encoding="utf-8"))
    del data["blueprint"]["migration_steps"]
    checkpoint.write_text(json.dumps(data), encoding="utf-8")

    loaded = load_state(state.run_id)

    assert loaded.blueprint.migration_steps == []   # default applied on load
    assert loaded.run_id == state.run_id
    # And the file on disk was not rewritten by loading it.
    assert "migration_steps" not in checkpoint.read_text(encoding="utf-8")


# ── 6. Reviewer: existing judgments carry the new disciplines ─────────────


def test_reviewer_flaw_detection_covers_decomposition_quality():
    prompt = _REV_PROMPT
    assert "DECOMPOSITION QUALITY" in prompt
    assert "exist only because a source module/package existed" in prompt
    # Weak decomposition is a concern…
    assert "fragmentation" in prompt and "boundary rationale" in prompt
    # …but a different decomposition than any reference passes.
    assert "do not insist services match" in prompt


def test_reviewer_adr_soundness_covers_technology_change_justification():
    prompt = _REV_PROMPT
    assert "TECHNOLOGY-CHANGE" in prompt
    assert "Technology novelty without a requirement-linked rationale" in prompt
    # Justified changes must be allowed.
    assert "must not be flagged merely for differing" in prompt


def test_reviewer_covers_migration_coherence_without_pm_detail():
    prompt = _REV_PROMPT
    assert "migration_steps" in prompt
    assert "big-bang rewrite" in prompt
    assert "coexist" in prompt
    # Architecture level only; greenfield not penalized.
    assert "do not demand project-management detail" in prompt
    assert "do not penalize a non-brownfield design" in prompt


def test_reviewer_schema_unchanged_five_judgments():
    """The disciplines fold into the EXISTING judgments — no new fields."""
    fields = set(rev.LLMJudgments.model_fields)
    assert fields == {
        "repo_grounding", "flaw_detection", "adr_soundness",
        "best_practice_grounding", "refinement_readiness",
    }


def test_architect_output_schema_carries_migration_steps():
    """The phase-2 response schema embeds Blueprint, so generation can
    produce the sequence without any second call or mapping layer."""
    assert "migration_steps" in Blueprint.model_fields
    assert arch.ArchitectureDesign.model_fields["blueprint"].annotation is Blueprint


# ── 7. Data-flow fidelity (audit of run 20260822T163342Z-953d62e7) ─────────
#
# A PASSING run still produced flows with a collapsed placeholder node,
# two Blueprint components absent from every flow, and unnamed aggregate
# event consumers — so the deterministic diagram showed a materially
# emptier architecture than the artifacts. The fix is this prompt contract.


def test_data_flow_rule_requires_concrete_component_names():
    prompt = _ARCH_PROMPT
    assert "DATA-FLOW FIDELITY" in prompt
    assert "concrete Blueprint component names" in prompt
    # …and the flows must carry the same architecture as the artifacts.
    assert "SAME" in prompt and "architecture a reader of the full artifacts" in prompt


def test_data_flow_rule_forbids_collapsed_grouping_nodes():
    prompt = _ARCH_PROMPT
    assert "Do NOT collapse multiple" in prompt
    # The generic forbidden examples are named so the failure mode is
    # unmistakable (placeholders, not any benchmark architecture).
    for example in ("Backend Services", "Downstream Systems"):
        assert example in prompt, example
    assert "write a separate flow per component" in prompt


def test_data_flow_rule_requires_named_event_owners():
    prompt = _ARCH_PROMPT
    assert "concrete named owners" in prompt
    assert "never unnamed aggregate labels" in prompt


def test_data_flow_rule_requires_component_coverage_with_exceptions():
    prompt = _ARCH_PROMPT
    assert "Every non-external Blueprint component must appear" in prompt
    assert "state that exception explicitly" in prompt
    # Architecture-level only, external participants preserved.
    assert "architecture-level" in prompt
    assert "explicit external participants" in prompt


def test_data_flow_rule_carries_diagram_fidelity_intent():
    prompt = _ARCH_PROMPT
    assert "feeds a deterministic diagram" in prompt


def test_data_flow_rule_adds_no_call_and_no_schema_change():
    """Prompt-only: the response schema still embeds Blueprint unchanged
    (no new field, no second model call, no mapping layer)."""
    design_fields = set(arch.ArchitectureDesign.model_fields)
    assert design_fields == {
        "blueprint", "adrs", "components", "directive_objection",
    }
    assert "data_flows" in Blueprint.model_fields  # the EXISTING field
    assert arch.ArchitectureDesign.model_fields["blueprint"].annotation is Blueprint


def test_data_flow_rule_examples_contain_no_benchmark_names():
    """The forbidden-placeholder examples are generic; no reference-repo
    architecture leaks into the prompt through the new rule."""
    prompt = _ARCH_PROMPT
    for name in _BENCHMARK_NAMES:
        assert name not in prompt, name
