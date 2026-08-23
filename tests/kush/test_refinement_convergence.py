"""test_refinement_convergence.py — the three convergence improvements from
the forensic audit of run 20260822T160643Z-8e49d732, without raising the
refinement cap.

  1. Constraint findings carry the EXACT uncovered requirement strings
     (group labels alone left the architect re-paraphrasing).
  2. The REFINEMENT-turn prompt carries blocker-by-blocker repair
     discipline; the initial-design prompt does not.
  3. The prompt anchors technology conservatism to the run's ACTUAL
     detected stack, derived from the repository analysis — never
     hard-coded.

All offline; no LLM calls. The real run's requirement strings appear only
as generic fixture data of the same SHAPE, per the audit.
"""

from __future__ import annotations

from pipeline.agents import architect as arch
from pipeline.review_checks import run_deterministic_checks
from pipeline.state import (
    ADR,
    Blueprint,
    ComponentDescription,
    ContextRecord,
    Feature,
    RepoRepresentation,
    RepoStructure,
    ReviewResult,
    TechStack,
    new_run,
)


# ── shared fixtures ────────────────────────────────────────────────────────


def _constraint_state(functional, existing, design_text_additions):
    """A state whose design text mentions everything in
    `design_text_additions` (plus fixed filler) but nothing else."""
    state = new_run("Modernize a monolithic shop.")
    state.context_record = ContextRecord(
        project_name="Shop",
        business_goal="Survive peaks.",
        problem_statement="The monolith saturates.",
        functional_requirements=functional,
        existing_systems=existing,
    )
    filler = " ".join(design_text_additions)
    state.features = [
        Feature(
            id="FEAT-001", name="Cart", scenario="Customers buy.",
            acceptance_criteria=["Cart works."],
        )
    ]
    state.blueprint = Blueprint(
        project_name="Shop",
        selected_pattern="Queue-buffered extraction",
        stakeholder_view=filler,
        technical_view=filler,
        addressed_feature_ids=["FEAT-001"],
    )
    state.adrs = [
        ADR(
            id="ADR-001", title="ADR-1: Buffer the peak",
            context=filler, decision="Queue the peak.",
            rationale=filler, alternatives_considered=["Scale up"],
            positive_consequences=["Isolation."],
            negative_consequences=["Eventual consistency."],
        )
    ]
    state.components = [
        ComponentDescription(
            id="COMP-001", name="Queue Front", purpose=filler,
            description=filler, related_feature_ids=["FEAT-001"],
            related_adr_ids=["ADR-001"],
        )
    ]
    return state


# ── 1. exact constraint diagnostics ───────────────────────────────────────


def test_uncovered_requirements_reported_individually_not_as_group_only():
    state = _constraint_state(
        functional=[
            "browsing products",          # covered below
            "executing checkouts",        # NOT covered
            "viewing order history",      # NOT covered
        ],
        existing=["Django-based monolithic backend", "React frontend"],
        design_text_additions=[
            "browsing products", "Django-based monolithic backend",
            "order management",  # not 'order history'
        ],
    )

    checks = run_deterministic_checks(state)

    # The exact strings, per group…
    assert "executing checkouts" in checks.constraints_uncovered["functional"]
    assert "viewing order history" in checks.constraints_uncovered["functional"]
    assert "React frontend" in checks.constraints_uncovered["existing_system"]
    # …covered requirements are NOT listed.
    assert "browsing products" not in checks.constraints_uncovered["functional"]
    assert "Django-based monolithic backend" not in (
        checks.constraints_uncovered["existing_system"]
    )


def test_constraint_issue_evidence_and_fix_carry_the_exact_strings():
    state = _constraint_state(
        functional=["executing checkouts"],
        existing=["React frontend"],
        design_text_additions=["nothing relevant"],
    )

    checks = run_deterministic_checks(state)
    issue = next(
        i for i in checks.issues if i.category == "constraint"
    )

    # The strings LEAD both fields, so the refinement instruction's tail
    # clipping cannot cut them.
    assert issue.evidence.startswith(
        "Uncovered requirement(s): functional: 'executing checkouts'"
    )
    assert "'React frontend'" in issue.evidence
    assert issue.suggested_fix.startswith("Name every uncovered requirement")
    assert "'executing checkouts'" in issue.suggested_fix
    assert "'React frontend'" in issue.suggested_fix


def test_constraint_scoring_and_verdict_semantics_unchanged():
    """Same pass/fail and same score as before the enrichment: a partially
    covered group set still scores 1, a fully covered one scores 2."""
    partial = _constraint_state(
        functional=["executing checkouts"],         # uncovered
        existing=["Django-based monolithic backend"],  # covered below
        design_text_additions=["Django-based monolithic backend"],
    )
    assert run_deterministic_checks(partial).score_constraint_coverage == 1

    full = _constraint_state(
        functional=["executing checkouts"],
        existing=["Django-based monolithic backend"],
        design_text_additions=["executing checkouts",
                               "Django-based monolithic backend"],
    )
    checks = run_deterministic_checks(full)
    assert checks.score_constraint_coverage == 2
    assert checks.constraints_uncovered["functional"] == []
    assert not any(i.category == "constraint" for i in checks.issues)


def test_wording_variants_that_already_pass_still_pass():
    """Token-overlap matching is untouched: 'shopping cart' evidence still
    covers 'managing shopping cart' (2/3 tokens)."""
    state = _constraint_state(
        functional=["managing shopping cart"],
        existing=[],
        design_text_additions=["the shopping cart is durable"],
    )
    checks = run_deterministic_checks(state)
    assert checks.constraints_covered["functional"] is True
    assert checks.constraints_uncovered["functional"] == []


# ── 2. refinement prompt discipline ───────────────────────────────────────


def _architect_state(repo: RepoRepresentation | None, revising: bool):
    state = new_run("Modernize a monolithic shop.")
    state.context_record = ContextRecord(
        project_name="Shop", business_goal="g", problem_statement="p",
        functional_requirements=["orders"],
    )
    state.repo_representation = repo
    state.features = [Feature(
        id="FEAT-001", name="Orders", scenario="s",
        acceptance_criteria=["a"],
    )]
    if revising:
        state.review = ReviewResult(
            overall_status="fail", requires_refinement=True,
            refinement_instruction="Tighten the design.",
        )
        state.blueprint = Blueprint(
            project_name="Shop", stakeholder_view="s", technical_view="t",
        )
        state.adrs = [ADR(
            id="ADR-001", title="ADR-1: x", context="c", decision="d",
            rationale="r", alternatives_considered=["a"],
            positive_consequences=["p"], negative_consequences=["n"],
        )]
        state.components = [ComponentDescription(
            id="COMP-001", name="Front", purpose="p", description="d",
        )]
    return state


def test_refinement_prompt_carries_the_discipline_checklist():
    prompt = arch._build_architecture_prompt(
        _architect_state(repo=None, revising=True),
        [Feature(id="FEAT-001", name="Orders", scenario="s",
                 acceptance_criteria=["a"])],
    )
    flat = " ".join(prompt.split())

    assert "<refinement_discipline>" in prompt
    assert "Address EVERY supplied HIGH/blocking finding" in flat
    assert "verify each supplied blocker" in flat
    assert "SMALLEST changes" in flat
    assert "Re-emit ALL required fields for EVERY ADR and Component" in flat
    assert "Never leave a required field blank merely because it was unchanged" in flat
    assert "Do not drop them due to response compression" in flat


def test_initial_design_prompt_has_no_refinement_discipline():
    prompt = arch._build_architecture_prompt(
        _architect_state(repo=None, revising=False),
        [Feature(id="FEAT-001", name="Orders", scenario="s",
                 acceptance_criteria=["a"])],
    )

    assert "<refinement_discipline>" not in prompt
    assert "REFINEMENT pass" not in prompt


# ── 3. detected-stack anchoring ───────────────────────────────────────────


def _repo(languages, frameworks):
    return RepoRepresentation(
        structure=RepoStructure(
            tech_stack=TechStack(languages=languages, frameworks=frameworks)
        )
    )


def _prompt_for(repo):
    return arch._build_architecture_prompt(
        _architect_state(repo=repo, revising=False),
        [Feature(id="FEAT-001", name="Orders", scenario="s",
                 acceptance_criteria=["a"])],
    )


def test_django_python_fixture_names_the_detected_stack():
    prompt = _prompt_for(
        _repo({"Python": 18000, "JavaScript": 4000}, ["Django", "React"])
    )
    flat = " ".join(prompt.split())

    assert "<detected_existing_stack>" in prompt
    assert "Python (18,000 LOC)" in prompt
    assert "Django, React" in flat
    assert "Preserve these technologies by default" in flat
    assert "'better scalability' is not enough" in flat


def test_other_stack_fixture_reflects_that_actual_stack():
    prompt = _prompt_for(_repo({"Java": 52000}, ["Spring Boot", "Hibernate"]))

    assert "Java (52,000 LOC)" in prompt
    assert "Spring Boot, Hibernate" in prompt
    # Not hard-coded to the first fixture's stack.
    assert "Django" not in prompt
    assert "Python" not in prompt


def test_missing_stack_degrades_gracefully_without_inventing():
    # Greenfield: no repo representation at all.
    assert "<detected_existing_stack>" not in _prompt_for(None)

    # Repo present but nothing recognised — honest sentence, no inventions.
    prompt = _prompt_for(_repo({}, []))
    assert "<detected_existing_stack>" in prompt
    assert "No specific language or framework was identified" in prompt
    assert "Django" not in prompt and "Python" not in prompt


def test_stack_block_present_on_refinement_turns_too():
    prompt = arch._build_architecture_prompt(
        _architect_state(repo=_repo({"Python": 1000}, ["Django"]), revising=True),
        [Feature(id="FEAT-001", name="Orders", scenario="s",
                 acceptance_criteria=["a"])],
    )

    assert "<detected_existing_stack>" in prompt
    assert "<refinement_discipline>" in prompt   # both anchors together


def test_no_benchmark_names_in_new_prompt_construction():
    prompt = _prompt_for(_repo({"Python": 100}, ["Django"]))
    for banned in ("ecommerce-microservice", "harsh020", "Cart Service",
                   "Ordering Aggregator"):
        assert banned not in prompt, banned
