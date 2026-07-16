"""Demo_reviewer.py — run the REAL Reviewer against use-case #1.

    python -m pipeline.demo_reviewer

Needs GEMINI_API_KEY in .env (one real Gemini call per scenario). Two runs:

  A. SEEDED FLAW — an architect output that looks complete (it passes every
     deterministic check: all artifacts, keywords covered, traceability intact)
     but keeps the monolith and "fixes" peak load with vertical scaling — i.e.
     it patches the ground-truth flaw instead of structurally fixing it. Only
     the LLM layer can catch this; the demo shows it does.

  B. SOUND DESIGN — the queue-decoupled decomposition that actually fixes the
     flaw. The Reviewer should pass it. Together A and B show the verdict is
     driven by the design's substance, not by formatting.

These crafted scenarios remain a manual Reviewer smoke demo. The reusable,
labeled agreement runner lives in eval/harness.py.
"""
from __future__ import annotations

from pipeline.agents import reviewer
from pipeline.review_checks import run_deterministic_checks
from pipeline.state import (
    ADR,
    ArchitectState,
    Blueprint,
    ComponentDescription,
    ContextRecord,
    Feature,
    KBChunk,
    Stage,
    new_run,
)

# Use-case #1 — same prompt as pipeline/run.py
UC1_PROMPT = (
    "Fix our monolithic online shop so it can scale for seasonal peak sales. "
    "It's on AWS, budget is medium, must stay GDPR-compliant, and needs to handle "
    "~50k concurrent users at peak. Repo: https://github.com/example/bugged-shop"
)


def _base_state() -> ArchitectState:
    """Use-case #1 up to the DESIGNING stage: locked context + researcher findings."""
    s = new_run(UC1_PROMPT)
    s.context_record = ContextRecord(
        summary="cloud: AWS\nbudget: medium\nscale: ~50k concurrent users at seasonal peak\n"
                "compliance: GDPR\nexisting system: monolithic online shop, brownfield "
                "(repo: https://github.com/example/bugged-shop)"
    )
    s.retrieved_knowledge = [
        KBChunk(content="Decouple services with an asynchronous message queue so bursts are "
                        "buffered instead of overwhelming synchronous request paths.",
                source="architecture_patterns.md"),
        KBChunk(content="Design web tiers stateless and externalize session state so instances "
                        "can scale horizontally behind a load balancer.",
                source="microservices-on-aws.pdf"),
    ]
    s.features = [
        Feature(id="F1", name="Survive seasonal peak load",
                scenario="Given a seasonal sale, when ~50k users browse and buy concurrently, "
                         "checkout completes without outage."),
        Feature(id="F2", name="GDPR-compliant order data",
                scenario="Given an EU customer order, personal data is stored encrypted in the EU "
                         "and can be deleted on request."),
    ]
    s.stage = Stage.DESIGNING
    return s


def seeded_flaw_state() -> ArchitectState:
    """Scenario A: complete-LOOKING design that PATCHES the flaw (vertical scaling)."""
    s = _base_state()
    s.blueprint = Blueprint(
        stakeholder_view="The existing shop stays as it is; we buy stronger servers so it "
                         "survives seasonal sales. Customers notice nothing new.",
        technical_view="Keep the monolith as one deployable on AWS. Upgrade the single EC2 "
                       "instance to a larger size before each seasonal peak and schedule nightly "
                       "restarts to clear memory. Sessions remain in process memory. Order data "
                       "stays in the existing PostgreSQL with disk encryption for GDPR "
                        "compliance. This scaling approach stays within the medium budget because "
                        "no code changes are needed for the expected concurrent peak load.",
        components=["Shop Monolith", "Order Module"],
        addressed_feature_ids=["F1", "F2"],
        constraints_addressed=["AWS", "medium budget", "peak load", "GDPR", "monolith"],
    )
    s.adrs = [
        ADR(id="ADR-001", title="ADR-1: Vertically scale the Shop Monolith for peak load",
            decision="Keep the existing monolith and move the Shop Monolith to a larger EC2 "
                      "instance class before sale events; cheapest option within the medium "
                      "budget and no migration risk.",
            related_feature_ids=["F1"],
            related_component_names=["Shop Monolith"]),
        ADR(id="ADR-002", title="ADR-2: Nightly restarts of the Order Module",
            decision="Schedule nightly restarts of the Shop Monolith so the Order Module's "
                      "memory leaks under load never accumulate; GDPR unaffected.",
            related_feature_ids=["F2"],
            related_component_names=["Order Module"]),
    ]
    s.components = [
        ComponentDescription(id="COMP-001", name="Shop Monolith",
                              description="The whole existing shop as one deployable on a large EC2 "
                                          "instance; handles browsing, checkout and orders (traces to F1).",
                              related_feature_ids=["F1"], related_adr_ids=["ADR-001"]),
        ComponentDescription(id="COMP-002", name="Order Module",
                              description="Order handling inside the monolith; stores GDPR-encrypted "
                                          "order data in PostgreSQL (traces to F1, F2).",
                              related_feature_ids=["F2"], related_adr_ids=["ADR-002"]),
    ]
    return s


def sound_design_state() -> ArchitectState:
    """Scenario B: design that structurally FIXES the flaw."""
    s = _base_state()
    s.blueprint = Blueprint(
        stakeholder_view="During seasonal sales, customers keep browsing and checking out without "
                         "outages: orders are accepted instantly and processed reliably in the "
                         "background, and EU customer data stays protected.",
        technical_view="Decompose the monolith around the load hotspot: a stateless web tier on "
                       "AWS ECS autoscaling behind an ALB (sessions in ElastiCache), a dedicated "
                       "CheckoutService that accepts orders and publishes them to an SQS queue, "
                       "and an OrderWorker consuming the queue asynchronously so peak bursts are "
                       "buffered instead of crashing the system. Order data lives in RDS "
                       "PostgreSQL (eu-central-1) encrypted at rest for GDPR. Autoscaling with "
                        "spot capacity keeps cost within the medium budget; the rest of the "
                        "existing monolith is migrated incrementally (strangler pattern).",
        components=["WebTier", "CheckoutService", "OrderWorker"],
        addressed_feature_ids=["F1", "F2"],
        constraints_addressed=["AWS", "medium budget", "peak load", "GDPR", "monolith migration"],
    )
    s.adrs = [
        ADR(id="ADR-001", title="ADR-1: Extract CheckoutService behind an SQS queue",
            decision="Split checkout out of the existing monolith and buffer orders through SQS "
                      "so ~50k concurrent peak users cannot saturate order processing "
                      "(retrieved: async queue decoupling, architecture_patterns.md). Trade-off: "
                      "eventual consistency for order confirmation, accepted for availability.",
            related_feature_ids=["F1"],
            related_component_names=["CheckoutService"]),
        ADR(id="ADR-002", title="ADR-2: Stateless WebTier with externalized sessions",
            decision="Run the WebTier stateless on ECS with sessions in ElastiCache so it "
                      "scales horizontally during peak load (retrieved: stateless web tier, "
                      "microservices-on-aws.pdf). Trade-off: added infrastructure cost, kept "
                      "within the medium budget via autoscaling down off-peak.",
            related_feature_ids=["F1"],
            related_component_names=["WebTier"]),
        ADR(id="ADR-003", title="ADR-3: OrderWorker with EU data residency for GDPR",
            decision="Process queued orders in a dedicated OrderWorker writing to RDS in "
                      "eu-central-1 with encryption at rest, satisfying GDPR compliance and "
                      "right-to-erasure. Trade-off: single-region latency, accepted.",
            related_feature_ids=["F2"],
            related_component_names=["OrderWorker"]),
    ]
    s.components = [
        ComponentDescription(id="COMP-001", name="WebTier",
                              description="Stateless storefront on ECS autoscaling behind an ALB; absorbs "
                                          "peak browsing load (traces to F1).",
                              related_feature_ids=["F1"], related_adr_ids=["ADR-002"]),
        ComponentDescription(id="COMP-002", name="CheckoutService",
                              description="Accepts checkouts and publishes orders to the SQS queue; keeps "
                                          "checkout responsive at peak (traces to F1).",
                              related_feature_ids=["F1"], related_adr_ids=["ADR-001"]),
        ComponentDescription(id="COMP-003", name="OrderWorker",
                              description="Consumes the queue asynchronously and persists GDPR-encrypted "
                                          "order data in the EU (traces to F1, F2).",
                              related_feature_ids=["F2"], related_adr_ids=["ADR-003"]),
    ]
    return s


def _run_scenario(label: str, state: ArchitectState) -> None:
    print(f"\n{'=' * 74}\nSCENARIO {label}\n{'=' * 74}")
    checks = run_deterministic_checks(state)
    print("\n-- Layer 1: deterministic checks (code, no LLM) --")
    print(f"  artifacts present       : {checks.artifacts_present}")
    print(f"  constraints covered     : {checks.constraints_covered}")
    print(f"  feature->component gaps : {checks.features_without_component or 'none'}")
    print(f"  component->ADR gaps     : {checks.components_without_adr or 'none'}")
    print(f"  [code] rubric scores    : artifacts={checks.score_all_artifacts_present} "
          f"constraints={checks.score_constraint_coverage} "
          f"traceability={checks.score_traceability} adr_presence={checks.score_adr_presence}")

    print("\n-- Layer 2: Gemini qualitative judgment + merge --")
    out = reviewer.reviewer_node(state)
    if out["stage"] is Stage.FAILED:
        print(f"  REVIEWER FAILED: {out.get('errors')}")
        return
    report = out["review"]
    print(report.model_dump_json(indent=2))
    print(f"\n  -> stage out: {out['stage'].value.upper()}"
          f"  |  tokens so far in/out: {state.input_tokens}/{state.output_tokens}")


if __name__ == "__main__":
    _run_scenario("A - seeded flaw (monolith + vertical-scaling patch)", seeded_flaw_state())
    _run_scenario("B - sound design (queue-decoupled decomposition)", sound_design_state())
