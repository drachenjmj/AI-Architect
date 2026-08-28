"""ui_demo.py — DEV ONLY. Fully-populated states for screenshots, no API calls.

    streamlit run ui.py -- --demo          # a clean passing run
    streamlit run ui.py -- --demo capped   # a run stopped on the refine budget

WHY THIS EXISTS
---------------
The DONE screen has to be filmed and screenshotted repeatedly for the slides
and the assessed video. Doing that with live runs burns API quota, takes a
minute each time, and — because our knowledge base currently returns zero
chunks and most demo prompts are greenfield — a live run cannot populate the
retrieval or repository sections at all. So this module builds the state a
*complete* run would have produced and drops it straight into
`st.session_state`.

NOT A TEST FIXTURE, NOT PIPELINE CODE. Nothing here runs during a real run;
`ui.py` only calls it when the `--demo` flag is on the command line. It makes
no LLM calls and writes no checkpoints.

HOW IT RELATES TO eval/scenarios.py
-----------------------------------
The two labeled scenarios there (`sound_design_state`, `seeded_flaw_state`)
already carry a context record, features, a blueprint, ADRs and components, so
they are the base. They stop at `DESIGNING` though — they exist to be fed TO
the reviewer, so they carry no review, no trace and no token counts. This
module layers those on top. `eval/scenarios.py` is left untouched so the
Reviewer alignment evaluation keeps measuring exactly what it measured before.
"""
from __future__ import annotations

from pipeline.state import (
    ArchitectState,
    DependencyEdge,
    JudgmentReasons,
    KBChunk,
    PartitionSummary,
    RepoBehavior,
    RepoStructure,
    ReviewIssue,
    ReviewResult,
    RubricScores,
    Stage,
    StepLog,
    TechStack,
)

# Model IDs are the REAL ones from pipeline/llm.py, so the demo's cost figures
# are computed by exactly the same code path as a live run's.
_LITE = "gemini-3.1-flash-lite"
_FLASH = "gemini-3.5-flash"

DEMO_VARIANTS = ("pass", "capped")


# ── shared enrichment ────────────────────────────────────────────────────
def _add_repo_detail(state: ArchitectState) -> None:
    """Fill in the parts of the repo representation the eval fixtures omit."""
    repo = state.repo_representation
    if repo is None:
        return
    repo.meta.commit_sha = "9f2c1ab4d77e0b3a5c81ee62f0d4a9b7c3e15082"
    repo.meta.clone_path = ".cache/repos/bugged-shop"
    repo.structure = RepoStructure(
        file_tree=(
            "shop/\n"
            "  app.py            412 LOC\n"
            "  checkout.py       288 LOC\n"
            "  orders.py         241 LOC\n"
            "  catalog.py        176 LOC\n"
            "  models.py         133 LOC\n"
            "templates/\n"
            "  checkout.html      64 LOC\n"
            "requirements.txt\n"
            "docker-compose.yml"
        ),
        repo_map=(
            "app.py: ShopApp — serves catalog, checkout and orders in one "
            "process\n"
            "  def create_app() -> Flask\n"
            "  def handle_checkout(cart: Cart) -> Order\n"
            "checkout.py: calls orders.py synchronously inside the request\n"
            "  def submit(cart: Cart) -> Order\n"
            "orders.py: writes orders, sends confirmation mail inline\n"
            "  def place(order: Order) -> None"
        ),
        dependency_edges=[
            DependencyEdge(source="shop/app.py", target="shop/checkout.py"),
            DependencyEdge(source="shop/app.py", target="shop/catalog.py"),
            DependencyEdge(source="shop/checkout.py", target="shop/orders.py"),
            DependencyEdge(source="shop/orders.py", target="shop/models.py"),
        ],
        architecture_diagram=(
            "graph TD\n"
            "    app[shop/app.py]\n"
            "    checkout[shop/checkout.py]\n"
            "    catalog[shop/catalog.py]\n"
            "    orders[shop/orders.py]\n"
            "    models[shop/models.py]\n"
            "    app --> checkout\n"
            "    app --> catalog\n"
            "    checkout --> orders\n"
            "    orders --> models"
        ),
        tech_stack=TechStack(
            languages={"Python": 1250, "HTML": 340, "SQL": 96},
            frameworks=["Flask", "SQLAlchemy"],
            dependencies=["flask", "sqlalchemy", "psycopg2", "stripe", "jinja2"],
            external_services=["PostgreSQL", "Redis"],
        ),
        integration_interface=(
            "POST /checkout      submit a cart, returns an order\n"
            "GET  /orders/{id}   read one order\n"
            "GET  /catalog       list products"
        ),
    )
    repo.behavior = RepoBehavior(
        overview=(
            "One Flask deployable serves browsing, checkout and order "
            "processing. Checkout calls order placement synchronously inside "
            "the request, and sessions are held in process memory, so the "
            "whole shop scales only as far as a single instance does."
        ),
        partitions=[
            PartitionSummary(
                name="shop/checkout",
                paths=["shop/checkout.py", "templates/checkout.html"],
                role="Owns the purchase path from cart to confirmed order.",
                functionality=(
                    "Validates the cart, takes payment through Stripe, then "
                    "calls orders.place() in the same request — the blocking "
                    "call that saturates the app under peak load."
                ),
            ),
            PartitionSummary(
                name="shop/orders",
                paths=["shop/orders.py", "shop/models.py"],
                role="Persists orders and notifies the customer.",
                functionality=(
                    "Writes the order row and sends confirmation mail inline, "
                    "so a slow mail server directly slows checkout."
                ),
            ),
            PartitionSummary(
                name="shop/catalog",
                paths=["shop/catalog.py"],
                role="Read-only product browsing.",
                functionality="Serves product lists and detail pages from PostgreSQL.",
            ),
        ],
    )


def _add_knowledge(state: ArchitectState) -> None:
    """Realistic retrieval results across all three boxes."""
    state.retrieved_knowledge = [
        KBChunk(
            content=(
                "Introduce a queue at an asynchronous boundary to buffer bursts. "
                "The producer accepts work at request rate while the consumer "
                "drains it at its own sustainable rate, so a traffic spike "
                "becomes queue depth rather than dropped requests."
            ),
            source="architecture_patterns.md",
            page=42,
            box=1,
            distance=0.1873,
        ),
        KBChunk(
            content=(
                "A stateless web tier scales horizontally: with no session data "
                "held in process, any instance can serve any request and "
                "instances can be added or removed freely behind a load balancer."
            ),
            source="architecture_patterns.md",
            page=17,
            box=1,
            distance=0.2140,
        ),
        KBChunk(
            content=(
                "Under GDPR, personal data of EU residents should be processed "
                "and stored in an EU region. Encrypt at rest and in transit, and "
                "record the lawful basis for each processing purpose."
            ),
            source="gdpr-for-architects.pdf",
            page=8,
            box=2,
            distance=0.3055,
        ),
        KBChunk(
            content=(
                "Amazon SQS standard queues offer effectively unlimited "
                "throughput with at-least-once delivery; pair them with an "
                "idempotent consumer and a dead-letter queue for poison messages."
            ),
            source="https://docs.aws.amazon.com/sqs/latest/dg/welcome.html",
            page=0,
            box=3,
            distance=None,
        ),
    ]


def _totals_from_history(state: ArchitectState) -> None:
    """Set the run totals from the trace, exactly as the reducers would.

    Keeps the demo self-consistent: the status strip, the per-agent table and
    the trace all add up, because they are all derived from the same steps.
    """
    state.input_tokens = sum(step.input_tokens for step in state.history)
    state.output_tokens = sum(step.output_tokens for step in state.history)


def _step(
    agent: str,
    stage_in: Stage,
    stage_out: Stage,
    note: str,
    clock: str,
    model: str = "",
    tokens: tuple[int, int] = (0, 0),
) -> StepLog:
    """One trace entry, priced by the same function the pipeline uses."""
    from pipeline.llm import estimate_cost_usd

    cost = estimate_cost_usd(model, tokens[0], tokens[1]) if model else 0.0
    return StepLog(
        agent=agent,
        stage_in=stage_in,
        stage_out=stage_out,
        note=note,
        timestamp=f"2026-08-17T{clock}+00:00",
        model=model,
        input_tokens=tokens[0],
        output_tokens=tokens[1],
        cost_usd=cost or 0.0,
    )


# ── variant: a clean passing run ─────────────────────────────────────────
def demo_pass_state() -> ArchitectState:
    """A complete run that passed review on the second pass."""
    from eval.scenarios import sound_design_state

    state = sound_design_state()
    _add_repo_detail(state)
    _add_knowledge(state)

    state.clarifying_questions = []
    state.clarification_answers = {
        "Which AWS region must EU order data stay in?": "eu-central-1 (Frankfurt).",
        "What is the acceptable checkout latency at peak?": (
            "Under 500 ms at the 95th percentile, even at 50k concurrent users."
        ),
    }
    state.context_record.business_goal = (
        "Capture seasonal peak revenue instead of losing it to outages."
    )
    state.context_record.users = [
        "Retail customers (up to 50k concurrent at peak)",
        "Warehouse operations team",
        "Customer support agents",
    ]
    state.context_record.functional_requirements = [
        "Browse the product catalog",
        "Complete checkout and payment",
        "Track order status after purchase",
    ]
    state.context_record.assumptions = [
        "Existing Stripe payment integration is retained.",
        "Order volume grows at most 3x year over year.",
    ]
    state.context_record.open_questions = [
        "Is a full catalog migration in scope for the same release?",
    ]

    # The eval fixtures carry only the fields the Reviewer scores, so the rest
    # of each artifact is filled in here — the demo screen exists precisely to
    # show every field, including the ones a thin run leaves empty.
    state.features[0].description = (
        "Checkout must stay available and responsive while seasonal traffic "
        "peaks, without manual intervention on the day."
    )
    state.features[0].acceptance_criteria = [
        "Checkout p95 latency stays under 500 ms at 50k concurrent users.",
        "No checkout request is dropped during a 3x traffic spike.",
        "Order processing catches up within 5 minutes of the peak ending.",
    ]
    state.features[0].related_requirement_ids = ["NFR-001", "NFR-002"]
    state.features[1].description = (
        "Order data belonging to EU customers must stay within the EU and be "
        "encrypted throughout its lifecycle."
    )
    state.features[1].priority = "must"
    state.features[1].acceptance_criteria = [
        "All order records are stored in eu-central-1.",
        "Order data is encrypted at rest and in transit.",
        "Data-deletion requests complete within 30 days.",
    ]
    state.features[1].related_requirement_ids = ["COMP-GDPR-001"]

    state.blueprint.project_name = "Seasonal Shop"
    state.blueprint.selected_pattern = "Event-driven services behind a managed queue"
    state.blueprint.rationale = (
        "The repository analysis shows one deployable whose checkout path calls "
        "order placement synchronously, so peak traffic saturates the whole "
        "shop at once. Putting a managed queue on that boundary lets the "
        "customer-facing tier scale horizontally and stay stateless, while "
        "order processing drains at its own rate — the retrieved "
        "queue-buffering pattern applied to the exact coupling found in the "
        "code. Managed services keep it inside the medium budget."
    )
    state.blueprint.data_flows = [
        "Customer → Checkout Service: cart submission over HTTPS",
        "Checkout Service → SQS: order placement message (EU region)",
        "SQS → Order Worker: at-least-once delivery, idempotent consumer",
        "Order Worker → PostgreSQL: encrypted order record in eu-central-1",
        "Order Worker → notification service: asynchronous confirmation mail",
    ]
    state.blueprint.assumptions = [
        "The existing Stripe integration is reused unchanged.",
        "Catalog browsing stays on the current stack for this release.",
    ]
    state.blueprint.open_risks = [
        "Eventual consistency means order confirmation is not instant; the "
        "customer-facing copy has to reflect that.",
        "Queue depth needs alerting, or a slow consumer hides as latency.",
    ]

    state.components[0].component_type = "service"
    state.components[0].inputs = [
        "Cart submission from the storefront",
        "Payment authorisation result from Stripe",
    ]
    state.components[0].outputs = [
        "Order placement message onto SQS",
        "Immediate acceptance response to the customer",
    ]
    state.components[0].dependencies = ["SQS order queue", "Stripe", "Catalog service"]
    state.components[0].security_considerations = [
        "TLS termination at the load balancer; no card data stored.",
        "Session state held in the client token, not in process memory.",
    ]
    state.components[1].component_type = "worker"
    state.components[1].inputs = ["Order placement messages from SQS"]
    state.components[1].outputs = [
        "Persisted order record",
        "Customer confirmation notification",
    ]
    state.components[1].dependencies = [
        "SQS order queue",
        "PostgreSQL (eu-central-1)",
        "Notification service",
    ]

    state.history = [
        _step("repo_ingestor", Stage.CREATED, Stage.INGESTING,
              "cloned bugged-shop at 9f2c1ab; mapped 8 files, 4 import edges",
              "09:12:04", _LITE, (2140, 812)),
        _step("clarifier", Stage.INGESTING, Stage.AWAITING_HUMAN,
              "missing 2 critical fact(s); asked 2", "09:12:19", _LITE, (612, 470)),
        _step("clarifier", Stage.AWAITING_HUMAN, Stage.CLARIFYING,
              "context locked; 2 assumption(s) recorded", "09:13:47", _LITE, (988, 356)),
        _step("researcher", Stage.CLARIFYING, Stage.RESEARCHING,
              "retrieved 4 KB chunks (3 curated, 1 web fallback)", "09:13:52"),
        _step("architect", Stage.RESEARCHING, Stage.DESIGNING,
              "derived 2 feature(s); generated blueprint, 2 ADR(s), 2 component(s)",
              "09:14:26", _FLASH, (4210, 2680)),
        _step("reviewer", Stage.DESIGNING, Stage.REFINING,
              "fail; 2 issue(s)", "09:14:44", _LITE, (3318, 402)),
        _step("refine_gate", Stage.REFINING, Stage.REFINING,
              "refine 1/2 → architect", "09:14:45"),
        _step("architect", Stage.REFINING, Stage.DESIGNING,
              "redesigned from reviewer instruction; 2 ADR(s), 2 component(s)",
              "09:15:11", _FLASH, (5024, 2912)),
        _step("reviewer", Stage.DESIGNING, Stage.DONE,
              "pass; 0 issue(s)", "09:15:29", _LITE, (3402, 388)),
    ]
    state.refine_iterations = 1
    state.stopped_on_cap = False
    state.review = ReviewResult(
        overall_status="pass",
        rubric_scores=RubricScores(
            all_artifacts_present=2,
            constraint_coverage=2,
            traceability=2,
            adr_presence=2,
            repo_grounding=True,
            flaw_detection=True,
            adr_soundness=True,
            best_practice_grounding=True,
            refinement_readiness=True,
        ),
        judgment_reasons=JudgmentReasons(
            repo_grounding=(
                "The design names the actual coupling found in the repository: "
                "checkout.py calling orders.place() synchronously inside the "
                "request, which is the documented cause of peak-load failure."
            ),
            flaw_detection=(
                "The synchronous checkout-to-order path is identified and "
                "removed by an SQS boundary, rather than patched by scaling the "
                "monolith vertically."
            ),
            adr_soundness=(
                "Both ADRs state a decision, the alternative rejected, and the "
                "cost of the trade-off; ADR-001 explicitly accepts eventual "
                "consistency in exchange for independent scaling."
            ),
            best_practice_grounding=(
                "The queue-at-an-asynchronous-boundary pattern and the stateless "
                "web-tier guidance are both cited from the retrieved chunks."
            ),
            refinement_readiness=(
                "Every component names a technology and a scaling approach, so "
                "an implementation team can act on the design without a further "
                "round of questions."
            ),
        ),
        issues=[],
        requires_refinement=False,
        refinement_instruction="",
    )
    _totals_from_history(state)
    state.stage = Stage.DONE
    return state


# ── variant: stopped on the refine budget ────────────────────────────────
def demo_capped_state() -> ArchitectState:
    """A run that burned both refine iterations and still did not pass.

    The honest failure case, and the one the old screen reported as a green
    "Design complete." This is the shape of the real capped run in
    `.cache/runs/20260817T062320Z-a7229bdb/`.
    """
    from eval.scenarios import seeded_flaw_state

    state = seeded_flaw_state()
    _add_repo_detail(state)
    _add_knowledge(state)

    state.clarification_answers = {
        "Which AWS region must EU order data stay in?": "eu-central-1 (Frankfurt).",
    }
    state.context_record.business_goal = (
        "Stop losing seasonal peak revenue to checkout outages."
    )
    state.context_record.users = ["Retail customers", "Warehouse operations team"]
    state.context_record.open_questions = [
        "Is there budget for a migration, or must the monolith be kept?",
        "Who owns the order data retention policy?",
    ]

    # Same enrichment as the passing variant, but describing the weaker design:
    # every field is populated so the screen can be checked field by field.
    state.features[0].description = (
        "The shop must survive seasonal peak traffic without falling over."
    )
    state.features[0].acceptance_criteria = [
        "The shop stays reachable during the seasonal peak.",
        "Checkout completes for 50k concurrent users.",
    ]
    state.features[0].related_requirement_ids = ["NFR-001"]
    state.features[1].description = (
        "EU order data must remain in the EU and be encrypted."
    )
    state.features[1].acceptance_criteria = [
        "Order data is encrypted at rest.",
        "Order records stay in an EU region.",
    ]
    state.features[1].related_requirement_ids = ["COMP-GDPR-001"]

    state.blueprint.project_name = "Seasonal Shop"
    state.blueprint.selected_pattern = "Vertically scaled monolith"
    state.blueprint.rationale = (
        "Keeping the existing deployable avoids migration cost and fits the "
        "medium budget. A larger instance class absorbs the seasonal peak, and "
        "a nightly restart clears the memory growth seen in production."
    )
    state.blueprint.data_flows = [
        "Customer → Shop Monolith: browsing and checkout in one process",
        "Shop Monolith → Order Module: synchronous in-process call",
        "Order Module → PostgreSQL: encrypted order record",
    ]
    state.blueprint.assumptions = [
        "Peak load stays within what one enlarged instance can serve.",
        "A nightly restart window is acceptable to the business.",
    ]
    state.blueprint.open_risks = [
        "The instance ceiling is a hard limit with no path beyond it.",
        "A single deployable remains a single point of failure.",
    ]

    state.components[0].component_type = "monolith"
    state.components[0].inputs = ["HTTP requests from customers"]
    state.components[0].outputs = ["Rendered pages", "Order records"]
    state.components[0].dependencies = ["PostgreSQL", "Redis", "Stripe"]
    state.components[0].security_considerations = [
        "TLS at the load balancer; sessions held in process memory.",
    ]
    state.components[1].component_type = "module"
    state.components[1].inputs = ["In-process order placement calls"]
    state.components[1].outputs = ["Persisted order record", "Confirmation mail"]
    state.components[1].dependencies = ["PostgreSQL", "SMTP relay"]
    state.components[1].scalability_considerations = [
        "Scales only with the monolith it lives inside.",
    ]

    state.history = [
        _step("repo_ingestor", Stage.CREATED, Stage.INGESTING,
              "cloned bugged-shop at 9f2c1ab; mapped 8 files, 4 import edges",
              "06:23:20", _LITE, (2140, 812)),
        _step("clarifier", Stage.INGESTING, Stage.AWAITING_HUMAN,
              "missing 2 critical fact(s); asked 2", "06:23:34", _LITE, (601, 471)),
        _step("clarifier", Stage.AWAITING_HUMAN, Stage.CLARIFYING,
              "context locked; 2 assumption(s) recorded", "06:24:52", _LITE, (693, 300)),
        _step("researcher", Stage.CLARIFYING, Stage.RESEARCHING,
              "retrieved 4 KB chunks (3 curated, 1 web fallback)", "06:24:56"),
        _step("architect", Stage.RESEARCHING, Stage.DESIGNING,
              "derived 2 feature(s); generated blueprint, 1 ADR(s), 3 component(s)",
              "06:25:29", _LITE, (1676, 1850)),
        _step("reviewer", Stage.DESIGNING, Stage.REFINING,
              "fail; 2 issue(s)", "06:25:47", _LITE, (3321, 332)),
        _step("refine_gate", Stage.REFINING, Stage.REFINING,
              "refine 1/2 → architect", "06:25:48"),
        _step("architect", Stage.REFINING, Stage.DESIGNING,
              "redesigned from reviewer instruction; 2 ADR(s), 4 component(s)",
              "06:26:14", _LITE, (1733, 2173)),
        _step("reviewer", Stage.DESIGNING, Stage.REFINING,
              "fail; 3 issue(s)", "06:26:31", _LITE, (3633, 442)),
        _step("refine_gate", Stage.REFINING, Stage.REFINING,
              "refine 2/2 → architect", "06:26:32"),
        _step("architect", Stage.REFINING, Stage.DESIGNING,
              "redesigned from reviewer instruction; 2 ADR(s), 2 component(s)",
              "06:26:58", _LITE, (1738, 2069)),
        _step("reviewer", Stage.DESIGNING, Stage.REFINING,
              "fail; 2 issue(s)", "06:27:15", _LITE, (3282, 405)),
        _step("refine_gate", Stage.REFINING, Stage.DONE,
              "cap reached (max_iterations (2)); accepting best-effort design",
              "06:27:16"),
    ]
    state.refine_iterations = 2
    state.stopped_on_cap = True
    state.review = ReviewResult(
        overall_status="fail",
        rubric_scores=RubricScores(
            all_artifacts_present=2,
            constraint_coverage=2,
            traceability=2,
            adr_presence=2,
            repo_grounding=True,
            flaw_detection=True,
            adr_soundness=True,
            best_practice_grounding=False,
            refinement_readiness=False,
        ),
        judgment_reasons=JudgmentReasons(
            repo_grounding=(
                "The design refers to the Shop Monolith and Order Module that "
                "the repository analysis actually found."
            ),
            flaw_detection=(
                "The peak-load bottleneck is named, though the response is to "
                "scale the same deployable vertically rather than to remove the "
                "synchronous coupling that causes it."
            ),
            adr_soundness=(
                "Both ADRs record a decision and a rejected alternative with "
                "their consequences."
            ),
            best_practice_grounding=(
                "Recommendations are not grounded in the retrieved knowledge: "
                "the queue-at-an-asynchronous-boundary chunk was retrieved and "
                "then not used, and no source is cited for the GDPR claims."
            ),
            refinement_readiness=(
                "Vertical scaling plus a nightly restart is an operational "
                "workaround, not an implementable architecture — no instance "
                "class, ceiling, or failover behaviour is specified."
            ),
        ),
        issues=[
            ReviewIssue(
                id="LLM-1",
                severity="high",
                category="grounding",
                finding="Retrieved best-practice knowledge was not applied.",
                evidence=(
                    "The retrieved chunk on buffering bursts behind a queue is "
                    "absent from the blueprint, which keeps the synchronous "
                    "checkout-to-order call in place."
                ),
                suggested_fix=(
                    "Introduce an asynchronous boundary between checkout and "
                    "order processing and cite the retrieved pattern in the ADR."
                ),
                requires_refinement=True,
            ),
            ReviewIssue(
                id="LLM-2",
                severity="medium",
                category="repo_alignment",
                finding=(
                    "Vertical scaling does not address the coupling the repo "
                    "analysis identified."
                ),
                evidence=(
                    "checkout.py calls orders.place() inside the request; a "
                    "larger EC2 instance raises the ceiling but leaves the "
                    "blocking call, so the failure mode returns at higher load."
                ),
                suggested_fix=(
                    "Extract order processing into an independently scalable "
                    "consumer instead of enlarging the single deployable."
                ),
                requires_refinement=True,
            ),
        ],
        requires_refinement=True,
        refinement_instruction=(
            "Replace the vertical-scaling decision with an asynchronous "
            "boundary between checkout and order processing, citing the "
            "retrieved queue-buffering pattern. Specify the compute and "
            "storage services concretely, and add an explicit GDPR source "
            "reference for the EU residency and encryption claims."
        ),
    )
    _totals_from_history(state)
    state.stage = Stage.DONE
    return state


def build_demo_state(variant: str = "pass") -> ArchitectState:
    """Return the demo state for `variant`; unknown names fall back to "pass"."""
    return demo_capped_state() if variant == "capped" else demo_pass_state()
