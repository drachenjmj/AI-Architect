"""Re-derive the verdict for every review already on disk. No LLM, no network.

The reviewer's pass/fail rule changes over time; the recorded runs do not. This
replays each stored `ReviewResult` through the CURRENT `derive_verdict` and
prints what the verdict would be now, beside what it actually was when the run
happened. The gap between those two columns is the measurement.

It deliberately does NOT reconstruct `DeterministicChecks` or re-ask the LLM.
Everything the rule needs - the code scores, five judgments, the issue
severities, and which criteria were not applicable - is already in the stored
report, which is exactly why `derive_verdict` was written to take only those.

    python -m eval.replay_reviews
    python -m eval.replay_reviews --run 20260818T194159Z-107ff26e
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from pipeline.agents.reviewer import derive_verdict
from pipeline.persistence import runs_dir
from pipeline.state import ReviewResult


# The verdict rule AS IT STOOD before refinement_readiness became advisory and
# best_practice_grounding became skippable. Frozen here on purpose: comparing
# against it is what lets this harness attribute a flipped verdict to a specific
# criterion instead of just reporting that the total moved. It is a historical
# artifact - do not "keep it in sync" with derive_verdict.
_HISTORICAL_CODE_SCORES = (
    "all_artifacts_present",
    "constraint_coverage",
    "traceability",
    "adr_presence",
)
_HISTORICAL_JUDGMENTS = (
    "repo_grounding",
    "flaw_detection",
    "adr_soundness",
    "best_practice_grounding",
    "refinement_readiness",
)


def historical_blocking(review: ReviewResult) -> list[str]:
    """What blocked this report under the pre-change rule. Pure."""

    blocking = [
        name for name in _HISTORICAL_CODE_SCORES
        if getattr(review.rubric_scores, name, 0) != 2
    ]
    blocking += [
        name for name in _HISTORICAL_JUDGMENTS
        if not getattr(review.rubric_scores, name, False)
    ]
    if any(issue.severity == "high" for issue in review.issues):
        blocking.append("high_severity_issue")
    return blocking


@dataclass(frozen=True)
class Replay:
    """One stored review, with the verdict it got and the verdict it gets now."""

    run_id: str
    checkpoint: str
    round_index: int
    recorded_pass: bool
    derived_pass: bool
    blocking_before: list[str]
    blocking: list[str]
    issue_count: int
    not_applicable: list[str]

    @property
    def changed(self) -> bool:
        return (not self.blocking_before) != self.derived_pass

    @property
    def unblocked(self) -> list[str]:
        """Criteria that blocked under the old rule and no longer do."""

        return [name for name in self.blocking_before if name not in self.blocking]


def _reviews_of(run_dir: Path) -> Iterator[tuple[str, ReviewResult, list[str]]]:
    """Yield each DISTINCT review in one run, oldest first, with its N/A set.

    A checkpoint is written per transition, so the same review is persisted
    several times over. Consecutive duplicates are collapsed, which makes the
    round index mean "the Nth review this run produced".

    `not_applicable` is DERIVED from the checkpoint's `retrieved_knowledge`
    rather than read from the stored report. Every run recorded before that
    field existed would otherwise come back with an empty set and be judged by a
    different rule than a fresh one - which would make the whole comparison
    meaningless. Deriving it applies one rule to all the evidence. For runs
    recorded since the change the two agree, because it is the same rule.
    """

    previous: str | None = None
    for path in sorted(run_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        raw = payload.get("review")
        if not raw:
            continue
        signature = json.dumps(raw, sort_keys=True)
        if signature == previous:
            continue
        previous = signature
        initial_request = payload.get("initial_request") or {}
        repo_expected = bool(str(initial_request.get("repo_url", "")).strip())
        repo_available = bool(payload.get("repo_representation"))
        not_applicable: list[str] = []
        if not repo_expected and not repo_available:
            not_applicable.append("repo_grounding")
        if not payload.get("retrieved_knowledge") and not repo_available:
            not_applicable.append("best_practice_grounding")
        yield path.name, ReviewResult(**raw), not_applicable


def replay(run_ids: list[str] | None = None) -> list[Replay]:
    """Re-derive every recorded verdict. Pure with respect to the run store."""

    rows: list[Replay] = []
    base = runs_dir()
    if not base.exists():
        return rows
    for run_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        if run_ids and run_dir.name not in run_ids:
            continue
        for index, (name, review, not_applicable) in enumerate(
            _reviews_of(run_dir), start=1
        ):
            derived, blocking = derive_verdict(
                review.rubric_scores, review.issues, not_applicable
            )
            rows.append(Replay(
                run_id=run_dir.name,
                checkpoint=name,
                round_index=index,
                recorded_pass=review.overall_status == "pass",
                derived_pass=derived,
                blocking_before=historical_blocking(review),
                blocking=blocking,
                issue_count=len(review.issues),
                not_applicable=not_applicable,
            ))
    return rows


def format_table(rows: list[Replay]) -> str:
    """Render the replay as fixed-width text."""

    if not rows:
        return "No recorded reviews found under " + str(runs_dir())

    header = (
        f"{'run':<26} {'rnd':>3} {'iss':>3} "
        f"{'before':<7} {'after':<6} {'':<3} unblocked / still blocking"
    )
    lines = [header, "-" * 108]
    for row in rows:
        marker = "-->" if row.changed else ""
        note = ""
        if row.unblocked:
            note = "-" + ", -".join(row.unblocked) + "   "
        note += "blocking: " + (", ".join(row.blocking) or "(nothing)")
        if row.not_applicable:
            note += "   [n/a: " + ", ".join(row.not_applicable) + "]"
        lines.append(
            f"{row.run_id:<26} {row.round_index:>3} {row.issue_count:>3} "
            f"{('PASS' if not row.blocking_before else 'fail'):<7} "
            f"{('PASS' if row.derived_pass else 'fail'):<6} {marker:<3} {note}"
        )
    return "\n".join(lines)


def summarise(rows: list[Replay]) -> str:
    """Say how many verdicts moved, and WHICH criterion stopped blocking.

    The per-criterion attribution is the whole point: "three rounds now pass" is
    not a finding until it says which change did it.
    """

    if not rows:
        return "Nothing to summarise."

    before_pass = sum(1 for row in rows if not row.blocking_before)
    after_pass = sum(1 for row in rows if row.derived_pass)
    flipped = [row for row in rows if row.changed and row.derived_pass]
    regressed = [row for row in rows if row.changed and not row.derived_pass]

    lines = [
        f"{len(rows)} recorded review(s) across "
        f"{len({row.run_id for row in rows})} run(s).",
        f"Verdict: {before_pass} passed under the old rule, "
        f"{after_pass} pass under the current one "
        f"({len(flipped)} fail -> PASS, {len(regressed)} PASS -> fail).",
    ]

    # Attribution: count each criterion that stopped blocking a FLIPPED round.
    credit: dict[str, int] = {}
    for row in flipped:
        for name in row.unblocked:
            credit[name] = credit.get(name, 0) + 1
    if credit:
        lines.append("")
        lines.append("Which criterion did the unblocking (rounds that flipped):")
        for name, count in sorted(credit.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"  {name:<28} {count} round(s)")

    # A criterion can stop blocking without flipping the round, when something
    # else still blocks it. Worth separating - it is progress, not a pass.
    partial: dict[str, int] = {}
    for row in rows:
        if row.changed and row.derived_pass:
            continue
        for name in row.unblocked:
            partial[name] = partial.get(name, 0) + 1
    if partial:
        lines.append("")
        lines.append("Stopped blocking, but the round still fails for other reasons:")
        for name, count in sorted(partial.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"  {name:<28} {count} round(s)")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m eval.replay_reviews",
        description=__doc__.split("\n\n")[0],
    )
    parser.add_argument(
        "--run", action="append", dest="runs", metavar="RUN_ID",
        help="replay only this run id; repeatable (default: every recorded run)",
    )
    args = parser.parse_args(argv)

    rows = replay(args.runs)
    print(format_table(rows))
    print()
    print(summarise(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
