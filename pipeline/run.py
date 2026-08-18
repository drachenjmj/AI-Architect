"""run.py — the command-line entry point: ONE run, prompt to artifacts (Kati).

WHAT THIS IS
------------
`python -m pipeline.run` is the non-interactive twin of `ui.py`. It drives the
SAME contract the UI drives — `new_run()` -> `run_pipeline_streaming()` -> read
the returned state — and adds the one thing the UI got for free from its form:
the answer loop. Nothing about the pipeline is special-cased here; this file is
a caller and a printer, nothing more.

WHY A LOOP IS NEEDED (the pause/resume contract)
------------------------------------------------
The pipeline pauses mid-run by RETURNING with `stage == AWAITING_HUMAN`. The
orchestrator routes that stage to END (`_route`) and expects the CALLER to
resolve the pause and re-invoke. WHICH resolution is owed is carried by
`state.pending_decision`, and this module handles both kinds:

  * CLARIFICATION — the clarifier asked questions. Collect answers, merge them
    into `clarification_answers`, re-invoke; `_entry_route` sends the state back
    to the clarifier, which re-judges with the new answers.
  * CONTEXT_LOCK  — a Context Record is frozen and waiting for approval. This
    one is resolved ENTIRELY OUTSIDE the graph (accept / edit / ask, see
    `pipeline/agents/clarifier.py`); the orchestrator refuses to be entered
    while it is still pending.

One call therefore never finishes a run. This module is that caller:

    run  ->  AWAITING_HUMAN?  ->  answer questions / approve the record  ->  run
              |                                                             |
              no                                                            |
              v                                                             |
      DONE / FAILED  ->  print everything            <----------------------+

THE CONTEXT GATE IS OPT-IN HERE, AND HAS TO BE
-----------------------------------------------
`--approve-context` is off by default, so `python -m pipeline.run` keeps
finishing unattended exactly as it did. The gate is a HUMAN pause: switching it
on for every headless caller — this command, the eval harness, the test suite —
would not have made those runs safer, it would have made them hang. With the
flag off, the clarifier locks and advances, which is precisely the behaviour
that existed before the gate, and the record still prints in full at the end.

With `--approve-context --no-input` there is a gate and nobody to work it, so
the run stops and says so rather than rubber-stamping its own ground truth.

Two rules the loop keeps, both of which `ui.py` keeps too:

  * MERGE answers, never replace. The clarifier can pause more than once and
    re-judges with the FULL Q&A history, so each round adds to
    `clarification_answers` rather than overwriting it.
  * Use the RETURNED state. `run_pipeline*` does not mutate the state you pass
    in (see DETERMINISM_MAP.md), so reading the object you passed in would read
    a state one round out of date.

BOUNDED, ALWAYS — the loop can never hang
------------------------------------------
Three separate exits, because a CLI that spins forever is worse than one that
quits:

  1. `MAX_CLARIFICATION_ROUNDS` caps how many times we will answer.
  2. A pause with an EMPTY question list is detected and reported. That is a
     KNOWN, UNFIXED clarifier bug (it can report `missing_critical` without
     filling `questions`); there is nothing to ask, so a naive loop would spin.
     We stop instead. Fixing the clarifier is a separate task — this only
     refuses to hang on it.
  3. A stop at any stage that is neither a pause nor terminal means the graph
     ended somewhere unwired. Re-running would end instantly at the same place,
     so we report it rather than loop.
  4. `MAX_USER_ROUNDS` (refine_gate) caps total POST-LOCK re-judges — an edit at
     the approval gate that reopens a gap. Answering questions and approving the
     record do not spend it: that counter bounds user-initiated REFINEMENT of a
     design, and rule 1 above is what bounds a clarifier that will not converge.

`--answers` exists for REPRODUCIBILITY: the same scenario re-run identically,
with no keyboard, for the report and for screenshots.

WHY THE OUTPUT IS DUPLICATED FROM ui_sections.py
------------------------------------------------
It is not imported: `ui_sections` renders Streamlit widgets and importing it
would drag Streamlit into a plain terminal run. The FIELD COVERAGE deliberately
mirrors it, so the CLI and the UI show the same run the same way, but the
formatting here is plain text written from `pipeline/state.py`.
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Sequence

from pipeline.agents import clarifier as clarifier_gate
from pipeline.llm import LLMError
from pipeline.orchestrator import MAX_STEPS, run_pipeline_streaming
from pipeline.agents.reviewer import ADVISORY_CRITERIA
from pipeline.persistence import runs_dir
from pipeline.refine_gate import MAX_USER_ROUNDS, begin_user_round
from pipeline.state import (
    ADR,
    ArchitectState,
    ComponentDescription,
    ContextEdits,
    Feature,
    NOT_APPLICABLE_REASONS,
    PendingDecision,
    Stage,
    new_run,
)

# NOTE: the repo URL in here is FICTIONAL, so the default run always fails to
# clone it. That is survivable by design — `repo_ingestor` degrades a dead URL
# into a recorded error and carries on (see its "failure philosophy"), so the
# loop below never sees it. Someone else owns picking the real repo; pass a
# real one with `--repo-url` in the meantime.
EXAMPLE_PROMPT = (
    "Fix our monolithic online shop so it can scale for seasonal peak sales. "
    "It's on AWS, budget is medium, must stay GDPR-compliant, and needs to handle "
    "~50k concurrent users at peak. Repo: https://github.com/example/bugged-shop"
)

# How many times we will answer a pause before giving up. The clarifier is meant
# to converge in one or two rounds; more than this means it is not converging,
# and looping on it would burn tokens without ever finishing.
MAX_CLARIFICATION_ROUNDS = 3

# Exit codes. Distinct on purpose: a run that failed is a different problem from
# a run we could not even drive.
EXIT_OK = 0             # reached Stage.DONE
EXIT_RUN_FAILED = 1     # reached Stage.FAILED — the pipeline itself failed
EXIT_CANNOT_CONTINUE = 2  # we could not answer / could not proceed (never DONE)

_WIDTH = 78


# ══════════════════════════════════════════════════════════════════════════
# Answers: where a reply to one clarifying question comes from
# ══════════════════════════════════════════════════════════════════════════
class AnswerFileError(RuntimeError):
    """`--answers` could not be read or is not one of the two supported shapes."""


def load_answer_file(path: Path) -> tuple[dict[str, str], list[str]]:
    """Read `--answers` into (mapping, positional queue). Exactly one is filled.

    Two shapes are supported, and which one you get is decided by the JSON's own
    top-level type — there is no flag to keep in sync with the file:

      * OBJECT  {"question text": "answer"} — matched by EXACT question text.
        Robust to the clarifier changing the ORDER it asks in.
      * ARRAY   ["answer", "answer"] — consumed positionally, in the order the
        questions are asked (across every round). Robust to the clarifier
        changing the WORDING, which is the more common drift.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AnswerFileError(f"cannot read {path}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AnswerFileError(f"{path} is not valid JSON: {exc}") from exc

    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items()}, []
    if isinstance(data, list):
        return {}, [str(v) for v in data]
    raise AnswerFileError(
        f"{path} must hold a JSON object (question text -> answer) or a JSON "
        f"array (answers in the order asked); found {type(data).__name__}."
    )


class AnswerSource:
    """Answers one clarifying question, from the file or from the keyboard.

    Both `--answers` shapes and the interactive fallback live behind one call so
    the loop never has to care which is in play. Returns `(answer, origin)` for
    a question it can answer and `None` for one it cannot — the loop turns that
    `None` into an honest failure rather than a silently skipped question.
    """

    def __init__(
        self,
        mapping: dict[str, str] | None = None,
        queue: Sequence[str] | None = None,
        allow_stdin: bool = True,
    ) -> None:
        self._mapping = dict(mapping or {})
        self._queue = list(queue or [])
        # Positional answers are consumed in the order questions are ASKED, and
        # that order spans rounds — so the cursor lives on the source, not on
        # one round's loop.
        self._cursor = 0
        self._allow_stdin = allow_stdin

    def answer(self, question: str, label: str = "A") -> tuple[str, str] | None:
        """The answer to `question` and where it came from, or None if there is none.

        `label` ("A1", "A2", ...) is only the prompt shown when the answer has
        to come from the keyboard.
        """
        if question in self._mapping:
            return self._mapping[question].strip(), "--answers (matched by text)"
        if self._cursor < len(self._queue):
            value = self._queue[self._cursor].strip()
            self._cursor += 1
            return value, f"--answers (position {self._cursor})"
        if not self._allow_stdin:
            return None
        return self._from_stdin(label)

    def line(self, label: str) -> str | None:
        """One raw line from the keyboard, or None when there is nobody there.

        The context gate reads through this rather than through `answer`, on
        purpose: `--answers` holds replies to CLARIFYING QUESTIONS, matched by
        text or consumed positionally, and letting a gate command eat one of them
        would silently mis-answer a later question. The gate is therefore
        interactive-only — with `--no-input` it reports that it cannot be worked
        instead of approving the record on the human's behalf.
        """
        if not self._allow_stdin:
            return None
        reply = self._from_stdin(label)
        return None if reply is None else reply[0]

    def _from_stdin(self, label: str) -> tuple[str, str] | None:
        """Read one answer from the keyboard. EOF (piped/closed stdin) = no answer.

        Treating EOF as "no answer" rather than letting `input()` raise is what
        stops an unattended run without `--no-input` from dying on a traceback:
        it fails the same clean way `--no-input` does.

        A PIPED stdin is echoed back, a real terminal is not: the terminal
        already shows what was typed, but a piped transcript would otherwise
        record the question and lose the answer.
        """
        try:
            value = input(f"  {label} > ").strip()
        except EOFError:
            return None
        if not sys.stdin.isatty():
            print(value)
        return value, "stdin"


# ══════════════════════════════════════════════════════════════════════════
# The answer loop
# ══════════════════════════════════════════════════════════════════════════
def stream_once(state: ArchitectState, max_steps: int) -> ArchitectState:
    """One pass through the graph, printing each new step as it lands.

    Mirrors what `ui.py:_run` does with the same generator: the LAST yielded
    state is the terminal one (identical to what `run_pipeline` would return),
    so we simply keep it. The pause/resume contract is untouched — what comes
    back is what gets inspected for `AWAITING_HUMAN`.

    `input_tokens` / `output_tokens` are reducer fields, so every snapshot
    already carries the run total so far; the cost line needs no accumulation
    here and the number is watchable as it climbs through the refine loop.

    Only steps not yet printed are printed, because `stream_mode="values"` emits
    the FULL state each time — and on a resume pass that state arrives already
    carrying the previous rounds' history.
    """
    latest = state
    printed = len(state.history)
    for snapshot in run_pipeline_streaming(state, max_steps):
        latest = snapshot
        for step in snapshot.history[printed:]:
            print(
                f"  {step.agent:<14s} {step.stage_in.value:<14s} -> "
                f"{step.stage_out.value}"
            )
            if step.note:
                print(_wrap(step.note, " " * 4))
            print(_wrap(_running_total(snapshot), " " * 4))
        # The step-cap path yields a copy of the ORIGINAL state plus one step,
        # so its history can be SHORTER than what we already printed. Nothing is
        # lost: it is the last emission, and the full trace is printed below.
        printed = len(snapshot.history)
    return latest


# ══════════════════════════════════════════════════════════════════════════
# The context gate: approve / edit / ask, before a research token is spent
# ══════════════════════════════════════════════════════════════════════════
# A tiny command grammar rather than a wizard. The UI can afford a form with a
# checkbox per assumption; a terminal cannot, and a nine-question interrogation
# every time somebody wants to approve an obviously-fine record would make the
# gate the first thing people switch off. So: press Enter to accept, or type one
# instruction. The same table is in --help, so --help IS the documentation.
_GATE_HELP = """\
  <Enter> or 'accept'      approve this record and continue to research
  strike N[,N...]          reject assumption N (it will not come back)
  set FIELD = value        replace a field (list fields: comma-separate)
  recommend FIELD          "I do not know - you propose one, with a reason"
  answer N text            answer open question N
  ? your question          ask about the record; nothing is changed or resolved
  quit                     stop here without approving"""


class GateAbort(RuntimeError):
    """The gate cannot be worked: no human on stdin, or the human asked to stop."""


def print_context_gate(state: ArchitectState) -> None:
    """Show the frozen record for approval, with its vetoable parts numbered.

    Reuses `print_context_record` so the gate and the final report can never show
    two different pictures of the same record; the numbering exists only because
    `strike 2` needs something to point at.
    """
    record = state.context_record
    print()
    print("=" * _WIDTH)
    print("CONTEXT LOCK - APPROVE BEFORE DESIGN BEGINS")
    print("=" * _WIDTH)
    print(
        _wrap(
            "This is the ground truth the whole design will be built on. "
            "Everything below was either taken from your request or assumed on "
            "your behalf. Nothing has been researched or designed yet."
        )
    )
    print_context_record(state)

    if record.assumptions:
        _sub("Assumptions you can strike")
        for index, assumption in enumerate(record.assumptions, start=1):
            print(_wrap(f"{index}. {assumption}", "  "))
    if record.open_questions:
        _sub("Open questions you can answer")
        for index, question in enumerate(record.open_questions, start=1):
            print(_wrap(f"{index}. {question}", "  "))

    _sub("What you can do")
    print(_GATE_HELP)


def _split_items(value: str) -> list[str]:
    """A list field typed on one line -> its items. Comma-separated, blanks dropped."""
    return [part.strip() for part in value.split(",") if part.strip()]


def _pick(numbers: str, population: Sequence[str], what: str) -> list[str]:
    """Resolve '2' or '1,3' against a numbered list. Raises ValueError on a miss."""
    chosen: list[str] = []
    for token in numbers.replace(" ", "").split(","):
        if not token:
            continue
        if not token.isdigit() or not 1 <= int(token) <= len(population):
            raise ValueError(
                f"'{token}' is not one of the {len(population)} {what} listed above."
            )
        chosen.append(population[int(token) - 1])
    if not chosen:
        raise ValueError(f"Name at least one of the {what} listed above, by number.")
    return chosen


def parse_gate_command(line: str, state: ArchitectState) -> tuple[str, object]:
    """One typed instruction -> `(action, payload)`. Pure; raises ValueError on nonsense.

    Actions: `accept` (payload None), `quit` (None), `ask` (the question text),
    `edit` (a `ContextEdits`). Kept separate from the loop below so the grammar is
    testable without a keyboard, which is the only way it stays trustworthy.
    """
    text = line.strip()
    record = state.context_record
    if not text or text.lower() == "accept":
        return "accept", None
    if text.lower() in {"quit", "exit"}:
        return "quit", None
    if text.startswith("?"):
        question = text[1:].strip()
        if not question:
            raise ValueError("Type your question after the '?'.")
        return "ask", question

    verb, _, rest = text.partition(" ")
    verb = verb.lower()
    rest = rest.strip()
    fields = clarifier_gate.EDITABLE_RECORD_FIELDS

    if verb == "strike":
        return "edit", ContextEdits(
            struck_assumptions=_pick(rest, record.assumptions, "assumptions")
        )
    if verb == "recommend":
        if rest not in fields:
            raise ValueError(
                f"'{rest}' is not a Context Record field. One of: {', '.join(fields)}."
            )
        return "edit", ContextEdits(recommend=[rest])
    if verb == "set":
        name, sep, value = rest.partition("=")
        name = name.strip()
        if not sep or not name:
            raise ValueError("Use: set FIELD = value")
        if name not in fields:
            raise ValueError(
                f"'{name}' is not a Context Record field. One of: {', '.join(fields)}."
            )
        is_list = isinstance(getattr(record, name), list)
        parsed = _split_items(value) if is_list else value.strip()
        return "edit", ContextEdits(fields={name: parsed})
    if verb == "answer":
        number, _, answer = rest.partition(" ")
        question = _pick(number, record.open_questions, "open questions")[0]
        if not answer.strip():
            raise ValueError("Type your answer after the question number.")
        return "edit", ContextEdits(answered_questions={question: answer.strip()})

    raise ValueError(f"'{verb}' is not one of the actions listed above.")


def resolve_context_gate(state: ArchitectState, answers: AnswerSource) -> bool:
    """Work the gate until it is resolved. Returns True when a re-judge is needed.

    False once the human has ACCEPTED: the pause is cleared and the caller
    re-enters the graph, which routes straight to the researcher. True when an
    edit opened a gap the clarifier has to re-judge; the caller re-enters the
    graph and lands back here on the next lock.

    Raises `GateAbort` when there is nobody on stdin or the human quits. Both are
    honest dead ends, and neither is a reason to approve the record for them.

    Advisory questions loop inside here WITHOUT resolving anything, which is the
    entire point of them: they run outside the graph, change no artifact and no
    stage, and the pause stays open across as many as you like.
    """
    print_context_gate(state)
    while True:
        line = answers.line("gate")
        if line is None:
            raise GateAbort(
                "The context gate needs a human: --approve-context pauses the run "
                "so the frozen Context Record can be approved, and stdin is closed "
                "(--no-input, or a piped/EOF stdin). Drop --approve-context to let "
                "the run auto-approve its own record, or run it interactively."
            )
        try:
            action, payload = parse_gate_command(line, state)
        except ValueError as exc:
            print(_wrap(f"! {exc}", "  "))
            continue

        if action == "quit":
            raise GateAbort(
                "Stopped at the context gate without approving. The run is "
                "checkpointed and can be resumed from the UI."
            )

        if action == "accept":
            clarifier_gate.accept_context_lock(state)
            print(_wrap("Approved. Continuing to research...", "  "))
            return False

        if action == "ask":
            # OUTSIDE the graph: no routing decision, no stage change,
            # pending_decision untouched. It costs tokens, so it lands in the
            # trace under its own agent name - see clarifier.ask_advisor.
            try:
                print()
                print(_wrap(clarifier_gate.ask_advisor(state, str(payload)), "  "))
                print()
            except LLMError as exc:
                # A failed side question must never cost the human their gate.
                print(_wrap(f"! Could not answer that: {exc}", "  "))
            continue

        # action == "edit"
        try:
            reasons = clarifier_gate.submit_context_edits(state, payload)
        except ValueError as exc:
            print(_wrap(f"! {exc}", "  "))
            continue

        if not reasons:
            # Filling or replacing a value cannot open a gap, so the record
            # re-froze with NO model call. Redraw and stay at the gate.
            print(_wrap("Recorded. No re-judge needed (nothing was emptied).", "  "))
            print_context_gate(state)
            continue

        print(_wrap("Re-judging, because " + "; ".join(reasons) + ".", "  "))
        allowed, why = begin_user_round(state)
        if not allowed:
            print(
                _wrap(
                    f"! Out of user rounds ({why}). Your edit is kept, but the "
                    f"clarifier will not re-judge it. Accept the record as it "
                    f"stands, or quit.",
                    "  ",
                )
            )
            continue
        clarifier_gate.open_for_rejudge(state)
        return True


def drive(
    state: ArchitectState,
    max_steps: int,
    answers: AnswerSource,
    max_rounds: int = MAX_CLARIFICATION_ROUNDS,
) -> tuple[ArchitectState, int]:
    """Run -> answer -> resume, until a terminal stage. Returns (state, exit code).

    This is the whole reason the command exists: `run_pipeline` alone returns at
    the first pause, which is why a single call only ever printed a two-step
    trace and produced no artifact.

    Both pause kinds land here and are told apart by `pending_decision`, not by
    the stage: a CONTEXT_LOCK is resolved outside the graph by
    `resolve_context_gate`, a CLARIFICATION by the answer loop below.
    """
    rounds = 0
    while True:
        state = stream_once(state, max_steps)

        if state.stage is Stage.DONE:
            return state, EXIT_OK
        if state.stage is Stage.FAILED:
            return state, EXIT_RUN_FAILED
        if state.stage is not Stage.AWAITING_HUMAN:
            # Neither a pause nor terminal: the graph ended at a stage that is
            # not wired in STAGE_TO_NODE. Re-running would end instantly at the
            # same place, so stopping is the only non-spinning option.
            _problem(
                f"The pipeline stopped at stage '{state.stage.value}', which is "
                f"neither a pause nor a terminal stage. That stage has no route "
                f"in orchestrator.STAGE_TO_NODE, so resuming would stop here "
                f"again. Nothing further to do."
            )
            return state, EXIT_CANNOT_CONTINUE

        if state.pending_decision is PendingDecision.CONTEXT_LOCK:
            # The record is frozen and waiting. Everything about resolving it
            # happens OUTSIDE the graph, and `resolve_context_gate` does not
            # return until it is resolved one way or the other — so there is no
            # path from here back into the loop with the lock still pending,
            # which is the state `_entry_route` refuses to be entered in.
            try:
                needs_rejudge = resolve_context_gate(state, answers)
            except GateAbort as exc:
                _problem(str(exc))
                return state, EXIT_CANNOT_CONTINUE
            # Approving costs no user round: it releases work that was already
            # waiting rather than asking for any to be redone. Only an edit that
            # sends the record back for re-judging is charged, inside
            # `resolve_context_gate`. See `ArchitectState.user_rounds`.
            print()
            print("Resuming...")
            continue

        rounds += 1
        if rounds > max_rounds:
            _problem(
                f"The clarifier is still asking after {max_rounds} round(s) of "
                f"answers, which is the cap in MAX_CLARIFICATION_ROUNDS. It is "
                f"not converging on this prompt — give it a more complete "
                f"prompt, or better answers, rather than more rounds."
            )
            return state, EXIT_CANNOT_CONTINUE

        questions = list(state.clarifying_questions)
        if not questions:
            # KNOWN, UNFIXED clarifier bug — see the module docstring. Paused
            # with nothing to ask, so there is no answer that could move it on.
            _problem(
                "The pipeline paused for clarification but asked NO questions "
                "(stage is awaiting_input and clarifying_questions is empty). "
                "This is a known clarifier bug: it can report missing critical "
                "facts without producing the matching questions. There is "
                "nothing to answer, so this run cannot continue."
            )
            return state, EXIT_CANNOT_CONTINUE

        # The WHOLE set first, then one prompt per question. Same shape as the
        # UI's form: you see everything you are about to be asked before you
        # answer the first, and a --no-input failure names the full set too.
        print()
        print(
            f"-- Clarification round {rounds} of {max_rounds}: "
            f"{len(questions)} question(s) --"
        )
        for index, question in enumerate(questions, start=1):
            print(_wrap(f"Q{index}. {question}", "  "))
        print()

        collected: dict[str, str] = {}
        for index, question in enumerate(questions, start=1):
            label = f"A{index}"
            reply = answers.answer(question, label)
            if reply is None:
                _problem(
                    f"No answer available for this question:\n\n"
                    f"    {question}\n\n"
                    f"Add it to the --answers file (as a key matching that "
                    f"exact text, or as one more entry in the positional "
                    f"array), or drop --no-input and answer it at the prompt."
                )
                return state, EXIT_CANNOT_CONTINUE
            text, origin = reply
            if origin != "stdin":
                # Echo what the file supplied, so a non-interactive transcript
                # reads exactly like an interactive one.
                print(_wrap(f"{label} > {text or '(blank)'}  [{origin}]", "  "))
            if text:
                collected[question] = text

        if not collected:
            # Blank answers are treated as "skipped", exactly as ui.py treats an
            # empty form field. Nothing new reaches the clarifier, so it will
            # re-judge identically and pause again — bounded by the round cap.
            print(
                "  (no answers recorded this round — the clarifier will "
                "re-judge with what it already had)"
            )

        # MERGE, never replace: the clarifier re-judges with the FULL history,
        # and this state already carries every earlier round. Same call ui.py makes.
        state.clarification_answers.update(collected)

        # `max_rounds` above is the ONLY cap on this loop. Answering the
        # clarifier does not spend a `user_rounds`: that counter bounds
        # post-lock refinement, and a run that measured four rounds of questions
        # here arrived at the architect with its whole refinement budget already
        # gone. Two different failures, two different caps — see
        # `ArchitectState.user_rounds`.
        print()
        print("Resuming...")


# ══════════════════════════════════════════════════════════════════════════
# Plain-text formatting helpers
# ══════════════════════════════════════════════════════════════════════════
def _wrap(body: str, indent: str = "") -> str:
    """Wrap prose to the page width, keeping the author's own line breaks."""
    lines: list[str] = []
    for line in str(body).splitlines() or [""]:
        if not line.strip():
            lines.append("")
            continue
        lines.extend(
            textwrap.wrap(
                line.strip(),
                width=_WIDTH,
                initial_indent=indent,
                subsequent_indent=indent,
            )
            or [indent + line.strip()]
        )
    return "\n".join(lines)


def _section(title: str) -> None:
    """A top-level heading — one per artifact."""
    print()
    print("=" * _WIDTH)
    print(title.upper())
    print("=" * _WIDTH)


def _sub(title: str) -> None:
    """A heading for one item within a section (one feature, one ADR, ...)."""
    print()
    print(title)
    print("-" * min(len(title), _WIDTH))


def _problem(message: str) -> None:
    """Report why the run cannot continue, on stderr, unmissably.

    stdout is flushed first: piped to a file it is block-buffered while stderr
    is not, so without this the complaint would surface ABOVE the run it is
    complaining about.
    """
    print()
    sys.stdout.flush()
    print("!" * _WIDTH, file=sys.stderr)
    print(_wrap(message), file=sys.stderr)
    print("!" * _WIDTH, file=sys.stderr)


def _text(label: str, value: str | None, indent: str = "") -> bool:
    """One prose field. Silent when empty; True when it drew something."""
    body = str(value or "").strip()
    if not body:
        return False
    print(f"{indent}{label}:")
    print(_wrap(body, indent + "  "))
    print()
    return True


def _bullets(label: str, values: Sequence[str] | None, indent: str = "") -> bool:
    """One list field, as bullets. Silent when empty."""
    items = [str(v).strip() for v in (values or []) if str(v).strip()]
    if not items:
        return False
    print(f"{indent}{label}:")
    for item in items:
        wrapped = textwrap.wrap(
            item,
            width=_WIDTH,
            initial_indent=indent + "  - ",
            subsequent_indent=indent + "    ",
        )
        print("\n".join(wrapped) or f"{indent}  - {item}")
    print()
    return True


def _inline(label: str, values: Sequence[str] | None, indent: str = "") -> bool:
    """One list of short IDs, on a single line. Silent when empty."""
    items = [str(v).strip() for v in (values or []) if str(v).strip()]
    if not items:
        return False
    print(_wrap(f"{label}: {', '.join(items)}", indent))
    print()
    return True


def _running_total(state: ArchitectState) -> str:
    """The token/cost line printed under each live step, as the run climbs."""
    total = state.input_tokens + state.output_tokens
    return (
        f"{total:,} tokens so far ({state.input_tokens:,} in / "
        f"{state.output_tokens:,} out) ~ ${state.total_cost_usd():,.4f} "
        f"list-price equivalent"
    )


# ══════════════════════════════════════════════════════════════════════════
# Output: the run, then every artifact, from pipeline/state.py
# ══════════════════════════════════════════════════════════════════════════
def print_trace(state: ArchitectState) -> None:
    """The step trace this command has always printed, unchanged in shape."""
    _section("Run trace")
    print(f"Final stage: {state.stage.value}")
    print(f"Errors: {state.errors or 'none'}")
    print()
    if not state.history:
        print("No steps were recorded for this run.")
        return
    for step in state.history:
        print(
            f"  {step.agent:11s} {step.stage_in.value:11s} -> "
            f"{step.stage_out.value:11s} | {step.note}"
        )


def print_clarification(state: ArchitectState) -> None:
    """The Q&A that unblocked the run — the input half of reproducing it."""
    _section("Clarification Q&A")
    if not state.clarification_answers:
        print("No clarifying questions were answered in this run.")
        return
    for index, (question, answer) in enumerate(
        state.clarification_answers.items(), start=1
    ):
        print(_wrap(f"Q{index}. {question}"))
        print(_wrap(f"A{index}. {answer}", "  "))
        print()


def print_context_record(state: ArchitectState) -> None:
    """The frozen context — every field, not just `summary`."""
    _section("Context Record")
    record = state.context_record
    if record is None:
        print(
            "Not produced: clarification never completed, so no context was "
            "locked for this run."
        )
        return
    drawn = [
        _text("Project name", record.project_name),
        _text("Business goal", record.business_goal),
        _text("Problem statement", record.problem_statement),
        _bullets("Users and stakeholders", record.users),
        _bullets("Functional requirements", record.functional_requirements),
        _bullets("Non-functional requirements", record.non_functional_requirements),
        _text("Cloud provider", record.cloud_provider),
        _text("Budget", record.budget),
        _bullets("Compliance requirements", record.compliance_requirements),
        _bullets("Existing systems", record.existing_systems),
        _bullets("Assumptions", record.assumptions),
        _bullets("Open questions", record.open_questions),
        _text("Summary", record.summary),
    ]
    if not any(drawn):
        print("The context record exists but every field is empty.")


def print_features(state: ArchitectState) -> None:
    """The derived features — what the system must do."""
    _section("Features")
    if not state.features:
        print(
            "None derived, so there is nothing for the design to trace back to."
        )
        return
    print(f"{len(state.features)} feature(s).")
    for feature in state.features:
        _print_feature(feature)


def _print_feature(feature: Feature) -> None:
    _sub(f"{feature.id}  {feature.name}  [{feature.priority.upper()}]")
    _text("Description", feature.description)
    _text("Scenario", feature.scenario)
    _bullets("Acceptance criteria", feature.acceptance_criteria)
    _inline("Traces to requirements", feature.related_requirement_ids)


def print_blueprint(state: ArchitectState) -> None:
    """The blueprint — both views plus the reasoning around them."""
    _section("Blueprint")
    blueprint = state.blueprint
    if blueprint is None:
        print("The architect produced no blueprint for this run.")
        return
    print(
        f"{blueprint.blueprint_id} | v{blueprint.version}"
        + (f" | {blueprint.project_name}" if blueprint.project_name else "")
    )
    print()
    _text("Selected pattern", blueprint.selected_pattern)
    _text("Rationale", blueprint.rationale)
    drew_view = _text("Stakeholder view (business-facing)", blueprint.stakeholder_view)
    drew_view |= _text("Technical view", blueprint.technical_view)
    if not drew_view:
        print("Both blueprint views are empty.")
        print()
    _bullets("Components", blueprint.components)
    _bullets("Data flows", blueprint.data_flows)
    _bullets("Constraints addressed", blueprint.constraints_addressed)
    _bullets("Assumptions", blueprint.assumptions)
    _bullets("Open risks", blueprint.open_risks)
    _inline("Addresses features", blueprint.addressed_feature_ids)


def print_adrs(state: ArchitectState) -> None:
    """Every ADR in full — context, rationale, alternatives, consequences."""
    _section("Architecture Decision Records")
    if not state.adrs:
        print("None written, so the design's decisions are unrecorded.")
        return
    print(f"{len(state.adrs)} ADR(s).")
    for adr in state.adrs:
        _print_adr(adr)


def _print_adr(adr: ADR) -> None:
    _sub(f"{adr.id}  {adr.title}  [{adr.status.upper()}]")
    _text("Context", adr.context)
    _text("Decision", adr.decision)
    _text("Rationale", adr.rationale)
    _bullets("Alternatives considered", adr.alternatives_considered)
    _bullets("Positive consequences", adr.positive_consequences)
    _bullets("Negative consequences", adr.negative_consequences)
    _inline("Related features", adr.related_feature_ids)
    _inline("Related components", adr.related_component_names)
    _bullets("Sources", adr.source_references)


def print_components(state: ArchitectState) -> None:
    """Every component in full — I/O, dependencies, security, scalability."""
    _section("Components")
    if not state.components:
        print("None described, so the blueprint has no parts to build.")
        return
    print(f"{len(state.components)} component(s).")
    for component in state.components:
        _print_component(component)


def _print_component(component: ComponentDescription) -> None:
    _sub(f"{component.id}  {component.name}  [{component.component_type}]")
    _text("Purpose", component.purpose)
    _text("Description", component.description)
    _bullets("Inputs", component.inputs)
    _bullets("Outputs", component.outputs)
    _bullets("Dependencies", component.dependencies)
    _bullets("Technology choices", component.technology_choices)
    _bullets("Security considerations", component.security_considerations)
    _bullets("Scalability considerations", component.scalability_considerations)
    _inline("Implements features", component.related_feature_ids)
    _inline("Justified by ADRs", component.related_adr_ids)


# The four code-owned rubric checks, in report order: 0-2 diagnostic scores, and
# the verdict requires every one of them to be 2 (see reviewer.py).
_CODE_CHECKS: list[tuple[str, str]] = [
    ("all_artifacts_present", "All artifacts present"),
    ("constraint_coverage", "Constraint coverage"),
    ("traceability", "Traceability"),
    ("adr_presence", "ADR presence"),
]
# The five LLM-owned judgments: binary, each paired with its written reason.
_LLM_CHECKS: list[tuple[str, str]] = [
    ("repo_grounding", "Repo grounding"),
    ("flaw_detection", "Flaw detection"),
    ("adr_soundness", "ADR soundness"),
    ("best_practice_grounding", "Best-practice grounding"),
    ("refinement_readiness", "Refinement readiness"),
]


def print_review(state: ArchitectState) -> None:
    """The quality gate, with the code-owned / LLM-owned split kept visible."""
    _section("Review report")
    review = state.review
    if review is None:
        print(
            "No review result: the design never reached the quality gate."
        )
        print()
        print(
            f"refine_iterations : {state.refine_iterations}\n"
            f"stopped_on_cap    : {state.stopped_on_cap}"
        )
        return

    print(f"Verdict: {review.overall_status.upper()}  "
          f"(owned by deterministic code, never by the LLM)")
    print(f"Requires refinement: {'yes' if review.requires_refinement else 'no'}")
    print()

    print("Code-owned rubric checks (deterministic Python; 0-2; all must be 2):")
    for field, label in _CODE_CHECKS:
        score = getattr(review.rubric_scores, field, 0)
        mark = "PASS" if score == 2 else "FAIL"
        print(f"  [{mark}] {label:<28s} {score}/2")
    print()

    print("LLM-owned judgments (binary; never summed into the verdict):")
    for field, label in _LLM_CHECKS:
        # not_applicable FIRST. The stored boolean is true for such a criterion
        # so nothing downstream breaks, so reading it first would render "no
        # evidence existed" as a pass - which is the one thing this must not do.
        if field in review.not_applicable:
            why = NOT_APPLICABLE_REASONS.get(field, "not applicable")
            print(f"  [ n/a] {label}  ({why}; excluded from the verdict)")
        else:
            passed = bool(getattr(review.rubric_scores, field, False))
            # An advisory criterion is asked and shown but cannot decide
            # anything, so a bare FAIL beside a PASS verdict would look like a
            # contradiction rather than the deliberate choice it is.
            note = "  (advisory; excluded from the verdict)" if (
                field in ADVISORY_CRITERIA and not passed
            ) else ""
            print(f"  [{'PASS' if passed else 'FAIL'}] {label}{note}")
        reason = (getattr(review.judgment_reasons, field, "") or "").strip()
        print(_wrap(reason or "(no reason recorded)", "        "))
    print()

    if review.issues:
        print(f"Issues ({len(review.issues)}):")
        for index, issue in enumerate(review.issues, start=1):
            print()
            print(
                f"  {index}. {issue.id or '(no id)'}  "
                f"[{issue.severity}/{issue.category}]"
            )
            _text("Finding", issue.finding, indent="     ")
            _text("Evidence", issue.evidence, indent="     ")
            _text("Suggested fix", issue.suggested_fix, indent="     ")
            print(
                f"     Requires refinement: "
                f"{'yes' if issue.requires_refinement else 'no'}"
            )
    else:
        print("Issues: none recorded.")
    print()

    print(f"refine_iterations : {state.refine_iterations}")
    print(
        f"stopped_on_cap    : {state.stopped_on_cap}"
        + (
            "  (the run ended on the cost guardrail, not on a clean pass)"
            if state.stopped_on_cap
            else "  (the run ended on the reviewer's verdict)"
        )
    )
    print()
    _text("Refinement instruction fed back to the architect",
          review.refinement_instruction)


def print_knowledge(state: ArchitectState) -> None:
    """What the researcher pulled out of the knowledge base — INCLUDING nothing.

    An empty result is a real finding about our knowledge base, so it is stated
    outright rather than left as an absent section.
    """
    _section("Retrieved knowledge")
    chunks = state.retrieved_knowledge
    if not chunks:
        print(
            _wrap(
                "NO KNOWLEDGE-BASE RESULTS FOR THIS RUN. The researcher ran and "
                "returned zero chunks, so the design above is grounded in the "
                "request and the repository only — not in the curated pattern "
                "library."
            )
        )
        return
    print(
        f"{len(chunks)} chunk(s) handed to the architect as grounding. "
        f"Box 1 = curated architecture patterns, box 2 = domain knowledge, "
        f"box 3 = live web-search fallback."
    )
    for index, chunk in enumerate(chunks, start=1):
        head = f"{index:02d}. {chunk.source or 'unknown source'}"
        if chunk.page:
            head += f", p. {chunk.page}"
        distance = (
            "no distance recorded (web-search result)"
            if chunk.distance is None
            else f"distance {chunk.distance:.4f} (lower is closer)"
        )
        _sub(f"{head}  [box {chunk.box}]  {distance}")
        print(_wrap(chunk.content or "(empty chunk)", "  "))


def print_cost(state: ArchitectState) -> None:
    """Tokens and cost for the whole run, and per agent.

    `total_cost_usd()` and `usage_by_agent()` are METHODS, not properties — both
    are derived from `history` on every call and never stored on the state.
    """
    _section("Tokens and cost")
    total = state.input_tokens + state.output_tokens
    print(
        f"Tokens: {total:,} total "
        f"({state.input_tokens:,} in / {state.output_tokens:,} out)"
    )
    print(f"Cost:   ~${state.total_cost_usd():,.4f} list-price-equivalent USD")
    print()
    print(
        _wrap(
            "That figure is what the run WOULD have cost at Google's published "
            "Gemini list prices. This project runs on free-tier keys, so it is "
            "NOT money spent."
        )
    )
    print()

    usage = state.usage_by_agent()
    if not usage:
        print("No per-agent usage recorded (no steps ran).")
        return
    print(f"  {'Agent':<14s}{'Input':>10s}{'Output':>10s}{'Total':>10s}{'Cost':>12s}")
    print(f"  {'-' * 56}")
    for agent, agent_usage in usage.items():
        print(
            f"  {agent:<14s}{agent_usage.input_tokens:>10,}"
            f"{agent_usage.output_tokens:>10,}{agent_usage.total_tokens:>10,}"
            f"{'~$' + format(agent_usage.cost_usd, ',.4f'):>12s}"
        )


def print_where(state: ArchitectState) -> None:
    """The run's identity and where every checkpoint of it landed.

    The directory is read from `persistence.runs_dir()`, never hardcoded, because
    `AI_ARCHITECT_RUNS_DIR` can move it and a printed path that is not the real
    one would be worse than printing none.
    """
    _section("Where to find this run")
    print(f"Run id      : {state.run_id}")
    try:
        location = (Path(runs_dir()) / state.run_id).resolve()
        print(f"Checkpoints : {location}")
        print()
        print(
            _wrap(
                "Every transition of this run was checkpointed there, newest "
                "file last. Reload it in the UI's 'Resume a previous run' "
                "picker, or with persistence.load_state(run_id)."
            )
        )
    except Exception as exc:  # noqa: BLE001 — a footer is never worth a crash
        print(f"Checkpoints : (could not resolve the runs directory: {exc})")


def print_report(state: ArchitectState) -> None:
    """Everything the run produced, in the order a reader should meet it."""
    print_trace(state)
    print_clarification(state)
    print_context_record(state)
    print_features(state)
    print_blueprint(state)
    print_adrs(state)
    print_components(state)
    print_review(state)
    print_knowledge(state)
    print_cost(state)
    print_where(state)


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════
# Indented one level for the epilog block it is quoted into.
_GATE_HELP_EPILOG = "\n".join("  " + line for line in _GATE_HELP.splitlines())

_EPILOG = f"""\
the --answers file (two shapes, chosen by the JSON's own top-level type)
-----------------------------------------------------------------------
  OBJECT - question text -> answer, matched EXACTLY:

      {{"What is your expected peak load?": "50k concurrent users",
       "Which cloud provider?": "AWS"}}

  ARRAY - answers consumed POSITIONALLY, in the order the questions are
  asked (the order carries across clarification rounds):

      ["50k concurrent users", "AWS"]

  A question with no matching answer is never silently skipped: interactively
  you are asked for it on stdin; with --no-input the command prints the
  unanswered question and exits {EXIT_CANNOT_CONTINUE}. An answer left blank counts as
  "skipped", exactly as an empty field does in the Streamlit UI.

clarification rounds
--------------------
  The clarifier can pause more than once; answers are MERGED across rounds so
  it re-judges with the full Q&A history. At most {MAX_CLARIFICATION_ROUNDS} rounds are answered
  (MAX_CLARIFICATION_ROUNDS); answering them does not spend the separate
  post-lock budget of {MAX_USER_ROUNDS} re-judges (refine_gate.MAX_USER_ROUNDS). A pause
  carrying no questions at all is a known clarifier bug: it is reported and the
  command exits rather than spinning.

--approve-context (the context lock)
------------------------------------
  OFF by default, so this command finishes unattended exactly as before: the
  clarifier locks the Context Record and goes straight on to research.

  With the flag, the run PAUSES once the record is frozen and shows it for
  approval before a single research, design or review token is spent on it:

{_GATE_HELP_EPILOG}

  'set' and 'answer' fill values in, which cannot open a gap, so they re-freeze
  the record with NO model call. 'strike' on an architecture-critical value and
  'recommend' DO open one, so the clarifier re-judges - and on a re-judge it may
  no longer ask questions: the gap comes back as a labelled assumption plus an
  open question, on the record you are already looking at.

  The gate is interactive: with --no-input there is nobody to work it, and the
  command stops rather than approving its own ground truth. Drop the flag for
  unattended runs.

the default prompt
------------------
  With no PROMPT argument, EXAMPLE_PROMPT is used. It names a FICTIONAL repo
  (github.com/example/bugged-shop), so the repo ingestor will fail to clone it
  and record the failure; the run continues without repository insight. Pass
  --repo-url (or your own prompt) for a real one.

exit codes
----------
  {EXIT_OK}  the run reached DONE
  {EXIT_RUN_FAILED}  the run reached FAILED
  {EXIT_CANNOT_CONTINUE}  the run could not be driven to a terminal stage (unanswerable
     question, round cap exceeded, a pause with no questions, or a context
     gate with nobody to work it)

examples
--------
  python -m pipeline.run
  python -m pipeline.run "Design an event-driven order pipeline for 10k rps."
  python -m pipeline.run --repo-url https://github.com/pallets/flask
  python -m pipeline.run --answers answers.json --no-input > run.txt
  python -m pipeline.run --approve-context
"""


def build_parser() -> argparse.ArgumentParser:
    """The command-line surface. Kept in one place so --help IS the documentation."""
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.run",
        description=(
            "Drive one AI-Architect run from the command line, answering the "
            "clarifier's questions until the pipeline reaches DONE or FAILED, "
            "then print the full trace and every artifact as plain text."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=EXAMPLE_PROMPT,
        metavar="PROMPT",
        help="the system to architect, in natural language "
             "(default: the built-in EXAMPLE_PROMPT, see below)",
    )
    parser.add_argument(
        "--repo-url",
        default="",
        metavar="URL",
        help="existing codebase to re-architect, passed straight to new_run(); "
             "omit for a greenfield project (a URL in PROMPT still works)",
    )
    parser.add_argument(
        "--answers",
        metavar="FILE",
        type=Path,
        help="JSON file of answers to the clarifier's questions, for "
             "reproducible non-interactive runs; object or array (see below)",
    )
    parser.add_argument(
        "--no-input",
        action="store_true",
        help="never prompt: a question with no answer in --answers fails the "
             "run instead, so this can run unattended",
    )
    parser.add_argument(
        "--approve-context",
        action="store_true",
        help="pause once the Context Record is locked and show it for approval "
             "before any research or design happens; needs a terminal, so it "
             "cannot be combined with --no-input (see below)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS,
        metavar="N",
        help="hard cap on graph steps per invocation, passed to "
             "run_pipeline_streaming. The default is DERIVED from the cost caps "
             "in refine_gate.py (default: %(default)s), so raising a cap never "
             "shows up as a recursion error",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse, drive, print. Returns the process exit code."""
    # BEFORE parsing, because --help prints and exits from inside parse_args.
    # Artifacts come back from an LLM and routinely contain characters the
    # Windows console codepage cannot encode; reconfigure rather than let a
    # finished run die printing its own output.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001 — encoding is never worth a crash
                pass

    args = build_parser().parse_args(argv)

    mapping: dict[str, str] = {}
    queue: list[str] = []
    if args.answers is not None:
        try:
            mapping, queue = load_answer_file(args.answers)
        except AnswerFileError as exc:
            _problem(f"--answers: {exc}")
            return EXIT_CANNOT_CONTINUE

    answers = AnswerSource(mapping, queue, allow_stdin=not args.no_input)

    state = new_run(
        raw_prompt=args.prompt,
        repo_url=args.repo_url,
        require_context_approval=args.approve_context,
    )
    print(_wrap(f"Prompt: {args.prompt}"))
    if args.repo_url:
        print(f"Repo:   {args.repo_url}")
    print(f"Run id: {state.run_id}")
    print()
    print("Running...")

    try:
        # run_pipeline* returns the FINAL state and does NOT mutate the state
        # passed in (a step-cap failure marks a copy). Always use what comes
        # back; never read the object handed in.
        state, code = drive(state, args.max_steps, answers)
    except KeyboardInterrupt:
        _problem(
            f"Interrupted. The run is checkpointed up to its last completed "
            f"step under run id {state.run_id} and can be resumed from the UI."
        )
        return EXIT_CANNOT_CONTINUE

    print_report(state)
    return code


if __name__ == "__main__":
    sys.exit(main())
