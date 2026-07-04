"""run.py — entry point for the skeleton. Run with:  python -m pipeline.run

Builds a state from an example prompt, runs the full pipeline, prints the trace.
This is the Week-1 exit gate: proves the skeleton runs end-to-end with stubs.

The example constants below are a stand-in for real user input. The UI and the
automated tests will call the SAME new_run() -> run_pipeline() -> read state
pattern; only these constants get replaced.
"""
from __future__ import annotations

from pipeline.state import new_run
from pipeline.orchestrator import run_pipeline

EXAMPLE_PROMPT = (
    "Fix our monolithic online shop so it can scale for seasonal peak sales. "
    "It's on AWS, budget is medium, must stay GDPR-compliant, and needs to handle "
    "~50k concurrent users at peak. Repo: https://github.com/example/bugged-shop"
)


def main() -> None:
    state = new_run(raw_prompt=EXAMPLE_PROMPT)
    run_pipeline(state)

    print(f"\nFinal stage: {state.stage.value}")
    print(f"Errors: {state.errors or 'none'}\n")
    print("Trace:")
    for step in state.history:
        print(
            f"  {step.agent:11s} {step.stage_in.value:11s} -> "
            f"{step.stage_out.value:11s} | {step.note}"
        )


if __name__ == "__main__":
    main()
