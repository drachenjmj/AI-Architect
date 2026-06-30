from pathlib import Path


def generate_adrs(context, pattern):
    adr = f"""# Architecture Decision Records

## ADR-001: Use {pattern["pattern_name"]}

### Status

Accepted

### Context

The project goal is to {context["business_goal"].lower()}. The system must support the selected business features, customer behavior tracking, and the stated compliance requirements.

### Decision

Use **{pattern["pattern_name"]}**.

### Reason

{pattern["reason"]}

### Alternatives Considered

- Layered Architecture
- Modular Monolith Architecture

### Consequences

Positive:
- Supports scalability
- Allows independent services
- Fits behavior/event-based recommendation logic

Negative:
- More complex than a simple layered system
- Requires monitoring and careful event handling

### Traceability

Related requirements:
- REQ-001
- REQ-002
- NFR-001
"""

    output_path = Path("outputs/adrs.md")

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(adr)

    print("ADRs generated successfully!")