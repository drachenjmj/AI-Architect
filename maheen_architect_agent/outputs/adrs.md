# Architecture Decision Records

## ADR-001: Use Event-Driven Microservices Architecture

### Status

Accepted

### Context

The project goal is to increase online sales through personalized product recommendations. The system must support the selected business features, customer behavior tracking, and the stated compliance requirements.

### Decision

Use **Event-Driven Microservices Architecture**.

### Reason

The system depends on customer behavior events such as clicks, purchases, and wishlists.

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
