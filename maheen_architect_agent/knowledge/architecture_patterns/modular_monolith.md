# Modular Monolith Architecture

## When to Use

Use this pattern when the system should stay in one deployable application but still be organized into clear modules.

## Good Fit For

- MVPs
- Medium-complexity systems
- Teams that want structure without microservice complexity
- Systems that may later evolve into microservices

## Typical Components

- Web Interface
- Application Core
- User Module
- Business Module
- Data Module
- Shared Database

## Strengths

- Easier than microservices
- Faster to build for MVP
- Clear module boundaries
- Can be refactored later

## Risks

- May become too large over time
- Scaling individual modules is harder
- Requires discipline to keep module boundaries clean

## Selection Clues

Choose this pattern if the context mentions:
- MVP
- medium budget
- moderate complexity
- one team
- faster delivery
- future scalability