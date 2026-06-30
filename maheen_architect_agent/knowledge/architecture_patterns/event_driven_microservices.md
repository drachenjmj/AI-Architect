# Event-Driven Microservices Architecture

## When to Use

Use this pattern when the system needs to react to events such as clicks, purchases, transactions, messages, or status changes.

## Good Fit For

- Recommendation systems
- Fraud detection systems
- Real-time tracking systems
- Systems with many independent services

## Typical Components

- Frontend
- API Gateway
- Event Collector
- Message Broker
- Business Service
- Data Store
- Monitoring
- Consent or Security Service

## Strengths

- Scales well
- Services can evolve independently
- Good for real-time behavior tracking
- Supports asynchronous processing

## Risks

- More complex than a simple architecture
- Requires monitoring
- Event failures must be handled carefully

## Selection Clues

Choose this pattern if the context mentions:
- clicks
- purchases
- user behavior
- real-time
- events
- tracking
- high scalability