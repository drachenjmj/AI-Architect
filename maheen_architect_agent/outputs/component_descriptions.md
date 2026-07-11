# Component Descriptions

| Component | Purpose | Input | Output | Feature Traceability | ADR Traceability |
|---|---|---|---|---|---|
| Frontend Website | Shows product recommendations to online customers. | Customer actions | Recommendation requests | FEAT-003 | ADR-001 |
| API Gateway | Routes requests between the website and backend services. | Website requests | Backend service calls | FEAT-003 | ADR-001 |
| Recommendation Service | Generates personalized product recommendations. | Customer behavior and product data | Recommended products | FEAT-001 | ADR-001 |
| Event Collector | Collects clicks, purchases, and wishlist events. | Customer behavior events | Stored behavior events | FEAT-002 | ADR-001 |
| Customer Behavior Database | Stores customer activity used for recommendations. | Behavior events | Customer behavior history | FEAT-002 | ADR-001 |
| Consent Management | Supports GDPR compliance by managing user consent. | Consent choices | Consent status | NFR-002 | ADR-001 |
