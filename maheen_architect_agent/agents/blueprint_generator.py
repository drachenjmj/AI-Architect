from pathlib import Path


def generate_blueprint(context, features, pattern):
    with open(pattern["pattern_file"], "r", encoding="utf-8") as file:
        pattern_knowledge = file.read()
    technical_components = [
        "Frontend",
        "API Gateway",
        "Recommendation Service",
        "Event Collector",
        "Customer Behavior Database",
        "Product Database",
        "Consent Management"
    ]

    mermaid_diagram = """```mermaid
flowchart TD
    A[Customer] --> B[Frontend Website]
    B --> C[API Gateway]
    C --> D[Recommendation Service]
    B --> E[Event Collector]
    E --> F[Customer Behavior Database]
    D --> F
    D --> G[Product Database]
    D --> H[Consent Management]
```"""

    blueprint = f"""# Architecture Blueprint

## Project

{context["project_name"]}

## Business Goal

{context["business_goal"]}

---

## Selected Architecture Pattern

**{pattern["pattern_name"]}**

**Reason:** {pattern["reason"]}

---

## Architecture Rationale

The selected architecture pattern fits the project because:

- {pattern["reason"]}
- It supports the required business features.
- It matches the customer's scalability and compliance needs.

---

## Stakeholder View

Customer → Website → Recommendation System → Personalized Product Suggestions → Increased Sales

---

## Technical View

{mermaid_diagram}

---

## Technical Components

"""

    for component in technical_components:
        blueprint += f"- {component}\n"

    blueprint += "\n---\n\n## Feature-to-Component Traceability\n\n"
    blueprint += "| Feature ID | Feature | Supporting Component |\n"
    blueprint += "|---|---|---|\n"

    mapping = {
        "FEAT-001": "Recommendation Service",
        "FEAT-002": "Event Collector + Customer Behavior Database",
        "FEAT-003": "Frontend + API Gateway"
    }

    for feature in features:
        component = mapping.get(feature["feature_id"], "To be defined")
        blueprint += f"| {feature['feature_id']} | {feature['name']} | {component} |\n"

    output_path = Path("outputs/architecture_blueprint.md")

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(blueprint)

    print("Architecture Blueprint generated successfully!")