from pathlib import Path


def generate_component_descriptions(context, features):
    components = [
        {
            "component": "Frontend Website",
            "purpose": "Shows product recommendations to online customers.",
            "input": "Customer actions",
            "output": "Recommendation requests",
            "related_feature": "FEAT-003",
            "related_adr": "ADR-001"
        },
        {
            "component": "API Gateway",
            "purpose": "Routes requests between the website and backend services.",
            "input": "Website requests",
            "output": "Backend service calls",
            "related_feature": "FEAT-003",
            "related_adr": "ADR-001"
        },
        {
            "component": "Recommendation Service",
            "purpose": "Generates personalized product recommendations.",
            "input": "Customer behavior and product data",
            "output": "Recommended products",
            "related_feature": "FEAT-001",
            "related_adr": "ADR-001"
        },
        {
            "component": "Event Collector",
            "purpose": "Collects clicks, purchases, and wishlist events.",
            "input": "Customer behavior events",
            "output": "Stored behavior events",
            "related_feature": "FEAT-002",
            "related_adr": "ADR-001"
        },
        {
            "component": "Customer Behavior Database",
            "purpose": "Stores customer activity used for recommendations.",
            "input": "Behavior events",
            "output": "Customer behavior history",
            "related_feature": "FEAT-002",
            "related_adr": "ADR-001"
        },
        {
            "component": "Consent Management",
            "purpose": "Supports GDPR compliance by managing user consent.",
            "input": "Consent choices",
            "output": "Consent status",
            "related_feature": "NFR-002",
            "related_adr": "ADR-001"
        }
    ]

    markdown = "# Component Descriptions\n\n"
    markdown += "| Component | Purpose | Input | Output | Feature Traceability | ADR Traceability |\n"
    markdown += "|---|---|---|---|---|---|\n"

    for item in components:
        markdown += (
            f"| {item['component']} "
            f"| {item['purpose']} "
            f"| {item['input']} "
            f"| {item['output']} "
            f"| {item['related_feature']} "
            f"| {item['related_adr']} |\n"
        )

    output_path = Path("outputs/component_descriptions.md")

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(markdown)

    print("Component Descriptions generated successfully!")