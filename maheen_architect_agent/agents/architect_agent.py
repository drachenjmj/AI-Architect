import json
from pathlib import Path

from pattern_selector import select_architecture_pattern
from blueprint_generator import generate_blueprint
from adr_generator import generate_adrs
from component_generator import generate_component_descriptions
from validator import validate_outputs
from llm.llm_interface import generate_with_llm


def load_context_record():
    file_path = Path("inputs/context_record.json")

    with open(file_path, "r", encoding="utf-8") as file:
        context = json.load(file)

    return context


def load_architect_prompt():
    prompt_path = Path("prompts/architect_prompt.md")

    with open(prompt_path, "r", encoding="utf-8") as file:
        prompt = file.read()

    return prompt


def derive_features(context):
    features = [
        {
            "feature_id": "FEAT-001",
            "name": "Product Recommendation",
            "description": "Recommend products based on customer behavior.",
            "related_requirement_id": "REQ-001"
        },
        {
            "feature_id": "FEAT-002",
            "name": "Customer Behavior Tracking",
            "description": "Track clicks, purchases, and wishlists.",
            "related_requirement_id": "REQ-002"
        },
        {
            "feature_id": "FEAT-003",
            "name": "Website Recommendation Display",
            "description": "Show personalized recommendations on the website.",
            "related_requirement_id": "REQ-003"
        }
    ]

    return features


if __name__ == "__main__":
    context_record = load_context_record()
    architect_prompt = load_architect_prompt()

    llm_ready_message = generate_with_llm(architect_prompt)

    print("LLM Interface Ready!")
    print("Context Record Loaded Successfully!")
    print("Architect Prompt Loaded Successfully!")
    print("Project Name:", context_record["project_name"])
    print("Business Goal:", context_record["business_goal"])

    features = derive_features(context_record)

    print("\nFeatures Derived:")
    for feature in features:
        print(f"- {feature['feature_id']}: {feature['name']}")

    selected_pattern = select_architecture_pattern(context_record, features)

    print("\nArchitecture Pattern Selected:")
    print("Pattern:", selected_pattern["pattern_name"])
    print("Reason:", selected_pattern["reason"])

    generate_blueprint(context_record, features, selected_pattern)
    generate_adrs(context_record, selected_pattern)
    generate_component_descriptions(context_record, features)
    validate_outputs(features, selected_pattern)