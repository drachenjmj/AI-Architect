def select_architecture_pattern(context, features):
    text = (
        context.get("business_goal", "") + " " +
        context.get("problem_statement", "") + " " +
        " ".join([f["description"] for f in features])
    ).lower()

    if any(word in text for word in ["click", "purchase", "wishlist", "behavior", "real-time", "event", "tracking"]):
        return {
            "pattern_id": "PAT-001",
            "pattern_name": "Event-Driven Microservices Architecture",
            "pattern_file": "knowledge/architecture_patterns/event_driven_microservices.md",
            "reason": "The system depends on customer behavior events such as clicks, purchases, and wishlists."
        }

    if any(word in text for word in ["dashboard", "internal", "simple", "crud", "stable"]):
        return {
            "pattern_id": "PAT-002",
            "pattern_name": "Layered Architecture",
            "pattern_file": "knowledge/architecture_patterns/layered_architecture.md",
            "reason": "The system appears simple and can be separated into clear layers."
        }

    return {
        "pattern_id": "PAT-003",
        "pattern_name": "Modular Monolith Architecture",
        "pattern_file": "knowledge/architecture_patterns/modular_monolith.md",
        "reason": "The system has medium complexity and can start as one structured application before evolving later."
    }