def validate_outputs(features, pattern):
    errors = []

    if not features:
        errors.append("No features were generated.")

    for feature in features:
        if "feature_id" not in feature:
            errors.append("A feature is missing feature_id.")
        if "name" not in feature:
            errors.append("A feature is missing name.")

    if not pattern.get("pattern_name"):
        errors.append("Architecture pattern is missing.")

    if not pattern.get("reason"):
        errors.append("Pattern selection reason is missing.")

    required_feature_ids = {"FEAT-001", "FEAT-002", "FEAT-003"}
    generated_feature_ids = {feature["feature_id"] for feature in features}

    missing_features = required_feature_ids - generated_feature_ids
    if missing_features:
        errors.append(f"Missing traceability for features: {missing_features}")

    if errors:
        print("\nValidation failed:")
        for error in errors:
            print("-", error)

        print("\nRepair suggestion:")
        print("Please regenerate the missing or incomplete output sections before continuing.")
        return False

    print("\nValidation passed: features, pattern selection, and traceability are complete enough for the prototype.")
    return True