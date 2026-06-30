# Architecture Blueprint Schema

## Purpose

The Architecture Blueprint shows how the proposed system should be structured. It must include both a simple business view and a more detailed technical view.

---

## Required Sections

### 1. Blueprint Metadata

- blueprint_id
- project_name
- version
- created_by
- created_date

### 2. Input Reference

- context_record_id
- selected_features
- selected_architecture_pattern

### 3. Stakeholder View

A simple explanation of the system for non-technical users.

Required fields:

- actors
- business_capabilities
- business_flow
- expected_business_value

### 4. Technical View

A technical explanation of how the system is built.

Required fields:

- frontend_components
- backend_components
- data_components
- ai_or_ml_components
- integration_components
- security_components
- monitoring_components

### 5. Architecture Diagram

Required formats:

- mermaid_diagram
- short_text_flow

### 6. Traceability

Each component should connect back to at least one feature.

Required fields:

- feature_id
- component_id
- reason_for_mapping

### 7. Open Risks

- risk
- impact
- mitigation