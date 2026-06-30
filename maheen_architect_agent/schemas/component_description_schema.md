# Component Description Schema

## Purpose

The Component Description explains each component used in the Architecture Blueprint. It describes what the component does, why it exists, what it receives as input, and what it produces as output.

---

## Required Sections

### 1. Component Metadata

- component_id
- component_name
- component_type
- related_blueprint_id

### 2. Purpose

Describe the purpose of this component.

Required fields:

- business_purpose
- technical_purpose

### 3. Inputs

What information does this component receive?

Required fields:

- input_data
- source

### 4. Outputs

What information does this component produce?

Required fields:

- output_data
- destination

### 5. Dependencies

Which components does this component interact with?

Required fields:

- upstream_components
- downstream_components

### 6. Technologies

Possible technologies used.

Examples:

- Python
- FastAPI
- PostgreSQL
- Azure
- OpenAI
- Docker

### 7. Traceability

Required fields:

- related_feature_id
- related_requirement_id
- related_adr_id