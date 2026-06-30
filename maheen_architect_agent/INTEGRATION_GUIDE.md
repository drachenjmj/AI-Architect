# Architect Agent Integration Guide

## Purpose

The Architect Agent is responsible for transforming a customer Context Record into architecture documentation.

It produces:

- Architecture Blueprint
- Architecture Decision Records (ADRs)
- Component Descriptions

The module also performs basic validation before completing execution.

---

# Expected Input

## Required

Input file:

```
inputs/context_record.json
```

The Context Record should contain:

- Project Name
- Business Goal
- Functional Requirements
- Non-functional Requirements
- Constraints

---

## Future Input (from Kush's RAG)

The Architect Agent can also receive retrieved architecture knowledge such as:

- Architecture Design Patterns
- Design Principles
- Technology Catalog
- Quality Attributes

This knowledge will be provided by the Research Agent.

---

## Future Input (from Kati's Orchestrator)

The Orchestrator will provide:

- Context Record
- Retrieved Knowledge
- Prompt Template
- LLM Response

through the LLM Interface.

---

# Current Workflow

Context Record

↓

Feature Design

↓

Architecture Pattern Selection

↓

Architecture Blueprint

↓

Architecture Decision Records

↓

Component Descriptions

↓

Validation

---

# Generated Outputs

The Architect Agent generates:

```
outputs/
│
├── architecture_blueprint.md
├── adrs.md
└── component_descriptions.md
```

---

# Traceability

The module maintains traceability between:

Requirement

↓

Feature

↓

Component

↓

ADR

---

# LLM Integration Point

Future AI models should be connected through:

```
agents/llm/llm_interface.py
```

No changes are required in the remaining Architect Agent modules.

---

# Current Status

Implemented

✅ Context Record

✅ Feature Design

✅ Pattern Selection

✅ Blueprint Generation

✅ ADR Generation

✅ Component Descriptions

✅ Traceability

✅ Validation

Prepared

⬜ LLM Integration

⬜ RAG Integration

⬜ Multi-Agent Orchestration