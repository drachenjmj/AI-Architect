# AI Architect Agent

## Project Overview

This project is part of the Multi-Agent Architecture Assistant project for BCG.  
The goal is to build an AI-based solution architect that can take a structured Context Record and generate architecture outputs.

This module focuses on Maheen’s part:

- Output schemas
- Architect / Writer Agent
- Architecture Blueprint generation
- ADR generation
- Component Description generation
- Traceability
- Basic validation

---

## Architect Agent Workflow

The current workflow is:

```text
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




AI_Architect_Agent/
│
├── agents/
│   ├── architect_agent.py
│   ├── pattern_selector.py
│   ├── blueprint_generator.py
│   ├── adr_generator.py
│   ├── component_generator.py
│   ├── validator.py
│   └── llm/
│       └── llm_interface.py
│
├── knowledge/
│   └── architecture_patterns/
│
├── outputs/
│   ├── architecture_blueprint.md
│   ├── adrs.md
│   └── component_descriptions.md
│
├── prompts/
│   └── architect_prompt.md
│
├── inputs/
│   └── context_record.json
│
├── schemas/
│   ├── context_record_schema.md
│   ├── architecture_blueprint_schema.md
│   ├── adr_schema.md
│   └── component_description_schema.md
│
└── README.md