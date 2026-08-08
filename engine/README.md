# Engine

The engine is the application-facing orchestration layer.

It connects retrieval to multistakeholder deliberation and produces a structured policy-analysis object that a model, API or Hugging Face Space can consume.

The current implementation deliberately stops before free-form LLM synthesis. This keeps the evidence and deliberation stages testable and auditable.
