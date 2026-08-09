---
license: apache-2.0
tags:
  - lora
  - qwen
  - policy
  - internet-governance
  - deliberation
pretty_name: Arwen Policy LoRA
---

# Arwen Policy LoRA

LoRA (Low-Rank Adaptation) adapter for the [Arwen Policy Base](https://huggingface.co/soyames/arwen-policy-base)
model. This adapter fine-tunes the base model for evidence-grounded policy
deliberation and stakeholder-aware synthesis on the
[Arwen Policy Corpus](https://huggingface.co/datasets/soyames/arwen-policy-corpus).

## Base Model

[soyames/arwen-policy-base](https://huggingface.co/soyames/arwen-policy-base)
(Qwen 3 configured for policy analysis via Ollama).

## Intended Use

- Evidence-grounded policy synthesis
- Stakeholder-aware deliberation
- Multistakeholder Internet governance analysis

## Limitations

- Task-specific adaptation; not a general-purpose model.
- Performance depends on the quality and coverage of the underlying corpus.
- Should not be used as an authoritative policy-making tool.

## Links

- **Base model**: https://huggingface.co/soyames/arwen-policy-base
- **Dataset**: https://huggingface.co/datasets/soyames/arwen-policy-corpus
- **Interactive demo**: https://huggingface.co/spaces/soyames/arwen-policy
- **Source code**: https://github.com/soyames/arwen-policy

## Citation

See [CITATION.cff](https://github.com/soyames/arwen-policy/blob/main/CITATION.cff)
in the project repository.
