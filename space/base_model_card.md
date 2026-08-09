---
license: apache-2.0
library_name: ollama
tags:
  - qwen
  - policy
  - internet-governance
  - deliberation
pretty_name: Arwen Policy Base
---

# Arwen Policy Base

Base language model for the Arwen Policy project — an evidence-grounded,
multistakeholder AI deliberation system for digital policy and Internet governance.

## Model

This repository packages the Arwen Policy configuration for use with
[Qwen 3](https://ollama.com/library/qwen3) via Ollama. The model is configured
for evidence-grounded policy analysis with a 40K context window, low temperature
(0.2), and a system prompt that instructs the model to attribute claims to
documented sources, preserve stakeholder disagreements, disclose missing
perspectives, and never manufacture consensus or evidence.

## Intended Use

- Evidence-grounded policy synthesis
- Stakeholder-aware argument analysis
- Multistakeholder deliberation
- Policy recommendation generation with provenance

## Training

This is the base Qwen 3 model with an Arwen Policy system prompt and inference
configuration. It has not been fine-tuned on the Arwen Policy Corpus.

For the task-adapted version, see [Arwen Policy LoRA](https://huggingface.co/soyames/arwen-policy-lora).

## Links

- **Dataset**: https://huggingface.co/datasets/soyames/arwen-policy-corpus
- **LoRA adapter**: https://huggingface.co/soyames/arwen-policy-lora
- **Interactive demo**: https://huggingface.co/spaces/soyames/arwen-policy
- **Source code**: https://github.com/soyames/arwen-policy

## Citation

See [CITATION.cff](https://github.com/soyames/arwen-policy/blob/main/CITATION.cff)
in the project repository.
