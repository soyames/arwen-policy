"""Inspect a v2 example and compare with v1."""
import json
from pathlib import Path

# V2
v2_path = Path("datasets/sft_v2/train.jsonl")
if not v2_path.exists():
    print("No v2 examples yet")
    exit()

v2 = [json.loads(l) for l in v2_path.read_text(encoding="utf-8").splitlines() if l.strip()]
print(f"V2 examples: {len(v2)}")
print()

# Show first v2 example
ex = v2[0]
print("=" * 60)
print("V2 EXAMPLE")
print("=" * 60)
print(f"Task: {ex['task_type']}")
print(f"Source: {ex['source_document_ids'][0][:30]}")
print(f"URL: {ex['source_urls'][0][:80]}")
for msg in ex["messages"]:
    print(f"\n[{msg['role'].upper()}]:")
    print(msg["content"][:1000])
print(f"\nReasoning: {ex.get('reasoning', '')[:500]}")
print(f"Uncertainty: {ex.get('uncertainty', '')[:300]}")
print(f"Evidence: {len(ex.get('evidence', []))} entries")
print(f"Confidence: {ex.get('teacher_confidence')}")

# Show second example
if len(v2) > 1:
    ex2 = v2[1]
    print()
    print("=" * 60)
    print("V2 EXAMPLE 2")
    print("=" * 60)
    print(f"Task: {ex2['task_type']}")
    for msg in ex2["messages"]:
        print(f"\n[{msg['role'].upper()}]:")
        print(msg["content"][:800])

# Compare with v1 if available
v1_path = Path("datasets/sft/train.jsonl")
if v1_path.exists():
    v1 = [json.loads(l) for l in v1_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print()
    print("=" * 60)
    print(f"V1 has {len(v1)} examples (v2 has {len(v2)})")
    print("V1 avg answer length:", sum(len(
        [m for m in ex["messages"] if m["role"] == "assistant"][0]["content"]
        if [m for m in ex["messages"] if m["role"] == "assistant"] else ""
    ) for ex in v1) // max(len(v1), 1), "chars")
    v2_avg = sum(len(
        [m for m in ex["messages"] if m["role"] == "assistant"][0]["content"]
        if [m for m in ex["messages"] if m["role"] == "assistant"] else ""
    ) for ex in v2) // max(len(v2), 1)
    print(f"V2 avg answer length: {v2_avg} chars")
    # V2 has reasoning, uncertainty fields
    v2_with_reasoning = sum(1 for ex in v2 if ex.get("reasoning"))
    v2_with_uncertainty = sum(1 for ex in v2 if ex.get("uncertainty"))
    print(f"V2 with reasoning: {v2_with_reasoning}/{len(v2)}")
    print(f"V2 with uncertainty: {v2_with_uncertainty}/{len(v2)}")
