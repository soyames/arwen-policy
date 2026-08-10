"""Quick test of the OpenRouter teacher model with .env loading."""
import os
import json
from pathlib import Path

# Load .env
env_path = Path(".env")
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            if key.strip() not in os.environ:
                os.environ[key.strip()] = val.strip()

key = os.environ.get("OPENROUTER_API_KEY", "")
print(f"Key loaded: {bool(key)} (length={len(key)})")

if not key:
    print("OPENROUTER_API_KEY not found in .env")
    exit(1)

# Test call
from arwen_etl.engine.openrouter_provider import call_openrouter

result = call_openrouter(
    [{"role": "user", "content": "Say hello in one sentence."}],
    max_tokens=50,
)
print(f"Success: {result['success']}")
print(f"Model: {result.get('model', '?')}")
print(f"Content: {result.get('content', '')[:200]}")
if result.get("error"):
    print(f"Error: {result['error']}")
print(f"Usage: {result.get('usage', {})}")
