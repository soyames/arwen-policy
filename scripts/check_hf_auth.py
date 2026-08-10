"""Check HF authentication and upload status."""
import os
import sys

from huggingface_hub import HfApi, whoami

try:
    user = whoami()
    print(f"Authenticated as: {user.get('name', '?')}")
    api = HfApi()

    # Check the dataset
    try:
        info = api.dataset_info("soyames/arwen-policy-corpus")
        print(f"Dataset exists: {info.id}")
        print(f"Last modified: {info.last_modified}")
    except Exception as e:
        print(f"Dataset info error: {e}")

except Exception as e:
    print(f"NOT authenticated: {e}")
    print(f"HF_TOKEN set: {bool(os.environ.get('HF_TOKEN'))}")
    print("\nTo authenticate, run: huggingface-cli login")
    sys.exit(1)
