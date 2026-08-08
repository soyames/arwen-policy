import os
import json
import pathlib
from pathlib import Path
from typing import List, Dict

# ---------------------------------------------------------------------------
# Utility: Load HF token (now fully environment-agnostic)
# ---------------------------------------------------------------------------
def get_hf_token() -> str:
    """
    Retrieve the Hugging Face authentication token.

    The function follows this precedence:
    1. Environment variable ``HF_TOKEN`` (standard approach)
    2. File path specified via ``TOKEN_FILE_PATH`` environment variable
    3. Raise an informative error if neither source provides a token

    Returns:
        str: The raw token string
    """
    # 1️⃣  Environment variable (standard)
    token = os.getenv("HF_TOKEN")
    if token:
        return token.strip()

    # 2️⃣  Token file path from environment variable
    token_path = os.getenv("TOKEN_FILE_PATH")
    if token_path:
        try:
            return Path(token_path).read_text(encoding="utf-8").strip()
        except Exception as exc:
            raise RuntimeError("Failed to read token from specified file path") from exc

    # 3️⃣  No token found
    raise RuntimeError(
        "Hugging Face token not found. Set HF_TOKEN or TOKEN_FILE_PATH environment variable and retry."
    )