"""OpenRouter teacher-model provider for dataset generation.

Uses the free OpenRouter endpoint for transforming policy documents
into supervised training examples. Never used for inference.

Credentials:
    OPENROUTER_API_KEY environment variable (never hard-coded).
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# OpenRouter free models — these rotate, but the free endpoint auto-selects
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_FREE_MODEL = "openrouter/free"  # auto-selects best free model
OPENROUTER_REFERER = "https://github.com/soyames/arwen-policy"

# Maximum tokens for teacher responses
MAX_RESPONSE_TOKENS = 2048
REQUEST_TIMEOUT = 120.0
MAX_RETRIES = 3
RETRY_DELAY = 5.0


def _get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable is not set. "
            "Set it to your OpenRouter API key."
        )
    return key


def _build_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_get_api_key()}",
        "Content-Type": "application/json",
        "HTTP-Referer": OPENROUTER_REFERER,
        "X-Title": "Arwen Policy Dataset Builder",
    }


def call_openrouter(
    messages: list[dict[str, str]],
    model: str = OPENROUTER_FREE_MODEL,
    max_tokens: int = MAX_RESPONSE_TOKENS,
    temperature: float = 0.1,
    response_format: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call the OpenRouter chat completions API.

    Returns:
        dict with keys: success, content, model, usage, error
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if response_format:
        payload["response_format"] = response_format

    last_error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = httpx.post(
                OPENROUTER_API_URL,
                json=payload,
                headers=_build_headers(),
                timeout=REQUEST_TIMEOUT,
            )

            if resp.status_code == 200:
                data = resp.json()
                choice = data.get("choices", [{}])[0]
                content = choice.get("message", {}).get("content", "")
                usage = data.get("usage", {})
                model_used = data.get("model", model)
                return {
                    "success": True,
                    "content": content,
                    "model": model_used,
                    "usage": usage,
                    "error": "",
                }
            elif resp.status_code == 429:
                # Rate limited — wait and retry
                wait = RETRY_DELAY * attempt * 2
                logger.warning(
                    "OpenRouter rate limited (attempt %d/%d), waiting %.0fs",
                    attempt, MAX_RETRIES, wait,
                )
                time.sleep(wait)
                last_error = f"Rate limited (429) on attempt {attempt}"
            elif resp.status_code == 402:
                return {
                    "success": False,
                    "content": "",
                    "model": model,
                    "usage": {},
                    "error": "OpenRouter quota exhausted (402). Free tier limit reached.",
                }
            else:
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                logger.warning("OpenRouter error: %s", last_error)
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)

        except httpx.TimeoutException:
            last_error = f"Timeout after {REQUEST_TIMEOUT}s"
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
        except Exception as e:
            last_error = str(e)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    return {
        "success": False,
        "content": "",
        "model": model,
        "usage": {},
        "error": last_error,
    }


def build_teacher_prompt(
    document_text: str,
    document_meta: dict[str, Any],
    task_type: str,
) -> list[dict[str, str]]:
    """Build a teacher prompt for generating a single training example.

    The teacher is instructed to use ONLY the source material,
    never invent claims, and always provide evidence provenance.
    """
    title = document_meta.get("title", "Untitled")
    source = document_meta.get("source", "Unknown")
    source_url = document_meta.get("source_url", "")
    doc_id = document_meta.get("document_id", "")
    pub_date = document_meta.get("published_at", "unknown")
    language = document_meta.get("language", "en")

    # Truncate document text to fit context window
    max_chars = 8000
    doc_excerpt = document_text[:max_chars]
    if len(document_text) > max_chars:
        doc_excerpt += f"\n\n[... document continues, {len(document_text)} chars total]"

    task_descriptions = {
        "policy_question": "Generate a specific policy question about this document and answer it using only the document content.",
        "stakeholder_position": "Identify a stakeholder position expressed or implied in this document and explain their stance with evidence from the text.",
        "evidence_extraction": "Extract a factual claim from this document and explain what evidence the document provides to support it.",
        "argument_identification": "Identify a policy argument made in this document and reconstruct its reasoning chain from the text.",
        "historical_context": "Explain the historical or institutional context of this document based on its content and metadata.",
        "institutional_role": "Explain the role and authority of the institution that produced this document, based on the text.",
        "document_understanding": "Explain what this document is about in a way that would help someone understand its policy significance.",
    }

    task_instruction = task_descriptions.get(
        task_type, task_descriptions["document_understanding"]
    )

    system_prompt = """You are a policy-analysis teacher model generating supervised training data.

CRITICAL RULES:
- Ground answers in the supplied document content as your primary source.
- You may use general policy knowledge to contextualize the document within broader Internet governance and digital policy frameworks, but clearly distinguish document content from broader context.
- Do not invent stakeholder positions, quotations, dates, or institutional roles that are not in the document.
- Every factual claim about the document must identify its supporting source evidence from the document text.
- If the document provides insufficient evidence for the specific task, respond with {"skip": true, "reason": "..."}.
- Return VALID JSON only, with no additional text.

Output format:
{
  "skip": false,
  "question": "A specific question about the document content",
  "answer": "A thorough answer grounded in the document",
  "evidence": [
    {
      "quote_or_excerpt": "Relevant excerpt from the document",
      "explanation": "How this supports the answer"
    }
  ],
  "stakeholders_mentioned": ["stakeholder names found in text"],
  "policy_topics": ["relevant policy topics"],
  "confidence": "high|medium|low"
}"""

    user_prompt = f"""TASK: {task_instruction}

DOCUMENT INFORMATION:
Title: {title}
Source: {source}
Source URL: {source_url}
Publication Date: {pub_date}
Document ID: {doc_id}
Language: {language}

DOCUMENT CONTENT:
{doc_excerpt}

Generate one training example of type: {task_type}
Remember: use ONLY the supplied source material. No invention."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def parse_teacher_response(
    content: str,
    task_type: str,
    doc_meta: dict[str, Any],
) -> dict[str, Any] | None:
    """Parse and validate a teacher model response.

    Returns None if the response is invalid or should be skipped.
    """
    if not content or not content.strip():
        return None

    # Try to extract JSON from the response
    json_str = content.strip()
    # Handle markdown code blocks
    if "```json" in json_str:
        start = json_str.index("```json") + 7
        end = json_str.index("```", start)
        json_str = json_str[start:end].strip()
    elif "```" in json_str:
        start = json_str.index("```") + 3
        end = json_str.index("```", start)
        json_str = json_str[start:end].strip()

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        # Try to find JSON object boundaries
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            if start_char in content and end_char in content:
                s = content.index(start_char)
                e = content.rindex(end_char) + 1
                try:
                    parsed = json.loads(content[s:e])
                    break
                except json.JSONDecodeError:
                    continue
        else:
            logger.warning("Could not parse teacher response as JSON")
            return None

    # Check skip flag
    if parsed.get("skip"):
        return None

    question = parsed.get("question", "").strip()
    answer = parsed.get("answer", "").strip()

    if not question or not answer:
        return None
    if len(question) < 10 or len(answer) < 20:
        return None

    # Build the example
    doc_id = doc_meta.get("document_id", "")
    doc_hash = doc_meta.get("artifact_sha256", "")
    source_url = doc_meta.get("source_url", "")

    evidence = []
    for ev in parsed.get("evidence", []):
        if isinstance(ev, dict) and ev.get("quote_or_excerpt"):
            evidence.append({
                "document_id": doc_id,
                "source_hash": doc_hash,
                "source_url": source_url,
                "quote_or_excerpt": str(ev.get("quote_or_excerpt", ""))[:500],
                "explanation": str(ev.get("explanation", ""))[:500],
            })

    example = {
        "task_type": task_type,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are Arwen Policy, a multistakeholder policy-analysis AI. "
                    "Combine policy reasoning with source evidence when available. "
                    "Distinguish between general stakeholder perspectives and "
                    "documented organizational positions. Preserve stakeholder "
                    "disagreement and disclose missing perspectives. "
                    "Attribute specific claims to documented sources. "
                    "Do not fabricate facts, dates, or organizational positions."
                ),
            },
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "source_document_ids": [doc_id],
        "source_hashes": [doc_hash],
        "source_urls": [source_url],
        "evidence": evidence,
        "stakeholders_mentioned": parsed.get("stakeholders_mentioned", []),
        "policy_topics": parsed.get("policy_topics", []),
        "teacher_confidence": parsed.get("confidence", "unknown"),
        "language": doc_meta.get("language", "en"),
    }

    return example
