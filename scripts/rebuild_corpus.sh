#!/usr/bin/env bash
# Rebuild the Arwen Policy Corpus from configured sources.
# Run from the repository root.
set -euo pipefail

echo "=== Arwen Policy Corpus Rebuild ==="
echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Source URLs — ordered by organization.
# Add new sources here as they become available.
SOURCES=(
  # ICANN
  "https://www.icann.org/resources/pages/governance/bylaws-en"
  "https://www.icann.org/en/announcements"

  # IETF
  "https://www.ietf.org/about/introduction/"
  "https://www.ietf.org/standards/rfcs/"
  "https://www.rfc-editor.org/rfc/rfc3935"

  # ITU
  "https://www.itu.int/en/about/Pages/default.aspx"
  "https://www.itu.int/en/mediacentre/Pages/default.aspx"
)

SUCCESS=0
FAILED=0

for url in "${SOURCES[@]}"; do
  echo ""
  echo "--- $url ---"
  if uv run arwen-etl ingest-url "$url" 2>&1 | head -3; then
    SUCCESS=$((SUCCESS + 1))
    echo "  -> OK"
  else
    FAILED=$((FAILED + 1))
    echo "  -> FAILED"
  fi
done

echo ""
echo "=== Rebuild Complete ==="
echo "Succeeded: $SUCCESS"
echo "Failed:    $FAILED"
echo "Extracted: $(ls data/extracted/*.json 2>/dev/null | wc -l) documents"
echo "Registry:  $(ls data/registry/*.json 2>/dev/null | wc -l) records"
echo "Finished:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
