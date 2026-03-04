#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

TARGETS=(
  "frontend/src/pages"
  "frontend/src/features"
)

PATTERNS=(
  "from ['\"][^'\"]*/api/(sync|schedule)['\"]"
  "from ['\"][^'\"]*/api/graph['\"]"
  "['\"]/api/(sync|schedule|repos)\\b"
  "['\"]/sync/(schedule|info|full-refresh|stars|embeddings|clustering|status|jobs)\\b"
)

existing_targets=()
for target in "${TARGETS[@]}"; do
  if [[ -d "${target}" ]]; then
    existing_targets+=("${target}")
  fi
done

if [[ "${#existing_targets[@]}" -eq 0 ]]; then
  echo "No target directories found, skipping legacy API gate."
  exit 0
fi

violations=0
for pattern in "${PATTERNS[@]}"; do
  if matches="$(rg -n --no-heading -e "${pattern}" "${existing_targets[@]}" 2>/dev/null)"; then
    if [[ -n "${matches}" ]]; then
      if [[ "${violations}" -eq 0 ]]; then
        echo "Found forbidden legacy API usage in page/feature layer:"
      fi
      violations=1
      echo "${matches}"
    fi
  fi
done

if [[ "${violations}" -ne 0 ]]; then
  echo
  echo "Use only frontend/src/api/v2/* in page/feature layer."
  exit 1
fi

echo "Legacy API usage gate passed."
