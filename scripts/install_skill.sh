#!/usr/bin/env bash
set -euo pipefail

repo_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
skill_source="${repo_root}/SKILL.md"
target_dir="${1:-${HOME}/.claude/skills/agentic-imodels}"

if [[ ! -f "${skill_source}" ]]; then
  printf 'SKILL.md not found at %s\n' "${skill_source}" >&2
  exit 1
fi

mkdir -p "${target_dir}"
cp "${skill_source}" "${target_dir}/SKILL.md"
printf 'Installed %s\n' "${target_dir}/SKILL.md"
