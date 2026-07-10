#!/usr/bin/env bash
set -euo pipefail

source_url="https://github.com/wfcommons/pegasus-instances.git"
source_commit="813a2a7d3e7273200805e89f5475f9126d903eab"
destination="data/pegasus-instances"

if [[ -e "$destination" ]]; then
  echo "Refusing to replace existing $destination" >&2
  exit 1
fi

git clone --depth 1 --branch v1.4 "$source_url" "$destination"
actual_commit="$(git -C "$destination" rev-parse HEAD)"
if [[ "$actual_commit" != "$source_commit" ]]; then
  echo "Unexpected workflow revision: $actual_commit" >&2
  exit 1
fi

echo "Downloaded workflow instances at $actual_commit"
