#!/usr/bin/env bash
# 2 つのコミットの間の変更が models/ の下だけか判定し、true か false を出力する。
#
#   model-only.sh BASE HEAD
#
# BASE が空・存在しないコミット（新しいブランチの最初の push の 0000… など）の場合や、変更がない場合は false
# （= すべて確認・ビルドする側に倒す）。
set -euo pipefail

base="${1:-}"
head="${2:?HEAD を指定してください}"

if [ -z "$base" ] || ! git cat-file -e "$base^{commit}" 2>/dev/null; then
  echo false
  exit 0
fi

changed=$(git diff --name-only "$base" "$head")
if [ -n "$changed" ] && ! grep -qv '^models/' <<<"$changed"; then
  echo true
else
  echo false
fi
