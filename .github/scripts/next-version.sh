#!/usr/bin/env bash
# 次のリリースのバージョン（例：0.1.3）を出力する。
#
#   next-version.sh
#
# Cargo.toml の version のメジャー・マイナー（例：0.1.0 → 0.1）を使い、既存のタグ v<メジャー>.<マイナー>.* の
# うち最大のパッチ番号に 1 を足す。該当するタグがなければパッチ番号は 0 から始める。
# 大きな変更のときは Cargo.toml の version のメジャー・マイナーを手で上げる（パッチ番号は無視される）。
set -euo pipefail

base=$(sed -n 's/^version *= *"\([0-9]*\.[0-9]*\)\.[0-9]*.*"/\1/p' Cargo.toml | head -n 1)
if [ -z "$base" ]; then
  echo "Cargo.toml から version を読み取れません" >&2
  exit 1
fi

latest=$(git tag --list "v$base.*" --sort=-v:refname | head -n 1)
if [ -z "$latest" ]; then
  echo "$base.0"
else
  echo "$base.$((${latest##*.} + 1))"
fi
