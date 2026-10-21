"""exe に組み込まれるライブラリの NOTICE ファイルを集めて 1 つのテキストにする。

Apache License 2.0 の第 4 条 (d) により、NOTICE ファイルを含むライブラリを再配布する場合は、
その内容を同梱する必要がある（cargo-about のライセンス表示には NOTICE は含まれない）。

    python .github/scripts/collect-notices.py OUTPUT [cargo metadata に渡す引数...]

例: python .github/scripts/collect-notices.py THIRD_PARTY_NOTICES.txt --features cuda --filter-platform x86_64-pc-windows-msvc
"""

import json
import re
import subprocess
import sys
from pathlib import Path

NOTICE_FILE = re.compile(r"^notice", re.IGNORECASE)


def linked_packages(metadata):
    """ルートのパッケージから通常の依存（ビルド・テスト用を除く）でたどれるパッケージ。ルート自身は除く。"""
    resolve = metadata["resolve"]
    nodes = {n["id"]: n for n in resolve["nodes"]}
    seen = set()
    stack = [resolve["root"]]
    while stack:
        pkg = stack.pop()
        if pkg in seen:
            continue
        seen.add(pkg)
        for dep in nodes[pkg]["deps"]:
            if any(kind["kind"] is None for kind in dep["dep_kinds"]):
                stack.append(dep["pkg"])
    seen.discard(resolve["root"])
    packages = {p["id"]: p for p in metadata["packages"]}
    return sorted((packages[i] for i in seen), key=lambda p: (p["name"], p["version"]))


def notice_files(package):
    """パッケージの直下にある NOTICE ファイル"""
    root = Path(package["manifest_path"]).parent
    return sorted(p for p in root.iterdir() if p.is_file() and NOTICE_FILE.match(p.name))


def main():
    # Windows では標準の文字コードが UTF-8 ではない（cp1252 など）ため、出力も cargo metadata の読み取りも UTF-8 に固定する
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    output = Path(sys.argv[1])
    metadata = json.loads(
        subprocess.run(
            ["cargo", "metadata", "--format-version", "1", *sys.argv[2:]],
            check=True,
            capture_output=True,
            encoding="utf-8",
        ).stdout
    )

    sections = []
    for package in linked_packages(metadata):
        for path in notice_files(package):
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            header = f"{package['name']} {package['version']}（{package.get('license') or 'ライセンス不明'}）: {path.name}"
            sections.append(f"{header}\n{'-' * len(header)}\n{text}\n")

    intro = (
        "othello.exe に組み込まれているライブラリのうち、NOTICE ファイルを含むものの NOTICE です。\n"
        "各ライブラリのライセンス文は THIRD_PARTY_LICENSES.html を参照してください。\n"
    )
    body = "\n".join(sections) if sections else "NOTICE ファイルを含むライブラリはありません。\n"
    output.write_text(f"{intro}\n{'=' * 60}\n\n{body}", encoding="utf-8")
    print(f"{len(sections)} 件の NOTICE を {output} に書き出しました")


if __name__ == "__main__":
    main()
