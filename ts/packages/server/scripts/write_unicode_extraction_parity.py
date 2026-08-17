"""Record Python's extraction of non-ASCII sources, so the offset conversion can be held to it.

`tests/fixtures/extractor_corpus/` is entirely ASCII, which leaves the single
most dangerous difference in the TypeScript port untested by the golden
snapshot: tree-sitter's Node binding reports **UTF-16 code-unit indices** where
the Python binding reports **UTF-8 byte offsets**. For `x = "eee"` written with
accents, the statement that follows starts at byte 15 and at code unit 11.

Those offsets are the durable contract -- they are digested into chunk ids,
stored on every chunk and reference row, and handed to callers as edit spans --
so getting them wrong corrupts data silently, and only for files containing a
non-ASCII character. An all-ASCII corpus would never notice.

This records what the shipping Python build extracts from sources that exercise
the three regimes that differ: two-byte characters (Latin-1 supplement),
three-byte characters (CJK), and four-byte astral characters (emoji, musical
symbols), the last of which is also where Python's code-point string indexing
parts company with JavaScript's UTF-16 indexing.

Run it from the repository root after touching either extractor:

    uv run python ts/packages/server/scripts/write_unicode_extraction_parity.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

from code_indexing_mcp.extractor import TreeSitterExtractor

# Each case pairs a language with a source whose byte offsets and code-unit
# offsets disagree from the first line onward, so an unconverted offset cannot
# accidentally still be right.
CASES: dict[str, tuple[str, str]] = {
    "latin1.py": (
        "python",
        'TITLE = "café-münchen"\n\n\n'
        "def naïve(größe, café=1):\n"
        '    """Füge größe hinzu."""\n'
        "    return größe + café\n\n\n"
        "class Wörterbuch:\n"
        "    def übersetzen(self, wort):\n"
        "        return naïve(wort)\n",
    ),
    "cjk.py": (
        "python",
        'DESCRIPTION = "説明文テキスト"\n\n\n'
        "def 計算する(引数):\n"
        "    return 引数 * 2\n\n\n"
        "class データ処理:\n"
        "    def 実行(self):\n"
        "        return 計算する(1)\n",
    ),
    "astral.py": (
        "python",
        # Astral characters are two UTF-16 units and four UTF-8 bytes, which is
        # where Python's code-point string indexing and JavaScript's UTF-16
        # indexing diverge as well as the byte/unit mapping.
        'EMOJI = "🎉🎊🎈 party 𝄞𝄢 clefs"\n\n\n'
        "def celebrate(count):\n"
        '    """Return 🎉 repeated."""\n'
        '    return "🎉" * count\n\n\n'
        "class Party:\n"
        "    def start(self):\n"
        "        return celebrate(3)\n",
    ),
    "mixed.ts": (
        "typescript",
        'const GREETING = "héllo 世界 🌍";\n\n'
        "export interface Café {\n"
        "  naïve: string;\n"
        "}\n\n"
        "export function grüße(café: Café): string {\n"
        "  return `${GREETING} ${café.naïve}`;\n"
        "}\n\n"
        "export class Wörterbuch {\n"
        "  übersetzen(wort: string): string {\n"
        "    return grüße({ naïve: wort });\n"
        "  }\n"
        "}\n",
    ),
    "bom.py": (
        "python",
        # A byte-order mark is stripped before extraction, so every offset below
        # is relative to the BOM-free bytes. Written here as a literal U+FEFF so
        # the fixture carries it.
        '﻿"""Módulo con BOM."""\n\n\ndef función(año):\n    return año\n',
    ),
    "wide_lines.py": (
        "python",
        # An oversized line built from multi-byte characters, so the chunk
        # splitter's code-point budget and its byte offsets have to disagree.
        "payload = '" + ("é" * 3000) + "'\n" + "tail = '" + ("🎈" * 1500) + "'\n",
    ),
}


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    extractor = TreeSitterExtractor()
    cases = []
    for name, (language, source) in CASES.items():
        result = extractor.extract(Path(name), language, source.encode("utf-8"))
        cases.append(
            {
                "name": name,
                "language": language,
                "source": source,
                "has_errors": result.has_errors,
                "chunks": [
                    [
                        chunk.kind,
                        chunk.symbol,
                        chunk.qualified_symbol,
                        chunk.parent_symbol,
                        chunk.start_byte,
                        chunk.end_byte,
                        chunk.start_line,
                        chunk.end_line,
                        chunk.part_index,
                        _digest(chunk.content),
                        _digest(chunk.embedding_text),
                        _digest(chunk.search_text),
                    ]
                    for chunk in result.chunks
                ],
                "references": [
                    [
                        reference.kind,
                        reference.target_name,
                        reference.written_name,
                        reference.start_byte,
                        reference.end_byte,
                        reference.start_line,
                        reference.end_line,
                        reference.source_qualified_symbol,
                    ]
                    for reference in result.references
                ],
                "declarations": [
                    [
                        declaration.qualified_symbol,
                        declaration.kind,
                        declaration.start_byte,
                        declaration.end_byte,
                        declaration.start_line,
                        declaration.end_line,
                        [
                            [parameter.name, parameter.kind, parameter.required]
                            for parameter in declaration.parameters
                        ],
                    ]
                    for declaration in result.declarations
                ],
            }
        )
    destination = (
        Path(__file__).resolve().parents[1] / "test" / "fixtures" / "unicode-extraction.json"
    )
    destination.write_text(json.dumps({"cases": cases}, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
