"""Documentation shape and link-resolution checks (pinned per Art of README)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def _check_links(doc: Path) -> None:
    text = doc.read_text(encoding="utf-8")
    for raw in LINK_RE.findall(text):
        target = raw.split("#", 1)[0].strip()
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        resolved = (doc.parent / target).resolve()
        assert resolved.exists(), f"{doc.name}: broken relative link -> {raw}"


def test_docs_links_and_translation_parity() -> None:
    en = ROOT / "README.md"
    zh = ROOT / "README.zh-CN.md"
    contrib = ROOT / "CONTRIBUTING.md"
    for p in (en, zh, contrib):
        assert p.is_file(), f"Missing doc: {p}"
        _check_links(p)

    en_txt = en.read_text(encoding="utf-8")
    zh_txt = zh.read_text(encoding="utf-8")

    en_h2 = [ln for ln in en_txt.splitlines() if ln.startswith("## ")]
    zh_h2 = [ln for ln in zh_txt.splitlines() if ln.startswith("## ")]
    assert len(en_h2) == len(zh_h2) >= 5, f"Section parity mismatch: {len(en_h2)} vs {len(zh_h2)}"

    # Cross-links at top
    assert "README.zh-CN.md" in "\n".join(en_txt.splitlines()[:10])
    assert "README.md" in "\n".join(zh_txt.splitlines()[:10])

    # Core funnel elements present
    for txt in (en_txt, zh_txt):
        assert "pip install" in txt
        assert "LICENSE" in txt
        assert "proofline" in txt

    # Ensure zero leaked personal paths anywhere in repo python/markdown files
    for fp in list(ROOT.rglob("*.py")) + list(ROOT.rglob("*.md")):
        content = fp.read_text(encoding="utf-8")
        assert "C:\\Users\\" not in content and "D:\\Myfile" not in content, f"Leaked path in {fp}"
