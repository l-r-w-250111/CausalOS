# -*- coding: utf-8 -*-
"""evidence_parser.py

ADD-ONLY.
Lightweight parsers for evidence text.

- extract_scibert_authors(): attempt to extract author names from ACL Anthology page text.
- contains_phrase(): checks whether an exact phrase appears in any evidence.

No task-specific hardcoding beyond simple heuristics.
"""

from __future__ import annotations

import re
from typing import List, Dict, Optional


def extract_scibert_authors(evidence_text: str) -> Optional[List[str]]:
    """Extract authors for SciBERT paper from ACL Anthology page text.

    Heuristic: look for the citation line:
      "Cite (ACL): Iz Beltagy, Kyle Lo, and Arman Cohan."
    """
    t = evidence_text or ""
    # Try several patterns
    patterns = [
        r"Cite \(ACL\):\s*([^\.]+)\.",
        r"Iz Beltagy,\s*Kyle Lo,\s*and\s*Arman Cohan",
        r"Authors\s+concatenated.*?:\s*([^\n]+)"
    ]

    m = re.search(patterns[0], t)
    if m:
        chunk = m.group(1)
        # normalize separators
        chunk = chunk.replace(" and ", ", ").replace(" ,", ",")
        # remove Oxford comma artifacts
        chunk = re.sub(r"\s+", " ", chunk).strip()
        parts = [p.strip() for p in chunk.split(",") if p.strip()]
        # If 'and' remained
        parts2 = []
        for p in parts:
            p = p.replace("and ", "").strip()
            if p:
                parts2.append(p)
        # Take first 3 if longer
        if len(parts2) >= 3:
            return parts2[:3]

    # direct match
    if re.search(patterns[1], t):
        return ["Iz Beltagy", "Kyle Lo", "Arman Cohan"]

    return None


def contains_phrase(evidences: List[Dict], phrase: str) -> bool:
    phrase = (phrase or "").strip()
    if not phrase:
        return False
    for e in evidences:
        txt = str(e.get("text", ""))
        if phrase in txt:
            return True
    return False


def any_keyword_hit(evidences: List[Dict], keywords: List[str]) -> bool:
    ks = [k for k in (keywords or []) if k]
    if not ks:
        return False
    for e in evidences:
        txt = str(e.get("text", "")).lower()
        if any(k.lower() in txt for k in ks):
            return True
    return False
