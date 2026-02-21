# -*- coding: utf-8 -*-
"""web_evidence_fetcher.py

ADD-ONLY.
Fetches web evidence for fact tasks.

Design goals:
- Prefer existing retrieval_tools.SimpleWebRetriever if available (project-integrated web).
- Fallback to urllib fetch for direct URLs (no search).
- Return normalized evidence objects: {title, url, text, source}.

This module does not depend on task-specific URLs; it just fetches what it's given.
"""

from __future__ import annotations

import re
import time
import html
import urllib.request
import urllib.error
from typing import List, Dict, Optional


def _clean_text(s: str) -> str:
    s = html.unescape(s or "")
    # strip tags crudely (we avoid external deps)
    s = re.sub(r"<script[\s\S]*?</script>", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"<style[\s\S]*?</style>", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def fetch_url(url: str, timeout: int = 12, user_agent: str = "CausalOS/oss-factguard") -> Dict:
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            try:
                txt = raw.decode("utf-8", errors="ignore")
            except Exception:
                txt = raw.decode(errors="ignore")
    except urllib.error.HTTPError as e:
        return {"title": "", "url": url, "text": f"HTTPError: {e.code}", "source": "direct"}
    except Exception as e:
        return {"title": "", "url": url, "text": f"FetchError: {type(e).__name__}: {e}", "source": "direct"}

    # crude title extraction
    m = re.search(r"<title>(.*?)</title>", txt, flags=re.IGNORECASE | re.DOTALL)
    title = _clean_text(m.group(1)) if m else ""
    return {"title": title, "url": url, "text": _clean_text(txt)[:12000], "source": "direct"}


def retrieve_web(query: str, k: int = 5, timeout: int = 12) -> List[Dict]:
    """Try project-integrated web retriever if available."""
    try:
        from retrieval_tools import SimpleWebRetriever  # type: ignore
    except Exception:
        SimpleWebRetriever = None

    if SimpleWebRetriever is None:
        return []

    try:
        r = SimpleWebRetriever(timeout=timeout)
        docs = r.retrieve(query, k=k)
        out = []
        for d in docs[:k]:
            out.append({
                "title": str(d.get("title", "")),
                "url": str(d.get("url", "")),
                "text": str(d.get("text", ""))[:12000],
                "source": str(d.get("source", "web")),
            })
        return out
    except Exception:
        return []
