# -*- coding: utf-8 -*-
"""CausalChatAgent_webkr_v2.py

Runner for CausalOS v5.3_full that uses WebKnowledgeRetriever.py (DuckDuckGo) for web search.

Improvements vs v1:
- If no sources -> return fixed "I don't know from the provided sources." (no LLM call)
- Query rewriting for DDG (shorter, keyword-focused)
- For identifier questions (DOI/arXiv/ISSN/ISBN/URL) extract from sources with regex;
  if not found -> don't know (prevents hallucinated identifiers)
- Citation guard: removes citations that reference non-existent source ids

ADD-ONLY: does not modify CausalOS_v5_3_full.py.

Requires:
- duckduckgo-search (pip install duckduckgo-search)
"""

import os
import re
import sys
import json
from typing import Dict, Optional, Tuple, List

import CausalOS_v5_3_full as causal
from WebKnowledgeRetriever import WebKnowledgeRetriever


def _norm(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _is_identifier_question(q: str) -> str:
    """Return one of: doi, arxiv, issn, isbn, url, other/''"""
    t = q.lower()
    if "doi" in t:
        return "doi"
    if "arxiv" in t or "arXiv" in q:
        return "arxiv"
    if "issn" in t:
        return "issn"
    if "isbn" in t:
        return "isbn"
    if "url" in t or "http" in t or "website" in t:
        return "url"
    return ""


def _extract_identifier(kind: str, text: str) -> Optional[str]:
    if not text:
        return None
    if kind == "doi":
        m = re.search(r"\b10\.\d{4,9}/[^\s\]\)\"\']+", text)
        return m.group(0) if m else None
    if kind == "arxiv":
        m = re.search(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", text)
        return m.group(0) if m else None
    if kind == "issn":
        m = re.search(r"\b\d{4}-\d{3}[\dX]\b", text, re.IGNORECASE)
        return m.group(0) if m else None
    if kind == "isbn":
        # very rough ISBN-10/13 patterns
        m = re.search(r"\b(?:97[89][- ]?)?\d{1,5}[- ]?\d{1,7}[- ]?\d{1,7}[- ]?[\dX]\b", text, re.IGNORECASE)
        return m.group(0) if m else None
    if kind == "url":
        m = re.search(r"https?://[^\s\]\)\"\']+", text)
        return m.group(0) if m else None
    return None


class DDGRetrieverAdapter:
    """Adapter to match CausalOS Retriever protocol: retrieve(query,k)->list[dict]."""

    def __init__(self, wkr: WebKnowledgeRetriever):
        self.wkr = wkr

    def _rewrite(self, query: str) -> str:
        q = _norm(query)
        q = q.rstrip('?')
        # If the question contains a quoted title, prefer that + keyword
        qt = re.findall(r'"([^"]{4,200})"', q)
        if qt:
            title = qt[0]
            if "doi" in q.lower():
                return f"{title} DOI"
            if "arxiv" in q.lower():
                return f"{title} arXiv"
            return title
        # Common patterns
        low = q.lower()
        if "doi" in low and " of " in low:
            tail = q.split(" of ", 1)[1]
            return _norm(tail + " DOI")
        if "arxiv" in low and " of " in low:
            tail = q.split(" of ", 1)[1]
            return _norm(tail + " arXiv")
        # remove leading question words to shorten
        q = re.sub(r"^(what is|provide|give|tell me)\b[: ]*", "", q, flags=re.IGNORECASE)
        return q

    def retrieve(self, query: str, k: int = 5):
        q1 = self._rewrite(query)
        res = self.wkr.search(q1, max_results=int(k))
        # fallback: original query if rewrite returned nothing
        if (not res) and q1 != query:
            res = self.wkr.search(query, max_results=int(k))
        out = []
        for r in res or []:
            out.append({
                "id": r.get("url", ""),
                "title": r.get("title", ""),
                "text": r.get("snippet", ""),
                "url": r.get("url", ""),
                "source": "duckduckgo",
            })
        return out


class CausalChatAgent:
    def __init__(self, osys: "causal.UnifiedCausalOSV5_3Full"):
        self.osys = osys
        self.buf: List[str] = []

    def _extract_options(self, text: str) -> Optional[Dict[str, str]]:
        if not text:
            return None
        t = text.replace("\r\n", "\n").replace("\r", "\n")
        pat = re.compile(r'^\s*([A-Z])\s*:\s*(.+?)\s*$', re.MULTILINE)
        opts = {}
        for m in pat.finditer(t):
            k = m.group(1)
            v = m.group(2).strip()
            if v:
                opts[k] = v
        return dict(sorted(opts.items(), key=lambda kv: kv[0])) if len(opts) >= 2 else None

    def _strip_options(self, text: str) -> str:
        t = (text or "").strip()
        m = re.search(r'(\s\n^)([A-Z])\s*:\s*', t)
        if m:
            return t[:m.start()].strip()
        return t

    def _extract_cf_pair(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        t = (text or "").strip()
        t = re.sub(r"\s+", " ", t)
        m = re.search(r'(.+?)\.\s*What\s+(?:if|would have happened if)\s+(.+?)\?', t, re.IGNORECASE)
        if m:
            return m.group(1).strip(), m.group(2).strip()
        return None, None

    def _dont_know(self, sources: List[dict]) -> str:
        msg = "I don't know from the provided sources."
        if os.environ.get("CAUSALOS_SHOW_TRACE", "0") == "1":
            msg += "\n\n[Trace]\n" + json.dumps({"retrieved": sources}, ensure_ascii=False, indent=2)
        return msg

    def _answer_with_web_sources(self, query: str) -> str:
        query = _norm(query)
        if not query:
            return ""

        docs = self.osys.retriever.retrieve(query, k=int(os.environ.get("CAUSALOS_WEB_MAX", "5")))
        sources = []
        for i, d in enumerate(docs[:5], 1):
            sources.append({
                "id": f"S{i}",
                "title": str(d.get("title", "")),
                "source": str(d.get("source", "")),
                "url": str(d.get("url", "")),
                "text": str(d.get("text", ""))[:1600],
            })

        # Hard stop: no sources => don't know
        if not sources:
            return self._dont_know(sources)

        # Identifier questions: extract from sources only; otherwise don't know
        kind = _is_identifier_question(query)
        if kind:
            blob = "\n".join([s.get("text", "") + "\n" + s.get("title", "") for s in sources])
            val = _extract_identifier(kind, blob)
            if not val:
                return self._dont_know(sources)
            # Return extracted id with a conservative citation
            ans = f"{val} [S1]"
            if os.environ.get("CAUSALOS_SHOW_TRACE", "0") == "1":
                ans += "\n\n[Trace]\n" + json.dumps({"retrieved": sources}, ensure_ascii=False, indent=2)
            return ans

        # Otherwise: sources-only generation
        prompt = (
            "You are a careful assistant. Use ONLY the provided SOURCES to answer the QUESTION.\n"
            "If the answer is not supported by the SOURCES, say EXACTLY: 'I don't know from the provided sources.' and stop.\n"
            "Do NOT invent citations. You may cite ONLY these ids: "
            + ", ".join([s["id"] for s in sources])
            + "\n"
            "Return: (1) a short answer, (2) citations as [S#].\n\n"
            f"QUESTION: {query}\n\n"
            f"SOURCES(JSON): {json.dumps(sources, ensure_ascii=False)}\n\n"
            "ANSWER:"
        )

        tok = self.osys.tokenizer(prompt, return_tensors="pt")
        tok = {k: v.to(self.osys.model_device) for k, v in tok.items()}
        with causal.torch.no_grad():
            out = self.osys.model.generate(
                **tok,
                max_new_tokens=220,
                do_sample=False,
                pad_token_id=self.osys.tokenizer.eos_token_id,
            )
        resp = self.osys.tokenizer.decode(out[0][tok["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

        # Citation guard: remove citations not in allowed set
        allowed = set([s["id"] for s in sources])
        def repl(m):
            return m.group(0) if m.group(0).strip('[]') in allowed else ''
        resp = re.sub(r"\[S\d+\]", repl, resp)
        resp = resp.strip()

        # If model output is empty after guard, fallback
        if not resp:
            return self._dont_know(sources)

        if os.environ.get("CAUSALOS_SHOW_TRACE", "0") == "1":
            resp += "\n\n[Trace]\n" + json.dumps({"retrieved": sources}, ensure_ascii=False, indent=2)

        return resp

    def process_block(self, block: str):
        block = (block or "").strip()
        if not block:
            return

        options = self._extract_options(block)
        main_text = self._strip_options(block) if options else block

        # Keep ingest (ADD-ONLY)
        self.osys.ingest_context(main_text, source="user", weight=0.80)

        factual, cf = self._extract_cf_pair(main_text)
        if factual and cf:
            pkt = self.osys.answer_counterfactual_B2(factual, cf, options=options)
            print(pkt.best_effort_answer)
            if os.environ.get("CAUSALOS_SHOW_TRACE", "0") == "1":
                print("\n[Trace]")
                print(json.dumps(pkt.reason_trace, ensure_ascii=False, indent=2))
            return

        # fact-like -> retrieval-first
        if os.environ.get("CAUSALOS_ENABLE_FACT_MODE", "0") == "1":
            if causal._is_exact_fact_task(main_text) or causal._contains_fact_like_patterns(main_text):
                print(self._answer_with_web_sources(main_text))
                return

        print("（反事実形式が検出できませんでした。例: 'X. What would have happened if Y?'）")

    def run_from_stdin(self):
        data = sys.stdin.read()
        if not data:
            return
        data = data.replace("\r\n", "\n").replace("\r", "\n").strip()
        blocks = re.split(r"\n\s*\n+", data)
        for b in blocks:
            self.process_block(b)

    def repl(self):
        print("Enter a multi-line block. Finish with an empty line.", flush=True)
        self.buf = []
        while True:
            try:
                line = input("\nUser: ")
            except EOFError:
                if self.buf:
                    self.process_block("\n".join(self.buf))
                break
            except KeyboardInterrupt:
                if self.buf:
                    self.process_block("\n".join(self.buf))
                break

            s = (line or "").rstrip("\n")
            if s.lower().strip() in ["exit", "quit", "q"]:
                if self.buf:
                    self.process_block("\n".join(self.buf))
                break
            if s.strip() == "":
                if self.buf:
                    self.process_block("\n".join(self.buf))
                self.buf = []
                continue
            self.buf.append(s)


def main():
    print("--- Starting CausalChatAgent (CausalOS v5.3_full + WebKnowledgeRetriever v2) ---", flush=True)

    model_id = os.environ.get("CAUSALOS_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

    # Initialize WebKnowledgeRetriever (DDG)
    wkr = WebKnowledgeRetriever(llm_model=None, tokenizer=None)
    retriever = DDGRetrieverAdapter(wkr)

    osys = causal.UnifiedCausalOSV5_3Full(
        model_id=model_id,
        init_n_nodes=256,
        init_slots_per_concept=2,
        expand_chunk=256,
        local_horizon=10,
        w0=0.7,
        w1=0.3,
        retriever=retriever,
        verifier=None,
    )

    agent = CausalChatAgent(osys)

    if not sys.stdin.isatty():
        agent.run_from_stdin()
        return

    agent.repl()


if __name__ == "__main__":
    main()
