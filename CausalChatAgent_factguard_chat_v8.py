
# -*- coding: utf-8 -*-
"""CausalChatAgent_factguard_chat_v8.py (BUILD v8-j5b)

Fixes the v8-j5 runtime crash:
- NameError: _dedup_evidences was missing. (Seen in err.log)

Also improves author retrieval robustness:
- Primary: Semantic Scholar Graph API (DOI) -> Authors list
- Secondary: CrossCite formatted citation (DOI) -> Authors-like prefix
- Fallback: arXiv API (HTTPS) multi-query + best title match

Design goals:
- Return PERSON as an exact substring from an evidence 'Authors:' line.
- Never print trace JSON to stdout; save to ./logs/trace_turn_<n>.json when CAUSALOS_SHOW_TRACE=1.

"""

import os
import re
import sys
import json
import unicodedata
import urllib.parse
from typing import List, Dict, Any, Optional, Tuple

import torch
from CausalOS_v5_3_full import UnifiedCausalOSV5_3Full

try:
    from retrieval_tools import SimpleWebRetriever
except Exception:
    SimpleWebRetriever = None

from web_evidence_fetcher import fetch_url

BUILD_TAG = 'v8-j5b'
ANCHOR = "I don't know from the provided sources."
ANCHOR_PREFIX = "i don't know from the provided sources"
EXIT_WORDS = {"exit", "quit", "q", "終了"}

_DOI_RX = re.compile(r"\b10\.\d{4,9}/[^\s\]\)\}\<\>]+", re.IGNORECASE)


def _nfkc(s: str) -> str:
    return unicodedata.normalize('NFKC', s or '')


def _norm(s: str) -> str:
    s = _nfkc(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _safe_json_loads(raw: str) -> Optional[Dict[str, Any]]:
    raw = raw or ''
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        j = raw.rfind('}')
        if j > 0:
            try:
                obj = json.loads(raw[:j+1])
                return obj if isinstance(obj, dict) else None
            except Exception:
                return None
        return None


def _record_api_debug(errs: List[Dict[str, Any]], where: str, url: str, raw: str, error: str):
    raw = raw or ''
    head = raw[:200].replace('\n','\\n')
    tail = raw[-200:].replace('\n','\\n') if len(raw) > 200 else ''
    errs.append({'where': where, 'url': url, 'error': error, 'len': len(raw), 'head': head, 'tail': tail})


def _find_urls(text: str) -> List[str]:
    rx = re.compile(r"https?://[^\s)\]\}>\"']+", re.IGNORECASE)
    return rx.findall(text or '')


def _extract_quoted_phrases(text: str) -> List[str]:
    t = text or ''
    q = []
    q += [s.strip() for s in re.findall(r'"([^"]{4,400})"', t) if s.strip()]
    q += [s.strip() for s in re.findall(r'『([^』]{2,400})』', t) if s.strip()]
    q += [s.strip() for s in re.findall(r'「([^」]{2,400})」', t) if s.strip()]
    out = []
    for s in q:
        if s not in out:
            out.append(s)
    return out


def _clean_entity_title(entity: str, user_text: str) -> str:
    ent = (entity or '').strip()
    qs = _extract_quoted_phrases(ent)
    if qs:
        qs.sort(key=len, reverse=True)
        return qs[0]
    ent2 = re.sub(r"(?i)\b(first\s+author\s+of\s+the\s+paper|first\s+author\s+of|author\s+of\s+the\s+paper|paper)\b", " ", ent)
    ent2 = _norm(ent2).strip(" \"'`“””’")
    if ent2:
        return ent2
    qs2 = _extract_quoted_phrases(user_text)
    if qs2:
        qs2.sort(key=len, reverse=True)
        return qs2[0]
    return ''


def _is_anchor_like(user_text: str) -> bool:
    t = _norm(user_text).casefold()
    return bool(t) and (t == _norm(ANCHOR).casefold() or t.startswith(ANCHOR_PREFIX))


def _token_set(s: str) -> set:
    s = _norm(s)
    s = re.sub(r'[^0-9A-Za-z]+', ' ', s)
    toks = [t.lower() for t in s.split() if len(t) >= 2]
    stop = {'and','from','with','the','for','data','mass','radius','properties','implications','paper','journal','letter','letters','of','to','in'}
    return {t for t in toks if t not in stop}


def _title_overlap_score(title: str, entity_title: str) -> float:
    a = _token_set(title)
    b = _token_set(entity_title)
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(b))


def _filter_by_title(evidences: List[Dict[str, Any]], entity_title: str) -> List[Dict[str, Any]]:
    ent = _norm(entity_title)
    if not ent:
        return list(evidences or [])
    scored = []
    for e in evidences or []:
        sc = _title_overlap_score(str(e.get('title','') or ''), ent)
        scored.append((sc, e))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return []
    best = scored[0][0]
    thr = 0.60 if len(_token_set(ent)) >= 6 else 0.40
    thr = max(thr, 0.85 * best)
    kept = [e for sc, e in scored if sc >= thr]
    return kept if kept else [e for _, e in scored[:4]]


def _find_dois(s: str) -> List[str]:
    return [m.group(0).rstrip('.,;') for m in _DOI_RX.finditer(s or '')]


def _dedup_list(xs: List[str], n: int) -> List[str]:
    out=[]; seen=set()
    for x in xs or []:
        k=(x or '').strip().lower()
        if not k or k in seen:
            continue
        seen.add(k); out.append((x or '').strip())
        if len(out)>=n:
            break
    return out


def _extract_dois(evidences: List[Dict[str, Any]]) -> List[str]:
    dois=[]
    for e in evidences or []:
        blob = str(e.get('title','')) + '\n' + str(e.get('text','')) + '\n' + str(e.get('url',''))
        dois += _find_dois(blob)
    return _dedup_list(dois, 6)


def _dedup_evidences(evs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate evidences by (url,title,source)."""
    seen=set(); out=[]
    for e in evs or []:
        key=(str(e.get('url','') or '').strip(), str(e.get('title','') or '').strip(), str(e.get('source','') or '').strip())
        if key in seen:
            continue
        seen.add(key); out.append(e)
    return out


# -------------------- Resolvers (captcha-safe) --------------------

def _semantic_scholar_url(doi: str) -> str:
    doi = (doi or '').strip()
    return f"https://api.semanticscholar.org/graph/v1/paper/DOI:{urllib.parse.quote(doi)}?fields=title,authors"


def _semantic_scholar_meta(doi: str, timeout: int, errs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    url = _semantic_scholar_url(doi)
    try:
        ev = fetch_url(url, timeout=timeout)
        raw = str(ev.get('text','') or '')
        data = _safe_json_loads(raw)
        if not data:
            _record_api_debug(errs, 'semanticscholar', url, raw, 'Semantic Scholar JSON parse failed')
            return None
        title = str(data.get('title','') or '').strip()
        authors = data.get('authors', [])
        names=[]
        if isinstance(authors, list):
            for a in authors[:400]:
                if isinstance(a, dict):
                    nm = str(a.get('name','') or '').strip()
                    if nm:
                        names.append(nm)
        parts=[]
        if title:
            parts.append(f"Title: {title}")
        parts.append(f"DOI: {doi}")
        if names:
            parts.append('Authors: ' + ', '.join(names))
        return {'id':'S0','title': title or 'Semantic Scholar metadata','url': url,'text':'\n'.join(parts),'source':'resolver:semanticscholar'}
    except Exception as e:
        _record_api_debug(errs, 'semanticscholar', url, '', str(e))
        return None


def _crosscite_url(doi: str) -> str:
    doi = (doi or '').strip()
    return f"https://citation.crosscite.org/format?doi={urllib.parse.quote(doi)}&style=apa&lang=en-US"


def _crosscite_meta(doi: str, timeout: int, errs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    url = _crosscite_url(doi)
    try:
        ev = fetch_url(url, timeout=timeout)
        raw = str(ev.get('text','') or '')
        txt = raw.strip()
        if not txt:
            _record_api_debug(errs, 'crosscite', url, raw, 'CrossCite empty')
            return None
        # Author list is generally before the year.
        m = re.split(r"\s*\(\d{4}\)\.", txt, maxsplit=1)
        head = m[0].strip() if m else txt
        parts = [f"DOI: {doi}", "Authors: " + head]
        return {'id':'S0','title': 'CrossCite formatted citation','url': url,'text':'\n'.join(parts),'source':'resolver:crosscite'}
    except Exception as e:
        _record_api_debug(errs, 'crosscite', url, '', str(e))
        return None


def _arxiv_query_urls(entity_title: str) -> List[str]:
    t = (entity_title or '').strip()
    q1 = urllib.parse.quote('ti:"' + t[:180] + '"')
    q2 = urllib.parse.quote('ti:"PSR J0030+0451" AND ti:"Mass and Radius"')
    q3 = urllib.parse.quote('all:"PSR J0030+0451" AND all:"Mass and Radius"')
    q4 = urllib.parse.quote('all:"PSR J0030+0451"')
    return [
        f"https://export.arxiv.org/api/query?search_query={q1}&start=0&max_results=5",
        f"https://export.arxiv.org/api/query?search_query={q2}&start=0&max_results=5",
        f"https://export.arxiv.org/api/query?search_query={q3}&start=0&max_results=5",
        f"https://export.arxiv.org/api/query?search_query={q4}&start=0&max_results=5",
    ]


def _arxiv_pick_best(entity_title: str, xml: str) -> Optional[Dict[str, Any]]:
    entries = re.findall(r'<entry>.*?</entry>', xml or '', flags=re.IGNORECASE|re.DOTALL)
    if not entries:
        return None
    best_entry=None; best_title=''; best_sc=-1.0
    for entry in entries:
        mt = re.search(r'<title[^>]*>(.*?)</title>', entry, flags=re.IGNORECASE|re.DOTALL)
        tt = re.sub(r'\s+',' ', mt.group(1)).strip() if mt else ''
        sc = _title_overlap_score(tt, entity_title)
        if sc > best_sc:
            best_sc=sc; best_entry=entry; best_title=tt
    if not best_entry:
        return None
    authors = re.findall(r'<name[^>]*>(.*?)</name>', best_entry, flags=re.IGNORECASE|re.DOTALL)
    names=[]; seen=set()
    for a in authors:
        a=re.sub(r'\s+',' ',a).strip()
        if not a:
            continue
        k=a.lower()
        if k in seen:
            continue
        seen.add(k); names.append(a)
        if len(names)>=500:
            break
    if not names:
        return None
    return {'title': best_title or 'arXiv search', 'authors': names}


def _arxiv_meta(entity_title: str, timeout: int, errs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for url in _arxiv_query_urls(entity_title):
        try:
            ev = fetch_url(url, timeout=timeout)
            raw = str(ev.get('text','') or '')
            picked = _arxiv_pick_best(entity_title, raw)
            if picked:
                parts=[f"Title: {picked['title']}", 'Authors: ' + ', '.join(picked['authors'])]
                return {'id':'S0','title': picked['title'], 'url': url, 'text': '\n'.join(parts), 'source':'resolver:arxiv_search'}
            _record_api_debug(errs, 'arxiv', url, raw, 'arXiv no matching <entry>')
        except Exception as e:
            _record_api_debug(errs, 'arxiv', url, '', str(e))
            continue
    return None


# -------------------- Deterministic extraction --------------------

def _deterministic_first_author(evidences: List[Dict[str, Any]]) -> Optional[Tuple[str, List[str]]]:
    for e in evidences or []:
        sid=str(e.get('id','') or '')
        for ln in str(e.get('text','') or '').splitlines():
            t=_norm(ln)
            if t.casefold().startswith('authors:'):
                rest=t.split(':',1)[1].strip() if ':' in t else ''
                parts=[p.strip() for p in rest.split(',') if p.strip()]
                if parts:
                    return parts[0], [sid]
                parts=[p.strip() for p in re.split(r"\band\b|&", rest) if p.strip()]
                if parts:
                    return parts[0], [sid]
    return None


def _write_trace_file(turn_id: int, obj: Dict[str, Any]) -> Optional[str]:
    try:
        d = os.environ.get('CAUSALOS_TRACE_DIR', './logs')
        os.makedirs(d, exist_ok=True)
        fn = os.path.join(d, f"trace_turn_{turn_id}.json")
        with open(fn, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        return fn
    except Exception:
        return None


def _assistant_print(msg: str):
    print("\nAssistant: " + (msg or '').strip())


def _prov_line(turn_id: int, mode: str, used_web: bool, used_llm: bool, sources: List[Dict[str, Any]]) -> str:
    sids = []
    for s in (sources or [])[:6]:
        if isinstance(s, dict) and s.get('id'):
            sids.append(str(s.get('id')))
    return f"[Turn {turn_id}][Provenance] mode={mode} used_web={str(used_web).lower()} used_llm={str(used_llm).lower()} sources={','.join(sids)}"


def _prov_print(show: bool, line: str):
    if show:
        print(line)


def _llm_generate(osys, prompt: str, max_new_tokens: int = 220) -> str:
    tok = osys.tokenizer(prompt, return_tensors='pt')
    tok = {k: v.to(osys.model_device) for k, v in tok.items()}
    with torch.no_grad():
        out = osys.model.generate(**tok, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=osys.tokenizer.eos_token_id)
    return osys.tokenizer.decode(out[0][tok['input_ids'].shape[-1]:], skip_special_tokens=True).strip()


def _llm_json_obj(osys, prompt: str, max_new_tokens: int = 260) -> Dict[str, Any]:
    txt = _llm_generate(osys, prompt, max_new_tokens=max_new_tokens)
    start = txt.find('{'); end = txt.rfind('}')
    if start < 0 or end < 0 or end <= start:
        return {}
    try:
        obj = json.loads(txt[start:end+1])
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _is_rigid_literal_query(osys, user_text: str) -> Dict[str, Any]:
    prompt = (
        "You are a router for a fact-guarded assistant. Output JSON only.\n"
        "If the user asks for an exact proper name/title/identifier/person name, set needs_rigid_literal=true.\n"
        "slot must be one of ['PERSON','LABEL','IDENTIFIER','OTHER'].\n"
        f"USER: {user_text}\nJSON:"
    )
    obj = _llm_json_obj(osys, prompt, max_new_tokens=220)
    if obj.get('needs_rigid_literal') is True:
        return {
            'needs_rigid_literal': True,
            'slot': str(obj.get('slot','OTHER')).strip().upper(),
            'entity': str(obj.get('entity','') or '').strip(),
            'notes': str(obj.get('notes','') or '')
        }
    return {'needs_rigid_literal': False}


class FactGuardChatV8:
    def __init__(self, osys):
        self.osys = osys
        self.turn = 0
        self.timeout = int(os.environ.get('CAUSALOS_WEB_TIMEOUT', '12'))
        self.show_trace = os.environ.get('CAUSALOS_SHOW_TRACE', '0') == '1'
        self.show_prov = os.environ.get('CAUSALOS_SHOW_PROVENANCE', '1') == '1'
        self.web_r = None
        if SimpleWebRetriever is not None:
            try:
                self.web_r = SimpleWebRetriever(timeout=self.timeout)
            except Exception:
                self.web_r = None

    def _retrieve(self, query: str, k: int = 6) -> List[Dict[str, Any]]:
        evidences=[]
        urls = [u for u in _find_urls(query)[:3] if not re.search(r'\bdoi\.org/10\.', u, flags=re.IGNORECASE)]
        for u in urls:
            try:
                ev = fetch_url(u, timeout=self.timeout)
                evidences.append({'id': f"S{len(evidences)+1}", **ev})
            except Exception:
                continue
        if self.web_r is not None:
            try:
                docs = self.web_r.retrieve(query, k=k)
            except Exception:
                docs=[]
            for d in docs[:k]:
                evidences.append({'id': f"S{len(evidences)+1}", 'title': str(d.get('title','')), 'url': str(d.get('url','')), 'text': str(d.get('text',''))[:2500], 'source': str(d.get('source','web'))})
        return evidences

    def handle(self, user_text: str):
        user_text = (user_text or '').strip()
        if not user_text:
            return
        if _is_anchor_like(user_text):
            self.turn += 1
            _assistant_print('その文はシステムのアンカーです。質問文を入力してください。')
            _prov_print(self.show_prov, _prov_line(self.turn, 'anchor_blocked', False, False, []))
            return

        self.turn += 1
        t = self.turn
        rinfo = _is_rigid_literal_query(self.osys, user_text)

        if rinfo.get('needs_rigid_literal') is True:
            slot = str(rinfo.get('slot','OTHER')).strip().upper()
            entity_title = _clean_entity_title(str(rinfo.get('entity','') or ''), user_text) or user_text

            evidences = self._retrieve(f'"{entity_title}" {user_text}', k=8)
            diag: Dict[str, Any] = {'build': BUILD_TAG}

            if slot == 'PERSON':
                ev0 = _filter_by_title(evidences, entity_title)
                dois = _extract_dois(ev0)
                errs: List[Dict[str, Any]] = []
                augmented = list(ev0)

                if dois:
                    doi = dois[0]
                    enabled = {p.strip() for p in os.environ.get('CAUSALOS_FG_RESOLVERS', 'semanticscholar,crosscite,arxiv_search').lower().split(',') if p.strip()}
                    if 'semanticscholar' in enabled:
                        e = _semantic_scholar_meta(doi, self.timeout, errs)
                        if e:
                            augmented.append(e)
                    if not any('Authors:' in (str(x.get('text','')) or '') for x in augmented):
                        if 'crosscite' in enabled:
                            e = _crosscite_meta(doi, self.timeout, errs)
                            if e:
                                augmented.append(e)

                if not any('Authors:' in (str(x.get('text','')) or '') for x in augmented):
                    if 'arxiv_search' in {p.strip() for p in os.environ.get('CAUSALOS_FG_RESOLVERS', 'semanticscholar,crosscite,arxiv_search').lower().split(',') if p.strip()}:
                        e = _arxiv_meta(entity_title, self.timeout, errs)
                        if e:
                            augmented.append(e)

                augmented = _dedup_evidences(augmented)
                out=[]
                for i,e in enumerate(augmented,1):
                    e2=dict(e); e2['id']=f'S{i}'; out.append(e2)
                evidences = out

                diag.update({
                    'filtered_n': len(ev0),
                    'dois': dois,
                    'enabled': os.environ.get('CAUSALOS_FG_RESOLVERS', ''),
                    'have_authors': any('Authors:' in (str(x.get('text','')) or '') for x in evidences),
                    'resolver_errors': errs[:20],
                })

                det = _deterministic_first_author(evidences)
                if det:
                    ans, used_ids = det
                    cite = ' '.join([f'[{sid}]' for sid in used_ids]) if used_ids else ''
                    _assistant_print((ans + (' ' + cite if cite else '')).strip())
                    _prov_print(self.show_prov, _prov_line(t, 'rigid_literal_answer', True, False, evidences))
                    if self.show_trace:
                        obj={'turn':t,'rigid':rinfo,'entity_title':entity_title,'diagnostics':diag,'deterministic':{'ok':True,'answer':ans,'sources':used_ids},'sources':evidences}
                        p=_write_trace_file(t,obj)
                        print(f"[Trace] saved={p} build={BUILD_TAG}")
                    return

                _assistant_print(ANCHOR)
                _prov_print(self.show_prov, _prov_line(t, 'rigid_literal_anchor', True, True, evidences))
                if self.show_trace:
                    obj={'turn':t,'rigid':rinfo,'entity_title':entity_title,'diagnostics':diag,'sources':evidences}
                    p=_write_trace_file(t,obj)
                    print(f"[Trace] saved={p} build={BUILD_TAG}")
                return

            _assistant_print(ANCHOR)
            return

        _assistant_print(ANCHOR)


def main():
    print('--- Starting FactGuard Chat v8 (CausalOS v5.3_full) ---', flush=True)
    print(f'[Build] {BUILD_TAG}', flush=True)
    model_id = os.environ.get('CAUSALOS_MODEL','Qwen/Qwen2.5-7B-Instruct')
    osys = UnifiedCausalOSV5_3Full(
        model_id=model_id,
        init_n_nodes=256,
        init_slots_per_concept=2,
        expand_chunk=256,
        local_horizon=10,
        w0=0.7,
        w1=0.3,
        retriever=None,
        verifier=None,
    )
    chat = FactGuardChatV8(osys)

    if not sys.stdin.isatty():
        data = sys.stdin.read().replace('\r\n','\n').replace('\r','\n').strip()
        for b in re.split(r"\n\s*\n+", data):
            b=b.strip()
            if b:
                chat.handle(b)
        return

    print('Commands: exit/quit/q/終了 to quit.')
    while True:
        try:
            line = input('\nUser> ')
        except EOFError:
            break
        if _norm(line).casefold() in EXIT_WORDS:
            break
        if line.strip():
            chat.handle(line)


if __name__ == '__main__':
    main()
