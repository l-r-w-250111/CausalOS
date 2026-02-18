# -*- coding: utf-8 -*-
"""
CausalOS_v5_3_full.py (robustpack_v8 FULL)
- Contrast option scoring (task-agnostic, constant criterion): Sim(option, CF) - Sim(option, F)
- Query B trigger uses constant margin gate OR IDS: (margin < M_THR) OR (IDS >= IDS_THR)
- prior_mask wiring: A_eff_mask = clamp(A_mask + prior_mask)
- enforce restored: extract -> enforce -> dedup(inclusion) -> dedup(embedding) -> score(content-only)
- ADD-ONLY philosophy: do not delete; use inactive flags, disabled_prior flags
- No keyword-based semantic classification; everything uses fixed numeric criteria and constant schemas.
"""

from __future__ import annotations

import os
import re
import sys
import json
import math
import time
import copy
from dataclasses import dataclass
from collections import defaultdict, deque
from typing import Any, Dict, List, Tuple, Optional, Protocol, runtime_checkable

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

BUILD_ID = "2026-02-18-v5.3_full+robustpack_v8plus_v11r4(cf_anchor+opts_debug+label_fix)"

print("[System] Checking hardware...", flush=True)
try:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[System] Using CUDA: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        device = torch.device("cpu")
        print("[System] Using CPU", flush=True)
except Exception as e:
    device = torch.device("cpu")
    print(f"[System] Hardware check error: {e}, using CPU", flush=True)

__all__ = ["BUILD_ID", "device", "UnifiedCausalOSV5_3Full"]


# ==========================================================
# Utilities
# ==========================================================
def _now_ts() -> float:
    return time.time()

def _normalize_text(x: Any) -> str:
    s = "" if x is None else str(x)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _norm_label(x: Any) -> str:
    return _normalize_text(x).lower()

def _clip_mag(x: float) -> float:
    return float(np.clip(float(x), -0.99, 0.99))

def _safe_tanh_inv(y: float) -> float:
    y = float(np.clip(float(y), -0.99, 0.99))
    return float(np.arctanh(y))

def _cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    a = a.float().view(-1)
    b = b.float().view(-1)
    na = float(torch.norm(a).item())
    nb = float(torch.norm(b).item())
    if na < eps or nb < eps:
        return 0.0
    return float(torch.dot(a, b).item() / (na * nb + eps))

def _tokenize_lenient(s: str) -> List[str]:
    s = _normalize_text(s)
    if not s:
        return []
    return [t for t in re.split(r"\s+", s) if t][:256]

def _strip_options_block(text: str) -> str:
    t = _normalize_text(text)
    m = re.search(r'(\s|^)([A-Z])\s*:\s*', t)
    if m:
        return t[:m.start()].strip()
    return t

def _extract_first_json_array(text: str) -> Optional[str]:
    if not text:
        return None
    t = text
    if "```" in t:
        parts = t.split("```")
        if len(parts) >= 3:
            t = parts[1]
            t = re.sub(r"^\s*json\s*", "", t, flags=re.IGNORECASE)

    start = t.find("[")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(t)):
        c = t[i]
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return t[start:i + 1]
    return None

def _extract_first_json_obj(text: str) -> Optional[str]:
    if not text:
        return None
    t = text
    if "```" in t:
        parts = t.split("```")
        if len(parts) >= 3:
            t = parts[1]
            t = re.sub(r"^\s*json\s*", "", t, flags=re.IGNORECASE)

    start = t.find("{")
    if start < 0:
        return None

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(t)):
        c = t[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return t[start:i + 1]
    return None

def _validate_triplet(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    if "cause" not in obj or "effect" not in obj:
        return False
    if not isinstance(obj["cause"], str) or not isinstance(obj["effect"], str):
        return False
    if "magnitude" in obj:
        try:
            float(obj["magnitude"])
        except Exception:
            return False
    return True


# ==========================================================
# Placeholder / schema-leak guard
# ==========================================================
_PLACEHOLDER_PATTERNS = [
    r"^\.\.\.$",
    r"\bpos\|neg\b",
    r"\bcan\|must\|may\|unknown\b",
    r"\bcannot\|must\|may\|unknown\b",
]

def _is_placeholder_text(s: Any) -> bool:
    t = _normalize_text(s).lower()
    if not t:
        return True
    if t in {"...", "pos|neg", "can|must|may|unknown", "cannot|must|may|unknown"}:
        return True
    for p in _PLACEHOLDER_PATTERNS:
        if re.search(p, t):
            return True
    return False

def _is_bad_label(lab: str) -> bool:
    lab = _norm_label(lab)
    if not lab:
        return True
    if lab in {"a", "b", "c", "d", "e", "f"}:
        return True
    if len(lab) <= 1:
        return True
    if os.environ.get("CAUSALOS_PLACEHOLDER_GUARD", "1") == "1" and _is_placeholder_text(lab):
        return True
    return False


def _frame_head(frame: Dict[str, Any], max_entities: int = 8, max_events: int = 8, max_states: int = 10) -> Dict[str, Any]:
    ents = frame.get("entities", []) if isinstance(frame.get("entities", []), list) else []
    evs = frame.get("events", []) if isinstance(frame.get("events", []), list) else []
    sts = frame.get("states", []) if isinstance(frame.get("states", []), list) else []
    cons = frame.get("constraints", []) if isinstance(frame.get("constraints", []), list) else []

    def _act(d: Dict[str, Any]) -> bool:
        if os.environ.get("CAUSALOS_IGNORE_INACTIVE", "1") == "1":
            return not bool(d.get("inactive", False))
        return True

    ents = [str(x) for x in ents[:max_entities]]

    evs2 = []
    for e in evs[:max_events]:
        if isinstance(e, dict) and _act(e):
            evs2.append({
                "predicate": e.get("predicate", ""),
                "order": e.get("order", 0),
                "polarity": e.get("polarity", ""),
                "modality": e.get("modality", ""),
                "args": (e.get("args", [])[:3] if isinstance(e.get("args", []), list) else [])
            })

    sts2 = []
    for s in sts[:max_states]:
        if isinstance(s, dict) and _act(s):
            sts2.append({
                "var": s.get("var", ""),
                "subject": s.get("subject", ""),
                "value": s.get("value", ""),
                "polarity": s.get("polarity", ""),
                "modality": s.get("modality", ""),
            })

    cons2 = []
    for c in cons[:6]:
        if isinstance(c, dict):
            cons2.append({"type": c.get("type", ""), "statement": c.get("statement", "")})

    return {"entities": ents, "events": evs2, "states": sts2, "constraints": cons2, "notes": frame.get("notes", "")}


# ==========================================================
# Answer protocol
# ==========================================================
@dataclass
class AnswerPacket:
    best_effort_answer: str
    confidence: float
    need_info_questions: List[str]
    reason_trace: Dict[str, Any]
    mode: str


# ==========================================================
# Universal skeleton: pluggable tools
# ==========================================================
@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]: ...

@runtime_checkable
class Verifier(Protocol):
    def verify(self, claims: List[str]) -> Dict[str, Any]: ...

class NullRetriever:
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        return []

class NullVerifier:
    def verify(self, claims: List[str]) -> Dict[str, Any]:
        return {"verified": [], "unverified": claims, "notes": "null_verifier"}


# ==========================================================
# Knowledge Policy (fact-mode disabled by default)
# ==========================================================
def _is_exact_fact_task(text: str) -> bool:
    t = (text or "").lower()
    keys = ["doi", "arxiv", "url", "paper title", "論文名", "著者", "isbn", "issn", "citation", "reference"]
    return any(k in t for k in keys)

def _contains_fact_like_patterns(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    pats = [
        r"\bdoi\b", r"\barxiv\b", r"\bhttp[s]?://", r"\bwww\.", r"\bisbn\b", r"\bissn\b",
        r"\b\d{4}\b", r"\bvol\.?\b", r"\bno\.?\b", r"\bpp\.?\b",
    ]
    return any(re.search(p, t) for p in pats)

class KnowledgePolicy:
    def __init__(self, beta_prior: float = 0.25):
        self.beta_prior = float(beta_prior)

    def choose_mode(self, user_text: str, anomaly_score: float = 0.0) -> str:
        if os.environ.get("CAUSALOS_ENABLE_FACT_MODE", "0") == "1":
            if _is_exact_fact_task(user_text) or _contains_fact_like_patterns(user_text):
                return "VERIFY_REQUIRED"
        if anomaly_score >= 1.0:
            return "CAUSAL_ONLY"
        return "OPEN"


# ==========================================================
# ConceptBank (namespace protection)
# ==========================================================
class ConceptBank:
    def __init__(self, osys: "UnifiedCausalOSV5_3Full",
                 init_slots_per_concept: int = 2,
                 sim_base_threshold: float = 0.82,
                 expand_chunk: int = 256):
        self.osys = osys
        self.init_slots_per_concept = int(init_slots_per_concept)
        self.sim_base_threshold = float(sim_base_threshold)
        self.expand_chunk = int(expand_chunk)

        self.concepts: Dict[int, Dict[str, Any]] = {}
        self.alias_to_cid: Dict[str, int] = {}
        self._cid_counter = 0
        self._recent_sims: deque = deque(maxlen=256)

    @staticmethod
    def _is_protected(lab: str) -> bool:
        return lab.startswith("state::") or lab.startswith("event::") or lab.startswith("question::")

    def _new_cid(self) -> int:
        cid = self._cid_counter
        self._cid_counter += 1
        return cid

    def _dynamic_threshold(self) -> float:
        if len(self._recent_sims) < 32:
            return self.sim_base_threshold
        arr = np.array(list(self._recent_sims), dtype=np.float32)
        mu = float(arr.mean())
        sd = float(arr.std() + 1e-6)
        thr = float(np.clip(mu + 0.5 * sd, 0.70, 0.92))
        return max(self.sim_base_threshold, thr)

    def _embed_label(self, label: Any) -> torch.Tensor:
        label = _normalize_text(label)
        if not label:
            return torch.zeros(1, dtype=torch.float32)
        tok = self.osys.tokenizer(str(label), return_tensors="pt", add_special_tokens=False)
        ids = tok["input_ids"].to(self.osys.model_device)
        with torch.no_grad():
            emb = self.osys.model.get_input_embeddings()(ids)[0]
            v = emb.mean(dim=0).float().detach().cpu()
        return v

    def _alloc_slots(self, k: int) -> List[int]:
        return [self.osys._alloc_node() for _ in range(int(k))]

    def resolve(self, label: Any) -> int:
        lab = _norm_label(label)
        if _is_bad_label(lab):
            lab = f"concept_{hash(str(label)) % 100000}"

        if lab in self.alias_to_cid:
            return self.alias_to_cid[lab]

        if self._is_protected(lab):
            cid = self._new_cid()
            slots = self._alloc_slots(self.init_slots_per_concept)
            self.concepts[cid] = {"cid": cid, "emb": self._embed_label(lab).float(), "aliases": set([lab]), "slots": slots, "usage": 0}
            self.alias_to_cid[lab] = cid
            return cid

        v = self._embed_label(lab)
        best_cid = None
        best_sim = -1.0
        for cid, c in self.concepts.items():
            sim = _cosine(v, c["emb"])
            if sim > best_sim:
                best_sim = sim
                best_cid = cid

        self._recent_sims.append(best_sim if best_sim >= 0 else 0.0)
        thr = self._dynamic_threshold()

        if best_cid is not None and best_sim >= thr:
            self.alias_to_cid[lab] = best_cid
            c = self.concepts[best_cid]
            c["emb"] = (0.9 * c["emb"] + 0.1 * v).float()
            c["aliases"].add(lab)
            return best_cid

        cid = self._new_cid()
        slots = self._alloc_slots(self.init_slots_per_concept)
        self.concepts[cid] = {"cid": cid, "emb": v.float(), "aliases": set([lab]), "slots": slots, "usage": 0}
        self.alias_to_cid[lab] = cid
        return cid

    def rep_slot(self, cid: int) -> int:
        slots = self.concepts[cid]["slots"]
        return int(slots[0]) if slots else 0


# ==========================================================
# VarNormalizer
# ==========================================================
class VarNormalizer:
    def __init__(self, osys: "UnifiedCausalOSV5_3Full", base_threshold: float = 0.84):
        self.osys = osys
        self.base_threshold = float(base_threshold)
        self._canon: Dict[str, torch.Tensor] = {}
        self._stats: Dict[str, int] = defaultdict(int)
        self._recent: deque = deque(maxlen=256)

    def _embed(self, s: str) -> torch.Tensor:
        return self.osys.concepts._embed_label(s)

    def _dyn_thr(self) -> float:
        if len(self._recent) < 32:
            return self.base_threshold
        arr = np.array(list(self._recent), dtype=np.float32)
        mu = float(arr.mean())
        sd = float(arr.std() + 1e-6)
        thr = float(np.clip(mu + 0.35 * sd, 0.75, 0.93))
        return max(self.base_threshold, thr)

    def canonicalize(self, var: str) -> str:
        var = _normalize_text(var)
        if not var:
            return var
        key = var.lower()
        v = self._embed(key)

        best = None
        best_sim = -1.0
        for canon, emb in self._canon.items():
            sim = _cosine(v, emb)
            if sim > best_sim:
                best_sim = sim
                best = canon

        self._recent.append(best_sim if best_sim >= 0 else 0.0)
        thr = self._dyn_thr()

        if best is not None and best_sim >= thr:
            self._canon[best] = (0.9 * self._canon[best] + 0.1 * v).float()
            self._stats[best] += 1
            return best

        self._canon[key] = v.float()
        self._stats[key] += 1
        return key

    def snapshot(self, max_items: int = 12) -> Dict[str, int]:
        items = sorted(self._stats.items(), key=lambda kv: kv[1], reverse=True)
        return {k: int(v) for k, v in items[:max_items]}


# ==========================================================
# GroundingChecker (content-only)
# ==========================================================
class GroundingChecker:
    def __init__(self, osys: "UnifiedCausalOSV5_3Full"):
        self.osys = osys
        self._emb_cache: Dict[str, torch.Tensor] = {}

    def _embed_text(self, text: str) -> torch.Tensor:
        key = text[:3000]
        if key in self._emb_cache:
            return self._emb_cache[key]
        tok = self.osys.tokenizer(str(key), return_tensors="pt", add_special_tokens=False)
        ids = tok["input_ids"].to(self.osys.model_device)
        with torch.no_grad():
            v = self.osys.model.get_input_embeddings()(ids)[0].mean(dim=0).float().detach()
        self._emb_cache[key] = v
        return v

    @staticmethod
    def _tokenize_mixed(s: str) -> List[str]:
        s = _norm_label(s)
        toks = re.split(r"[^a-z0-9]+", s)
        toks = [t for t in toks if len(t) >= 2]
        return toks[:64]

    @staticmethod
    def _char_bigrams(s: str) -> List[str]:
        s = _norm_label(s)
        s = re.sub(r"\s+", "", s)
        if len(s) < 2:
            return [s] if s else []
        return [s[i:i + 2] for i in range(min(len(s) - 1, 64))]

    @staticmethod
    def overlap_score(a: str, b: str) -> float:
        ta = GroundingChecker._tokenize_mixed(a)
        tb = GroundingChecker._tokenize_mixed(b)
        if ta and tb:
            sa, sb = set(ta), set(tb)
            return float(len(sa & sb) / max(1, len(sa | sb)))
        ba = set(GroundingChecker._char_bigrams(a))
        bb = set(GroundingChecker._char_bigrams(b))
        if not ba or not bb:
            return 0.0
        return float(len(ba & bb) / max(1, len(ba | bb)))

    def score_item(self, item: str, source: str) -> float:
        item_n = _norm_label(item)
        src_n = _norm_label(source)
        if not item_n:
            return 0.0
        if os.environ.get("CAUSALOS_PLACEHOLDER_GUARD", "1") == "1" and _is_placeholder_text(item_n):
            return 0.0
        if item_n in src_n:
            return 1.0

        ov = self.overlap_score(item_n, src_n) if os.environ.get("CAUSALOS_GROUND_TOKEN_OVERLAP", "1") == "1" else 0.0
        vi = self._embed_text(item_n)
        vs = self._embed_text(src_n)
        emb = float(np.clip(_cosine(vi, vs), 0.0, 1.0))
        return float(np.clip(0.55 * emb + 0.45 * ov, 0.0, 1.0))

    def score_frame(self, frame: Dict[str, Any], source: str) -> Dict[str, float]:
        def _act(d: Dict[str, Any]) -> bool:
            if os.environ.get("CAUSALOS_IGNORE_INACTIVE", "1") == "1":
                return not bool(d.get("inactive", False))
            return True

        items_full: List[str] = []
        items_content: List[str] = []

        for e in (frame.get("events", []) or []):
            if isinstance(e, dict) and _act(e):
                pred = str(e.get("predicate", ""))
                items_full.append(pred)
                items_content.append(pred)
                for a in (e.get("args", []) or []):
                    if isinstance(a, dict):
                        items_full.append(str(a.get("role", "")))
                        items_full.append(str(a.get("value", "")))
                        items_content.append(str(a.get("value", "")))

        for s in (frame.get("states", []) or []):
            if isinstance(s, dict) and _act(s):
                items_full.append(str(s.get("var", "")))
                items_full.append(str(s.get("subject", "")))
                items_full.append(str(s.get("value", "")))
                items_content.append(str(s.get("subject", "")))
                items_content.append(str(s.get("value", "")))

        for ent in (frame.get("entities", []) or []):
            items_full.append(str(ent))
            items_content.append(str(ent))

        items_full = [x for x in items_full if _normalize_text(x)]
        items_content = [x for x in items_content if _normalize_text(x)]

        if not items_full:
            return {"avg": 0.0, "min": 0.0, "n": 0, "avg_full": 0.0, "min_full": 0.0, "n_full": 0,
                    "avg_content": 0.0, "min_content": 0.0, "n_content": 0}

        scores_full = [self.score_item(it, source) for it in items_full]
        avg_full = float(np.mean(scores_full))
        min_full = float(np.min(scores_full))
        n_full = int(len(scores_full))

        if not items_content:
            avg_c = 0.0
            min_c = 0.0
            n_c = 0
        else:
            scores_c = [self.score_item(it, source) for it in items_content]
            avg_c = float(np.mean(scores_c))
            min_c = float(np.min(scores_c))
            n_c = int(len(scores_c))

        use_content = os.environ.get("CAUSALOS_GROUND_CONTENT_ONLY", "1") == "1"
        if use_content:
            return {"avg": avg_c, "min": min_c, "n": n_c,
                    "avg_full": avg_full, "min_full": min_full, "n_full": n_full,
                    "avg_content": avg_c, "min_content": min_c, "n_content": n_c}
        return {"avg": avg_full, "min": min_full, "n": n_full,
                "avg_full": avg_full, "min_full": min_full, "n_full": n_full,
                "avg_content": avg_c, "min_content": min_c, "n_content": n_c}


# ==========================================================
# EdgeBank
# ==========================================================
class EdgeBank:
    def __init__(self):
        self.strong: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.prior: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.prior_meta: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.disabled_prior: set = set()

    def _update(self, store: Dict, e: int, c: int, m: float, w: float, source: str):
        m = _clip_mag(m)
        w = float(max(0.0, w))
        key = (e, c)
        rec = store.get(key)
        if rec is None:
            store[key] = {"m": float(m), "w": float(w), "src": defaultdict(float)}
            store[key]["src"][source] += w
        else:
            m_old = float(rec["m"])
            w_old = float(rec["w"])
            rec["m"] = float((m_old * w_old + m * w) / max(w_old + w, 1e-6))
            rec["w"] = float(w_old + w)
            rec["src"][source] += w

    def update_edge(self, effect_cid: int, cause_cid: int, m: float, w: float,
                    source: str = "user", layer: str = "strong", meta: Optional[Dict[str, Any]] = None):
        if layer == "strong":
            self._update(self.strong, effect_cid, cause_cid, m, w, source)
        else:
            self._update(self.prior, effect_cid, cause_cid, m, w, source)
            if meta is not None:
                self.prior_meta[(effect_cid, cause_cid)] = dict(meta)

    def disable_prior_edge(self, effect_cid: int, cause_cid: int):
        self.disabled_prior.add((effect_cid, cause_cid))


# ==========================================================
# CausalCoreV5 (prior_mask supported)
# ==========================================================
class CausalCoreV5(nn.Module):
    def __init__(self, n_nodes: int = 256, p_r0: float = 0.20):
        super().__init__()
        self.n_nodes = int(n_nodes)

        self.x = nn.Parameter(torch.randn(self.n_nodes, 2, device=device) * 0.02)
        self.raw_S = nn.Parameter(torch.zeros(self.n_nodes, self.n_nodes, device=device))
        self.raw_phase = nn.Parameter(torch.zeros(self.n_nodes, self.n_nodes, device=device))

        p = float(np.clip(p_r0, 0.01, 0.99))
        init_logit = math.log(p / (1 - p))
        self.raw_r = nn.Parameter(torch.full((self.n_nodes, self.n_nodes), init_logit, device=device))

        self.register_buffer("A_mask", torch.zeros(self.n_nodes, self.n_nodes, device=device))
        self.register_buffer("G_gate", torch.ones(self.n_nodes, self.n_nodes, device=device))
        with torch.no_grad():
            self.A_mask.fill_(0.0)
            self.A_mask.diagonal().fill_(1.0)

        self.register_buffer("omega", torch.tensor(0.1, device=device))

        self.do_values: Dict[int, torch.Tensor] = {}
        self.do_cut_in: set = set()

    def resize(self, new_n: int, p_r0: float = 0.20):
        new_n = int(new_n)
        if new_n <= self.n_nodes:
            return

        p = float(np.clip(p_r0, 0.01, 0.99))
        init_logit = math.log(p / (1 - p))

        def expand_square(old: torch.Tensor, fill: float) -> torch.Tensor:
            new = torch.full((new_n, new_n), fill, device=old.device, dtype=old.dtype)
            new[:self.n_nodes, :self.n_nodes] = old
            return new

        with torch.no_grad():
            oldx = self.x.data
            newx = torch.zeros(new_n, 2, device=oldx.device, dtype=oldx.dtype)
            newx[:self.n_nodes] = oldx
            newx[self.n_nodes:] = torch.randn(new_n - self.n_nodes, 2, device=oldx.device) * 0.02
        self.x = nn.Parameter(newx)

        self.raw_S = nn.Parameter(expand_square(self.raw_S.data, 0.0))
        self.raw_phase = nn.Parameter(expand_square(self.raw_phase.data, 0.0))
        self.raw_r = nn.Parameter(expand_square(self.raw_r.data, init_logit))

        oldA = self.A_mask
        oldG = self.G_gate
        newA = torch.zeros(new_n, new_n, device=oldA.device, dtype=oldA.dtype)
        newG = torch.ones(new_n, new_n, device=oldG.device, dtype=oldG.dtype)
        newA[:self.n_nodes, :self.n_nodes] = oldA
        newG[:self.n_nodes, :self.n_nodes] = oldG
        newA.diagonal().fill_(1.0)

        self.A_mask = newA
        self.G_gate = newG
        self.n_nodes = new_n

    def reset_do(self):
        self.do_values = {}
        self.do_cut_in = set()

    def apply_do_cut_in(self, node_idx: int):
        self.do_cut_in.add(int(node_idx))

    def apply_do_value(self, node_idx: int, v_real: float, v_imag: float = 0.0):
        self.do_values[int(node_idx)] = torch.tensor([float(v_real), float(v_imag)], device=device)

    def get_S_eff(self, beta: float = 0.0, S_prior: Optional[torch.Tensor] = None,
                  prior_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        S = torch.tanh(self.raw_S)
        if S_prior is not None and beta > 0.0:
            S = torch.clamp(S + beta * S_prior, -0.99, 0.99)
        r = torch.sigmoid(self.raw_r)

        Aeff = self.A_mask
        if prior_mask is not None:
            Aeff = torch.clamp(Aeff + prior_mask, 0.0, 1.0)

        Aamp = Aeff * self.G_gate * S * r

        if self.do_cut_in:
            Aamp = Aamp.clone()
            for j in self.do_cut_in:
                if 0 <= j < self.n_nodes:
                    Aamp[j, :].fill_(0.0)
        return Aamp

    def step(self, x: torch.Tensor, t: int, beta: float = 0.0,
             S_prior: Optional[torch.Tensor] = None,
             prior_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        n = self.n_nodes
        x_real = x[:, 0].view(1, n)
        x_imag = x[:, 1].view(1, n)

        Aamp = self.get_S_eff(beta=beta, S_prior=S_prior, prior_mask=prior_mask)
        theta = self.raw_phase + self.omega * float(t)
        cosT = torch.cos(theta)
        sinT = torch.sin(theta)

        out_real = torch.matmul(Aamp * cosT, x_real.t()).view(n) - torch.matmul(Aamp * sinT, x_imag.t()).view(n)
        out_imag = torch.matmul(Aamp * sinT, x_real.t()).view(n) + torch.matmul(Aamp * cosT, x_imag.t()).view(n)

        x_next = torch.stack([torch.tanh(out_real), torch.tanh(out_imag)], dim=-1)

        if self.do_values:
            for idx, v in self.do_values.items():
                if 0 <= idx < n:
                    x_next[idx] = v
        return x_next

    def rollout(self, steps: int, x0: Optional[torch.Tensor] = None,
                beta: float = 0.0, S_prior: Optional[torch.Tensor] = None,
                prior_mask: Optional[torch.Tensor] = None,
                require_grad: bool = False) -> torch.Tensor:
        if x0 is None:
            x = self.x if require_grad else self.x.detach()
        else:
            x = x0 if require_grad else x0.detach()

        traj = [x]
        for t in range(int(steps)):
            x = self.step(x, t=t, beta=beta, S_prior=S_prior, prior_mask=prior_mask)
            traj.append(x)
        return torch.stack(traj, dim=0)


# ==========================================================
# WorkspaceGate
# ==========================================================
class WorkspaceGate:
    def __init__(self, core: CausalCoreV5):
        self.core = core
        self._saved_A = None
        self._saved_G = None

    def __enter__(self):
        self._saved_A = self.core.A_mask.clone()
        self._saved_G = self.core.G_gate.clone()
        return self

    def activate_nodes(self, active: List[int]):
        n = self.core.n_nodes
        active_set = set([int(a) for a in active if 0 <= int(a) < n])
        A_prev = self._saved_A
        with torch.no_grad():
            self.core.A_mask.fill_(0.0)
            self.core.A_mask.diagonal().fill_(1.0)
            self.core.G_gate.fill_(1.0)
            for j in active_set:
                for i in active_set:
                    if i == j:
                        continue
                    if float(A_prev[j, i].item()) > 0.5:
                        self.core.A_mask[j, i] = 1.0

    def __exit__(self, exc_type, exc, tb):
        if self._saved_A is not None:
            with torch.no_grad():
                self.core.A_mask.copy_(self._saved_A)
                self.core.G_gate.copy_(self._saved_G)
        return False


# ==========================================================
# OmegaLocalizer (prior_mask passed through)
# ==========================================================
class OmegaLocalizer:
    def __init__(self, horizon: int = 10, w0: float = 0.7, w1: float = 0.3,
                 alpha: float = 0.45, beta: float = 0.25, gamma: float = 0.30,
                 topk_edges: int = 250, hop: int = 2):
        self.horizon = int(horizon)
        self.w0 = float(w0)
        self.w1 = float(w1)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.topk_edges = int(topk_edges)
        self.hop = int(hop)

    @staticmethod
    def _edge_list_from_topk(score_mat: torch.Tensor, k: int) -> List[Tuple[int, int, float]]:
        n = score_mat.shape[0]
        flat = score_mat.view(-1)
        k = min(int(k), flat.numel())
        vals, idx = torch.topk(flat, k=k)
        edges = []
        for v, idv in zip(vals.tolist(), idx.tolist()):
            j = idv // n
            i = idv % n
            edges.append((j, i, float(v)))
        return edges

    @staticmethod
    def _build_adj_from_mat(mat: torch.Tensor, eps: float = 1e-4) -> List[List[int]]:
        n = mat.shape[0]
        adj = [[] for _ in range(n)]
        mm = mat.detach().abs()
        nz = torch.nonzero(mm > eps, as_tuple=False)
        for j, i in nz.tolist():
            adj[i].append(j)
        return adj

    @staticmethod
    def _reachability_edge_scores(S_eff: torch.Tensor, Q: List[int], T: List[int], eps: float = 1e-4) -> torch.Tensor:
        n = S_eff.shape[0]
        adj = OmegaLocalizer._build_adj_from_mat(S_eff, eps=eps)
        radj = [[] for _ in range(n)]
        for i in range(n):
            for j in adj[i]:
                radj[j].append(i)

        def bfs(starts: List[int], graph: List[List[int]]) -> List[bool]:
            vis = [False] * n
            dq = deque()
            for s in starts:
                if 0 <= s < n and not vis[s]:
                    vis[s] = True
                    dq.append(s)
            while dq:
                u = dq.popleft()
                for v in graph[u]:
                    if not vis[v]:
                        vis[v] = True
                        dq.append(v)
            return vis

        Rfwd = bfs(Q, adj)
        Rrev = bfs(T, radj)

        score = torch.zeros_like(S_eff)
        absS = S_eff.detach().abs()
        nz = torch.nonzero(absS > eps, as_tuple=False)
        for j, i in nz.tolist():
            if Rfwd[i] and Rrev[j]:
                score[j, i] = absS[j, i]
        return score

    def localize(self, core: CausalCoreV5, S_prior: Optional[torch.Tensor],
                 Q: List[int], T: List[int], beta_prior: float = 0.0,
                 prior_mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        n = core.n_nodes
        core.zero_grad(set_to_none=True)

        traj = core.rollout(steps=self.horizon, x0=core.x, beta=beta_prior,
                            S_prior=S_prior, prior_mask=prior_mask, require_grad=True)
        xT = traj[-1]

        loss = torch.tensor(0.0, device=device)
        for tidx in T:
            if 0 <= tidx < n:
                v = xT[tidx]
                loss = loss + self.w0 * v[0] + self.w1 * torch.norm(v, p=2)
        if float(loss.detach().item()) == 0.0:
            loss = self.w1 * torch.norm(xT, p=2)
        loss.backward()

        grad_rawS = core.raw_S.grad
        grad_score = grad_rawS.detach().abs() if grad_rawS is not None else torch.zeros(n, n, device=device)

        S_eff = core.get_S_eff(beta=beta_prior, S_prior=S_prior, prior_mask=prior_mask)
        src = torch.norm(xT.detach(), dim=-1)
        contrib = S_eff.detach().abs() * src.view(1, n)

        edges_top = self._edge_list_from_topk(contrib, k=self.topk_edges)
        seed_nodes = set()
        for j, i, _ in edges_top:
            seed_nodes.add(i); seed_nodes.add(j)

        eps = 1e-4
        und = [[] for _ in range(n)]
        absS = S_eff.detach().abs()
        nz = torch.nonzero(absS > eps, as_tuple=False)
        for j, i in nz.tolist():
            und[i].append(j); und[j].append(i)

        OmegaA_nodes = set(seed_nodes)
        frontier = set(seed_nodes)
        for _ in range(max(1, self.hop)):
            new_front = set()
            for u in frontier:
                for v in und[u]:
                    if v not in OmegaA_nodes:
                        OmegaA_nodes.add(v)
                        new_front.add(v)
            frontier = new_front
            if not frontier:
                break

        maskA = torch.zeros(n, n, device=device)
        for j in OmegaA_nodes:
            maskA[j, :] = 1.0

        reach = self._reachability_edge_scores(S_eff * maskA, Q=Q, T=T, eps=eps)
        grad_in = grad_score * maskA

        def norm01(x: torch.Tensor) -> torch.Tensor:
            mx = float(x.max().item())
            if mx <= 1e-8:
                return torch.zeros_like(x)
            return x / mx

        cN = norm01(contrib) * maskA
        rN = norm01(reach) * maskA
        gN = norm01(grad_in) * maskA
        combined = self.alpha * cN + self.beta * rN + self.gamma * gN

        Omega_edges = self._edge_list_from_topk(combined, k=max(50, self.topk_edges // 2))
        return {"Omega_edges": Omega_edges, "traj": traj.detach(), "OmegaA_nodes": list(sorted(OmegaA_nodes))}


# ==========================================================
# ImpossibilityController
# ==========================================================
class ImpossibilityController:
    def __init__(self, kappa: float = 10.0, tau: float = 0.65,
                 div_window: int = 6, rho_beta: float = 6.0):
        self.kappa = float(kappa)
        self.tau = float(tau)
        self.div_window = int(div_window)
        self.rho_beta = float(rho_beta)

    def _sigmoid(self, x: float) -> float:
        return float(1.0 / (1.0 + math.exp(-float(x))))

    def local_divergence(self, traj: torch.Tensor) -> float:
        T = traj.shape[0]
        w = min(self.div_window, T - 1)
        if w <= 1:
            return 0.0
        E = torch.norm(traj[-w:].reshape(w, -1), dim=-1)
        E0 = float(E[0].item())
        E1 = float(E[-1].item())
        if not np.isfinite(E0) or not np.isfinite(E1):
            return 1.0
        rel = (E1 - E0) / max(abs(E0), 1e-6)
        return float(np.clip(rel / 1.0, 0.0, 1.0))

    def local_spectral_risk(self, S_eff: torch.Tensor, Omega_nodes: List[int]) -> float:
        if not Omega_nodes:
            return 0.0
        idx = torch.tensor(Omega_nodes, device=S_eff.device, dtype=torch.long)
        sub = S_eff.detach().abs()[idx][:, idx]
        if sub.numel() == 0 or sub.shape[0] < 2:
            return 0.0
        try:
            vals = torch.linalg.eigvals(sub).abs()
            rho = float(torch.max(vals).item())
        except Exception:
            v = torch.randn(sub.shape[0], 1, device=sub.device)
            for _ in range(10):
                v = sub @ v
                v = v / (torch.norm(v) + 1e-8)
            rho = float(torch.norm(sub @ v).item())
        return float(np.clip(self._sigmoid(self.rho_beta * (rho - 1.0)), 0.0, 1.0))

    def constraint_violation(self, traj: torch.Tensor) -> float:
        if torch.isnan(traj).any() or torch.isinf(traj).any():
            return 1.0
        x = traj[-1]
        sat = float((x.abs() > 0.995).float().mean().item())
        return float(np.clip(sat, 0.0, 1.0))

    def combine_u(self, u_div: float, u_rho: float, u_c: float) -> float:
        u = 1.0 - (1.0 - u_div) * (1.0 - u_rho) * (1.0 - u_c)
        return float(np.clip(u, 0.0, 1.0))


# ==========================================================
# CausalTripletExtractor
# ==========================================================
class CausalTripletExtractor:
    def __init__(self, osys: "UnifiedCausalOSV5_3Full"):
        self.osys = osys

    def extract(self, text: str, max_triplets: int = 12) -> List[Dict[str, Any]]:
        text = _normalize_text(text)
        if not text:
            return []
        if os.environ.get("CAUSALOS_NO_LLM_GRAPH", "0") == "1":
            return []

        prompt = f"""Analyze causal relationships in the text.
Return ONLY a JSON array (<= {max_triplets} items) of objects:
{{"cause":"...","effect":"...","magnitude":0.7}}
Rules:
- Do NOT use option labels A/B/C/D.
- Do NOT output placeholder strings like "..." or "pos|neg".
Text: "{text}"
JSON:"""
        tok = self.osys.tokenizer(str(prompt), return_tensors="pt")
        tok = {k: v.to(self.osys.model_device) for k, v in tok.items()}
        with torch.no_grad():
            out = self.osys.model.generate(**tok, max_new_tokens=260, do_sample=False,
                                           pad_token_id=self.osys.tokenizer.eos_token_id)
        resp = self.osys.tokenizer.decode(out[0][tok["input_ids"].shape[-1]:], skip_special_tokens=True)
        arr = _extract_first_json_array(resp)
        if not arr:
            return []
        try:
            data = json.loads(arr)
            if not isinstance(data, list):
                return []
        except Exception:
            return []

        clean = []
        for obj in data:
            if not _validate_triplet(obj):
                continue
            c = _norm_label(obj.get("cause", ""))
            e = _norm_label(obj.get("effect", ""))
            if _is_bad_label(c) or _is_bad_label(e):
                continue
            m = float(obj.get("magnitude", 0.5))
            clean.append({"cause": c, "effect": e, "magnitude": _clip_mag(m)})
            if len(clean) >= max_triplets:
                break
        return clean


# ==========================================================
# FrameExtractorLLM
# ==========================================================
class FrameExtractorLLM:
    def __init__(self, osys: "UnifiedCausalOSV5_3Full"):
        self.osys = osys

    def _pick_varnorm(self, kind: str) -> VarNormalizer:
        return self.osys.varnorm_opt if kind == "option" else self.osys.varnorm_main

    def _generate_raw(self, prompt: str, max_new_tokens: int = 420) -> str:
        tok = self.osys.tokenizer(prompt, return_tensors="pt")
        tok = {k: v.to(self.osys.model_device) for k, v in tok.items()}
        with torch.no_grad():
            out = self.osys.model.generate(**tok, max_new_tokens=max_new_tokens, do_sample=False,
                                           pad_token_id=self.osys.tokenizer.eos_token_id)
        resp = self.osys.tokenizer.decode(out[0][tok["input_ids"].shape[-1]:], skip_special_tokens=True)
        if os.environ.get("CAUSALOS_DEBUG_FRAME_RAW", "0") == "1":
            head = resp[:260].replace("\n", "\\n")
            print(f"[DBG][FRAME_RAW] head={head}", file=sys.stderr, flush=True)
        return resp

    def _generate(self, prompt: str) -> Dict[str, Any]:
        resp = self._generate_raw(prompt, max_new_tokens=420)
        js = _extract_first_json_obj(resp)
        if not js:
            return {}
        try:
            return json.loads(js)
        except Exception:
            return {}

    @staticmethod
    def _schema_typed() -> str:
        return """{
  "entities": ["string"],
  "events": [{"predicate":"string", "args":[{"role":"string","value":"string"}], "order":0, "polarity":"pos|neg", "modality":"string"}],
  "states": [{"var":"string", "subject":"string", "value":"string", "polarity":"pos|neg", "modality":"string"}],
  "constraints": [{"type":"cannot|must|may|unknown","statement":"string"}],
  "notes":"string"
}"""

    @staticmethod
    def _fix_polarity(pol: Any) -> str:
        p = _norm_label(pol)
        if p in {"pos", "positive", "+"}:
            return "pos"
        if p in {"neg", "negative", "-"}:
            return "neg"
        return "pos"

    @staticmethod
    def _fix_modality(mod: Any) -> str:
        m = _normalize_text(mod)
        if not m:
            return "unknown"
        if os.environ.get("CAUSALOS_PLACEHOLDER_GUARD", "1") == "1" and _is_placeholder_text(m):
            return "unknown"
        return m

    def _sanitize(self, obj: Dict[str, Any], text_fallback: str, kind: str) -> Dict[str, Any]:
        vn = self._pick_varnorm(kind)
        obj = obj if isinstance(obj, dict) else {}
        obj["entities"] = obj.get("entities") if isinstance(obj.get("entities"), list) else []
        obj["events"] = obj.get("events") if isinstance(obj.get("events"), list) else []
        obj["states"] = obj.get("states") if isinstance(obj.get("states"), list) else []
        obj["constraints"] = obj.get("constraints") if isinstance(obj.get("constraints"), list) else []
        obj["notes"] = str(obj.get("notes", ""))

        ents = []
        for ent in obj["entities"]:
            s = _normalize_text(ent)
            if not s:
                continue
            if os.environ.get("CAUSALOS_PLACEHOLDER_GUARD", "1") == "1" and _is_placeholder_text(s):
                continue
            ents.append(s)
        obj["entities"] = ents

        evs = []
        for e in obj["events"]:
            if not isinstance(e, dict):
                continue
            pred = str(e.get("predicate", "")).strip()
            if os.environ.get("CAUSALOS_PLACEHOLDER_GUARD", "1") == "1" and _is_placeholder_text(pred):
                pred = ""
            if pred:
                pol = self._fix_polarity(e.get("polarity", "pos"))
                mod = self._fix_modality(e.get("modality", "unknown"))
                args = e.get("args", [])
                args = args if isinstance(args, list) else []
                if os.environ.get("CAUSALOS_PLACEHOLDER_GUARD", "1") == "1":
                    clean_args = []
                    for a in args:
                        if isinstance(a, dict):
                            rv = str(a.get("role", "")).strip()
                            vv = str(a.get("value", "")).strip()
                            if _is_placeholder_text(rv) and _is_placeholder_text(vv):
                                continue
                            clean_args.append({"role": rv, "value": vv})
                    args = clean_args
                evs.append({
                    "predicate": pred, "polarity": pol, "order": int(e.get("order", 0)),
                    "args": args, "modality": mod, "inactive": bool(e.get("inactive", False))
                })
        obj["events"] = evs

        sts = []
        for s in obj["states"]:
            if not isinstance(s, dict):
                continue
            var = str(s.get("var", "")).strip()
            subj = str(s.get("subject", "")).strip()
            val = str(s.get("value", "")).strip()
            if os.environ.get("CAUSALOS_PLACEHOLDER_GUARD", "1") == "1":
                if _is_placeholder_text(var) or _is_placeholder_text(subj):
                    continue
                if _is_placeholder_text(val):
                    val = ""
            if var and subj:
                var = vn.canonicalize(var)
                sts.append({
                    "var": var, "subject": subj, "value": val,
                    "polarity": self._fix_polarity(s.get("polarity", "pos")),
                    "modality": self._fix_modality(s.get("modality", "unknown")),
                    "inactive": bool(s.get("inactive", False))
                })
        obj["states"] = sts

        cons = []
        for c in obj["constraints"]:
            if not isinstance(c, dict):
                continue
            typ = str(c.get("type", "unknown")).strip() or "unknown"
            st = str(c.get("statement", "")).strip()
            if os.environ.get("CAUSALOS_PLACEHOLDER_GUARD", "1") == "1" and _is_placeholder_text(st):
                st = ""
            cons.append({"type": typ, "statement": st})
        obj["constraints"] = cons

        if not obj["events"]:
            obj["events"] = [{"predicate": text_fallback, "polarity": "pos", "order": 0, "args": [], "modality": "fallback", "inactive": False}]
            obj["notes"] = (obj["notes"] + " | fallback_event").strip()

        if len(obj.get("states", [])) == 0 and os.environ.get("CAUSALOS_STATE_FALLBACK", "1") == "1":
            subj0 = obj["entities"][0] if obj.get("entities") else "input"
            created = []
            for ev in (obj.get("events", []) or [])[:2]:
                if isinstance(ev, dict) and not bool(ev.get("inactive", False)):
                    pred = _normalize_text(ev.get("predicate", ""))
                    if not pred:
                        continue
                    var = vn.canonicalize("ev=" + pred[:60])
                    created.append({
                        "var": var, "subject": subj0, "value": pred,
                        "polarity": _norm_label(ev.get("polarity", "pos")) or "pos",
                        "modality": self._fix_modality(ev.get("modality", "unknown")),
                        "inactive": False
                    })
            if created:
                obj["states"] = created
                obj["notes"] = (obj["notes"] + " | deterministic_state_fallback").strip()

        return obj

    def _extract_atomic_predicate(self, text: str, kind: str) -> Optional[str]:
        schema = self._schema_typed()
        prompt = f"""Return ONLY JSON with schema:
{schema}

Rules:
- Output exactly ONE event.
- The event.predicate MUST be a short phrase copied from the input (ideally 1-8 tokens).
- Do NOT output placeholders like "...".
- Do NOT add new words not in the input.

Input({kind}): {text}
JSON:"""
        obj = self._sanitize(self._generate(prompt), text, kind)
        evs = obj.get("events", []) or []
        if evs and isinstance(evs[0], dict):
            p = _normalize_text(evs[0].get("predicate", ""))
            if p and _norm_label(p) in _norm_label(text):
                return p
        return None

    def extract_frame(self, text: str, kind: str = "generic", strict_level: int = 0) -> Dict[str, Any]:
        text = _normalize_text(text)
        if not text:
            return {"entities": [], "events": [], "states": [], "constraints": [], "notes": ""}

        if os.environ.get("CAUSALOS_NO_LLM_FRAME", "0") == "1":
            return self._sanitize({"entities": [], "events": [{"predicate": text}], "states": [], "constraints": [], "notes": "no_llm_frame"}, text, kind)

        schema = self._schema_typed()
        forbid = 'Do NOT output placeholder strings like "..." or "pos|neg" literally. Choose "pos" or "neg".'

        ladder = []
        ladder.append("Use words from the input as much as possible.")
        ladder.append(forbid)
        if strict_level >= 1:
            ladder.append("Every predicate/subject/value MUST be grounded in the input text. Prefer copying exact spans.")
            ladder.append("If uncertain, output fewer items rather than placeholders.")
        if strict_level >= 2:
            ladder.append("You MUST NOT output any of these tokens anywhere: ..., pos|neg, can|must|may|unknown, cannot|must|may|unknown.")
            ladder.append("For polarity, output exactly 'pos' or 'neg'. For modality, output a short string like 'past/present/unknown'.")
        if strict_level >= 3:
            ladder.append("Hard rule: event.predicate and state.value should be substrings of input when possible.")
            ladder.append("If you cannot satisfy the rule, output one fallback event with predicate equal to full input sentence.")

        ladder_txt = "\n".join([f"- {x}" for x in ladder])
        p1 = f"""You are a semantic parser. Return ONLY JSON with the schema (types shown, not templates):
{schema}

Rules:
{ladder_txt}

Input({kind}): {text}
JSON:"""
        obj = self._sanitize(self._generate(p1), text, kind)

        if os.environ.get("CAUSALOS_DEFALLBACK_ATOMIC", "1") == "1":
            ev0 = (obj.get("events", []) or [{}])[0]
            if isinstance(ev0, dict):
                pred0 = _normalize_text(ev0.get("predicate", ""))
                if pred0 and len(_tokenize_lenient(pred0)) > 7:
                    ap = self._extract_atomic_predicate(text, kind=kind)
                    if ap:
                        obj = copy.deepcopy(obj)
                        obj["events"][0]["predicate"] = ap
                        obj["events"][0]["modality"] = "atomic_defallback"
                        obj["notes"] = (_normalize_text(obj.get("notes", "")) + " | atomic_predicate_defallback").strip()
                        obj = self._sanitize(obj, text, kind)

        return obj


# ==========================================================
# NOTE: Part 2 continues from here
# ==========================================================
# ==========================================================
# InterventionIR_B2
# ==========================================================
class InterventionIR_B2:
    @staticmethod
    def diff_frames(factual: Dict[str, Any], counterfactual: Dict[str, Any]) -> List[Dict[str, Any]]:
        ops: List[Dict[str, Any]] = []

        def _act(d: Dict[str, Any]) -> bool:
            if os.environ.get("CAUSALOS_IGNORE_INACTIVE", "1") == "1":
                return not bool(d.get("inactive", False))
            return True

        f_states = [s for s in (factual.get("states", []) or []) if isinstance(s, dict) and _act(s)]
        c_states = [s for s in (counterfactual.get("states", []) or []) if isinstance(s, dict) and _act(s)]

        f_map = {}
        for s in f_states:
            var = _norm_label(s.get("var", ""))
            sub = _norm_label(s.get("subject", ""))
            if var and sub:
                f_map[(var, sub)] = s

        used = set()
        for s2 in c_states:
            var2 = _norm_label(s2.get("var", ""))
            sub2 = _norm_label(s2.get("subject", ""))
            if not var2 or not sub2:
                continue
            k = (var2, sub2)
            s1 = f_map.get(k)
            if s1 is None:
                ops.append({"op": "SET_STATE", "payload": {"from": None, "to": s2}})
            else:
                used.add(k)
                if (_norm_label(s1.get("value", "")) != _norm_label(s2.get("value", "")) or
                    _norm_label(s1.get("polarity", "")) != _norm_label(s2.get("polarity", "")) or
                    _norm_label(s1.get("modality", "")) != _norm_label(s2.get("modality", ""))):
                    ops.append({"op": "SET_STATE", "payload": {"from": s1, "to": s2}})

        for k, s1 in f_map.items():
            if k not in used:
                ops.append({"op": "UNSET_STATE", "payload": {"state": s1}})

        f_events = [e for e in (factual.get("events", []) or []) if isinstance(e, dict) and _act(e)]
        c_events = [e for e in (counterfactual.get("events", []) or []) if isinstance(e, dict) and _act(e)]

        def ev_sig(e: Dict[str, Any]) -> Tuple[str, str]:
            pred = _norm_label(e.get("predicate", ""))
            pol = _norm_label(e.get("polarity", "pos"))
            return (pred, pol)

        f_set = set([ev_sig(e) for e in f_events if ev_sig(e)[0]])
        c_set = set([ev_sig(e) for e in c_events if ev_sig(e)[0]])

        for sig in c_set - f_set:
            ops.append({"op": "ADD_EVENT", "payload": {"predicate": sig[0], "polarity": sig[1]}})
        for sig in f_set - c_set:
            ops.append({"op": "REMOVE_EVENT", "payload": {"predicate": sig[0], "polarity": sig[1]}})

        for con in (counterfactual.get("constraints", []) or []):
            if isinstance(con, dict):
                ops.append({"op": "MODALITY", "payload": {"type": con.get("type", "unknown"), "statement": con.get("statement", "")}})

        if not ops:
            ops = [{"op": "NOOP", "payload": {}}]
        return ops


# ==========================================================
# AtomicMapper_B2
# ==========================================================
class AtomicMapper_B2:
    def __init__(self, osys: "UnifiedCausalOSV5_3Full"):
        self.osys = osys

    def _state_key(self, s: Dict[str, Any]) -> str:
        return f"state::{_normalize_text(s.get('var',''))}::{_normalize_text(s.get('subject',''))}".strip()

    def _event_key(self, pred: str) -> str:
        return f"event::{_normalize_text(pred)}".strip()

    def _value_to_vec2(self, value: str, polarity: str) -> torch.Tensor:
        value = _normalize_text(value)
        pol = _norm_label(polarity)
        if not value:
            v2 = torch.zeros(2, device=device, dtype=torch.float32)
            if pol == "neg":
                v2 = -v2
            return v2.detach()

        tok = self.osys.tokenizer(str(value), return_tensors="pt", add_special_tokens=False)
        ids = tok["input_ids"].to(self.osys.model_device)
        with torch.no_grad():
            v = self.osys.model.get_input_embeddings()(ids)[0].mean(dim=0).float().detach().to(device)
        v2 = (self.osys._proj_W @ v.view(-1, 1)).view(2)
        v2 = torch.tanh(v2)
        if pol == "neg":
            v2 = -v2
        return v2.detach()

    def apply(self, ops: List[Dict[str, Any]], core: CausalCoreV5, workspace_nodes: List[int]) -> Dict[str, Any]:
        info = {"clamped": [], "cut_in": [], "events": [], "modality": []}
        for op in ops:
            kind = op.get("op")
            payload = op.get("payload", {}) or {}

            if kind == "SET_STATE":
                s2 = payload.get("to", {}) or {}
                key = self._state_key(s2)
                cid = self.osys.concepts.resolve(key)
                node = self.osys.concepts.rep_slot(cid)
                if node not in workspace_nodes:
                    workspace_nodes.append(node)
                vec = self._value_to_vec2(str(s2.get("value", "")), str(s2.get("polarity", "pos")))
                core.apply_do_cut_in(node)
                core.apply_do_value(node, float(vec[0].item()), float(vec[1].item()))
                info["clamped"].append({"node": node, "key": key})
                info["cut_in"].append(node)

            elif kind == "ADD_EVENT":
                pred = str(payload.get("predicate", ""))
                key = self._event_key(pred)
                cid = self.osys.concepts.resolve(key)
                node = self.osys.concepts.rep_slot(cid)
                if node not in workspace_nodes:
                    workspace_nodes.append(node)
                core.apply_do_cut_in(node)
                core.apply_do_value(node, 0.8, 0.0)
                info["events"].append({"add": key, "node": node})

            elif kind == "REMOVE_EVENT":
                pred = str(payload.get("predicate", ""))
                key = self._event_key(pred)
                cid = self.osys.concepts.resolve(key)
                node = self.osys.concepts.rep_slot(cid)
                if node not in workspace_nodes:
                    workspace_nodes.append(node)
                core.apply_do_cut_in(node)
                core.apply_do_value(node, 0.0, 0.0)
                info["events"].append({"remove": key, "node": node})

            elif kind == "MODALITY":
                info["modality"].append(payload)

        return info


# ==========================================================
# ScaffoldProjector
# ==========================================================
class ScaffoldProjector:
    def __init__(self, osys: "UnifiedCausalOSV5_3Full"):
        self.osys = osys

    def project(self, frame: Dict[str, Any], strength: float = 0.35):
        if os.environ.get("CAUSALOS_DISABLE_SCAFFOLD", "0") == "1":
            return

        def _act(d: Dict[str, Any]) -> bool:
            if os.environ.get("CAUSALOS_IGNORE_INACTIVE", "1") == "1":
                return not bool(d.get("inactive", False))
            return True

        core = self.osys.core
        n = core.n_nodes
        ents = frame.get("entities", []) or []
        evs = [ev for ev in (frame.get("events", []) or []) if isinstance(ev, dict) and _act(ev)]
        sts = [st for st in (frame.get("states", []) or []) if isinstance(st, dict) and _act(st)]

        ent_nodes = []
        for ent in ents:
            cid = self.osys.concepts.resolve(ent)
            ent_nodes.append(self.osys.concepts.rep_slot(cid))

        ev_nodes = []
        for ev in evs:
            pred = ev.get("predicate", "")
            if pred:
                cid = self.osys.concepts.resolve(f"event::{pred}")
                ev_nodes.append(self.osys.concepts.rep_slot(cid))

        st_nodes = []
        for st in sts:
            key = f"state::{st.get('var','')}::{st.get('subject','')}"
            cid = self.osys.concepts.resolve(key)
            st_nodes.append(self.osys.concepts.rep_slot(cid))

        def set_edge(j: int, i: int, m: float):
            if 0 <= j < n and 0 <= i < n and j != i:
                val = _safe_tanh_inv(_clip_mag(m))
                with torch.no_grad():
                    core.raw_S.data[j, i] = 0.9 * core.raw_S.data[j, i] + 0.1 * val
                    core.A_mask[j, i] = 1.0
                    rr = float(np.clip(abs(m), 0.20, 0.90))
                    core.raw_r.data[j, i] = 0.9 * core.raw_r.data[j, i] + 0.1 * math.log(rr / (1 - rr))

        for i in ev_nodes:
            for j in st_nodes:
                set_edge(j, i, +0.35 * strength)
        for i in ent_nodes:
            for j in ev_nodes:
                set_edge(j, i, +0.20 * strength)
            for j in st_nodes:
                set_edge(j, i, +0.15 * strength)


# ==========================================================
# ReconstructionChecker
# ==========================================================
class ReconstructionChecker:
    @staticmethod
    def apply_ir(f_frame: Dict[str, Any], ops: List[Dict[str, Any]]) -> Dict[str, Any]:
        def _act(d: Dict[str, Any]) -> bool:
            if os.environ.get("CAUSALOS_IGNORE_INACTIVE", "1") == "1":
                return not bool(d.get("inactive", False))
            return True

        out = {
            "entities": list(f_frame.get("entities", []) or []),
            "events": [dict(e) for e in (f_frame.get("events", []) or []) if isinstance(e, dict) and _act(e)],
            "states": [dict(s) for s in (f_frame.get("states", []) or []) if isinstance(s, dict) and _act(s)],
            "constraints": [dict(c) for c in (f_frame.get("constraints", []) or []) if isinstance(c, dict)],
            "notes": "reconstructed"
        }

        def ev_key(e: Dict[str, Any]) -> Tuple[str, str]:
            return (_norm_label(e.get("predicate", "")), _norm_label(e.get("polarity", "pos")))

        evset = {ev_key(e) for e in out["events"] if ev_key(e)[0]}
        stmap = {}
        for s in out["states"]:
            k = (_norm_label(s.get("var", "")), _norm_label(s.get("subject", "")))
            if k[0] and k[1]:
                stmap[k] = s

        for op in ops:
            kind = op.get("op")
            payload = op.get("payload", {}) or {}
            if kind == "SET_STATE":
                to = payload.get("to", {}) or {}
                k = (_norm_label(to.get("var", "")), _norm_label(to.get("subject", "")))
                if k[0] and k[1]:
                    stmap[k] = dict(to)
            elif kind == "UNSET_STATE":
                st = payload.get("state", {}) or {}
                k = (_norm_label(st.get("var", "")), _norm_label(st.get("subject", "")))
                if k in stmap:
                    del stmap[k]
            elif kind == "ADD_EVENT":
                p = _norm_label(payload.get("predicate", ""))
                pol = _norm_label(payload.get("polarity", "pos"))
                if p:
                    evset.add((p, pol))
            elif kind == "REMOVE_EVENT":
                p = _norm_label(payload.get("predicate", ""))
                pol = _norm_label(payload.get("polarity", "pos"))
                if p and (p, pol) in evset:
                    evset.remove((p, pol))
            elif kind == "MODALITY":
                out["constraints"].append({"type": payload.get("type", "unknown"), "statement": payload.get("statement", "")})

        out["events"] = [{"predicate": p, "polarity": pol, "order": 0, "args": [], "modality": "reconstructed", "inactive": False}
                         for (p, pol) in sorted(list(evset))]
        out["states"] = list(stmap.values())
        return out

    @staticmethod
    def score(frame_hat: Dict[str, Any], c_frame: Dict[str, Any]) -> Dict[str, float]:
        def _act(d: Dict[str, Any]) -> bool:
            if os.environ.get("CAUSALOS_IGNORE_INACTIVE", "1") == "1":
                return not bool(d.get("inactive", False))
            return True

        def evset(fr):
            s = set()
            for e in (fr.get("events", []) or []):
                if isinstance(e, dict) and _act(e):
                    p = _norm_label(e.get("predicate", ""))
                    pol = _norm_label(e.get("polarity", "pos"))
                    if p:
                        s.add((p, pol))
            return s

        Eh = evset(frame_hat)
        Ec = evset(c_frame)
        ev_jacc = float(len(Eh & Ec) / max(1, len(Eh | Ec)))

        def stmap(fr):
            m = {}
            for s in (fr.get("states", []) or []):
                if isinstance(s, dict) and _act(s):
                    k = (_norm_label(s.get("var", "")), _norm_label(s.get("subject", "")))
                    if k[0] and k[1]:
                        m[k] = (_norm_label(s.get("value", "")), _norm_label(s.get("polarity", "pos")))
            return m

        Sh = stmap(frame_hat)
        Sc = stmap(c_frame)
        keys = set(Sh.keys()) | set(Sc.keys())
        st_acc = float(sum(1 for k in keys if k in Sh and k in Sc and Sh[k] == Sc[k]) / len(keys)) if keys else 0.0
        overall = float(np.clip(0.50 * ev_jacc + 0.50 * st_acc, 0.0, 1.0))
        return {"ev_jacc": ev_jacc, "st_acc": st_acc, "overall": overall}


# ==========================================================
# OptionScorer_B2 (contrast scoring)
# ==========================================================
class OptionScorer_B2:
    def __init__(self, osys: "UnifiedCausalOSV5_3Full"):
        self.osys = osys

    def _embed_text(self, text: str) -> torch.Tensor:
        tok = self.osys.tokenizer(str(text), return_tensors="pt", add_special_tokens=False)
        ids = tok["input_ids"].to(self.osys.model_device)
        with torch.no_grad():
            v = self.osys.model.get_input_embeddings()(ids)[0].mean(dim=0).float().detach().to(device)
        return v

    @staticmethod
    def _scenario_relevance(option_text: str, scenario_text: str) -> float:
        opt = _norm_label(option_text)
        scn = _norm_label(scenario_text)
        if not opt or not scn:
            return 0.0
        ta = set([t for t in re.split(r"[^a-z0-9]+", opt) if len(t) >= 2][:64])
        tb = set([t for t in re.split(r"[^a-z0-9]+", scn) if len(t) >= 2][:128])
        tok = float(len(ta & tb) / max(1, len(ta | tb))) if ta and tb else 0.0

        def bigr(s):
            s = re.sub(r"\s+", "", s)
            if len(s) < 2:
                return set([s]) if s else set()
            return set([s[i:i + 2] for i in range(min(len(s) - 1, 64))])

        ba = bigr(opt); bb = bigr(scn)
        ch = float(len(ba & bb) / max(1, len(ba | bb))) if ba and bb else 0.0
        return float(np.clip(0.6 * tok + 0.4 * ch, 0.0, 1.0))

    def _combine_rel(self, overlap_rel: float, emb_rel: float) -> float:
        mode = str(os.environ.get("CAUSALOS_REL_COMB", "max")).strip().lower()
        if mode == "max":
            return float(np.clip(max(overlap_rel, emb_rel), 0.0, 1.0))
        w = float(os.environ.get("CAUSALOS_REL_EMB_W", "0.80"))
        w = float(np.clip(w, 0.0, 1.0))
        return float(np.clip((1.0 - w) * overlap_rel + w * emb_rel, 0.0, 1.0))

    def score(
        self,
        predicted_cf: Dict[str, torch.Tensor],
        options: Dict[str, str],
        scenario_text: str = "",
        ops_signature_text: str = "",
        predicted_f: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[Optional[str], Dict[str, float]]:
        if not options:
            return None, {}

        mode = str(os.environ.get("CAUSALOS_OPT_MODE", "contrast")).strip().lower()
        if mode not in {"contrast", "legacy"}:
            mode = "contrast"

        def _pred_summary(pred: Dict[str, torch.Tensor]) -> str:
            items = [(k, v.detach().cpu().tolist()) for k, v in pred.items()]
            return json.dumps({"predicted_states": items}, ensure_ascii=False)

        v_cf = self._embed_text(_pred_summary(predicted_cf))
        v_f = None
        if mode == "contrast":
            if predicted_f is None:
                v_f = torch.zeros_like(v_cf)
            else:
                v_f = self._embed_text(_pred_summary(predicted_f))

        rel_on = os.environ.get("CAUSALOS_OPT_SCENARIO_REL", "1") == "1"
        w_rel = float(os.environ.get("CAUSALOS_OPT_SCENARIO_W", "0.65"))
        use_emb_rel = os.environ.get("CAUSALOS_OPT_SCENARIO_EMB", "1") == "1"
        v_scn = self._embed_text(scenario_text) if (use_emb_rel and scenario_text) else None

        ops_on = os.environ.get("CAUSALOS_OPT_OPS_ALIGN", "1") == "1"
        w_ops = float(os.environ.get("CAUSALOS_OPT_OPS_W", "0.70"))
        v_ops = self._embed_text(ops_signature_text) if (ops_on and ops_signature_text) else None

        scores: Dict[str, float] = {}
        strict_max = int(os.environ.get("CAUSALOS_FRAME_STRICT_MAX", "3"))

        for k, text in options.items():
            frame = self.osys.frames.extract_frame(text, kind="option", strict_level=min(2, strict_max))
            v_opt = self._embed_text(json.dumps(frame, ensure_ascii=False))

            sim_cf = _cosine(v_cf, v_opt)
            if mode == "legacy":
                sim = sim_cf
            else:
                sim_f = _cosine(v_f, v_opt) if v_f is not None else 0.0
                sim = sim_cf - sim_f

            if rel_on and scenario_text:
                rel_ov = self._scenario_relevance(text, scenario_text)
                rel_emb = float(np.clip(_cosine(self._embed_text(text), v_scn), 0.0, 1.0)) if (use_emb_rel and v_scn is not None) else 0.0
                rel = self._combine_rel(rel_ov, rel_emb)
                sim *= float(np.clip((1.0 - w_rel) + w_rel * rel, 0.20, 1.00))

            if ops_on and ops_signature_text:
                rel_ov = self._scenario_relevance(text, ops_signature_text)
                rel_emb = float(np.clip(_cosine(self._embed_text(text), v_ops), 0.0, 1.0)) if (v_ops is not None) else 0.0
                rel_ops = self._combine_rel(rel_ov, rel_emb)
                sim *= float(np.clip((1.0 - w_ops) + w_ops * rel_ops, 0.20, 1.00))

            scores[k] = float(sim)

        best = max(scores.items(), key=lambda kv: kv[1])[0] if scores else None
        return best, scores



# ==========================================================
# LikelyYesNoScorer_B11 (task-agnostic, constant criterion)
# - Score(option) = Lik(CF, option) - Lik(F, option) - λ * max(0, Lik(EMPTY, option))
# - Lik(world, option) = logP(Yes|prompt) - logP(No|prompt)
# - Relevance scaling: score *= clamp((1-w)+w*Rel, floor, 1)
# - Prior signature appended to WORLD so QueryB priors can affect scoring
# ==========================================================
class LikelyYesNoScorer_B11:
    def __init__(self, osys: "UnifiedCausalOSV5_3Full"):
        self.osys = osys

    @staticmethod
    def _act(d: Dict[str, Any]) -> bool:
        if os.environ.get("CAUSALOS_IGNORE_INACTIVE", "1") == "1":
            return not bool(d.get("inactive", False))
        return True

    @staticmethod
    def _scenario_relevance(option_text: str, scenario_text: str) -> float:
        opt = _norm_label(option_text)
        scn = _norm_label(scenario_text)
        if not opt or not scn:
            return 0.0
        ta = set([t for t in re.split(r"[^a-z0-9]+", opt) if len(t) >= 2][:64])
        tb = set([t for t in re.split(r"[^a-z0-9]+", scn) if len(t) >= 2][:128])
        tok = float(len(ta & tb) / max(1, len(ta | tb))) if ta and tb else 0.0

        def bigr(s: str):
            s = re.sub(r"\s+", "", s)
            if len(s) < 2:
                return set([s]) if s else set()
            return set([s[i:i+2] for i in range(min(len(s)-1, 64))])

        ba = bigr(opt)
        bb = bigr(scn)
        ch = float(len(ba & bb) / max(1, len(ba | bb))) if ba and bb else 0.0
        return float(np.clip(0.6 * tok + 0.4 * ch, 0.0, 1.0))

    def _prior_signature(self, max_edges: int = 6) -> str:
        try:
            pri = list(self.osys.edge_bank.prior.items())
        except Exception:
            pri = []
        if not pri:
            return ""
        scored = []
        for (e_cid, c_cid), rec in pri:
            try:
                m = float(rec.get('m', 0.0))
                w = float(rec.get('w', 0.0))
            except Exception:
                continue
            scored.append((abs(m) * w, e_cid, c_cid, m, w))
        scored.sort(reverse=True)
        lines = []
        prior_meta = getattr(self.osys.edge_bank, 'prior_meta', {}) if hasattr(self.osys.edge_bank, 'prior_meta') else {}
        for _, e_cid, c_cid, m, w in scored[:max_edges]:
            meta = prior_meta.get((e_cid, c_cid), {}) if isinstance(prior_meta, dict) else {}
            c_lab = str(meta.get('cause', f'cid{c_cid}'))
            e_lab = str(meta.get('effect', f'cid{e_cid}'))
            ev = str(meta.get('evidence', ''))
            if ev:
                lines.append(f"prior: {c_lab} -> {e_lab} (m={m:.2f}, w={w:.2f}, ev={ev})")
            else:
                lines.append(f"prior: {c_lab} -> {e_lab} (m={m:.2f}, w={w:.2f})")
        return " | ".join(lines)[:800]

    def _logprob_continuation(self, prompt: str, continuation: str) -> float:
        tok = self.osys.tokenizer
        model = self.osys.model
        dev = self.osys.model_device

        enc_p = tok(prompt, return_tensors="pt", add_special_tokens=False)
        enc_c = tok(continuation, return_tensors="pt", add_special_tokens=False)

        input_ids = torch.cat([enc_p["input_ids"], enc_c["input_ids"]], dim=1).to(dev)
        attn = torch.ones_like(input_ids, device=dev)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn)
            logits = out.logits

        cont_ids = enc_c["input_ids"].to(dev)
        Lp = enc_p["input_ids"].shape[1]
        Lc = cont_ids.shape[1]
        if Lc == 0:
            return 0.0

        start = max(0, Lp - 1)
        end = Lp + Lc - 1
        logits_slice = logits[:, start:end, :]
        logp = torch.log_softmax(logits_slice, dim=-1)
        token_logp = logp.gather(-1, cont_ids.unsqueeze(-1)).squeeze(-1)
        return float(token_logp.sum().item())

    def _label_logprob(self, prompt: str, variants: List[str]) -> float:
        vals = [self._logprob_continuation(prompt, v) for v in variants]
        return float(max(vals)) if vals else -1e9

    def _yes_no_logodds(self, prompt: str) -> float:
        yes = str(os.environ.get("CAUSALOS_ENTAIL_YES", "Yes"))
        no = str(os.environ.get("CAUSALOS_ENTAIL_NO", "No"))
        yes_vars = [" " + yes, yes]
        no_vars = [" " + no, no]
        lp_y = self._label_logprob(prompt, yes_vars)
        lp_n = self._label_logprob(prompt, no_vars)
        return float(lp_y - lp_n)

    def world_from_frame(self, frame: Dict[str, Any], raw_text: str = "") -> str:
        parts: List[str] = []
        if raw_text:
            parts.append(_normalize_text(raw_text))
        for ent in (frame.get("entities", []) or []):
            s = _normalize_text(ent)
            if s:
                parts.append(s)
        for e in (frame.get("events", []) or []):
            if isinstance(e, dict) and self._act(e):
                p = _normalize_text(e.get("predicate", ""))
                if p:
                    parts.append(p)
        for st in (frame.get("states", []) or []):
            if isinstance(st, dict) and self._act(st):
                sub = _normalize_text(st.get("subject", ""))
                val = _normalize_text(st.get("value", ""))
                if sub and val:
                    parts.append(f"{sub}: {val}")
                elif val:
                    parts.append(val)
        s = " | ".join([p for p in parts if p])
        prior_sig = self._prior_signature(max_edges=int(os.environ.get("CAUSALOS_PRIOR_SIG_MAX", "6")))
        if prior_sig:
            s = (s + " | " + prior_sig)
        return s[:950]

    def _prompt(self, mode: str, world: str, intervention: str, statement: str) -> str:
        return (
            f"MODE: {mode}\\n"
            f"WORLD:\\n{world}\\n"
            f"INTERVENTION:\\n{intervention}\\n"
            f"STATEMENT:\\n{statement}\\n"
            f"QUESTION: Given the WORLD under MODE, is the STATEMENT likely/expected? Answer Yes or No.\\n"
            f"ANSWER:"
        )

    def score(self, options: Dict[str, str], world_f: str, world_cf: str, intervention: str) -> Tuple[Optional[str], Dict[str, float], Dict[str, Any]]:
        if not options:
            return None, {}, {"gen_pos": {}, "best_gen_pos": 0.0, "rel": {}, "best_rel": 0.0}

        use_generic = os.environ.get("CAUSALOS_GENERIC_PENALTY", "1") == "1"
        lam = float(os.environ.get("CAUSALOS_GENERIC_LAMBDA", "0.8"))
        lam = float(np.clip(lam, 0.0, 3.0))

        rel_on = os.environ.get("CAUSALOS_LIKELY_REL", "1") == "1"
        w_rel = float(os.environ.get("CAUSALOS_LIKELY_REL_W", "0.80"))
        rel_floor = float(os.environ.get("CAUSALOS_LIKELY_REL_FLOOR", "0.15"))
        w_rel = float(np.clip(w_rel, 0.0, 1.0))
        rel_floor = float(np.clip(rel_floor, 0.0, 0.50))

        scores: Dict[str, float] = {}
        gen_pos_map: Dict[str, float] = {}
        rel_map: Dict[str, float] = {}
        part_map: Dict[str, Dict[str, float]] = {}

        scenario_all = (world_cf + " " + world_f + " " + intervention)

        for k, text in options.items():
            s = text.strip()
            p_cf = self._prompt("COUNTERFACTUAL", world_cf, intervention, s)
            p_f = self._prompt("FACTUAL", world_f, "(none)", s)

            lik_cf = self._yes_no_logodds(p_cf)
            lik_f = self._yes_no_logodds(p_f)
            score = lik_cf - lik_f

            # counterfactual-likelihood anchor (task-agnostic): prefer statements that are themselves likely in CF
            cf_w = float(os.environ.get("CAUSALOS_LIKELY_CF_W", "0.50"))
            cf_w = float(np.clip(cf_w, 0.0, 2.0))
            score = score + cf_w * float(lik_cf)

            gen_pos = 0.0
            if use_generic:
                p0 = self._prompt("EMPTY", "", "(none)", s)
                gen = self._yes_no_logodds(p0)
                gen_pos = float(max(0.0, gen))
                score = score - lam * gen_pos

            rel = 1.0
            if rel_on:
                rel = self._scenario_relevance(s, scenario_all)
                scale = float(np.clip((1.0 - w_rel) + w_rel * rel, rel_floor, 1.00))
                score = score * scale

            part_map[k] = {'lik_cf': float(lik_cf), 'lik_f': float(lik_f), 'gen_pos': float(gen_pos), 'rel': float(rel if rel_on else 0.0), 'cf_term': float(cf_w * float(lik_cf))}

            scores[k] = float(score)
            gen_pos_map[k] = float(gen_pos)
            rel_map[k] = float(rel) if rel_on else 0.0

        best = max(scores.items(), key=lambda kv: kv[1])[0] if scores else None
        best_gen_pos = float(gen_pos_map.get(best, 0.0)) if best else 0.0
        best_rel = float(rel_map.get(best, 0.0)) if best else 0.0
        return best, scores, {"gen_pos": gen_pos_map, "best_gen_pos": best_gen_pos, "rel": rel_map, "best_rel": best_rel, "parts": part_map}

# ==========================================================
# PriorCandidateGenerator (Query B)
# ==========================================================
class PriorCandidateGenerator:
    def __init__(self, osys: "UnifiedCausalOSV5_3Full"):
        self.osys = osys

    @staticmethod
    def _schema() -> str:
        return """{
  "edges":[
    {
      "cause":"string",
      "effect":"string",
      "polarity":"pos|neg",
      "strength":0.0,
      "confidence":0.0,
      "evidence":{"type":"grounded|commonsense|analogy","note":"string"}
    }
  ],
  "notes":"string"
}"""

    def _generate(self, prompt: str) -> Dict[str, Any]:
        tok = self.osys.tokenizer(prompt, return_tensors="pt")
        tok = {k: v.to(self.osys.model_device) for k, v in tok.items()}
        with torch.no_grad():
            out = self.osys.model.generate(**tok, max_new_tokens=420, do_sample=False,
                                           pad_token_id=self.osys.tokenizer.eos_token_id)
        resp = self.osys.tokenizer.decode(out[0][tok["input_ids"].shape[-1]:], skip_special_tokens=True)
        js = _extract_first_json_obj(resp)
        if not js:
            return {}
        try:
            return json.loads(js)
        except Exception:
            return {}

    def propose(self, cause_candidates: List[str], effect_candidates: List[str], context: str, max_edges: int = 10) -> Dict[str, Any]:
        schema = self._schema()
        prompt = f"""You propose plausible causal edges for a causal memory prior.
Return ONLY JSON with schema:
{schema}

Rules:
- Use ONLY provided candidate strings; do not invent new identifiers.
- strength and confidence are in [0,1].
- evidence.type is one of: grounded, commonsense, analogy.
- Do NOT output placeholders like "...".
- Output at most {max_edges} edges.

CAUSE_CANDIDATES: {json.dumps(cause_candidates[:24], ensure_ascii=False)}
EFFECT_CANDIDATES: {json.dumps(effect_candidates[:24], ensure_ascii=False)}
CONTEXT: {context[:600]}

JSON:"""
        obj = self._generate(prompt)
        if not isinstance(obj, dict):
            return {"edges": [], "notes": "bad_obj"}
        edges = obj.get("edges", [])
        if not isinstance(edges, list):
            edges = []
        clean = []
        for e in edges:
            if not isinstance(e, dict):
                continue
            c = _normalize_text(e.get("cause", ""))
            eff = _normalize_text(e.get("effect", ""))
            if not c or not eff:
                continue
            pol = _norm_label(e.get("polarity", "pos"))
            pol = "neg" if pol == "neg" else "pos"
            try:
                strength = float(e.get("strength", 0.0))
                conf = float(e.get("confidence", 0.0))
            except Exception:
                continue
            strength = float(np.clip(strength, 0.0, 1.0))
            conf = float(np.clip(conf, 0.0, 1.0))
            ev = e.get("evidence", {}) if isinstance(e.get("evidence", {}), dict) else {}
            ev_type = _norm_label(ev.get("type", "commonsense"))
            if ev_type not in {"grounded", "commonsense", "analogy"}:
                ev_type = "commonsense"
            note = _normalize_text(ev.get("note", ""))[:120]
            clean.append({
                "cause": c, "effect": eff, "polarity": pol,
                "strength": strength, "confidence": conf,
                "evidence": {"type": ev_type, "note": note}
            })
            if len(clean) >= max_edges:
                break
        return {"edges": clean, "notes": _normalize_text(obj.get("notes", ""))[:160]}


# ==========================================================
# UnifiedCausalOSV5_3Full
# ==========================================================
class UnifiedCausalOSV5_3Full:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
        init_n_nodes: int = 256,
        init_slots_per_concept: int = 2,
        expand_chunk: int = 256,
        local_horizon: int = 10,
        w0: float = 0.7,
        w1: float = 0.3,
        retriever: Optional[Retriever] = None,
        verifier: Optional[Verifier] = None,
    ):
        print(f"[CausalOS v5.3_full] BUILD_ID={BUILD_ID}", flush=True)
        print(f"[CausalOS v5.3_full] Loading model: {model_id}", flush=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map="auto")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        try:
            gc = self.model.generation_config
            gc.do_sample = False
            gc.temperature = None
            gc.top_p = None
            gc.top_k = None
        except Exception:
            pass

        self.policy = KnowledgePolicy(beta_prior=0.25)

        self.core = CausalCoreV5(n_nodes=init_n_nodes, p_r0=0.20).to(device)
        self.expand_chunk = int(expand_chunk)
        self._n_used = 0

        self.concepts = ConceptBank(self, init_slots_per_concept=init_slots_per_concept, sim_base_threshold=0.82, expand_chunk=expand_chunk)
        self._proj_W = self._init_projection_matrix()

        self.varnorm_main = VarNormalizer(self, base_threshold=0.84)
        self.varnorm_opt = VarNormalizer(self, base_threshold=0.84)

        self.ground = GroundingChecker(self)

        self.edge_bank = EdgeBank()
        self._cache_prior_S: Optional[torch.Tensor] = None
        self._cache_prior_version = 0
        self._prior_version = 0

        self.triplets = CausalTripletExtractor(self)
        self.localizer = OmegaLocalizer(horizon=local_horizon, w0=w0, w1=w1)
        self.impossible = ImpossibilityController(kappa=10.0, tau=0.65)

        self.frames = FrameExtractorLLM(self)
        self.ir_b2 = InterventionIR_B2()
        self.atomic_b2 = AtomicMapper_B2(self)
        self.scaffold = ScaffoldProjector(self)
        self.recon = ReconstructionChecker()
        self.opt_scorer_b2 = OptionScorer_B2(self)

        self.opt_scorer_likely_b11 = LikelyYesNoScorer_B11(self)
        self.prior_gen = PriorCandidateGenerator(self)

        self.retriever: Retriever = retriever if retriever is not None else NullRetriever()
        self.verifier: Verifier = verifier if verifier is not None else NullVerifier()

        self._emb_cache: Dict[str, torch.Tensor] = {}

    @property
    def model_device(self):
        return next(self.model.parameters()).device

    def _init_projection_matrix(self) -> torch.Tensor:
        with torch.no_grad():
            hidden = self.model.get_input_embeddings().weight.shape[1]
        g = torch.Generator(device="cpu")
        g.manual_seed(42)
        W = torch.randn(2, hidden, generator=g, dtype=torch.float32) * 0.02
        return W.to(device)

    def _alloc_node(self) -> int:
        if self._n_used >= self.core.n_nodes:
            new_n = self.core.n_nodes + self.expand_chunk
            print(f"[CausalOS v5.3_full] Expanding n_nodes: {self.core.n_nodes} -> {new_n}", flush=True)
            self.core.resize(new_n, p_r0=0.20)
        idx = int(self._n_used)
        self._n_used += 1
        return idx

    def _embed_text(self, text: str) -> torch.Tensor:
        key = text[:2000]
        if key in self._emb_cache:
            return self._emb_cache[key]
        tok = self.tokenizer(str(key), return_tensors="pt", add_special_tokens=False)
        ids = tok["input_ids"].to(self.model_device)
        with torch.no_grad():
            v = self.model.get_input_embeddings()(ids)[0].mean(dim=0).float().detach()
        self._emb_cache[key] = v
        return v

    def _bump_prior_version(self):
        self._prior_version += 1
        self._cache_prior_S = None

    def _ensure_cache_prior_S(self) -> torch.Tensor:
        if self._cache_prior_S is not None and self._cache_prior_version == self._prior_version:
            return self._cache_prior_S
        n = self.core.n_nodes
        Sprior = torch.zeros(n, n, device=device)
        for (e_cid, c_cid), rec in self.edge_bank.prior.items():
            if (e_cid, c_cid) in self.edge_bank.disabled_prior:
                continue
            m = float(rec["m"])
            ej = self.concepts.rep_slot(e_cid)
            ci = self.concepts.rep_slot(c_cid)
            if 0 <= ej < n and 0 <= ci < n and ej != ci:
                Sprior[ej, ci] += float(m)
        Sprior = torch.clamp(Sprior, -0.99, 0.99)
        self._cache_prior_S = Sprior
        self._cache_prior_version = self._prior_version
        return Sprior

    # ---------- prior_mask ----------
    def _build_prior_mask(self, S_prior: Optional[torch.Tensor]) -> Tuple[Optional[torch.Tensor], Dict[str, int]]:
        if S_prior is None:
            return None, {"nonzero": 0, "topk": 0, "added_to_A": 0}

        abs_thr = float(os.environ.get("CAUSALOS_PRIOR_ABS_THR", "0.01"))
        topk = int(os.environ.get("CAUSALOS_PRIOR_TOPK", "64"))
        abs_thr = float(max(0.0, abs_thr))
        topk = int(max(0, topk))

        A = self.core.A_mask.detach()
        Sp = S_prior.detach()
        absSp = Sp.abs()
        n = Sp.shape[0]

        mask_cand = (absSp >= abs_thr)
        diag = torch.eye(n, device=Sp.device, dtype=torch.bool)
        mask_cand = mask_cand & (~diag)

        idx = torch.nonzero(mask_cand, as_tuple=False)
        nonzero = int(idx.shape[0])
        if nonzero == 0 or topk == 0:
            return None, {"nonzero": nonzero, "topk": 0, "added_to_A": 0}

        vals = absSp[mask_cand]
        k = min(topk, vals.numel())
        top_vals, top_pos = torch.topk(vals.view(-1), k=k)

        idx_list = idx.tolist()
        chosen = [idx_list[p] for p in top_pos.tolist()]

        prior_mask = torch.zeros_like(Sp)
        for j, i in chosen:
            prior_mask[j, i] = 1.0

        added_to_A = int((prior_mask.bool() & (A == 0.0).bool()).sum().item())
        return prior_mask, {"nonzero": nonzero, "topk": k, "added_to_A": added_to_A}

    # ---------- ingest_context ----------
    def ingest_context(self, text: Any, source: str = "user", weight: float = 0.85):
        text = _normalize_text(text)
        if not text:
            return
        clean_text = _strip_options_block(text)
        triplets = self.triplets.extract(clean_text)
        if triplets:
            for tr in triplets:
                c_label = tr["cause"]; e_label = tr["effect"]; m = float(tr["magnitude"])
                if _is_bad_label(c_label) or _is_bad_label(e_label):
                    continue
                c_cid = self.concepts.resolve(c_label)
                e_cid = self.concepts.resolve(e_label)
                self.edge_bank.update_edge(e_cid, c_cid, m=m, w=float(weight), source=source, layer="strong")
        self._project_strong_edges_to_core()

    def _project_strong_edges_to_core(self):
        n = self.core.n_nodes
        with torch.no_grad():
            for (e_cid, c_cid), rec in self.edge_bank.strong.items():
                m = float(rec["m"])
                ej = self.concepts.rep_slot(e_cid)
                ci = self.concepts.rep_slot(c_cid)
                if ej >= n or ci >= n or ej == ci:
                    continue
                val = _safe_tanh_inv(m)
                self.core.raw_S.data[ej, ci] = 0.7 * self.core.raw_S.data[ej, ci] + 0.3 * val
                self.core.A_mask[ej, ci] = 1.0
                rr = float(np.clip(abs(m), 0.25, 0.95))
                self.core.raw_r.data[ej, ci] = 0.7 * self.core.raw_r.data[ej, ci] + 0.3 * math.log(rr / (1 - rr))

    # ---------- helpers ----------
    def _nodes_for_state_keys(self, keys: List[str]) -> List[int]:
        nodes = []
        for k in keys:
            cid = self.concepts.resolve(k)
            nodes.append(self.concepts.rep_slot(cid))
        return [int(x) for x in dict.fromkeys(nodes)]

    def _collect_predicted_states(self, state_keys: List[str], x_final: torch.Tensor) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for k in state_keys:
            cid = self.concepts.resolve(k)
            node = self.concepts.rep_slot(cid)
            if 0 <= node < x_final.shape[0]:
                out[k] = x_final[node].detach()
        return out

    def _frame_quality(self, frame: Dict[str, Any]) -> Dict[str, float]:
        def _act(d: Dict[str, Any]) -> bool:
            if os.environ.get("CAUSALOS_IGNORE_INACTIVE", "1") == "1":
                return not bool(d.get("inactive", False))
            return True
        items = []
        for e in (frame.get("events", []) or []):
            if isinstance(e, dict) and _act(e):
                items.append(str(e.get("predicate", "")))
        for s in (frame.get("states", []) or []):
            if isinstance(s, dict) and _act(s):
                items += [str(s.get("subject", "")), str(s.get("value", ""))]
        if not items:
            return {"placeholder_ratio": 1.0, "density": 0.0}
        bad = sum(1 for it in items if os.environ.get("CAUSALOS_PLACEHOLDER_GUARD", "1") == "1" and _is_placeholder_text(it))
        pr = bad / max(1, len(items))
        density = float(np.clip((len(frame.get("states", []) or []) + len(frame.get("events", []) or [])) / 6.0, 0.0, 1.0))
        return {"placeholder_ratio": float(pr), "density": density}

    def _confidence(self, u: float, target_vecs: List[torch.Tensor], opt_margin: Optional[float],
                    recon_overall: float, ground_avg: float, fq: Dict[str, float]) -> float:
        stab = float(np.clip(1.0 - u, 0.0, 1.0))
        norms = [float(torch.norm(v).item()) for v in target_vecs] if target_vecs else [0.0]
        mean_norm = float(np.mean(norms))
        y0 = 0.25
        dec = float(np.clip(mean_norm / y0, 0.0, 1.0))
        conf = 0.15 + 0.75 * stab * (0.30 + 0.70 * dec)
        conf *= float(np.clip(0.55 + 0.65 * recon_overall, 0.20, 1.10))
        conf *= float(np.clip(0.55 + 0.65 * ground_avg, 0.20, 1.10))
        if opt_margin is not None:
            conf *= float(np.clip(0.85 + 0.30 * opt_margin, 0.75, 1.10))
        pr = float(fq.get("placeholder_ratio", 0.0))
        dens = float(fq.get("density", 1.0))
        conf *= float(np.clip((1.0 - 0.90 * pr) * (0.55 + 0.45 * dens), 0.10, 1.00))
        return float(np.clip(conf, 0.0, 1.0))

    # ---------- enforce/span ----------
    def _span_specificity_penalty(self, source: str, span: str) -> float:
        if os.environ.get("CAUSALOS_SPAN_SPECIFICITY", "1") != "1":
            return 0.0
        toks = _tokenize_lenient(span)
        n = len(toks)
        penalty = 0.0
        if n <= 1:
            penalty += 0.18
        if n == 2:
            penalty += 0.04
        src = _norm_label(source)
        sp = _norm_label(span)
        if src and sp:
            freq = src.count(sp)
            if freq >= 2:
                penalty += 0.07 * min(freq, 5)
        chars = [c for c in sp if c.isalnum()]
        if chars:
            uniq = len(set(chars)) / max(1, len(chars))
            if uniq < 0.45:
                penalty += 0.08 * (0.45 - uniq) / 0.45
        return float(np.clip(penalty, 0.0, 0.45))

    def _best_span_from_source(self, source: str, target: str) -> Optional[str]:
        src = _normalize_text(source)
        tgt = _normalize_text(target)
        if not src:
            return None
        toks = _tokenize_lenient(src)
        if not toks:
            return None

        min_tok = int(os.environ.get("CAUSALOS_SPAN_MIN_TOK", "2"))
        max_tok = int(os.environ.get("CAUSALOS_SPAN_MAX_TOK", "8"))
        min_tok = max(1, min(min_tok, 6))
        max_tok = max(min_tok, min(max_tok, 10))

        v_t = self._embed_text(tgt) if tgt else self._embed_text(src)

        best = None
        best_score = -1.0
        for n in range(min_tok, max_tok + 1):
            for i in range(0, max(1, len(toks) - n + 1)):
                cand = " ".join(toks[i:i + n]).strip()
                if not cand:
                    continue
                ov = GroundingChecker.overlap_score(cand, tgt) if tgt else 0.0
                emb = _cosine(self._embed_text(cand), v_t)
                score = 0.55 * emb + 0.45 * ov
                score -= 0.015 * (n - min_tok)
                score -= self._span_specificity_penalty(src, cand)
                if score > best_score:
                    best_score = score
                    best = cand

        if best is None and min_tok > 1:
            for i in range(len(toks)):
                cand = toks[i].strip()
                if not cand:
                    continue
                ov = GroundingChecker.overlap_score(cand, tgt) if tgt else 0.0
                emb = _cosine(self._embed_text(cand), v_t)
                score = 0.55 * emb + 0.45 * ov - self._span_specificity_penalty(src, cand) - 0.25
                if score > best_score:
                    best_score = score
                    best = cand

        return best if (best_score > 0.08 and best) else None

    def _enforce_grounded_frame(self, frame: Dict[str, Any], source: str, kind: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if os.environ.get("CAUSALOS_ENFORCE_GROUND", "1") != "1":
            return frame, {"changed": 0, "details": []}

        thr = float(os.environ.get("CAUSALOS_ENFORCE_THR", "0.55"))
        fr = copy.deepcopy(frame)
        details = []
        changed = 0

        def _act(d: Dict[str, Any]) -> bool:
            if os.environ.get("CAUSALOS_IGNORE_INACTIVE", "1") == "1":
                return not bool(d.get("inactive", False))
            return True

        for idx, e in enumerate(fr.get("events", []) or []):
            if not (isinstance(e, dict) and _act(e)):
                continue
            pred = _normalize_text(e.get("predicate", ""))
            if not pred:
                continue
            s = self.ground.score_item(pred, source)
            if s >= thr:
                continue
            ap = None
            if os.environ.get("CAUSALOS_DEFALLBACK_ATOMIC", "1") == "1":
                ap = self.frames._extract_atomic_predicate(source, kind=kind)
            if not ap:
                ap = self._best_span_from_source(source, pred)
            if not ap:
                ap = source
            if ap and ap != pred:
                fr["events"][idx]["predicate"] = ap
                fr["events"][idx]["modality"] = "enforced"
                changed += 1
                details.append({"type": "event_predicate", "old": pred, "new": ap, "score": s})

        for idx, st in enumerate(fr.get("states", []) or []):
            if not (isinstance(st, dict) and _act(st)):
                continue
            val = _normalize_text(st.get("value", ""))
            if not val:
                evs = [ev for ev in (fr.get("events", []) or []) if isinstance(ev, dict) and _act(ev)]
                if evs:
                    val2 = _normalize_text(evs[0].get("predicate", ""))
                    if val2:
                        fr["states"][idx]["value"] = val2
                        fr["states"][idx]["modality"] = "enforced"
                        changed += 1
                        details.append({"type": "state_value_empty", "old": "", "new": val2})
                continue
            s = self.ground.score_item(val, source)
            if s >= thr:
                continue
            bv = self._best_span_from_source(source, val) or source
            if bv and bv != val:
                fr["states"][idx]["value"] = bv
                fr["states"][idx]["modality"] = "enforced"
                changed += 1
                details.append({"type": "state_value", "old": val, "new": bv, "score": s})

        if changed:
            fr["notes"] = (_normalize_text(fr.get("notes", "")) + " | enforce_ground_v8").strip()

        return fr, {"changed": changed, "details": details}

    # ---------- dedup ----------
    def _inactive_dedup_inclusion(self, frame: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if os.environ.get("CAUSALOS_INACTIVE_DEDUP", "1") != "1":
            return frame, {"changed": 0, "events": 0, "states": 0}

        fr = copy.deepcopy(frame)
        changed = 0; de = 0; ds = 0

        def _act(d: Dict[str, Any]) -> bool:
            return not bool(d.get("inactive", False))

        evs = [e for e in (fr.get("events", []) or []) if isinstance(e, dict)]
        preds = [(i, _normalize_text(e.get("predicate", "")), _norm_label(e.get("predicate", ""))) for i, e in enumerate(evs) if _act(e)]
        for i, pi, pli in preds:
            if not pi:
                continue
            for j, pj, plj in preds:
                if i == j or not pj:
                    continue
                if pli and plj and pli in plj and len(pi) < len(pj):
                    if _act(fr["events"][i]):
                        fr["events"][i]["inactive"] = True
                        fr["events"][i]["modality"] = (_normalize_text(fr["events"][i].get("modality", "")) + "|inactive_inclusion").strip()
                        changed += 1; de += 1

        sts = [s for s in (fr.get("states", []) or []) if isinstance(s, dict)]
        vals = []
        for i, s in enumerate(sts):
            if not _act(s):
                continue
            subj = _norm_label(s.get("subject", ""))
            val = _normalize_text(s.get("value", ""))
            v = _norm_label(val)
            if subj and v:
                vals.append((i, subj, val, v))
        for i, si, vali, vli in vals:
            for j, sj, valj, vlj in vals:
                if i == j or si != sj:
                    continue
                if vli in vlj and len(vali) < len(valj):
                    if _act(fr["states"][i]):
                        fr["states"][i]["inactive"] = True
                        fr["states"][i]["modality"] = (_normalize_text(fr["states"][i].get("modality", "")) + "|inactive_inclusion").strip()
                        changed += 1; ds += 1

        if changed:
            fr["notes"] = (_normalize_text(fr.get("notes", "")) + " | inactive_inclusion").strip()
        return fr, {"changed": changed, "events": de, "states": ds}

    def _inactive_dedup_embedding(self, frame: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if os.environ.get("CAUSALOS_INACTIVE_DEDUP", "1") != "1":
            return frame, {"changed": 0, "events": 0, "states": 0}

        fr = copy.deepcopy(frame)
        changed = 0; de = 0; ds = 0
        thr = float(os.environ.get("CAUSALOS_DEDUP_SIM_THR", "0.92"))

        def _act(d: Dict[str, Any]) -> bool:
            return not bool(d.get("inactive", False))

        evs = [e for e in (fr.get("events", []) or []) if isinstance(e, dict)]
        reps: List[Tuple[str, torch.Tensor]] = []
        for i, e in enumerate(evs):
            if not _act(e):
                continue
            p = _normalize_text(e.get("predicate", ""))
            if not p:
                continue
            vp = self._embed_text(p)
            merged = False
            for rp, rv in reps:
                if _norm_label(p) == _norm_label(rp) or _cosine(vp, rv) >= thr:
                    fr["events"][i]["inactive"] = True
                    fr["events"][i]["modality"] = (_normalize_text(fr["events"][i].get("modality", "")) + "|inactive_dedup").strip()
                    changed += 1; de += 1
                    merged = True
                    break
            if not merged:
                reps.append((p, vp))

        sts = [s for s in (fr.get("states", []) or []) if isinstance(s, dict)]
        reps2: List[Tuple[str, str, torch.Tensor]] = []
        for i, s in enumerate(sts):
            if not _act(s):
                continue
            subj = _normalize_text(s.get("subject", ""))
            val = _normalize_text(s.get("value", ""))
            if not subj and not val:
                continue
            vv = self._embed_text(val) if val else self._embed_text(subj)
            key = (subj.lower(), val.lower())
            merged = False
            for (rsub, rval, rv) in reps2:
                if key == (rsub, rval) or _cosine(vv, rv) >= thr:
                    fr["states"][i]["inactive"] = True
                    fr["states"][i]["modality"] = (_normalize_text(fr["states"][i].get("modality", "")) + "|inactive_dedup").strip()
                    changed += 1; ds += 1
                    merged = True
                    break
            if not merged:
                reps2.append((key[0], key[1], vv))

        if changed:
            fr["notes"] = (_normalize_text(fr.get("notes", "")) + " | inactive_dedup").strip()
        return fr, {"changed": changed, "events": de, "states": ds}

    # ---------- IDS / QueryB ----------
    def _compute_ids(self, margin: Optional[float], ground_min: float, density: float, coverage: float, u: float) -> float:
        margin_ref = float(os.environ.get("CAUSALOS_IDS_MARGIN_REF", "0.05"))
        margin_ref = max(1e-6, margin_ref)
        m = float(margin if margin is not None else 0.0)
        m_norm = float(np.clip(m / margin_ref, 0.0, 1.0))
        ids = (
            0.35 * (1.0 - m_norm) +
            0.20 * (1.0 - float(np.clip(ground_min, 0.0, 1.0))) +
            0.20 * (1.0 - float(np.clip(density, 0.0, 1.0))) +
            0.15 * (1.0 - float(np.clip(coverage, 0.0, 1.0))) +
            0.10 * float(np.clip(u, 0.0, 1.0))
        )
        return float(np.clip(ids, 0.0, 1.0))

    def _inject_prior_edges(self, edges: List[Dict[str, Any]], source_tag: str = "prior_llm") -> Dict[str, Any]:
        base_w = float(os.environ.get("CAUSALOS_PRIOR_BASE_W", "0.20"))
        w_max = float(os.environ.get("CAUSALOS_PRIOR_W_MAX", "0.25"))
        base_w = float(np.clip(base_w, 0.0, 1.0))
        w_max = float(np.clip(w_max, 0.0, 1.0))

        added = 0
        metas = []
        for e in edges:
            c = _normalize_text(e.get("cause", ""))
            eff = _normalize_text(e.get("effect", ""))
            if not c or not eff:
                continue
            pol = _norm_label(e.get("polarity", "pos"))
            strength = float(np.clip(float(e.get("strength", 0.0)), 0.0, 1.0))
            conf = float(np.clip(float(e.get("confidence", 0.0)), 0.0, 1.0))
            ev = e.get("evidence", {}) if isinstance(e.get("evidence", {}), dict) else {}
            ev_type = _norm_label(ev.get("type", "commonsense"))
            if ev_type not in {"grounded", "commonsense", "analogy"}:
                ev_type = "commonsense"

            w = float(min(base_w * strength * conf, w_max))
            m = +min(0.90, 0.25 + 0.65 * strength)
            if pol == "neg":
                m = -m

            c_cid = self.concepts.resolve(c)
            e_cid = self.concepts.resolve(eff)
            self.edge_bank.update_edge(e_cid, c_cid, m=m, w=w, source=source_tag, layer="prior",
                                       meta={"evidence": ev_type, "conf": conf, "strength": strength, "cause": c, "effect": eff})
            metas.append({"cause": c, "effect": eff, "m": m, "w": w, "evidence": ev_type, "conf": conf, "strength": strength})
            added += 1

        if added:
            self._bump_prior_version()
        return {"added": added, "edges": metas[:12]}

    # ======================================================
    # answer_counterfactual_B2 (contrast scoring integrated)
    # ======================================================
    def answer_counterfactual_B2(self, factual: str, counterfactual: str,
                                 options: Optional[Dict[str, str]] = None) -> AnswerPacket:
        factual = _normalize_text(factual)
        counterfactual = _normalize_text(counterfactual)

        thr = float(os.environ.get("CAUSALOS_GROUND_THR", "0.45"))
        max_retry = int(os.environ.get("CAUSALOS_GROUND_RETRY", "3"))
        strict_max = int(os.environ.get("CAUSALOS_FRAME_STRICT_MAX", "3"))
        min_margin = float(os.environ.get("CAUSALOS_OPT_MIN_MARGIN", "0.03"))

        def _act(d: Dict[str, Any]) -> bool:
            if os.environ.get("CAUSALOS_IGNORE_INACTIVE", "1") == "1":
                return not bool(d.get("inactive", False))
            return True

        def extract_grounded(text: str, kind: str):
            best = None
            best_score = -1.0
            best_stats = {"avg": 0.0, "min": 0.0, "n": 0}
            best_try = 0
            best_fq = {"placeholder_ratio": 1.0, "density": 0.0}
            best_enf = {"changed": 0, "details": []}
            best_ddi = {"changed": 0, "events": 0, "states": 0}
            best_dde = {"changed": 0, "events": 0, "states": 0}

            for t in range(max_retry):
                strict_level = min(strict_max, t)
                fr = self.frames.extract_frame(text, kind=kind, strict_level=strict_level)

                fr, enf = self._enforce_grounded_frame(fr, text, kind=kind)
                fr, ddi = self._inactive_dedup_inclusion(fr)
                fr, dde = self._inactive_dedup_embedding(fr)

                stats = self.ground.score_frame(fr, text)
                fq = self._frame_quality(fr)

                score = 0.75 * stats["avg"] + 0.25 * stats["min"]
                score += 0.06 * min(6, len([s for s in (fr.get("states", []) or []) if isinstance(s, dict) and _act(s)]))
                score += 0.03 * min(4, len([e for e in (fr.get("events", []) or []) if isinstance(e, dict) and _act(e)]))
                score -= 0.80 * fq["placeholder_ratio"]

                if score > best_score:
                    best = fr
                    best_score = score
                    best_stats = stats
                    best_try = t + 1
                    best_fq = fq
                    best_enf = enf
                    best_ddi = ddi
                    best_dde = dde

                if stats["avg"] >= thr and stats["min"] >= thr * 0.6 and fq["placeholder_ratio"] <= 0.25:
                    break

            if best is None:
                best = {"entities": [], "events": [], "states": [], "constraints": [], "notes": "ground_fail"}
            return best, best_stats, best_try, best_fq, best_enf, best_ddi, best_dde

        f_frame, f_ground, f_try, f_fq, f_enf, f_ddi, f_dde = extract_grounded(factual, "factual")
        c_frame, c_ground, c_try, c_fq, c_enf, c_ddi, c_dde = extract_grounded(counterfactual, "counterfactual")

        self.scaffold.project(f_frame, strength=0.50)
        self.scaffold.project(c_frame, strength=0.50)

        ops = self.ir_b2.diff_frames(f_frame, c_frame)

        def ops_signature(ops_list: List[Dict[str, Any]]) -> str:
            parts = []
            for op in ops_list:
                kind = str(op.get("op", ""))
                payload = op.get("payload", {}) or {}
                parts.append(kind)
                if kind == "SET_STATE":
                    to = payload.get("to", {}) or {}
                    parts.append(_normalize_text(to.get("subject", "")))
                    parts.append(_normalize_text(to.get("value", "")))
                    parts.append(_normalize_text(to.get("var", "")))
                elif kind in ("ADD_EVENT", "REMOVE_EVENT"):
                    parts.append(_normalize_text(payload.get("predicate", "")))
                elif kind == "MODALITY":
                    parts.append(_normalize_text(payload.get("statement", "")))
            s = " | ".join([p for p in parts if p])
            return s[:800]
        ops_sig_text = ops_signature(ops)

        frame_hat = self.recon.apply_ir(f_frame, ops)
        recon_score = self.recon.score(frame_hat, c_frame)

        ws_nodes = []
        def add_frame_nodes(fr: Dict[str, Any]):
            for ent in fr.get("entities", []) or []:
                cid = self.concepts.resolve(ent)
                ws_nodes.append(self.concepts.rep_slot(cid))
            for ev in fr.get("events", []) or []:
                if isinstance(ev, dict) and _act(ev):
                    pred = ev.get("predicate", "")
                    if pred:
                        cid = self.concepts.resolve(f"event::{pred}")
                        ws_nodes.append(self.concepts.rep_slot(cid))
            for st in fr.get("states", []) or []:
                if isinstance(st, dict) and _act(st):
                    key = f"state::{st.get('var','')}::{st.get('subject','')}"
                    cid = self.concepts.resolve(key)
                    ws_nodes.append(self.concepts.rep_slot(cid))

        add_frame_nodes(f_frame)
        add_frame_nodes(c_frame)
        ws_nodes = [int(x) for x in dict.fromkeys(ws_nodes) if 0 <= int(x) < self.core.n_nodes]
        if not ws_nodes:
            cid = self.concepts.resolve("question::" + (factual + "|" + counterfactual)[:80])
            ws_nodes = [self.concepts.rep_slot(cid)]

        state_keys = []
        for st in (c_frame.get("states", []) or []):
            if isinstance(st, dict) and _act(st):
                state_keys.append(f"state::{st.get('var','')}::{st.get('subject','')}")
        state_keys = list(dict.fromkeys(state_keys))

        if not state_keys and os.environ.get("CAUSALOS_TARGET_FALLBACK", "1") == "1":
            ents = c_frame.get("entities", []) or []
            subj0 = ents[0] if ents else "input"
            for ev in (c_frame.get("events", []) or [])[:2]:
                if isinstance(ev, dict) and _act(ev):
                    pred = _normalize_text(ev.get("predicate", ""))
                    if not pred:
                        continue
                    var = self.varnorm_main.canonicalize("ev=" + pred[:60])
                    state_keys.append(f"state::{var}::{subj0}")
            state_keys = list(dict.fromkeys(state_keys))

        target_nodes = self._nodes_for_state_keys(state_keys) if state_keys else ws_nodes[:3]
        for tn in target_nodes:
            if tn not in ws_nodes:
                ws_nodes.append(tn)

        ground_avg = float(np.clip(0.5 * (f_ground["avg"] + c_ground["avg"]), 0.0, 1.0))
        ph = float(np.clip(0.5 * (f_fq["placeholder_ratio"] + c_fq["placeholder_ratio"]), 0.0, 1.0))
        dens = float(np.clip(0.5 * (f_fq["density"] + c_fq["density"]), 0.0, 1.0))
        anomaly_score = float(np.clip((1.0 - ground_avg) + 0.9 * ph + 0.4 * (1.0 - dens), 0.0, 2.0))

        mode_guess = self.policy.choose_mode(factual + " " + counterfactual, anomaly_score=anomaly_score)
        beta = self.policy.beta_prior if mode_guess == "OPEN" else 0.0
        Sprior = self._ensure_cache_prior_S() if beta > 0.0 else None
        prior_mask, pm_info = self._build_prior_mask(Sprior) if Sprior is not None else (None, {"nonzero": 0, "topk": 0, "added_to_A": 0})

        scenario_text = (factual + " " + counterfactual).strip()

        def run_once(_Sprior, _pmask, _pm_info):
            self.core.reset_do()
            loc_f = self.localizer.localize(self.core, S_prior=_Sprior, Q=ws_nodes, T=target_nodes, beta_prior=beta, prior_mask=_pmask)
            x_f = loc_f["traj"][-1]

            self.core.reset_do()
            atomic_info = self.atomic_b2.apply(ops, self.core, ws_nodes)

            loc_c = self.localizer.localize(self.core, S_prior=_Sprior, Q=ws_nodes, T=target_nodes, beta_prior=beta, prior_mask=_pmask)
            traj_c = loc_c["traj"]
            x_c = traj_c[-1]

            S_eff = self.core.get_S_eff(beta=beta, S_prior=_Sprior, prior_mask=_pmask)
            u_div = self.impossible.local_divergence(traj_c)
            u_rho = self.impossible.local_spectral_risk(S_eff, loc_c.get("OmegaA_nodes", []))
            u_cst = self.impossible.constraint_violation(traj_c)
            u = self.impossible.combine_u(u_div, u_rho, u_cst)

            predicted_cf = self._collect_predicted_states(state_keys, x_c) if state_keys else {}
            predicted_f = self._collect_predicted_states(state_keys, x_f) if state_keys else {}

            if (not predicted_cf) and os.environ.get("CAUSALOS_LATENT_OPT", "1") == "1":
                for i, tn in enumerate(target_nodes[:3]):
                    if 0 <= tn < x_c.shape[0]:
                        predicted_cf[f"latent::target{i}"] = x_c[tn].detach()
                    if 0 <= tn < x_f.shape[0]:
                        predicted_f[f"latent::target{i}"] = x_f[tn].detach()

            # option scoring (v8+v11 selectable)

            best_opt, opt_scores = (None, {})

            opt_margin = None

            top2 = None

            top1_score = None

            scorer_mode = str(os.environ.get('CAUSALOS_OPT_SCORER', 'likely_yesno')).strip().lower()

            best_gen_pos = 0.0

            best_rel = 0.0

            opt_parts = {}


            best_rel = 0.0

            if options:

                if scorer_mode in ('likely_yesno','yesno','likely'):

                    world_f_txt = self.opt_scorer_likely_b11.world_from_frame(f_frame, raw_text=factual)

                    world_cf_txt = self.opt_scorer_likely_b11.world_from_frame(c_frame, raw_text=counterfactual)

                    best, opt_scores, meta = self.opt_scorer_likely_b11.score(options=options, world_f=world_f_txt, world_cf=world_cf_txt, intervention=ops_sig_text)

                    best_gen_pos = float(meta.get('best_gen_pos', 0.0) or 0.0)

                    best_rel = float(meta.get('best_rel', 0.0) or 0.0)


                    opt_parts = meta.get('parts', {}) if isinstance(meta, dict) else {}
                else:

                    best, opt_scores = self.opt_scorer_b2.score(predicted_cf=predicted_cf, predicted_f=predicted_f, options=options, scenario_text=scenario_text, ops_signature_text=ops_sig_text)

                if opt_scores and len(opt_scores) >= 2:

                    sorted_items = sorted(opt_scores.items(), key=lambda kv: kv[1], reverse=True)

                    top2 = sorted_items[:2]

                    top1_score = float(sorted_items[0][1])

                    opt_margin = float(sorted_items[0][1] - sorted_items[1][1])

                    if opt_margin < 0.0:

                        opt_margin = 0.0

                    if opt_margin >= min_margin:

                        best_opt = best

                    else:

                        best_opt = None

                else:

                    best_opt = best

            # choose target vecs from CF prediction
            target_vecs = [predicted_cf[k] for k in predicted_cf] if predicted_cf else ([x_c[target_nodes[0]]] if target_nodes else [])

            fq = {"placeholder_ratio": ph, "density": dens}
            conf = self._confidence(u=u, target_vecs=target_vecs, opt_margin=opt_margin,
                                    recon_overall=recon_score["overall"], ground_avg=ground_avg, fq=fq)

            expected = max(1, len(state_keys) if state_keys else len(target_nodes))
            coverage = float(np.clip(len(predicted_cf) / expected, 0.0, 1.0))

            return {
                "x_f": x_f, "x_c": x_c,
                "predicted_cf": predicted_cf, "predicted_f": predicted_f,
                "u": u, "coverage": coverage,
                "atomic_info": atomic_info,
                "best_opt": best_opt, "opt_scores": opt_scores,
                "opt_margin": opt_margin, "opt_top2": top2, "opt_top1": top1_score, "opt_scorer_mode": scorer_mode, "opt_best_genpos": best_gen_pos, "opt_best_rel": best_rel, "opt_parts": opt_parts,
                "conf": conf,
                "prior_mask_info": _pm_info
            }

        # first pass
        with WorkspaceGate(self.core) as wg:
            wg.activate_nodes(ws_nodes)
            result = run_once(Sprior, prior_mask, pm_info)

        # Query B trigger (margin gate OR IDS)
        ids_thr = float(os.environ.get("CAUSALOS_IDS_THR", "0.55"))
        budget = int(os.environ.get("CAUSALOS_QUERY_B_BUDGET", "1"))
        enable_qb = os.environ.get("CAUSALOS_ENABLE_QUERY_B", "1") == "1"
        m_thr = float(os.environ.get("CAUSALOS_QB_MARGIN_THR", "0.02"))
        gen_thr = float(os.environ.get("CAUSALOS_QB_GEN_THR", "1.0"))
        rel_thr = float(os.environ.get("CAUSALOS_QB_REL_THR", "0.25"))
        beta_min = float(os.environ.get("CAUSALOS_QB_BETA_MIN", "0.25"))
        margin_now = float(result.get("opt_margin", 0.0) or 0.0)
        best_genpos_now = float(result.get("opt_best_genpos", 0.0) or 0.0)
        best_rel_now = float(result.get("opt_best_rel", 1.0) or 1.0)

        ids = self._compute_ids(
            margin=margin_now,
            ground_min=float(np.clip(min(f_ground["min"], c_ground["min"]), 0.0, 1.0)),
            density=dens,
            coverage=result.get("coverage", 0.0),
            u=result.get("u", 0.0),
        )
        qb_info = {"triggered": False, "ids": ids, "added": 0, "edges": [], "margin_now": margin_now, "m_thr": m_thr}

        if enable_qb and budget > 0 and (margin_now < m_thr or ids >= ids_thr or best_genpos_now > gen_thr or best_rel_now < rel_thr):
            if beta <= 0.0:
                beta = beta_min
            def active_event_texts(fr):
                out = []
                for ev in (fr.get("events", []) or []):
                    if isinstance(ev, dict) and _act(ev):
                        p = _normalize_text(ev.get("predicate", ""))
                        if p:
                            out.append(f"event::{p}")
                return out

            def active_state_texts(fr):
                out = []
                for st in (fr.get("states", []) or []):
                    if isinstance(st, dict) and _act(st):
                        var = _normalize_text(st.get("var", ""))
                        sub = _normalize_text(st.get("subject", ""))
                        if var and sub:
                            out.append(f"state::{var}::{sub}")
                return out

            cause_candidates = []
            effect_candidates = []

            for op in ops:
                if op.get("op") in ("ADD_EVENT", "REMOVE_EVENT"):
                    p = _normalize_text((op.get("payload", {}) or {}).get("predicate", ""))
                    if p:
                        cause_candidates.append(f"event::{p}")
                elif op.get("op") == "SET_STATE":
                    to = (op.get("payload", {}) or {}).get("to", {}) or {}
                    sub = _normalize_text(to.get("subject", ""))
                    var = _normalize_text(to.get("var", ""))
                    if var and sub:
                        cause_candidates.append(f"state::{var}::{sub}")

            cause_candidates += active_event_texts(f_frame) + active_event_texts(c_frame)
            cause_candidates += active_state_texts(f_frame)
            effect_candidates += active_state_texts(c_frame)
            effect_candidates += [k for k in state_keys[:12]]

            def uniq(xs):
                seen = set(); out = []
                for x in xs:
                    t = _normalize_text(x)
                    if not t or t in seen:
                        continue
                    seen.add(t); out.append(t)
                return out

            cause_candidates = uniq(cause_candidates)[:24]
            effect_candidates = uniq(effect_candidates)[:24]

            qb = self.prior_gen.propose(cause_candidates, effect_candidates, context=scenario_text, max_edges=10)
            inj = self._inject_prior_edges(qb.get("edges", []), source_tag="prior_llm")
            qb_info = {"triggered": True, "ids": ids, "query_notes": qb.get("notes", ""), **inj}

            Sprior = self._ensure_cache_prior_S() if beta > 0.0 else None
            prior_mask, pm_info = self._build_prior_mask(Sprior) if Sprior is not None else (None, {"nonzero": 0, "topk": 0, "added_to_A": 0})

            with WorkspaceGate(self.core) as wg:
                wg.activate_nodes(ws_nodes)
                result2 = run_once(Sprior, prior_mask, pm_info)

            # choose better (fixed criterion)
            def key_score(r):
                m = r.get("opt_margin", 0.0) or 0.0
                c = r.get("conf", 0.0) or 0.0
                return float(0.6 * m + 0.4 * c)

            if key_score(result2) >= key_score(result):
                result = result2

        # compose
        lines = []
        lines.append("【反事実推論（CausalOS v5.3_full / robustpack_v8+v11r4）】")
        lines.append(f"確信度: {result['conf']:.2f}")
        lines.append(f"Grounding: factual(avg={f_ground['avg']:.2f},min={f_ground['min']:.2f},try={f_try}) "
                     f"cf(avg={c_ground['avg']:.2f},min={c_ground['min']:.2f},try={c_try})")
        lines.append(f"Grounding(full): factual(min_full={f_ground.get('min_full', 0):.2f}) cf(min_full={c_ground.get('min_full', 0):.2f})")
        lines.append(f"FrameQuality: ph_ratio={ph:.2f}, density={dens:.2f}, anomaly={anomaly_score:.2f}")

        pmi = result.get("prior_mask_info", {"nonzero": 0, "topk": 0, "added_to_A": 0})
        lines.append(f"PriorMask: nonzero={pmi.get('nonzero',0)} topk={pmi.get('topk',0)} added_to_A={pmi.get('added_to_A',0)}")

        lines.append(f"Enforce: factual={f_enf.get('changed',0)} cf={c_enf.get('changed',0)} | "
                     f"Dedup: f_incl={f_ddi.get('changed',0)} f_emb={f_dde.get('changed',0)} "
                     f"c_incl={c_ddi.get('changed',0)} c_emb={c_dde.get('changed',0)} | "
                     f"IDS={ids:.2f} QB={int(qb_info.get('triggered',False))} QB_added={qb_info.get('added',0)}")

        top1 = result.get('opt_top1', None)

        mrg = float(result.get('opt_margin', 0.0) or 0.0)

        smode = str(result.get('opt_scorer_mode', 'contrast')).strip()

        gpos = float(result.get('opt_best_genpos', 0.0) or 0.0)

        relv = float(result.get('opt_best_rel', 0.0) or 0.0)

        lines.append('Score: top1={} margin={:.3f} scorer={} gen_pos={:.2f} rel={:.2f}'.format(top1 if top1 is not None else 'na', mrg, smode, gpos, relv))

        lines.append(f"再構成スコア: overall={recon_score['overall']:.2f} (ev={recon_score['ev_jacc']:.2f}, st={recon_score['st_acc']:.2f})")
        lines.append("")
        lines.append("推定された介入（IR）:")
        for op in ops[:12]:
            lines.append(f"- {op.get('op')}: {str(op.get('payload', {}))[:180]}")

        if options:
            # OPTS debug (single-line; grep-friendly)
            try:
                parts = result.get('opt_parts', {}) or {}
                items = []
                for lab in sorted(list(options.keys())):
                    sc = float((result.get('opt_scores', {}) or {}).get(lab, 0.0))
                    pr = parts.get(lab, {}) if isinstance(parts, dict) else {}
                    lik_cf = float(pr.get('lik_cf', 0.0))
                    lik_f = float(pr.get('lik_f', 0.0))
                    genp = float(pr.get('gen_pos', 0.0))
                    relv = float(pr.get('rel', 0.0))
                    cfterm = float(pr.get('cf_term', 0.0))
                    items.append(f"{lab}:sc={sc:.3f},rel={relv:.2f},gen={genp:.2f},lik_cf={lik_cf:.2f},lik_f={lik_f:.2f},cfT={cfterm:.2f}")
                lines.append('OPTS: ' + ' | '.join(items)[:900])
            except Exception:
                pass

            lines.append("")
            if result.get("best_opt"):
                lines.append(f"【選択肢との整合】最も整合する候補: {result['best_opt']} : {options.get(result['best_opt'],'')}")
            else:
                if result.get("opt_top2"):
                    a, b = result["opt_top2"][0], result["opt_top2"][1]
                    lines.append(f"【選択肢との整合】僅差で拮抗（margin={float(result.get('opt_margin',0.0) or 0.0):.3f} < {min_margin:.3f}）:")
                    lines.append(f"- 1位 {a[0]}: {options.get(a[0],'')} (score={a[1]:.3f})")
                    lines.append(f"- 2位 {b[0]}: {options.get(b[0],'')} (score={b[1]:.3f})")

        need_q = []
        if recon_score["overall"] < 0.55 or ground_avg < thr:
            need_q = [
                "結果として知りたい状態を1つだけ明示できますか？（例：旅が終わる/火傷の有無など）",
                "反実で固定する要素（不変）と変更する要素（介入）を短く区別できますか？"
            ]
            lines.append("")
            lines.append("より正確な回答のため、次を教えてください（短くでOK）:")
            for i, q in enumerate(need_q[:3], 1):
                lines.append(f"{i}) {q}")

        mode = "ANSWER" if result["conf"] >= 0.80 and not need_q else ("TENTATIVE" if need_q else "ANSWER")

        trace = {
            "build_id": BUILD_ID,
            "ops_signature_text": ops_sig_text,
            "ids": ids,
            "queryB": qb_info,
            "prior_mask_info": pmi,
            "opt_scores": result.get("opt_scores", {}),
            "opt_margin": result.get("opt_margin", None),
            "best_opt": result.get("best_opt", None),
            "grounding": {"factual": f_ground, "counterfactual": c_ground, "thr": thr},
        }
        if os.environ.get("CAUSALOS_TRACE_FRAMES", "1") == "1":
            trace["frames_head"] = {"factual": _frame_head(f_frame), "counterfactual": _frame_head(c_frame)}

        return AnswerPacket("\n".join(lines), float(result["conf"]), need_q[:3], trace, mode)