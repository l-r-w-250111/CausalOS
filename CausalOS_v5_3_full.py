# -*- coding: utf-8 -*-
"""
CausalOS_v5_3_full.py (2026-02-09)

v5.3_full = v5.2 (full components) + B2 (FRAME-based IR diff including EVENTS) ADD-ONLY.

User constraints:
- Do NOT remove previously introduced concepts/components. If concerns exist, use flags / comment-out.
- Avoid keyword-based intervention-type classification. (No "if contains not => NEGATION" etc.)
- Preserve LLM flexibility; add certainty via causal memory + computation.
- Do-intervention should be explicit (before/after), but without brittle keyword typing.

This file includes:
- ConceptBank (embedding centroid + variable slots; grow only)
- EdgeBank strong/prior with representative-slot projection to core
- CausalCoreV5: complex-like propagation (real/imag), raw_S/raw_phase/raw_r, A_mask/G_gate, do clamps
- OmegaLocalizer: contrib + reachability + gradient (always-on), horizon fixed by config
- ImpossibilityController: local u -> hardening gate (soft->hard continuous)
- Structured facts + heuristic injection (kept; can be disabled by flags)
- LLM causal triplet extraction (kept; can be disabled by flags)
- WorkspaceGate: per-question local gating to avoid cross-question interference (added; can be disabled)
- B2 addition:
  - FrameExtractorLLM (structured semantic parsing)
  - InterventionIR diff (states + events + constraints)
  - AtomicMapper (deterministic mapping IR -> do_cut_in/do_value + optional phase operations; flagged)
  - OptionScorer (frame-based similarity; no keyword hardcoding)

Flags (env vars):
- CAUSALOS_DISABLE_HEURISTICS=1 : disable deterministic heuristic edge injection
- CAUSALOS_DISABLE_STRUCTURED_FACTS=1 : disable structured fact ingestion
- CAUSALOS_NO_LLM_GRAPH=1 : disable LLM causal triplet extraction (graph)
- CAUSALOS_NO_LLM_FRAME=1 : disable LLM frame extraction (B2)
- CAUSALOS_DISABLE_WORKSPACE=1 : disable per-question workspace gating
- CAUSALOS_ENABLE_EVENT_PHASE=1 : enable event REORDER phase operations (experimental; off by default)
- CAUSALOS_SHOW_TRACE=1 : print reason_trace (agent side)
"""

from __future__ import annotations

import os
import re
import json
import math
import time
from dataclasses import dataclass
from collections import defaultdict, deque
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

BUILD_ID = "2026-02-09-v5.3_full"

# -----------------------------
# Device
# -----------------------------
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
                return t[start:i+1]
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
                    return t[start:i+1]
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

def _is_question_like(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if "?" in t or "？" in t:
        return True
    if re.search(r"(ですか|ますか|でしょうか|かな)\s*$", t):
        return True
    if re.search(r"\b(what|why|how|which|who|when|where)\b", t, re.IGNORECASE):
        return True
    return False

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

def _strip_options_block(text: str) -> str:
    t = _normalize_text(text)
    m = re.search(r'(\s|^)([A-Z])\s*:\s*', t)
    if m:
        return t[:m.start()].strip()
    return t

def _is_bad_label(lab: str) -> bool:
    lab = _norm_label(lab)
    if not lab:
        return True
    if lab in {"a","b","c","d","e","f"}:
        return True
    if len(lab) <= 1:
        return True
    return False


# ==========================================================
# Answer protocol
# ==========================================================
@dataclass
class AnswerPacket:
    best_effort_answer: str
    confidence: float
    need_info_questions: List[str]
    reason_trace: Dict[str, Any]
    mode: str  # "ANSWER" | "TENTATIVE" | "NEED_INFO" | "ABSTAIN"


# ==========================================================
# Knowledge Policy
# ==========================================================
class KnowledgePolicy:
    def __init__(self, beta_prior: float = 0.25):
        self.beta_prior = float(beta_prior)

    def choose_mode(self, user_text: str, anomaly_score: float = 0.0) -> str:
        if _is_exact_fact_task(user_text) or _contains_fact_like_patterns(user_text):
            return "VERIFY_REQUIRED"
        if anomaly_score >= 1.0:
            return "CAUSAL_ONLY"
        return "OPEN"


# ==========================================================
# ConceptBank
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
            emb_layer = self.osys.model.get_input_embeddings()
            embs = emb_layer(ids)[0]
            v = embs.mean(dim=0).float().detach().cpu()
        return v

    def _alloc_slots(self, k: int) -> List[int]:
        return [self.osys._alloc_node() for _ in range(int(k))]

    def resolve(self, label: Any) -> int:
        lab = _norm_label(label)
        if _is_bad_label(lab):
            lab = f"concept_{hash(str(label))%100000}"

        if lab in self.alias_to_cid:
            return self.alias_to_cid[lab]

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
        self.concepts[cid] = {
            "cid": cid,
            "emb": v.float(),
            "aliases": set([lab]),
            "slots": slots,
            "capacity": len(slots),
            "created_ts": _now_ts(),
            "last_used_ts": _now_ts(),
            "usage": 0,
            "conflicts": 0,
        }
        self.alias_to_cid[lab] = cid
        return cid

    def touch(self, cid: int):
        if cid in self.concepts:
            self.concepts[cid]["last_used_ts"] = _now_ts()
            self.concepts[cid]["usage"] += 1

    def slots_of(self, cid: int) -> List[int]:
        return list(self.concepts[cid]["slots"])

    def rep_slot(self, cid: int) -> int:
        slots = self.slots_of(cid)
        return int(slots[0]) if slots else 0

    def label_of(self, cid: int) -> str:
        if cid not in self.concepts:
            return f"concept_{cid}"
        aliases = sorted(list(self.concepts[cid]["aliases"]))
        aliases = [a for a in aliases if len(a) >= 2 and a not in {"a","b","c","d"}]
        return aliases[0] if aliases else f"concept_{cid}"

    def maybe_grow(self, cid: int, reason: str = "", add: int = 2):
        if cid not in self.concepts:
            return
        c = self.concepts[cid]
        new_slots = self._alloc_slots(int(add))
        c["slots"].extend(new_slots)
        c["capacity"] = len(c["slots"])
        c["conflicts"] += 1
        print(f"[ConceptBank] Grew concept {cid} slots +{add} ({reason}), now {c['capacity']}")


# ==========================================================
# EdgeBank (strong/prior)
# ==========================================================
class EdgeBank:
    def __init__(self):
        self.strong: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.prior: Dict[Tuple[int, int], Dict[str, Any]] = {}

    def _update(self, store: Dict, e: int, c: int, m: float, w: float, source: str):
        m = _clip_mag(m)
        w = float(max(0.0, w))
        key = (e, c)
        rec = store.get(key)
        if rec is None:
            store[key] = {"m": float(m), "w": float(w), "src": defaultdict(float), "ts": _now_ts()}
            store[key]["src"][source] += w
        else:
            m_old = float(rec["m"])
            w_old = float(rec["w"])
            m_new = (m_old * w_old + m * w) / max(w_old + w, 1e-6)
            rec["m"] = float(m_new)
            rec["w"] = float(w_old + w)
            rec["src"][source] += w
            rec["ts"] = _now_ts()

    def update_edge(self, effect_cid: int, cause_cid: int, m: float, w: float,
                    source: str = "user", layer: str = "strong"):
        if layer == "strong":
            self._update(self.strong, effect_cid, cause_cid, m, w, source)
        else:
            self._update(self.prior, effect_cid, cause_cid, m, w, source)


# ==========================================================
# CausalCoreV5 (complex-like propagation)
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

        # structural mask A sparse init, dynamic gate G open
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

    def get_S_eff(self, beta: float = 0.0, S_prior: Optional[torch.Tensor] = None) -> torch.Tensor:
        S = torch.tanh(self.raw_S)
        if S_prior is not None and beta > 0.0:
            S = torch.clamp(S + beta * S_prior, -0.99, 0.99)
        r = torch.sigmoid(self.raw_r)
        Aamp = self.A_mask * self.G_gate * S * r
        if self.do_cut_in:
            Aamp = Aamp.clone()
            for j in self.do_cut_in:
                if 0 <= j < self.n_nodes:
                    Aamp[j, :].fill_(0.0)
        return Aamp

    def step(self, x: torch.Tensor, t: int, beta: float = 0.0, S_prior: Optional[torch.Tensor] = None) -> torch.Tensor:
        n = self.n_nodes
        x_real = x[:, 0].view(1, n)
        x_imag = x[:, 1].view(1, n)

        Aamp = self.get_S_eff(beta=beta, S_prior=S_prior)
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
                require_grad: bool = False) -> torch.Tensor:
        if x0 is None:
            x = self.x if require_grad else self.x.detach()
        else:
            x = x0 if require_grad else x0.detach()

        traj = [x]
        for t in range(int(steps)):
            x = self.step(x, t=t, beta=beta, S_prior=S_prior)
            traj.append(x)
        return torch.stack(traj, dim=0)


# ==========================================================
# WorkspaceGate (ADD-ONLY)
# ==========================================================
class WorkspaceGate:
    """
    Per-question workspace to avoid cross-question interference:
    - Save A_mask/G_gate
    - Reset A_mask to self-loops and re-open only edges among active nodes that exist in saved A_mask
    - Restore after
    """
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
# OmegaLocalizer (kept, always-on)
# ==========================================================
class OmegaLocalizer:
    def __init__(self, horizon: int = 10,
                 w0: float = 0.7, w1: float = 0.3,
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
            edges.append((j, i, float(v)))  # i -> j
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
                 Q: List[int], T: List[int], beta_prior: float = 0.0) -> Dict[str, Any]:
        n = core.n_nodes

        core.zero_grad(set_to_none=True)
        traj = core.rollout(steps=self.horizon, x0=core.x, beta=beta_prior, S_prior=S_prior, require_grad=True)
        xT = traj[-1]

        # loss = sum_T (w0*x_T.real + w1*||x_T||)
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

        S_eff = core.get_S_eff(beta=beta_prior, S_prior=S_prior)

        src = torch.norm(xT.detach(), dim=-1)  # [n]
        contrib = S_eff.detach().abs() * src.view(1, n)  # [j,i]

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

        return {
            "Omega_edges": Omega_edges,
            "traj": traj.detach(),
            "loss": float(loss.detach().item()),
            "OmegaA_nodes": list(sorted(OmegaA_nodes)),
            "scores": {"contrib": cN.detach(), "reach": rN.detach(), "grad": gN.detach(), "combined": combined.detach()},
        }


# ==========================================================
# ImpossibilityController (kept)
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

    def g(self, u: float) -> float:
        return self._sigmoid(self.kappa * (self.tau - float(u)))

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

    def apply_local_gate(self, core: CausalCoreV5, Omega_edges: List[Tuple[int, int, float]], gval: float):
        with torch.no_grad():
            for j, i, _ in Omega_edges:
                if 0 <= j < core.n_nodes and 0 <= i < core.n_nodes:
                    core.G_gate[j, i] *= float(gval)


# ==========================================================
# LLM causal triplet extraction (kept)
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
Decompose into atomic 'Cause -> Effect' pairs with magnitude (-1.0..1.0).
Return ONLY a JSON array (<= {max_triplets} items).
Rules:
- Do NOT use option labels like "A/B/C/D" as causes/effects.
- Use meaningful phrases (>=2 characters).

Text: "{text}"
JSON:"""

        tok = self.osys.tokenizer(str(prompt), return_tensors="pt")
        tok = {k: v.to(self.osys.model_device) for k, v in tok.items()}

        with torch.no_grad():
            out = self.osys.model.generate(
                **tok,
                max_new_tokens=260,
                do_sample=False,
                pad_token_id=self.osys.tokenizer.eos_token_id,
            )
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
# B2: Frame extractor (LLM) - ADD-ONLY
# ==========================================================
class FrameExtractorLLM:
    """
    Extract a semantic frame. This is NOT used for keyword classification.
    Output schema is fixed; OS handles diff deterministically.
    """
    def __init__(self, osys: "UnifiedCausalOSV5_3Full"):
        self.osys = osys

    def extract_frame(self, text: str, kind: str = "generic") -> Dict[str, Any]:
        text = _normalize_text(text)
        if not text:
            return {"entities": [], "events": [], "states": [], "constraints": [], "notes": ""}

        if os.environ.get("CAUSALOS_NO_LLM_FRAME", "0") == "1":
            # minimal fallback
            return {"entities": [], "events": [{"id":"e0","predicate":text,"args":[]}], "states": [], "constraints": [], "notes":"fallback"}

        prompt = f"""You are a semantic parser.
Extract a compact structured FRAME from the input text.

Return ONLY valid JSON with this schema:
{{
  "entities": ["..."],
  "events": [
    {{"id":"e1","predicate":"...", "args":[{{"role":"agent|patient|theme|location|instrument|other","value":"..."}}, ...],
      "time":"", "order":0, "polarity":"pos|neg", "modality":"asserted|hypothetical|counterfactual|unknown"}}
  ],
  "states": [
    {{"var":"...", "subject":"...", "value":"...", "polarity":"pos|neg", "modality":"asserted|hypothetical|counterfactual|unknown"}}
  ],
  "constraints": [
    {{"type":"cannot|must|may|unknown", "statement":"..."}}
  ],
  "notes":""
}}

Rules:
- Do NOT copy option labels (A/B/C/D) as entities/states.
- Do not invent facts not stated.
- Keep it short but include the core changed elements if implied.
- Output JSON only.

Input ({kind}): "{text}"
JSON:"""

        tok = self.osys.tokenizer(str(prompt), return_tensors="pt")
        tok = {k: v.to(self.osys.model_device) for k, v in tok.items()}
        with torch.no_grad():
            out = self.osys.model.generate(
                **tok,
                max_new_tokens=360,
                do_sample=False,
                pad_token_id=self.osys.tokenizer.eos_token_id
            )
        resp = self.osys.tokenizer.decode(out[0][tok["input_ids"].shape[-1]:], skip_special_tokens=True)
        js = _extract_first_json_obj(resp)
        if not js:
            return {"entities": [], "events": [{"id":"e0","predicate":text,"args":[]}], "states": [], "constraints": [], "notes":"parse_fail"}

        try:
            obj = json.loads(js)
        except Exception:
            return {"entities": [], "events": [{"id":"e0","predicate":text,"args":[]}], "states": [], "constraints": [], "notes":"json_fail"}

        # sanitize
        obj["entities"] = obj.get("entities") if isinstance(obj.get("entities"), list) else []
        obj["events"] = obj.get("events") if isinstance(obj.get("events"), list) else []
        obj["states"] = obj.get("states") if isinstance(obj.get("states"), list) else []
        obj["constraints"] = obj.get("constraints") if isinstance(obj.get("constraints"), list) else []
        obj["notes"] = str(obj.get("notes", ""))

        obj["entities"] = [e for e in obj["entities"] if not _is_bad_label(e)]
        # drop obviously bad states
        fixed_states = []
        for s in obj["states"]:
            if isinstance(s, dict):
                if not _is_bad_label(str(s.get("var", ""))):
                    fixed_states.append(s)
        obj["states"] = fixed_states

        # ensure event order integer
        fixed_events = []
        for e in obj["events"]:
            if isinstance(e, dict):
                if _is_bad_label(str(e.get("predicate", ""))):
                    continue
                if "order" not in e:
                    e["order"] = 0
                try:
                    e["order"] = int(e["order"])
                except Exception:
                    e["order"] = 0
                fixed_events.append(e)
        obj["events"] = fixed_events

        return obj


# ==========================================================
# B2: IR diff (states + events + constraints) - deterministic
# ==========================================================
class InterventionIR_B2:
    @staticmethod
    def diff_frames(factual: Dict[str, Any], counterfactual: Dict[str, Any]) -> List[Dict[str, Any]]:
        ops: List[Dict[str, Any]] = []

        # --- States (same key = var+subject) ---
        f_states = factual.get("states", []) or []
        c_states = counterfactual.get("states", []) or []

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
            key = (var2, sub2)
            s1 = f_map.get(key)
            if s1 is None:
                ops.append({"op": "SET_STATE", "payload": {"from": None, "to": s2}})
            else:
                used.add(key)
                if (_norm_label(s1.get("value", "")) != _norm_label(s2.get("value", "")) or
                    _norm_label(s1.get("polarity", "")) != _norm_label(s2.get("polarity", "")) or
                    _norm_label(s1.get("modality", "")) != _norm_label(s2.get("modality", ""))):
                    ops.append({"op": "SET_STATE", "payload": {"from": s1, "to": s2}})

        for key, s1 in f_map.items():
            if key not in used:
                ops.append({"op": "UNSET_STATE", "payload": {"state": s1}})

        # --- Events (B2): add/remove + reorder detection (no keyword typing) ---
        f_events = factual.get("events", []) or []
        c_events = counterfactual.get("events", []) or []

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

        # Reorder: if same set but different order sequence
        f_order = [ev_sig(e)[0] for e in sorted(f_events, key=lambda x: int(x.get("order", 0))) if ev_sig(e)[0]]
        c_order = [ev_sig(e)[0] for e in sorted(c_events, key=lambda x: int(x.get("order", 0))) if ev_sig(e)[0]]
        if f_order and c_order and set(f_order) == set(c_order) and f_order != c_order:
            ops.append({"op": "REORDER_EVENTS", "payload": {"from": f_order, "to": c_order}})

        # --- Constraints / modality ---
        for con in (counterfactual.get("constraints", []) or []):
            if isinstance(con, dict):
                ops.append({"op": "MODALITY", "payload": {"type": con.get("type", "unknown"), "statement": con.get("statement", "")}})

        if not ops:
            ops = [{"op": "NOOP", "payload": {}}]
        return ops


# ==========================================================
# B2: Atomic mapping (deterministic; no keyword type classification)
# ==========================================================
class AtomicMapper_B2:
    """
    Map IR ops to atomic operations on core:
    - SET_STATE: do_cut_in(node), do_value(node)=proj(embed(value))
    - ADD_EVENT: create event node and clamp it to +1-ish
    - REMOVE_EVENT: create event node and clamp it to 0
    - REORDER_EVENTS: optional phase shift among event edges (EXPERIMENTAL; off by default)
    - MODALITY: record; may influence impossibility gate by raising u (handled in caller)
    """
    def __init__(self, osys: "UnifiedCausalOSV5_3Full"):
        self.osys = osys

    def _state_key(self, s: Dict[str, Any]) -> str:
        var = _normalize_text(s.get("var", ""))
        sub = _normalize_text(s.get("subject", ""))
        return f"state::{var}::{sub}".strip()

    def _event_key(self, pred: str) -> str:
        pred = _normalize_text(pred)
        return f"event::{pred}".strip()

    def _value_to_vec2(self, value: str, polarity: str) -> torch.Tensor:
        value = _normalize_text(value)
        pol = _norm_label(polarity)

        tok = self.osys.tokenizer(str(value), return_tensors="pt", add_special_tokens=False)
        ids = tok["input_ids"].to(self.osys.model_device)
        with torch.no_grad():
            emb_layer = self.osys.model.get_input_embeddings()
            embs = emb_layer(ids)[0]
            v = embs.mean(dim=0).float().detach().to(device)

        v2 = (self.osys._proj_W @ v.view(-1, 1)).view(2)
        v2 = torch.tanh(v2)
        if pol == "neg":
            v2 = -v2
        return v2.detach()

    def apply(self, ops: List[Dict[str, Any]], core: CausalCoreV5, workspace_nodes: List[int]) -> Dict[str, Any]:
        info = {"clamped": [], "cut_in": [], "events": [], "modality": [], "reorder": None}
        enable_phase = os.environ.get("CAUSALOS_ENABLE_EVENT_PHASE", "0") == "1"

        for op in ops:
            kind = op.get("op")
            payload = op.get("payload", {}) or {}

            if kind == "SET_STATE":
                s2 = payload.get("to", {}) or {}
                key = self._state_key(s2)
                cid = self.osys.concepts.resolve(key)
                self.osys.concepts.touch(cid)
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
                # activate event node
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
                # deactivate event node
                core.apply_do_cut_in(node)
                core.apply_do_value(node, 0.0, 0.0)
                info["events"].append({"remove": key, "node": node})

            elif kind == "REORDER_EVENTS":
                info["reorder"] = payload
                # Optional phase tweak among event edges (experimental).
                if enable_phase:
                    # small deterministic phase shift for edges between reordered event nodes
                    from_list = payload.get("from", []) or []
                    to_list = payload.get("to", []) or []
                    # map predicate -> node
                    node_map = {}
                    for pred in set(from_list) | set(to_list):
                        cid = self.osys.concepts.resolve(self._event_key(pred))
                        node_map[pred] = self.osys.concepts.rep_slot(cid)
                    with torch.no_grad():
                        # apply pi/8 shift to edges between consecutive "to" events
                        for a, b in zip(to_list, to_list[1:]):
                            ia = node_map.get(a, None)
                            ib = node_map.get(b, None)
                            if ia is None or ib is None:
                                continue
                            if 0 <= ib < core.n_nodes and 0 <= ia < core.n_nodes and ib != ia:
                                core.raw_phase.data[ib, ia] += (math.pi / 8.0)

            elif kind == "MODALITY":
                info["modality"].append(payload)

            else:
                pass

        return info


# ==========================================================
# Option scoring (frame-based, no keyword hardcoding)
# ==========================================================
class OptionScorer_B2:
    def __init__(self, osys: "UnifiedCausalOSV5_3Full"):
        self.osys = osys

    def _embed_text(self, text: str) -> torch.Tensor:
        tok = self.osys.tokenizer(str(text), return_tensors="pt", add_special_tokens=False)
        ids = tok["input_ids"].to(self.osys.model_device)
        with torch.no_grad():
            emb_layer = self.osys.model.get_input_embeddings()
            embs = emb_layer(ids)[0]
            v = embs.mean(dim=0).float().detach().to(device)
        return v

    def score(self, predicted_states: Dict[str, torch.Tensor], options: Dict[str, str]) -> Tuple[Optional[str], Dict[str, float]]:
        if not options:
            return None, {}

        pred_items = [(k, v.detach().cpu().tolist()) for k, v in predicted_states.items()]
        pred_summary = json.dumps({"predicted_states": pred_items}, ensure_ascii=False)
        v_pred = self._embed_text(pred_summary)

        scores: Dict[str, float] = {}
        for k, text in options.items():
            frame = self.osys.frames.extract_frame(text, kind="option")
            opt_summary = json.dumps(frame, ensure_ascii=False)
            v_opt = self._embed_text(opt_summary)
            scores[k] = _cosine(v_pred, v_opt)

        best = max(scores.items(), key=lambda kv: kv[1])[0] if scores else None
        return best, scores


# ==========================================================
# Unified OS v5.3_full
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
    ):
        print(f"[CausalOS v5.3_full] BUILD_ID={BUILD_ID}", flush=True)
        print(f"[CausalOS v5.3_full] Loading model: {model_id}", flush=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map="auto")

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # deterministic config to avoid transformers warnings
        try:
            gc = self.model.generation_config
            gc.do_sample = False
            gc.temperature = None
            gc.top_p = None
            gc.top_k = None
        except Exception:
            pass

        self.policy = KnowledgePolicy(beta_prior=0.25)
        self.edge_bank = EdgeBank()

        self.core = CausalCoreV5(n_nodes=init_n_nodes, p_r0=0.20).to(device)
        self.expand_chunk = int(expand_chunk)
        self._n_used = 0

        # concept bank
        self.concepts = ConceptBank(self, init_slots_per_concept=init_slots_per_concept, sim_base_threshold=0.82, expand_chunk=expand_chunk)

        # projection matrix for value->2D (deterministic)
        self._proj_W = self._init_projection_matrix()

        # memory
        self._recent_context = deque(maxlen=24)

        # answered facts (kept; can be disabled)
        self.answered_facts = {
            "barrier_solid": False,
            "talk_includes_approach": False,
            "predator_unpredictable": False,
        }

        # lexical anchors for heuristics (kept; can be disabled)
        self.lex = {
            "lion": ["lion", "ライオン", "獅子"],
            "human": ["man", "human", "person", "人", "男性", "男"],
            "barrier": ["barrier", "隔たり", "柵", "ガラス", "壁", "安全な距離", "distance", "距離"],
            "proximity": ["close", "near", "近く", "近い", "接近", "近づく"],
            "harm": ["eaten", "attack", "injury", "harm", "致命", "傷害", "襲", "食べ", "噛", "殺", "burn", "burned", "burnt"],
            "no_barrier": ["without a barrier", "no barrier", "隔たりが無い", "隔たりがない", "柵がない", "バリアがない"],
        }

        # cache prior S
        self._cache_prior_S: Optional[torch.Tensor] = None
        self._cache_prior_version = 0
        self._prior_version = 0

        # engines (kept)
        self.triplets = CausalTripletExtractor(self)
        self.localizer = OmegaLocalizer(horizon=local_horizon, w0=w0, w1=w1)
        self.impossible = ImpossibilityController(kappa=10.0, tau=0.65)

        # B2 additions (ADD-ONLY)
        self.frames = FrameExtractorLLM(self)
        self.ir_b2 = InterventionIR_B2()
        self.atomic_b2 = AtomicMapper_B2(self)
        self.opt_scorer_b2 = OptionScorer_B2(self)

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

    # -------------------------
    # Node allocation
    # -------------------------
    def _alloc_node(self) -> int:
        if self._n_used >= self.core.n_nodes:
            new_n = self.core.n_nodes + self.expand_chunk
            print(f"[CausalOS v5.3_full] Expanding n_nodes: {self.core.n_nodes} -> {new_n}", flush=True)
            self.core.resize(new_n, p_r0=0.20)
        idx = int(self._n_used)
        self._n_used += 1
        return idx

    # -------------------------
    # Prior projection (rep-slot)
    # -------------------------
    def _ensure_cache_prior_S(self) -> torch.Tensor:
        if self._cache_prior_S is not None and self._cache_prior_version == self._prior_version:
            return self._cache_prior_S

        n = self.core.n_nodes
        Sprior = torch.zeros(n, n, device=device)

        for (e_cid, c_cid), rec in self.edge_bank.prior.items():
            m = float(rec["m"])
            ej = self.concepts.rep_slot(e_cid)
            ci = self.concepts.rep_slot(c_cid)
            if 0 <= ej < n and 0 <= ci < n and ej != ci:
                Sprior[ej, ci] += float(m)

        Sprior = torch.clamp(Sprior, -0.99, 0.99)
        self._cache_prior_S = Sprior
        self._cache_prior_version = self._prior_version
        return Sprior

    # -------------------------
    # Structured facts ingestion (kept)
    # -------------------------
    def _ingest_structured_facts(self, text: str, weight: float = 0.9):
        if os.environ.get("CAUSALOS_DISABLE_STRUCTURED_FACTS", "0") == "1":
            return
        t = _norm_label(text)

        # barrier solid
        if ("バリア" in t or "隔たり" in t or "barrier" in t) and ("完全" in t or "物理" in t or "ガラス" in t or "壁" in t):
            self.answered_facts["barrier_solid"] = True
            cid_bar = self.concepts.resolve("barrier")
            cid_prox = self.concepts.resolve("proximity_without_barrier")
            cid_harm = self.concepts.resolve("human_harmed")
            self.edge_bank.update_edge(cid_prox, cid_bar, m=-0.9, w=weight, source="user", layer="strong")
            self.edge_bank.update_edge(cid_harm, cid_bar, m=-0.8, w=weight, source="user", layer="strong")

        # talk includes approach
        if ("話しかけ" in t or "talk" in t) and ("近づく" in t or "接近" in t or "approach" in t):
            self.answered_facts["talk_includes_approach"] = True
            cid_talk = self.concepts.resolve("talk_to_lion")
            cid_prox = self.concepts.resolve("proximity_without_barrier")
            self.edge_bank.update_edge(cid_prox, cid_talk, m=+0.7, w=0.8 * weight, source="user", layer="strong")

        # predator unpredictable
        if ("猛獣" in t or "dangerous" in t or "predator" in t) and ("どうであれ" in t or "状態はどう" in t or "何が起こるかわから" in t):
            self.answered_facts["predator_unpredictable"] = True
            cid_lion = self.concepts.resolve("lion")
            cid_harm = self.concepts.resolve("human_harmed")
            self.edge_bank.update_edge(cid_harm, cid_lion, m=+0.6, w=0.7 * weight, source="user", layer="strong")

    # -------------------------
    # Deterministic heuristics (kept; disable via flag)
    # -------------------------
    def _inject_heuristic_edges(self, text: str, source: str, weight: float):
        if os.environ.get("CAUSALOS_DISABLE_HEURISTICS", "0") == "1":
            return
        t = _norm_label(text)
        if not t:
            return

        has_lion = any(w in t for w in self.lex["lion"])
        has_harm = any(w in t for w in self.lex["harm"]) or ("傷害" in t) or ("致命" in t)
        has_prox = any(w in t for w in self.lex["proximity"])
        has_bar = any(w in t for w in self.lex["barrier"])
        has_no_bar = any(w in t for w in self.lex["no_barrier"])

        cid_prox = self.concepts.resolve("proximity_without_barrier")
        cid_harm = self.concepts.resolve("human_harmed")
        cid_bar = self.concepts.resolve("barrier")
        cid_lion = self.concepts.resolve("lion")
        cid_talk = self.concepts.resolve("talk_to_lion")

        # lion-like
        if has_lion and (has_prox or has_no_bar) and has_harm:
            self.edge_bank.update_edge(cid_harm, cid_prox, m=+0.95, w=weight, source=source, layer="strong")

        if has_bar and ("安全" in t or "距離" in t or "隔たり" in t or "barrier" in t):
            self.edge_bank.update_edge(cid_prox, cid_bar, m=-0.85, w=0.8 * weight, source=source, layer="strong")
            self.edge_bank.update_edge(cid_harm, cid_bar, m=-0.75, w=0.7 * weight, source=source, layer="strong")

        if has_lion and ("talk" in t or "話しかけ" in t):
            self.edge_bank.update_edge(cid_prox, cid_talk, m=+0.55, w=0.5 * weight, source=source, layer="strong")

        if has_lion:
            self.edge_bank.update_edge(cid_harm, cid_lion, m=+0.35, w=0.3 * weight, source=source, layer="strong")

        # fire-like: we keep as heuristic concept injection ONLY if heuristics enabled
        # (still not "intervention type classification"; just a stable causal prior for common physics)
        if "fire" in t or "火" in t:
            cid_touch = self.concepts.resolve("touch_fire")
            cid_burn = self.concepts.resolve("burned")
            if "touch" in t or "触れ" in t or "触っ" in t:
                self.edge_bank.update_edge(cid_burn, cid_touch, m=+0.9, w=0.6*weight, source=source, layer="strong")

    # -------------------------
    # Project strong edges to core (rep-slot, no dilution)
    # -------------------------
    def _project_strong_edges_to_core(self):
        n = self.core.n_nodes
        with torch.no_grad():
            for (e_cid, c_cid), rec in self.edge_bank.strong.items():
                m = float(rec["m"])
                ej = self.concepts.rep_slot(e_cid)
                ci = self.concepts.rep_slot(c_cid)
                if ej >= n or ci >= n or ej == ci:
                    continue

                target = float(m)
                val = _safe_tanh_inv(target)

                # fast adopt (user facts)
                self.core.raw_S.data[ej, ci] = 0.7 * self.core.raw_S.data[ej, ci] + 0.3 * val
                self.core.A_mask[ej, ci] = 1.0

                rr = float(np.clip(abs(target), 0.25, 0.95))
                self.core.raw_r.data[ej, ci] = 0.7 * self.core.raw_r.data[ej, ci] + 0.3 * math.log(rr / (1 - rr))

    # -------------------------
    # Ingest context (kept)
    # -------------------------
    def ingest_context(self, text: Any, source: str = "user", weight: float = 0.85, allow_llm_prior: bool = True):
        text = _normalize_text(text)
        if not text:
            return

        self._recent_context.append({"role": source, "content": text})

        if source == "user":
            self._ingest_structured_facts(text, weight=float(weight))
            self._inject_heuristic_edges(text, source=source, weight=float(weight))

        # LLM triplets for graph update (strip options)
        clean_text = _strip_options_block(text)
        triplets = self.triplets.extract(clean_text)

        if triplets:
            for tr in triplets:
                c_label = tr["cause"]
                e_label = tr["effect"]
                m = float(tr["magnitude"])
                if _is_bad_label(c_label) or _is_bad_label(e_label):
                    continue
                c_cid = self.concepts.resolve(c_label)
                e_cid = self.concepts.resolve(e_label)
                self.concepts.touch(c_cid); self.concepts.touch(e_cid)

                if source == "user":
                    self.edge_bank.update_edge(e_cid, c_cid, m=m, w=float(weight), source=source, layer="strong")
                else:
                    if allow_llm_prior:
                        self.edge_bank.update_edge(e_cid, c_cid, m=m, w=float(weight), source=source, layer="prior")
                        self._prior_version += 1
                        self._cache_prior_version = -1

        self._project_strong_edges_to_core()

    # -------------------------
    # Workspace nodes from frame (B2) - ADD-ONLY
    # -------------------------
    def _workspace_nodes_from_frame(self, frame: Dict[str, Any]) -> List[int]:
        nodes: List[int] = []
        for ent in frame.get("entities", []) or []:
            cid = self.concepts.resolve(ent)
            nodes.append(self.concepts.rep_slot(cid))
        for ev in frame.get("events", []) or []:
            pred = ev.get("predicate", "")
            if pred:
                cid = self.concepts.resolve(f"event::{pred}")
                nodes.append(self.concepts.rep_slot(cid))
        for st in frame.get("states", []) or []:
            key = f"state::{st.get('var','')}::{st.get('subject','')}"
            cid = self.concepts.resolve(key)
            nodes.append(self.concepts.rep_slot(cid))
        n = self.core.n_nodes
        nodes = [int(x) for x in dict.fromkeys(nodes) if 0 <= int(x) < n]
        return nodes

    def _target_state_keys(self, frame: Dict[str, Any], max_k: int = 6) -> List[str]:
        keys = []
        for st in frame.get("states", []) or []:
            k = f"state::{st.get('var','')}::{st.get('subject','')}"
            if k not in keys:
                keys.append(k)
            if len(keys) >= max_k:
                break
        return keys

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

    # -------------------------
    # Confidence (stability * decisiveness * option-margin) - FIXED
    # -------------------------
    def _confidence(self, u: float, target_vecs: List[torch.Tensor], opt_margin: Optional[float]) -> float:
        stab = float(np.clip(1.0 - u, 0.0, 1.0))
        norms = [float(torch.norm(v).item()) for v in target_vecs] if target_vecs else [0.0]
        mean_norm = float(np.mean(norms))
        y0 = 0.25
        dec = float(np.clip(mean_norm / y0, 0.0, 1.0))
        conf = 0.15 + 0.75 * stab * (0.30 + 0.70 * dec)
        if opt_margin is not None:
            conf *= float(np.clip(0.85 + 0.30 * opt_margin, 0.75, 1.10))
        return float(np.clip(conf, 0.0, 1.0))

    # -------------------------
    # B2: explicit counterfactual do-evaluation (ADD-ONLY)
    # -------------------------
    def answer_counterfactual_B2(self, factual: str, counterfactual: str,
                                 options: Optional[Dict[str, str]] = None) -> AnswerPacket:
        factual = _normalize_text(factual)
        counterfactual = _normalize_text(counterfactual)

        if _is_exact_fact_task(factual + " " + counterfactual) or _contains_fact_like_patterns(factual + " " + counterfactual):
            return AnswerPacket(
                best_effort_answer="【回答保留（誤答防止）】厳密事実は検証なしに断定しません。参照情報を提示してください。",
                confidence=0.0,
                need_info_questions=["参照情報（URL/DOI等）を提示できますか？"],
                reason_trace={"mode": "VERIFY_REQUIRED"},
                mode="ABSTAIN",
            )

        # frames (LLM semantics)
        f_frame = self.frames.extract_frame(factual, kind="factual")
        c_frame = self.frames.extract_frame(counterfactual, kind="counterfactual")

        # IR diff (B2: include events)
        ops = self.ir_b2.diff_frames(f_frame, c_frame)

        # workspace nodes
        ws_nodes = list(dict.fromkeys(self._workspace_nodes_from_frame(f_frame) + self._workspace_nodes_from_frame(c_frame)))
        if not ws_nodes:
            cid = self.concepts.resolve("question::" + (factual + "|" + counterfactual)[:80])
            ws_nodes = [self.concepts.rep_slot(cid)]

        # targets: states in counterfactual preferred
        state_keys = self._target_state_keys(c_frame, max_k=6) or self._target_state_keys(f_frame, max_k=6)
        target_nodes = self._nodes_for_state_keys(state_keys) if state_keys else ws_nodes[:3]
        for tn in target_nodes:
            if tn not in ws_nodes:
                ws_nodes.append(tn)

        # policy beta / prior (kept)
        mode_guess = self.policy.choose_mode(factual + " " + counterfactual, anomaly_score=0.0)
        beta = self.policy.beta_prior if mode_guess == "OPEN" else 0.0
        Sprior = self._ensure_cache_prior_S() if beta > 0.0 else None

        # optional workspace gating
        use_workspace = os.environ.get("CAUSALOS_DISABLE_WORKSPACE", "0") != "1"

        def _run_under_workspace():
            # factual
            self.core.reset_do()
            loc_f = self.localizer.localize(self.core, S_prior=Sprior, Q=ws_nodes, T=target_nodes, beta_prior=beta)
            x_f = loc_f["traj"][-1]

            # apply do
            self.core.reset_do()
            atomic_info = self.atomic_b2.apply(ops, self.core, ws_nodes)

            # counterfactual
            loc_c = self.localizer.localize(self.core, S_prior=Sprior, Q=ws_nodes, T=target_nodes, beta_prior=beta)
            traj_c = loc_c["traj"]
            x_c = traj_c[-1]

            # impossibility u (kept)
            S_eff = self.core.get_S_eff(beta=beta, S_prior=Sprior)
            u_div = self.impossible.local_divergence(traj_c)
            u_rho = self.impossible.local_spectral_risk(S_eff, loc_c.get("OmegaA_nodes", []))
            u_cst = self.impossible.constraint_violation(traj_c)

            # modality penalty (B2): if constraints contain "cannot", increase u slightly (not keyword typing of intervention; feasibility cue)
            # NOTE: This is not classification; it's a feasibility modifier based on extracted constraint.type.
            u_mod = 0.0
            for op in ops:
                if op.get("op") == "MODALITY":
                    tpe = _norm_label(op.get("payload", {}).get("type", ""))
                    if tpe == "cannot":
                        u_mod = max(u_mod, 0.25)

            u = self.impossible.combine_u(u_div, u_rho, u_cst)
            u = float(np.clip(u + u_mod, 0.0, 1.0))
            gval = self.impossible.g(u)

            applied_gate = False
            if u >= 0.70:
                self.impossible.apply_local_gate(self.core, loc_c["Omega_edges"], gval=gval)
                applied_gate = True
                loc_c2 = self.localizer.localize(self.core, S_prior=Sprior, Q=ws_nodes, T=target_nodes, beta_prior=beta)
                traj_c = loc_c2["traj"]
                x_c = traj_c[-1]

            predicted = self._collect_predicted_states(state_keys, x_c) if state_keys else {}
            target_vecs = [predicted[k] for k in predicted] if predicted else [x_c[target_nodes[0]]] if target_nodes else []

            # options via B2 scorer (frame-based)
            best_opt, opt_scores = (None, {})
            opt_margin = None
            if options:
                best_opt, opt_scores = self.opt_scorer_b2.score(predicted, options)
                if opt_scores and len(opt_scores) >= 2:
                    s_sorted = sorted(opt_scores.values(), reverse=True)
                    opt_margin = float(np.clip(s_sorted[0] - s_sorted[1], 0.0, 1.0))
                elif opt_scores and len(opt_scores) == 1:
                    opt_margin = 0.2

            conf = self._confidence(u=u, target_vecs=target_vecs, opt_margin=opt_margin)

            return {
                "x_f": x_f, "x_c": x_c, "predicted": predicted,
                "u": u, "g": gval, "applied_gate": applied_gate,
                "atomic_info": atomic_info, "best_opt": best_opt, "opt_scores": opt_scores,
                "conf": conf, "mode_guess": mode_guess, "beta": beta
            }

        if use_workspace:
            with WorkspaceGate(self.core) as wg:
                wg.activate_nodes(ws_nodes)
                result = _run_under_workspace()
        else:
            result = _run_under_workspace()

        # compose response (templated; no LLM answer generation)
        lines = []
        lines.append("【反事実推論（CausalOS v5.3_full / B2）】")
        lines.append(f"確信度: {result['conf']:.2f}")
        lines.append("")
        lines.append("推定された介入（IR: state/event/constraint diff）:")
        for op in ops[:8]:
            lines.append(f"- {op.get('op')}: {str(op.get('payload', {}))[:140]}")
        if result["applied_gate"]:
            lines.append("")
            lines.append("注: 局所的な不安定/矛盾の兆候が検知されたため、該当領域の接続を抑制して再評価しました。")

        if options and result["best_opt"]:
            lines.append("")
            lines.append(f"【選択肢との整合】最も整合する候補: {result['best_opt']} : {options.get(result['best_opt'], '')}")

        need_q = []
        if result["conf"] < 0.70 or not state_keys:
            need_q = [
                "結果として知りたい状態（例：火傷の有無、旅の終了など）を明示できますか？",
                "反実で固定する要素と変更する要素を明確にできますか？",
            ]
            lines.append("")
            lines.append("より正確な回答のため、次を教えてください（短くでOK）:")
            for i, q in enumerate(need_q[:3], 1):
                lines.append(f"{i}) {q}")

        mode = "ANSWER" if result["conf"] >= 0.80 and not need_q else ("TENTATIVE" if need_q else "ANSWER")

        trace = {
            "build_id": BUILD_ID,
            "policy_mode": result["mode_guess"],
            "beta_prior": result["beta"],
            "workspace_nodes": ws_nodes[:60],
            "state_keys": state_keys,
            "target_nodes": target_nodes,
            "ops": ops,
            "atomic_info": result["atomic_info"],
            "u": result["u"],
            "applied_gate": result["applied_gate"],
            "best_opt": result["best_opt"],
            "opt_scores": result["opt_scores"],
        }

        return AnswerPacket("\n".join(lines), float(result["conf"]), need_q[:3], trace, mode)

    # -------------------------
    # Default answer() kept (v5.2-style general), but now can route counterfactual
    # -------------------------
    def answer(self, user_text: str, options: Optional[Dict[str, str]] = None) -> AnswerPacket:
        user_text = _normalize_text(user_text)

        if _is_exact_fact_task(user_text) or _contains_fact_like_patterns(user_text):
            return AnswerPacket(
                best_effort_answer="【回答保留（誤答防止）】厳密事実は検証なしに断定しません。参照情報を提示してください。",
                confidence=0.0,
                need_info_questions=["参照情報（URL/DOI等）を提示できますか？"],
                reason_trace={"mode": "VERIFY_REQUIRED"},
                mode="ABSTAIN",
            )

        # fallback: use v5.2-style rollout without explicit IR
        mode_guess = self.policy.choose_mode(user_text, anomaly_score=0.0)
        beta = self.policy.beta_prior if mode_guess == "OPEN" else 0.0
        Sprior = self._ensure_cache_prior_S() if beta > 0.0 else None

        # focus/target: use a generic "question concept"
        qcid = self.concepts.resolve("question::" + user_text[:80])
        qnode = self.concepts.rep_slot(qcid)
        targets = [qnode]

        use_workspace = os.environ.get("CAUSALOS_DISABLE_WORKSPACE", "0") != "1"
        nodes = [qnode]

        def _run():
            self.core.reset_do()
            loc = self.localizer.localize(self.core, S_prior=Sprior, Q=nodes, T=targets, beta_prior=beta)
            traj = loc["traj"]
            x_final = traj[-1]

            S_eff = self.core.get_S_eff(beta=beta, S_prior=Sprior)
            u_div = self.impossible.local_divergence(traj)
            u_rho = self.impossible.local_spectral_risk(S_eff, loc.get("OmegaA_nodes", []))
            u_cst = self.impossible.constraint_violation(traj)
            u = self.impossible.combine_u(u_div, u_rho, u_cst)

            # decision vectors: target node vector
            target_vecs = [x_final[qnode].detach()]
            conf = self._confidence(u=u, target_vecs=target_vecs, opt_margin=None)

            lines = []
            lines.append("【推論（因果メモリ + 伝播）】")
            lines.append(f"確信度: {conf:.2f}")
            if options:
                # keep options but do not force an answer in general mode
                lines.append("")
                lines.append("【選択肢】入力を受け取りました（一般推論モードでは選択肢照合は実施しません）。")

            need_q = []
            if conf < 0.65:
                need_q = [
                    "前提（状況・範囲・制約）をもう少し具体化できますか？",
                    "何を結果として判定したいか（状態変数）を明示できますか？"
                ]
                lines.append("")
                lines.append("より正確な回答のため、次を教えてください（短くでOK）:")
                for i, q in enumerate(need_q[:3], 1):
                    lines.append(f"{i}) {q}")

            trace = {"build_id": BUILD_ID, "policy_mode": mode_guess, "beta_prior": beta, "u": u}
            mode = "ANSWER" if conf >= 0.75 and not need_q else ("TENTATIVE" if need_q else "ANSWER")

            return AnswerPacket("\n".join(lines), conf, need_q[:3], trace, mode)

        if use_workspace:
            with WorkspaceGate(self.core) as wg:
                wg.activate_nodes(nodes)
                return _run()
        else:
            return _run()


if __name__ == "__main__":
    mid = os.environ.get("CAUSALOS_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    osys = UnifiedCausalOSV5_3Full(model_id=mid, init_n_nodes=256, expand_chunk=256, local_horizon=10, w0=0.7, w1=0.3)
    print(f"[System] Loaded CausalOS v5.3_full BUILD_ID={BUILD_ID}")

    # Simple self-test: fire counterfactual with options
    factual = "A woman sees a fire."
    cf = "the woman had touched the fire"
    options = {
        "A": "She would have not been burned.",
        "B": "Everything would have been fine.",
        "C": "She would have been burned.",
        "D": "She would have seen fire."
    }
    pkt = osys.answer_counterfactual_B2(factual, cf, options=options)
    print(pkt.best_effort_answer)
    if os.environ.get("CAUSALOS_SHOW_TRACE", "0") == "1":
        print(pkt.reason_trace)