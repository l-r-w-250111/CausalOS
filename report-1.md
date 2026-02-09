# CausalOS v5.x: Causal Memory + Intervention IR for Robust Counterfactual Reasoning

> **Goal**: Build a “causal computation substrate” that augments LLMs with **causal memory** and **intervention-aware computation**, prioritizing *robustness against confident errors* and *long-horizon causal flow preservation*.

---

## 1. Motivation

LLMs excel at flexible language understanding but often fail at:
- **Long-horizon causal flow** (forgetting constraints and earlier premises)
- **Counterfactual / do-intervention reasoning** (especially when assumptions must be held fixed)
- **Confident hallucination** (producing plausible but wrong factual strings)

CausalOS aims to complement LLMs with:
- **Causal Memory** (S-matrix / causal graph as persistent substrate)
- **Intervention IR** (a deterministic “edit script” derived from semantic frames)
- **Local stability control** (impossibility detection → local gating)
- **Evidence-oriented output policy** (ask for missing assumptions rather than guessing)

---

## 2. Architectural Novelty (Key Contributions)

### (A) Two-tier causal reasoning loop: **LLM ≠ reasoner**
LLM is used only for **small, structured tasks**:
1) **Semantic Frame extraction** (entities/events/states/constraints)
2) **(optional)** causal triplet extraction into causal memory

All high-stakes decisions (graph updates, do-application, stability gating, scoring) are handled by the OS deterministically.

### (B) Causal Memory as **S-matrix + masks + dynamics**
Causal structure is stored as a matrix (or multi-parameter edge field):
- Topology / permission: **A-mask** (structural adjacency)
- Dynamic gate: **G-gate** (contextual/diagnostic gating, do-cut, impossibility hardening)
- Strength: **S**
- Decay: **r**
- Phase/time: **φ, ω**

This separates:
- “Is this connection allowed?” (A)
- “Is it currently active?” (G)
- “How strong is it?” (S, r)
- “How does it evolve in time?” (φ, ω)

### (C) Local anomaly → local repair (not global thresholds)
Instead of global thresholding, CausalOS localizes anomalies to a subgraph **Ω** using:
- Contribution (energy flow proxy)
- Reachability (query→target path relevance)
- Gradient sensitivity (∂L/∂S)

This supports a physically-inspired principle:
> **Local rupture can propagate into global uncertainty**, so we gate locally and quantify reachability loss.

### (D) Intervention IR (B2): **diff as the primitive, not label classification**
Rather than hard-coded “NEGATION/INTENSITY/TENSE” classification, CausalOS builds:
- **IR = deterministic diff** between factual and counterfactual semantic frames  
  (states + events + constraints)
- **Atomic Interventions** = deterministic mapping from IR to parameter-level operations

This preserves open-set generality: unknown interventions remain representable as edit scripts.

---

## 3. Core Formulation (with equations)

### 3.1 Complex-like causal propagation
Node state \(x_j(t)\in\mathbb{R}^2\) (real/imag) evolves by:

\[
x_{j}(t+1) = \sigma\Big( \sum_{i} A_{ji}\,G_{ji}(t)\,S_{ji}\,r_{ji}\,e^{i(\phi_{ji}+\omega t)}\,x_{i}(t)\Big)
\]

- \(A_{ji}\in\{0,1\}\): structural permission
- \(G_{ji}(t)\in[0,1]\): dynamic gate (do-cut, local impossibility hardening)
- \(S_{ji}\in[-1,1]\): signed causal strength
- \(r_{ji}\in[0,1]\): decay/amplitude
- \(\phi_{ji}\): phase (interference / polarity)
- \(\omega\): temporal modulation

### 3.2 Explicit do-evaluation (counterfactual)
CausalOS evaluates counterfactuals by explicit “before/after” computation:

1) **Factual rollout**: \(x^{(f)}(T)=\mathrm{Rollout}(S, A, G, \text{no-do})\)
2) **Apply do** via atomic interventions → modified gates/clamps
3) **Counterfactual rollout**: \(x^{(cf)}(T)=\mathrm{Rollout}(S, A, G', \text{do})\)
4) **Effect**: \(\Delta x(T)=x^{(cf)}(T)-x^{(f)}(T)\)

### 3.3 Local instability (impossibility) and hardening gate
We compute a local impossibility score \(u\in[0,1]\) from:
- divergence (energy growth)
- local spectral risk (stability of Ω)
- constraint violations (NaN/Inf/saturation)

\[
u = 1-(1-u_{\mathrm{div}})(1-u_{\rho})(1-u_C)
\]

Then apply a smooth hardening gate (soft→hard continuum):

\[
g(u)=\sigma(\kappa(\tau-u)),\quad G_{ji}\leftarrow G_{ji}\cdot g(u)\;\; \forall (j,i)\in \Omega
\]

---

## 4. Ω Localization (Reachability + Gradient always-on)

We define an Ω score per edge using three signals:

1) Contribution:
\[
c_{ji}\approx |S^{\mathrm{eff}}_{ji}|\,\|x_i\|
\]

2) Reachability:
edges on paths from query focus Q to targets T are scored higher.

3) Gradient sensitivity:
\[
g_{ji}=\left|\frac{\partial L}{\partial S_{ji}}\right|
\]
where \(L = w_0 x_T[0] + w_1\|x_T\|\).

Final edge score:
\[
\mathrm{score}_{ji}= \alpha\,\tilde c_{ji}+\beta\,\tilde r_{ji}+\gamma\,\tilde g_{ji}
\]
Ω is chosen by top-k edges under this combined score.

---

## 5. What v5.3_full Achieved (Progress)

### ✅ Robust separation of roles
- LLM is used for **frame extraction**, not for final answer generation.
- OS executes deterministic IR diff → atomic interventions → explicit do-evaluation.

### ✅ Workspace graph prevents cross-question contamination
Each question builds a local workspace that opens only the relevant nodes/edges, avoiding interference between unrelated benchmarks.

### ✅ Option evaluation without hard-coded keyword rules
Options are parsed into frames and compared against predicted outcomes via embeddings/structure, enabling flexible matching without brittle heuristics.

---

## 6. Current Limitation (Observed in tests)

Example:
> “A family starts a trip. What if the family had ended the trip?”

Result:
- IR became `NOOP`
- Option scorer still preferred A (semantic similarity)
- Confidence stayed low and CausalOS asked for clarification

**Interpretation**:
- The **Frame schema / extraction** did not expose the start/end difference as explicit `states/events` in a way that the deterministic diff could capture.
- Therefore the OS could not form a non-trivial intervention IR.

This is a *good failure mode* under safety-first design: CausalOS refuses to guess the causal intervention without a supported IR.

---

## 7. Next Steps (Non-destructive, Add-only)

1) **Strengthen frame schema** (still LLM-only parsing):
   - enforce mutually exclusive state variables when implied (e.g., trip_status)
   - add reconstruction consistency check:
     \[
     \text{Frame}(f) + \text{IR} \to \widehat{\text{Frame}}(cf),\;\; \text{match}(\widehat{}, cf)\uparrow
     \]
2) **IR candidate generation + selection** (self-consistency):
   - sample multiple frames, choose IR maximizing reconstruction accuracy and minimality.
3) Maintain the “no-removal” rule:
   - new features are gated by flags; no existing component is deleted.

---

## 8. Summary

CausalOS introduces a **causal computation substrate** that:
- preserves long-horizon causal flow via S-matrix memory,
- performs explicit do-intervention computation via deterministic IR,
- localizes instability via Ω and stabilizes via gating,
- keeps LLM flexibility while adding verifiable structure.

This design targets the gap between **language competence** and **causal competence**, aiming to raise reliability without sacrificing generality.
