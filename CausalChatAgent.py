# -*- coding: utf-8 -*-
"""
CausalChatAgent.py (demo for CausalOS_v5_3_full)

- Extract options A/B/C/D (multiline or inline)
- Extract factual/counterfactual from pattern:
    "X. What would have happened if Y?"
    "X. What if Y?"
- If detected -> call osys.answer_counterfactual_B2(factual, counterfactual, options)
- Otherwise -> call osys.answer(main_text, options)

IMPORTANT:
- We ingest user messages (without options pollution) into long-term causal memory.
"""

import os
import re
from typing import Dict, Optional, Tuple

import CausalOS_v5_3_full as causal


class CausalChatAgent:
    def __init__(self, osys: "causal.UnifiedCausalOSV5_3Full"):
        self.osys = osys

    def _extract_options(self, text: str) -> Optional[Dict[str, str]]:
        if not text:
            return None
        t = text.replace("\r\n", "\n").replace("\r", "\n")

        # multiline options
        line_pat = re.compile(r'^\s*([A-Z])\s*:\s*(.+?)\s*$')
        opts = {}
        for line in t.split("\n"):
            m = line_pat.match(line)
            if m:
                k = m.group(1)
                v = m.group(2).strip()
                if v:
                    opts[k] = v
        if len(opts) >= 2:
            return dict(sorted(opts.items(), key=lambda kv: kv[0]))

        # inline options
        boundary_pat = re.compile(r'(?:(?<=\n)|(?<=\s)|^)([A-Z])\s*:\s*')
        matches = list(boundary_pat.finditer(t))
        if len(matches) < 2:
            return None

        options = {}
        for i, m in enumerate(matches):
            key = m.group(1)
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(t)
            val = t[start:end].strip()
            val = re.sub(r"\s+", " ", val)
            if val:
                options[key] = val

        return dict(sorted(options.items(), key=lambda kv: kv[0])) if len(options) >= 2 else None

    def _strip_options(self, text: str) -> str:
        t = (text or "").strip()
        m = re.search(r'(\s|^)([A-Z])\s*:\s*', t)
        if m:
            return t[:m.start()].strip()
        return t

    def _extract_cf_pair(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        t = (text or "").strip()
        t = re.sub(r"\s+", " ", t)
        m = re.search(r'(.+?)\.\s*What\s+(?:if|would have happened if)\s+(.+?)\?', t, re.IGNORECASE)
        if m:
            factual = m.group(1).strip()
            cf = m.group(2).strip()
            return factual, cf
        return None, None

    def chat(self, user_input: str):
        user_input = (user_input or "").strip()
        if not user_input:
            return ""

        options = self._extract_options(user_input)
        main_text = self._strip_options(user_input) if options else user_input

        # ingest only main text (avoid A/B/C/D pollution)
        self.osys.ingest_context(main_text, source="user", weight=0.90)

        factual, cf = self._extract_cf_pair(main_text)

        if factual and cf:
            pkt = self.osys.answer_counterfactual_B2(factual, cf, options=options)
        else:
            pkt = self.osys.answer(main_text, options=options)

        print(pkt.best_effort_answer)
        if os.environ.get("CAUSALOS_SHOW_TRACE", "0") == "1":
            print("\n[Trace]")
            print(pkt.reason_trace)

        return pkt.best_effort_answer


def main():
    print("--- Starting CausalChatAgent (CausalOS v5.3_full) ---", flush=True)
    model_id = os.environ.get("CAUSALOS_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

    osys = causal.UnifiedCausalOSV5_3Full(
        model_id=model_id,
        init_n_nodes=256,
        init_slots_per_concept=2,
        expand_chunk=256,
        local_horizon=10,
        w0=0.7,
        w1=0.3,
    )
    agent = CausalChatAgent(osys)

    print("Ready! Type your message (or 'exit' to quit).", flush=True)
    while True:
        try:
            msg = input("\nUser: ").strip()
            if msg.lower() in ["exit", "quit", "q"]:
                break
            agent.chat(msg)
        except KeyboardInterrupt:
            break
        except Exception as e:
            if os.environ.get("CAUSALOS_TRACEBACK", "0") == "1":
                import traceback
                traceback.print_exc()
            else:
                print(f"[Error] {e}")


if __name__ == "__main__":
    main()