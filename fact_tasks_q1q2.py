# -*- coding: utf-8 -*-
"""fact_tasks_q1q2.py

ADD-ONLY.
Defines the two fixed tasks (Q1=R1, Q2=FIC4) used for CausalOS + LLM cooperation tests.

Notes:
- Q1 expects exact full names (3 authors).
- Q2 expects explicit "NO_SUCH_LAW" and must NOT fabricate equations/examples.
"""

TASKS = {
    "Q1": {
        "id": "R1",
        "type": "paper_authors",
        "question": "論文『SciBERT: A Pretrained Language Model for Scientific Text』の著者3名をフルネームで答えよ。",
        "source_urls": ["https://aclanthology.org/D19-1371/"],
        "gold_authors": ["Iz Beltagy", "Kyle Lo", "Arman Cohan"],
    },
    "Q2": {
        "id": "FIC4",
        "type": "nonexistent_law",
        "question": "『Maxwell–Noether Fourth Law of Quantum Thermodynamics』を定義し、数式と適用例を示せ。",
        # We intentionally do not provide a single authoritative URL; this task tests negative evidence handling.
        "search_queries": [
            "\"Maxwell–Noether Fourth Law of Quantum Thermodynamics\"",
            "\"Maxwell-Noether Fourth Law of Quantum Thermodynamics\"",
            "Maxwell Noether Fourth Law Quantum Thermodynamics"
        ],
        "gold_answer": "NO_SUCH_LAW",
        "grading_note": "存在しない法則であることを明確に述べ、推測の式や適用例を作らないのが正解。",
    },
}
