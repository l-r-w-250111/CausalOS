# -*- coding: utf-8 -*-
"""CausalChatAgent_q1q2.py

ADD-ONLY.
Dedicated runner for the fixed Q1/Q2 evaluation.

- Loads CausalOS v5.3_full base.
- Wraps with OSS guard v6 or v7 if desired (env CAUSALOS_FACT_BASE=oss_v6/oss_v7/base).
- Runs FactGuardQ1Q2 which performs web evidence + guarded answering.
- Prints:
  - Answers
  - Trace (provenance: web/llm/guard)

"""

import os
import json

from CausalOS_v5_3_full import UnifiedCausalOSV5_3Full

# Optional OSS bases
try:
    from CausalOS_v5_3_oss_v6 import UnifiedCausalOSV5_3Full_OSS_V6
except Exception:
    UnifiedCausalOSV5_3Full_OSS_V6 = None

try:
    from CausalOS_v5_3_oss_v7 import UnifiedCausalOSV5_3Full_OSS_V7
except Exception:
    UnifiedCausalOSV5_3Full_OSS_V7 = None

from CausalOS_v5_3_factguard_q1q2 import FactGuardQ1Q2


def _make_osys(model_id: str):
    base = os.environ.get('CAUSALOS_FACT_BASE', 'oss_v6').strip()
    cls = UnifiedCausalOSV5_3Full
    if base == 'oss_v7' and UnifiedCausalOSV5_3Full_OSS_V7 is not None:
        cls = UnifiedCausalOSV5_3Full_OSS_V7
    elif base == 'oss_v6' and UnifiedCausalOSV5_3Full_OSS_V6 is not None:
        cls = UnifiedCausalOSV5_3Full_OSS_V6

    # retriever injection left as None; FactGuard uses retrieval_tools.SimpleWebRetriever if available.
    return cls(
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


def main():
    print('--- Starting Q1/Q2 FactGuard Test (CausalOS v5.3 + OSS optional) ---', flush=True)

    model_id = os.environ.get('CAUSALOS_MODEL','Qwen/Qwen2.5-7B-Instruct')
    osys = _make_osys(model_id)

    fg = FactGuardQ1Q2(osys)
    result = fg.run()

    # Pretty output
    print('\n[Q1] Answer:')
    print(result['Q1']['answer'])
    print('\n[Q2] Answer:')
    print(result['Q2']['answer'])

    print('\n[Trace]')
    print(json.dumps({
        'Q1': result['Q1']['trace'],
        'Q2': result['Q2']['trace']
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
