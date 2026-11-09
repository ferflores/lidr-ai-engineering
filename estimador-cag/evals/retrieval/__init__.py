"""Retrieval eval (session 10 pre-work): the four configurations of `POST /search` against a golden set.

    golden_set.json   five project descriptions with the budgets that are genuinely relevant, annotated by hand
    metrics.py        precision@k and relevant-budget recall@k over budget ids; latency percentiles
    run.py            runs A (vector), B (hybrid), C (vector + rerank), D (hybrid + rerank) against a live
                      service, repeats each query for latency, writes results.csv and REPORT.md
"""
