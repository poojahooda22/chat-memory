"""Offline evaluation harness (P0).

Measures the memory engine on the LOCOMO benchmark against two baselines (naive RAG,
full-context) on three axes: answer quality (LLM-as-judge J score), latency (p50/p95),
and token cost. This package is NOT part of the request-serving app — it is a batch job run
from the command line (`python -m eval.run`), never imported by app/ at runtime.
"""