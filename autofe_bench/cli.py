"""Linha de comando:  python -m autofe_bench {check,run,report,estimate,methods,datasets} ..."""
from __future__ import annotations

import argparse
import sys


def _client(spec, rpm, tpm):
    from .llm import parse_llm_spec
    if spec == "mock":
        from .mockllm import HeuristicMockLLM
        return HeuristicMockLLM()
    kw = {}
    if rpm:
        kw["rpm"] = rpm
    if tpm:
        kw["tpm"] = tpm
    return parse_llm_spec(spec, **kw)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="autofe_bench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    def common(p):
        p.add_argument("--llm", default="none", help="groq:MODELO | ollama:MODELO | openai:MODELO@URL | mock | none")
        p.add_argument("--rpm", type=int); p.add_argument("--tpm", type=int)
    c = sub.add_parser("check"); c.add_argument("method"); c.add_argument("--dataset", default="breast_cancer"); common(c)
    r = sub.add_parser("run"); r.add_argument("--methods", nargs="+", required=True); r.add_argument("--datasets", nargs="+", required=True)
    r.add_argument("--protocol", default="dev"); r.add_argument("--store", default="results/results.jsonl"); r.add_argument("--logs", default="logs"); common(r)
    rp = sub.add_parser("report"); rp.add_argument("--store", default="results/results.jsonl"); rp.add_argument("--baseline", default="sem_fe"); rp.add_argument("--model", default="rf")
    e = sub.add_parser("estimate"); e.add_argument("--methods", nargs="+", required=True); e.add_argument("--datasets", nargs="+", required=True)
    e.add_argument("--protocol", default="v1"); e.add_argument("--tpd", type=int); e.add_argument("--rpd", type=int); e.add_argument("--store")
    sub.add_parser("methods"); sub.add_parser("datasets")
    mo = sub.add_parser("models"); mo.add_argument("--provider", choices=["ollama", "groq"], default="ollama"); mo.add_argument("--host")
    mg = sub.add_parser("merge"); mg.add_argument("files", nargs="+"); mg.add_argument("-o", "--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "check":
        from .check import check_method
        ok, _ = check_method(a.method, a.dataset, client=_client(a.llm, a.rpm, a.tpm))
        return 0 if ok else 1
    if a.cmd == "run":
        from .runner import run_benchmark
        run_benchmark(a.methods, a.datasets, a.protocol, _client(a.llm, a.rpm, a.tpm), a.store, a.logs)
    elif a.cmd == "report":
        from .report import render
        print(render(a.store, a.baseline, a.model))
    elif a.cmd == "estimate":
        from .runner import estimate
        print(estimate(a.methods, a.datasets, a.protocol, a.store, a.tpd, a.rpd)["text"])
    elif a.cmd == "models":
        from .llm import list_groq_models, list_ollama_models
        print("\n".join(list_ollama_models(a.host) if a.provider == "ollama" else list_groq_models()))
    elif a.cmd == "merge":
        from .store import merge_stores
        print(f"{merge_stores(a.files, a.out)} jobs gravados em {a.out}")
    elif a.cmd == "methods":
        from .methods import available_methods
        print("\n".join(available_methods()))
    elif a.cmd == "datasets":
        from .data import LOADERS, OFFLINE
        print("\n".join(f"{k}{'  (offline)' if k in OFFLINE else ''}" for k in LOADERS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
