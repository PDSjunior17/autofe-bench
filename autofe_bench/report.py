"""Relatorio: tabelas por base, ganho pareado sobre o baseline com IC bootstrap, ranks, custo e avisos."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .store import ResultStore

PRIMARY = [("auc", True), ("logloss", False), ("rmse", False)]


def long_table(store_path_or_store, only_ok=True) -> pd.DataFrame:
    store = store_path_or_store if isinstance(store_path_or_store, ResultStore) else ResultStore(store_path_or_store)
    df = store.dataframe(only_ok=only_ok)
    if df.empty:
        return df
    recs = []
    for _, r in df.iterrows():
        for model in ("lr", "rf", "hgb", "xgb"):
            for metric, hib in PRIMARY:
                v = r.get(f"{model}_{metric}")
                if v is not None and not pd.isna(v):
                    recs.append({**{k: r[k] for k in ("method", "dataset", "seed", "llm", "n_new", "inductive_declared",
                                                      "transductive_detected", "llm_calls", "prompt_tokens",
                                                      "completion_tokens", "total_seconds", "protocol")
                                    if k in r},
                                 "model": model, "metric": metric, "value": float(v), "higher_better": hib})
                    break
    return pd.DataFrame(recs)


def paired_gains(lt: pd.DataFrame, baseline="sem_fe") -> pd.DataFrame:
    """gain > 0 = melhor que o baseline na mesma (base, divisao, modelo)."""
    b = lt[lt.method == baseline][["dataset", "seed", "model", "value"]].rename(columns={"value": "base"})
    m = lt[lt.method != baseline].merge(b, on=["dataset", "seed", "model"])
    sign = np.where(m.higher_better, 1.0, -1.0)
    m["gain"] = sign * (m.value - m.base)
    return m


def bootstrap_ci(x, n=2000, alpha=0.05, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return tuple(np.quantile(means, [alpha / 2, 1 - alpha / 2]))


def summary_tables(path, baseline="sem_fe", model="rf") -> dict:
    lt = long_table(path)
    out = {}
    if lt.empty:
        return out
    t = lt[lt.model == model]
    out["media"] = t.pivot_table(index="dataset", columns="method", values="value", aggfunc="mean")
    out["dp"] = t.pivot_table(index="dataset", columns="method", values="value", aggfunc="std")
    g = paired_gains(lt, baseline)
    g = g[g.model == model]
    rows = []
    for (ds, meth), d in g.groupby(["dataset", "method"]):
        lo, hi = bootstrap_ci(d.gain)
        rows.append({"dataset": ds, "method": meth, "gain_medio": d.gain.mean(), "ic95_inf": lo, "ic95_sup": hi,
                     "vitorias": int((d.gain > 0).sum()), "n": len(d)})
    out["ganho"] = pd.DataFrame(rows)
    # rank medio por base (1 = melhor), usando a media por divisao
    ranks = []
    for ds, d in t.groupby("dataset"):
        hib = bool(d.higher_better.iloc[0])
        m = d.groupby("method").value.mean()
        ranks.append(m.rank(ascending=not hib).rename(ds))
    if ranks:
        out["rank"] = pd.concat(ranks, axis=1).T
        out["rank_medio"] = out["rank"].mean().sort_values()
    c = lt.drop_duplicates(["method", "dataset", "seed"])
    out["custo"] = c.groupby("method").agg(chamadas=("llm_calls", "sum"), tokens_prompt=("prompt_tokens", "sum"),
                                           tokens_resp=("completion_tokens", "sum"), segundos=("total_seconds", "sum"),
                                           novas_feat_media=("n_new", "mean"))
    flags = c.groupby("method").agg(transdutivo_declarado=("inductive_declared", lambda s: int((~s.astype(bool)).sum())),
                                    transdutivo_detectado=("transductive_detected", lambda s: int(s.astype(bool).sum())),
                                    jobs=("seed", "count"))
    out["avisos"] = flags
    err = ResultStore(path).dataframe(only_ok=False)
    if "status" in err:
        out["erros"] = err[err.status != "ok"][["method", "dataset", "seed", "status", "error"]] if (err.status != "ok").any() else pd.DataFrame()
    return out


def render(path, baseline="sem_fe", model="rf", fmt="text") -> str:
    t = summary_tables(path, baseline, model)
    if not t:
        return "(sem resultados)"
    parts = [f"# Resultados (modelo final: {model}; baseline: {baseline})",
             "Metrica primaria: AUC (binaria, maior melhor), logloss (multiclasse) ou RMSE (regressao), menores melhores.",
             "\n## Media por base\n" + t["media"].round(4).to_string()]
    if "ganho" in t and len(t["ganho"]):
        parts.append("\n## Ganho pareado sobre o baseline (positivo = melhor)\n" + t["ganho"].round(4).to_string(index=False))
    if "rank_medio" in t:
        parts.append("\n## Rank medio (1 = melhor)\n" + t["rank_medio"].round(2).to_string())
    parts.append("\n## Custo\n" + t["custo"].round(1).to_string())
    parts.append("\n## Avisos (metodos transdutivos nao sao estritamente comparaveis)\n" + t["avisos"].to_string())
    if len(t.get("erros", [])):
        parts.append("\n## Jobs com erro\n" + t["erros"].to_string(index=False))
    return "\n".join(parts)
