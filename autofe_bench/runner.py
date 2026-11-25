"""Corredor: gera os jobs (metodo x base x divisao), roda cada um sob o protocolo e grava uma linha por job."""
from __future__ import annotations

import re
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .contract import Budget, Context, Task, new_columns, to_jsonable
from .data import Dataset, load_dataset, make_splits, split_hash
from .llm import BudgetedLLM, LLMClient, QuotaExhausted
from .prep import Preprocessor, evaluate_models
from .protocol import Protocol, get_protocol
from .sandbox import time_limit
from .store import ResultStore, job_key


def _frames_equal(a: pd.DataFrame, b: pd.DataFrame, rtol=1e-6, atol=1e-9) -> bool:
    cols = [c for c in a.columns if c in b.columns]
    if not cols:
        return False
    for c in cols:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if len(x) != len(y):
            return False
        try:
            xf, yf = x.astype(float), y.astype(float)
            if not np.allclose(xf, yf, rtol=rtol, atol=atol, equal_nan=True):
                return False
        except (ValueError, TypeError):
            if not all((p == q) or (pd.isna(p) and pd.isna(q)) for p, q in zip(x, y)):
                return False
    return True


def inductive_probe(fitted, Xte: pd.DataFrame, Tte: pd.DataFrame, seed: int, n_batch: int = 20, n_single: int = 3) -> bool:
    """True se `transform` de poucas linhas de teste reproduz o `transform` do teste inteiro (nao e transdutivo)."""
    rng = np.random.RandomState(seed)
    n = len(Xte)
    pos = np.sort(rng.choice(n, size=min(n_batch, n), replace=False))
    sub = fitted.transform(Xte.iloc[pos].copy()).reset_index(drop=True)
    if not _frames_equal(sub, Tte.iloc[pos].reset_index(drop=True)):
        return False
    for p in rng.choice(n, size=min(n_single, n), replace=False):
        one = fitted.transform(Xte.iloc[[p]].copy()).reset_index(drop=True)
        if not _frames_equal(one, Tte.iloc[[p]].reset_index(drop=True)):
            return False
    return True


def _safe(s) -> str:
    return re.sub(r"[^0-9a-zA-Z_.-]+", "_", str(s))


def run_job(method, ds: Dataset, seed: int, tr, te, protocol: Protocol, client: LLMClient | None,
            log_dir=None, llm_mode: str = "record", verbose: bool = True) -> dict:
    """Roda UM job e devolve a linha de resultado (status 'ok' ou 'error'). QuotaExhausted sobe para o chamador."""
    t_start = time.time()
    llm_id = f"{client.provider}:{client.model}" if (client is not None and method.needs_llm) else "none"
    row = {"method": method.name, "method_version": method.version, "llm": llm_id, "dataset": ds.name,
           "dataset_hash": ds.data_hash(), "seed": int(seed), "split_hash": split_hash(te),
           "protocol": protocol.name, "protocol_hash": protocol.hash(), "harness": __version__,
           "n_train": int(len(tr)), "n_test": int(len(te)), "n_features_in": int(ds.X.shape[1]),
           "task_kind": ds.task.kind, "params": to_jsonable(getattr(method, "params", {})),
           "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    Xtr, ytr = ds.X.iloc[tr].reset_index(drop=True), ds.y.iloc[tr].reset_index(drop=True)
    Xte, yte = ds.X.iloc[te].reset_index(drop=True), ds.y.iloc[te].reset_index(drop=True)
    budget: Budget = protocol.budget()
    llm = None
    if method.needs_llm:
        if client is None:
            row.update(status="error", error="o metodo precisa de LLM e nenhum foi configurado")
            return row
        log_path = None
        if log_dir is not None:
            log_path = Path(log_dir) / f"{_safe(method.name)}__{_safe(ds.name)}__{seed}__{_safe(llm_id)}.jsonl"
        llm = BudgetedLLM(client, max_calls=budget.max_llm_calls, max_total_tokens=budget.max_total_tokens,
                          log_path=log_path, mode=llm_mode)
    ctx = Context(llm, budget, seed, ds.task, log=(print if verbose else None))
    try:
        t0 = time.time()
        with time_limit(budget.time_limit_s, mem_gb=None):
            fitted = method.fit(Xtr.copy(), ytr.copy(), ds.task, ctx)
        row["fit_seconds"] = round(time.time() - t0, 1)

        t0 = time.time()
        Ttr = fitted.transform(Xtr.copy())
        Tte = fitted.transform(Xte.copy())
        if not isinstance(Ttr, pd.DataFrame) or not isinstance(Tte, pd.DataFrame):
            raise TypeError("transform deve devolver um DataFrame")
        if len(Ttr) != len(Xtr) or len(Tte) != len(Xte):
            raise ValueError("transform alterou o numero de linhas")
        if set(Ttr.columns) != set(Tte.columns):
            raise ValueError("treino e teste sairam com colunas diferentes do transform")
        Ttr, Tte = Ttr.reset_index(drop=True), Tte[Ttr.columns].reset_index(drop=True)
        new = new_columns(Xtr, Ttr)
        row["n_new_raw"] = len(new)
        if len(new) > budget.max_new_features:
            drop = new[budget.max_new_features:]
            Ttr, Tte = Ttr.drop(columns=drop), Tte.drop(columns=drop)
            row["truncated"] = len(drop)
            new = new[: budget.max_new_features]
        row["n_new"] = len(new)
        row["inductive_declared"] = bool(getattr(fitted, "inductive", True))
        row["transductive_detected"] = not inductive_probe(fitted, Xte, Tte, seed)
        row["transform_seconds"] = round(time.time() - t0, 1)

        prep = Preprocessor().fit(Ttr)
        Ftr, Fte = prep.transform(Ttr), prep.transform(Tte)
        row["n_out"] = int(Ftr.shape[1])
        row["metrics"] = evaluate_models(Ftr, ytr, Fte, yte, ds.task.kind, protocol.models, seed,
                                         protocol.rf_trees, protocol.rf_depth, protocol.hgb_iter)
        errs = {m: d["error"] for m, d in row["metrics"].items() if "error" in d}
        row["status"] = "ok" if not errs else "error"
        if errs:
            row["error"] = str(errs)
        try:
            row["report"] = to_jsonable(fitted.report())
        except Exception as e:  # noqa: BLE001
            row["report"] = {"report_error": str(e)}
    except QuotaExhausted:
        raise
    except Exception as e:  # noqa: BLE001
        row.update(status="error", error=f"{type(e).__name__}: {e}"[:500], traceback=traceback.format_exc()[-1500:])
    if llm is not None:
        row.update(llm.stats())
        if row.get("status") == "ok" and llm.calls == 0:
            # o metodo engoliu a falha do LLM e devolveu 'sem features': isso NAO e um resultado valido
            row.update(status="error", error=f"nenhuma chamada ao LLM teve sucesso. Ultimo erro: {llm.last_error}")
    row["total_seconds"] = round(time.time() - t_start, 1)
    return row


def _with_baseline(meths, include_baseline=True):
    """O baseline 'sem_fe' entra sempre (uma vez so): sem ele o relatorio nao tem com o que comparar."""
    from .methods import SemFE

    meths = list(meths)
    if include_baseline and not any(m.name == "sem_fe" for m in meths):
        meths.insert(0, SemFE())
    seen, out = set(), []
    for m in meths:
        if m.name not in seen:
            seen.add(m.name)
            out.append(m)
    return out


def run_benchmark(methods, datasets, protocol: Protocol | str = "v1", client: LLMClient | None = None,
                  store_path="results/results.jsonl", log_dir="logs", retry_errors=True, llm_mode="record",
                  verbose=True, dataset_loader=load_dataset, mirror_path=None, include_baseline=True):
    """Roda todos os jobs que ainda nao tem resultado. Pode ser interrompido e retomado a qualquer momento."""
    from .methods import resolve_method

    protocol = get_protocol(protocol) if isinstance(protocol, str) else protocol
    meths = _with_baseline([resolve_method(m) for m in methods], include_baseline)
    store = ResultStore(store_path, mirror_path)
    stopped = None
    n_run = n_skip = n_err = 0
    for dname in datasets:
        ds = dataset_loader(dname, protocol.max_rows) if isinstance(dname, str) else dname
        splits = make_splits(ds.y, ds.task.kind, protocol.seeds, protocol.test_size)
        for seed, tr, te in splits:
            for m in meths:
                llm_id = f"{client.provider}:{client.model}" if (client is not None and m.needs_llm) else "none"
                probe = {"method": m.name, "method_version": m.version, "llm": llm_id, "dataset": ds.name,
                         "seed": seed, "protocol_hash": protocol.hash()}
                if store.done(job_key(probe), retry_errors):
                    n_skip += 1
                    continue
                if verbose:
                    print(f"[{ds.name} | seed {seed} | {m.name} v{m.version} | llm={llm_id}] rodando...")
                try:
                    row = run_job(m, ds, seed, tr, te, protocol, client, log_dir, llm_mode, verbose=False)
                except QuotaExhausted as e:
                    stopped = str(e)
                    break
                store.add(row)
                n_run += 1
                n_err += row["status"] != "ok"
                if verbose:
                    prim = ds.task.primary
                    sc = {k: round(v.get(prim, float("nan")), 4) for k, v in (row.get("metrics") or {}).items()}
                    print(f"   -> {row['status']} | {prim}={sc} | novas={row.get('n_new')} | "
                          f"llm_calls={row.get('llm_calls', 0)} tokens={row.get('prompt_tokens', 0) + row.get('completion_tokens', 0)} "
                          f"| {row['total_seconds']}s" + (f" | ERRO: {row.get('error', '')[:120]}" if row['status'] != 'ok' else ""))
            if stopped:
                break
        if stopped:
            break
    summary = {"rodados": n_run, "pulados": n_skip, "com_erro": n_err, "parado_por_cota": stopped}
    if verbose:
        print(f"\nResumo: {summary}")
        if stopped:
            print("A cota do provedor acabou. Os resultados ja estao salvos: rode o mesmo comando depois "
                  "(amanha, se a cota for diaria) e o harness continua de onde parou.")
    return store, summary


# ------------------------------------------------------------------------------ estimativa de custo
DEFAULT_TOKENS_PER_CALL = {"caafe": 3000, "llmfe": 2500, "guided": 2600, "featcopilot_llm": 3000}


def estimate(methods, datasets, protocol: Protocol | str = "v1", store_path=None, tpd: int | None = None,
             rpd: int | None = None):
    """Estima chamadas e tokens do benchmark e quantos dias levaria numa cota diaria (ex.: Groq gratuito)."""
    from .methods import resolve_method

    protocol = get_protocol(protocol) if isinstance(protocol, str) else protocol
    meths = [resolve_method(m) for m in methods]
    observed = {}
    if store_path and Path(store_path).exists():
        df = ResultStore(store_path).dataframe()
        if len(df) and "llm_calls" in df:
            d = df[df["llm_calls"] > 0]
            for name, g in d.groupby("method"):
                observed[name] = float((g["prompt_tokens"] + g["completion_tokens"]).sum() / g["llm_calls"].sum())
    lines, tot_calls, tot_tokens = [], 0, 0
    n_jobs = len(datasets) * len(protocol.seeds)
    for m in meths:
        if not m.needs_llm:
            lines.append(f"- {m.name}: sem LLM ({n_jobs} jobs)")
            continue
        calls = n_jobs * protocol.max_llm_calls
        tpc = observed.get(m.name, DEFAULT_TOKENS_PER_CALL.get(m.name, 2500))
        src = "medido" if m.name in observed else "padrao"
        toks = int(calls * tpc)
        lines.append(f"- {m.name}: {n_jobs} jobs x {protocol.max_llm_calls} chamadas = {calls} chamadas, "
                     f"~{toks:,} tokens ({tpc:.0f} por chamada, {src})")
        tot_calls += calls
        tot_tokens += toks
    out = [f"Protocolo {protocol.name}: {len(datasets)} bases x {len(protocol.seeds)} divisoes = {n_jobs} jobs por metodo", *lines,
           f"Total: {tot_calls} chamadas, ~{tot_tokens:,} tokens"]
    if tpd:
        out.append(f"Com {tpd:,} tokens/dia: ~{tot_tokens / tpd:.1f} dias")
    if rpd:
        out.append(f"Com {rpd:,} requisicoes/dia: ~{tot_calls / rpd:.1f} dias")
    text = "\n".join(out)
    return {"calls": tot_calls, "tokens": tot_tokens, "text": text}
