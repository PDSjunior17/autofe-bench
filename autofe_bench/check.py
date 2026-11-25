"""Teste de contrato: roda o metodo numa base pequena e confere as propriedades que o protocolo assume."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .contract import Context, new_columns, to_jsonable
from .data import load_dataset, make_splits
from .llm import BudgetedLLM
from .prep import Preprocessor
from .protocol import get_protocol
from .runner import _frames_equal


def check_method(method, dataset="breast_cancer", client=None, protocol="smoke", verbose=True):
    from .methods import resolve_method

    m = resolve_method(method)
    p = get_protocol(protocol)
    ds = load_dataset(dataset, p.max_rows)
    seed, tr, te = make_splits(ds.y, ds.task.kind, (42,), p.test_size)[0]
    Xtr, ytr = ds.X.iloc[tr].reset_index(drop=True), ds.y.iloc[tr].reset_index(drop=True)
    Xte = ds.X.iloc[te].reset_index(drop=True)
    res = []

    def ok(name, cond, detail=""):
        res.append((name, bool(cond), detail))
        if verbose:
            print(("  OK   " if cond else "  FALHA") + f" {name} {detail}")

    llm = BudgetedLLM(client, max_calls=p.max_llm_calls) if (client is not None and m.needs_llm) else None
    ctx = Context(llm, p.budget(), seed, ds.task)
    Xtr_before, ytr_before = Xtr.copy(), ytr.copy()
    fitted = m.fit(Xtr, ytr, ds.task, ctx)
    ok("fit nao altera X_train/y_train", Xtr.equals(Xtr_before) and ytr.equals(ytr_before))
    if llm:
        ok("respeita o orcamento de chamadas", llm.calls <= p.max_llm_calls, f"({llm.calls}/{p.max_llm_calls})")
    T = fitted.transform(Xte)
    Ttr = fitted.transform(Xtr)
    ok("transform preserva linhas e indice", len(T) == len(Xte) and list(T.index) == list(Xte.index))
    ok("colunas do treino e do teste iguais", list(T.columns) == list(Ttr.columns))
    nn = new_columns(Xte, T)
    ok("respeita max_new_features", len(nn) <= p.max_new_features, f"({len(nn)}/{p.max_new_features})")
    ok("nao usa o alvo (nenhuma coluna igual ao rotulo)",
       not any(np.array_equal(pd.to_numeric(T[c], errors="coerce").to_numpy(), pd.to_numeric(ds.y.iloc[te], errors="coerce").to_numpy()) for c in nn))
    perm = np.random.default_rng(0).permutation(len(Xte))
    Tp = fitted.transform(Xte.iloc[perm].reset_index(drop=True))
    same = _frames_equal(Tp.reset_index(drop=True), T.iloc[perm].reset_index(drop=True))
    if getattr(fitted, "inductive", True):
        ok("invariante a ordem das linhas", same)
        single = pd.concat([fitted.transform(Xte.iloc[[i]]) for i in range(3)], ignore_index=True)
        ok("linha sozinha = linha em lote", _frames_equal(single, T.iloc[:3].reset_index(drop=True)))
    else:
        ok("metodo declara-se transdutivo (aviso, nao falha)", True, "(invariancia nao exigida)")
    try:
        F = Preprocessor().fit(Xtr if not len(nn) else fitted.transform(Xtr)).transform(T)
        ok("saida passa pelo pre-processamento comum", F.shape[0] == len(Xte))
    except Exception as e:  # noqa: BLE001
        ok("saida passa pelo pre-processamento comum", False, str(e)[:120])
    try:
        json.dumps(to_jsonable(fitted.report()))
        ok("report() e serializavel em JSON", True)
    except Exception as e:  # noqa: BLE001
        ok("report() e serializavel em JSON", False, str(e)[:120])
    failed = [n for n, c, _ in res if not c]
    if verbose:
        print("RESULTADO:", "tudo certo" if not failed else f"{len(failed)} falha(s): {failed}")
    return not failed, res
