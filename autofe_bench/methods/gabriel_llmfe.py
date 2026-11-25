"""Adaptacao do LLM_based_FE (github.com/GabrielFrederico1/LLM_based_FE) ao contrato do autofe-bench.

Desenho preservado do original (feature_engineering_llm.py):
  - o LLM propoe uma funcao `add_features(df)` que cria colunas com prefixo `fe_`, so com pandas/numpy;
  - guarda contra vazamento: a feature de uma linha nao pode depender de outras linhas (checada reexecutando
    a funcao num subconjunto e comparando);
  - cada coluna nova so entra se elevar a AUC da validacao cruzada INTERNA (ganho > min_delta), greedy;
  - a aceitacao nunca usa o conjunto de teste.

Simplificacoes em relacao ao original:
  - o original roda `exec` sem sandbox forte; aqui o codigo passa pelo sandbox do autofe-bench (sem imports, sem
    atributos privados, com limite de tempo);
  - o original repete o prompt com historico de aceitos/rejeitados; aqui sao iteracoes independentes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..contract import AutoFEMethod, FittedFE
from ..prep import Preprocessor
from ..sandbox import extract_code, run_function_program
from .legacy import columns_line, sample_rows_text

_FN = "add_features"


class GabrielFitted(FittedFE):
    inductive = True  # a funcao e row-wise (verificada no fit)

    def __init__(self, programs: list[tuple[str, list[str]]], info: dict | None = None):
        self.programs = list(programs)  # [(codigo, [colunas aceitas])]
        self.info = info or {}

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for code, cols in self.programs:
            new = run_function_program(code, X, fn_name=_FN, timeout=60)
            for c in cols:
                out[c] = new[c].to_numpy()
        return out

    def report(self) -> dict:
        return {"accepted": [{"cols": c, "code": code} for code, c in self.programs]}


def _is_rowwise(code: str, X: pd.DataFrame, new_cols: list[str]) -> bool:
    """Reexecuta a funcao num subconjunto e compara com o resultado no conjunto inteiro."""
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(X), size=min(60, len(X) // 2), replace=False))
    full = run_function_program(code, X, fn_name=_FN, timeout=60)
    sub = run_function_program(code, X.iloc[idx].copy(), fn_name=_FN, timeout=60)
    for c in new_cols:
        a = pd.to_numeric(full[c].iloc[idx], errors="coerce").to_numpy(float)
        b = pd.to_numeric(sub[c], errors="coerce").to_numpy(float)
        if not np.allclose(a, b, rtol=1e-6, atol=1e-9, equal_nan=True):
            return False
    return True


class GabrielLLMFE(AutoFEMethod):
    name = "gabriel_llmfe"
    version = "1.0"
    needs_llm = True

    def __init__(self, iterations: int = 3, min_delta: float = 0.003, folds: int = 5):
        self.iterations, self.min_delta, self.folds = iterations, min_delta, folds
        self.params = {"iterations": iterations, "min_delta": min_delta}

    def fit(self, X_train, y_train, task, ctx):
        if ctx.llm is None:
            return GabrielFitted([])
        pos = ctx.sample_rows(y_train)
        dfw = X_train.iloc[pos].reset_index(drop=True)
        y = y_train.iloc[pos].reset_index(drop=True)
        splits = ctx.inner_splits(y, self.folds)
        max_new = ctx.budget.max_new_features
        per_iter = max(1, max_new // max(1, self.iterations))

        def score(frame):
            F = Preprocessor().fit_transform(frame)
            if F.shape[1] == 0:
                return -np.inf
            return float(np.mean(ctx.cv_scores(F, y, splits)))

        cur = dfw.copy()
        cur_score = score(cur)
        programs: list[tuple[str, list[str]]] = []
        log = []
        n_new = 0
        for it in range(self.iterations):
            if ctx.remaining_calls() <= 0 or n_new >= max_new:
                break
            prompt = self._prompt(task, dfw, y, max_new, it)
            reply = ctx.llm.chat([{"role": "user", "content": prompt}])
            code = extract_code(reply)
            entry: dict = {"iter": it}
            try:
                out = run_function_program(code, dfw, fn_name=_FN, timeout=60)
                new = [c for c in out.columns if c not in dfw.columns]
                new = [c for c in new if str(c).startswith("fe_") and c not in cur.columns]
                new = [c for c in new if pd.to_numeric(out[c], errors="coerce").notna().mean() >= 0.5
                       and out[c].nunique() > 1]
                new = new[:max_new - n_new]
                if not new:
                    entry["status"] = "sem features validas"
                elif not _is_rowwise(code, dfw, new):
                    entry["status"] = "rejeitada: depende de outras linhas"
                else:
                    accepted_cols = []
                    for c in new:
                        cand = cur.copy()
                        cand[c] = out[c].to_numpy()
                        sc = score(cand)
                        gain = sc - cur_score
                        if gain > self.min_delta:
                            cur, cur_score = cand, sc
                            accepted_cols.append(c)
                            n_new += 1
                    entry["status"] = "ok"
                    entry["accepted"] = accepted_cols
                    if accepted_cols:
                        programs.append((code, accepted_cols))
            except Exception as e:  # noqa: BLE001
                entry["status"] = f"invalido: {type(e).__name__}: {e}"[:200]
            log.append(entry)

        return GabrielFitted(programs, info={"log": log, "score_final": cur_score})

    def _prompt(self, task, dfw, y, n_features, it):
        return (
            f"You are an expert in data science and feature engineering for tabular {task.kind_words} models.\n\n"
            f"TASK\nPropose new features that may improve a {task.kind_words} model.\n\n"
            f"DATASET\n{task.describe()}\n\n"
            f"{columns_line(dfw)}\n"
            f"Some labelled rows:\n{sample_rows_text(dfw, y, 5, it, task.target, task.kind)}\n\n"
            "REQUIREMENTS\n"
            f"1. Define exactly one function: def {_FN}(df): returning a copy of df with new columns added.\n"
            f"2. Propose at most {n_features} new features. Name every new column with the prefix fe_ and make it numeric.\n"
            "3. Row-wise only: a value must depend only on its own row. No mean, median, std, min, max, groupby, "
            "rank or scaling over other rows. Fixed domain thresholds are allowed.\n"
            "4. The target is not available. Never reference it.\n"
            "5. Handle NaN without crashing. Avoid division by zero and infinity.\n"
            "6. Use only pandas and numpy (as pd and np). No imports, no file or network access.\n"
            "7. Do not modify or drop the original columns.\n"
            "8. Reply with ONE python code block containing only the function."
        )
