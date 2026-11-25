"""LLM-FE (Abhyankar, Shojaee e Reddy, 2025) reimplementado e generalizado.

Busca evolutiva com ilhas: o LLM propoe programas `modify_features(df)`; cada programa valido e pontuado por validacao
cruzada (RF, amostra do treino) e entra numa ilha; novos prompts mostram k programas ja pontuados (do pior ao melhor).
Do artigo: 3 ilhas, 3 programas por prompt, temperatura 0.8, selecao de Boltzmann. Escolhas nossas (nao informadas no
artigo): k=2 exemplos, tau=0.05, reiniciar a pior ilha a cada 4 iteracoes. O orcamento total e o do protocolo."""
from __future__ import annotations

import ast
import random
import re
import textwrap
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..contract import AutoFEMethod
from ..llm import BudgetExceeded, QuotaExhausted
from ..prep import Preprocessor
from ..sandbox import extract_code, run_function_program
from .legacy import LegacyProgramFitted, columns_line, sample_rows_text

INITIAL = '''def modify_features(df):
    """Initial program: keep the dataframe unchanged."""
    df = df.copy()
    return df
'''


@dataclass
class _Island:
    programs: list = field(default_factory=list)

    def add(self, code, score):
        self.programs.append((code, score))

    def best(self):
        return max(self.programs, key=lambda p: (p[1], -len(p[0]))) if self.programs else None


def _sample_examples(island: _Island, k: int, tau: float, rng: random.Random):
    clusters = {}
    for code, sc in island.programs:
        clusters.setdefault(round(sc, 3), []).append((code, sc))
    keys = list(clusters)
    scores = np.array(keys, dtype=float)
    chosen, avail = [], list(range(len(keys)))
    for _ in range(min(k, len(keys))):
        s = scores[avail]
        p = np.exp((s - s.max()) / max(tau, 1e-6))
        p /= p.sum()
        j = rng.choices(avail, weights=p.tolist(), k=1)[0]
        avail.remove(j)
        members = clusters[keys[j]]
        w = [1.0 / (1 + len(c)) for c, _ in members]
        chosen.append(rng.choices(members, weights=w, k=1)[0])
    return sorted(chosen, key=lambda p: p[1])


def _body(code: str) -> str:
    try:
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                seg = ast.get_source_segment(code, node) or ""
                return textwrap.dedent("\n".join(seg.splitlines()[1:]))
    except Exception:  # noqa: BLE001
        pass
    return "df = df.copy()\nreturn df"


def _normalize(code: str) -> str:
    code = re.sub(r"def\s+modify_features\w*\s*\(", "def modify_features(", code)
    return code.strip("\n") + "\n"


def llmfe_prompt(examples, df, y, task, seed, n_rows=10):
    score_desc = {"binary": "mean(accuracy, ROC AUC)", "multiclass": "mean(accuracy, one-vs-rest ROC AUC)",
                  "regression": "R2"}[task.kind]
    parts = [f"""You are an expert data scientist doing feature engineering for a tabular {task.kind_words} task.

Task description:
{task.describe()}

{columns_line(df)}

A few labelled samples of the data (raw columns):
{sample_rows_text(df, y, n_rows, seed, task.target, task.kind)}

Programs are Python functions `modify_features(df)` that receive a pandas DataFrame with the raw columns above
and return a DataFrame with new or transformed columns (pandas as `pd` and numpy as `np` are available; no imports,
no printing, never use the target column "{task.target}", never change the number of rows).
Each program is scored by the validation score = {score_desc} of a random forest trained on the
engineered features (higher is better, cross-validation).
Missing values (NaN) are imputed automatically; text columns with many distinct values are dropped, so extract
useful information from them. Below are previous programs with their scores, from worst to best. Improve on them:
use your domain knowledge about this data, think step by step, create novel and meaningful features, and keep the
code robust to NaN.
"""]
    for i, (code, score) in enumerate(examples):
        parts.append(f"```python\n# score: {score:.4f}\ndef modify_features_v{i}(df):\n"
                     + "\n".join("    " + ln for ln in _body(code).splitlines()) + "\n```")
    parts.append(f"Write the next version `modify_features_v{len(examples)}` that achieves a HIGHER score. "
                 "Start with a short comment explaining the idea, then the function. Answer with a single ```python code block.")
    return "\n\n".join(parts)


class LLMFE(AutoFEMethod):
    name = "llmfe"
    version = "1.0"
    needs_llm = True

    def __init__(self, b: int = 3, n_islands: int = 3, k: int = 2, tau: float = 0.05, temperature: float = 0.8,
                 reset_every: int = 4, folds: int = 5, verbose: bool = False):
        self.b, self.n_islands, self.k, self.tau = b, n_islands, k, tau
        self.temperature, self.reset_every, self.folds, self.verbose = temperature, reset_every, folds, verbose
        self.params = {"b": b, "n_islands": n_islands, "k": k, "tau": tau, "temperature": temperature,
                       "reset_every": reset_every}

    def fit(self, X_train, y_train, task, ctx):
        rng = random.Random(ctx.seed)
        pos = ctx.sample_rows(y_train)
        dfw = X_train.iloc[pos].reset_index(drop=True).copy()
        y = y_train.iloc[pos].reset_index(drop=True)
        splits = ctx.inner_splits(y, self.folds)

        def score(frame):
            F = Preprocessor().fit_transform(frame)
            if F.shape[1] == 0:
                raise ValueError("nenhuma coluna numerica restou")
            return float(np.mean(ctx.cv_scores(F, y, splits)))

        base = score(dfw)
        islands = [_Island([(INITIAL, base)]) for _ in range(self.n_islands)]
        seen = {INITIAL.strip()}
        log = [{"iter": 0, "status": "inicial", "score": base}]
        it = 0
        while ctx.remaining_calls() > 0:
            it += 1
            isl_id = rng.randrange(self.n_islands)
            examples = _sample_examples(islands[isl_id], self.k, self.tau, rng)
            prompt = llmfe_prompt(examples, dfw, y, task, ctx.seed * 1000 + it)
            stop_all = False
            for j in range(self.b):
                if ctx.remaining_calls() <= 0:
                    stop_all = True
                    break
                entry = {"iter": it, "sample": j, "island": isl_id}
                try:
                    reply = ctx.llm.chat([{"role": "user", "content": prompt}], temperature=self.temperature,
                                         seed=ctx.seed * 100000 + it * 100 + j)
                except QuotaExhausted:
                    raise
                except BudgetExceeded:
                    stop_all = True
                    break
                except Exception as e:  # noqa: BLE001
                    entry.update(status="llm_error", error=f"{type(e).__name__}: {e}"[:200])
                    log.append(entry)
                    stop_all = True
                    break
                code = _normalize(extract_code(reply))
                entry["code"] = code
                if code.strip() in seen:
                    entry["status"] = "duplicado"
                    log.append(entry)
                    continue
                seen.add(code.strip())
                try:
                    out = run_function_program(code, dfw, forbidden=(task.target,))
                    sc = score(out)
                    islands[isl_id].add(code, sc)
                    entry.update(status="valido", score=sc, n_features=int(out.shape[1]))
                except QuotaExhausted:
                    raise
                except Exception as e:  # noqa: BLE001
                    entry.update(status="invalido", error=f"{type(e).__name__}: {e}"[:200])
                log.append(entry)
            if stop_all:
                break
            if it % self.reset_every == 0:
                bests = [isl.best()[1] for isl in islands]
                worst, top = int(np.argmin(bests)), int(np.argmax(bests))
                if worst != top:
                    islands[worst] = _Island([islands[top].best()])
        best_code, best_score = max((isl.best() for isl in islands), key=lambda p: (p[1], -len(p[0])))
        info = {"log": log, "score_inicial": base, "score_melhor": best_score,
                "n_valid_programs": sum(1 for e in log if e.get("status") == "valido"),
                "n_programs": sum(1 for e in log if e.get("iter", 0) > 0)}
        program = "" if best_code.strip() == INITIAL.strip() else best_code
        return LegacyProgramFitted(program, "function", X_train, forbidden=(task.target,), info=info)
