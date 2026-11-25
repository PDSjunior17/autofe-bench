"""Aplicacao de programas no estilo 'df -> df' (CAAFE, LLM-FE): sao TRANSDUTIVOS, pois o programa ve as linhas
que transforma (estatisticas de grupo, por exemplo). O harness detecta isso com a sonda e marca o resultado."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..contract import FittedFE
from ..sandbox import run_block, run_function_program


class LegacyProgramFitted(FittedFE):
    inductive = False

    def __init__(self, code: str, kind: str, ref: pd.DataFrame, forbidden=(), info=None):
        self.code, self.kind, self.forbidden = code or "", kind, tuple(forbidden)
        self.ref = ref.reset_index(drop=True).copy()
        self.info = info or {}
        self._cols = None

    def _train_cols(self):
        """Colunas que o programa produz no treino. Teste e treino saem SEMPRE com estas (ex.: get_dummies que
        cria colunas diferentes quando o conjunto muda deixaria treino e teste incompativeis)."""
        if self._cols is None:
            self._cols = list(self._run(self.ref.copy()).columns)
        return self._cols

    def _run(self, df):
        if not self.code.strip():
            return df.copy()
        fn = run_block if self.kind == "blocks" else run_function_program
        return fn(self.code, df, forbidden=self.forbidden, timeout=120)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.reset_index(drop=True)
        cols = self._train_cols()
        if len(X) == len(self.ref) and X.equals(self.ref):
            return self._run(X)[cols]
        both = pd.concat([self.ref, X], axis=0, ignore_index=True)
        out = self._run(both).iloc[len(self.ref):].reset_index(drop=True)
        return out.reindex(columns=cols)

    def report(self):
        return {"program": self.code, **self.info}


def dtype_name(s: pd.Series) -> str:
    return str(s.dtype)


def sample_rows_text(df: pd.DataFrame, y: pd.Series, n: int, seed: int, target_name: str, kind: str) -> str:
    idx = df.sample(min(n, len(df)), random_state=seed).index
    out = []
    for i in idx:
        feats = ", ".join(f"{c} is {_py(df.at[i, c])!r}" for c in df.columns)
        tv = _py(y.at[i])
        tv = round(float(tv), 4) if kind == "regression" else int(tv)
        out.append(f"If {feats}. Then {target_name} is {tv}.")
    return "\n".join(out)


def _py(v):
    return v.item() if hasattr(v, "item") else v


def columns_line(df: pd.DataFrame) -> str:
    """'Columns: a (int64), b (object)'. Linha padrao presente em todos os prompts (o LLM simulado a le)."""
    return "Columns: " + ", ".join(f"{c} ({dtype_name(df[c])})" for c in df.columns)
