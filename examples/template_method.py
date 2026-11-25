"""Modelo para testar o SEU método no harness. Copie este arquivo, renomeie a classe e implemente `fit`.

Rodar o teste de contrato (poucas chamadas):
    python -m autofe_bench check examples/template_method.py:MeuMetodo --llm ollama:qwen3-coder:30b
Rodar a comparacao com os outros:
    run_benchmark(["examples/template_method.py:MeuMetodo", "caafe", "llmfe", "guided"], ["titanic"], protocol="dev", client=client)

REGRAS DO PROTOCOLO (o `check` confere a maioria):
  1. `fit` so enxerga o treino (X_train, y_train). O teste so aparece em `transform`.
  2. `transform(X)` devolve as colunas ORIGINAIS + as novas, com o mesmo indice e numero de linhas de X.
  3. Estatisticas (medias por grupo, frequencias, quantis) devem ser calculadas no TREINO e guardadas no FittedFE.
     Nunca calcule estatistica sobre as linhas que chegam em transform (isso e transdutivo; o harness avisa).
  4. Use `ctx.llm.chat([...])` para chamar o LLM: ele conta as chamadas e respeita o orcamento (ctx.budget).
  5. No maximo ctx.budget.max_new_features colunas novas. Nao use o alvo para criar colunas do teste.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from autofe_bench import AutoFEMethod, FittedFE


class MeuFitted(FittedFE):
    inductive = True  # declare False se seu metodo calcula estatisticas sobre as linhas de teste

    def __init__(self, num_cols, medians):
        self.num_cols, self.medians = num_cols, medians

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for c in self.num_cols:                      # exemplo: indicador de ausente + preenchimento pela mediana do TREINO
            out[f"{c}_ausente"] = X[c].isna().astype(int)
        return out

    def report(self) -> dict:                         # vai para o results.jsonl (precisa ser serializavel em JSON)
        return {"num_cols": self.num_cols}


class MeuMetodo(AutoFEMethod):
    name = "meu_metodo"       # nome que aparece no relatorio
    version = "1"             # MUDE ao alterar o metodo: resultados de versoes diferentes ficam separados
    needs_llm = False         # True se usar ctx.llm

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, task, ctx) -> FittedFE:
        num = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c]) and X_train[c].isna().any()]
        num = num[: ctx.budget.max_new_features]
        # com LLM:  reply = ctx.llm.chat([{"role": "user", "content": "..."}]);  ctx.remaining_calls() diz quantas restam
        return MeuFitted(num, {c: float(X_train[c].median()) for c in num})
