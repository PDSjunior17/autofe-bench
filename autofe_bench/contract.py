"""Contrato entre o harness e os metodos de AutoFE.

Um metodo implementa `AutoFEMethod.fit(X_train, y_train, task, ctx)` e devolve um `FittedFE`
com `transform(X)`. O harness nunca entrega o teste ao `fit`; o teste so aparece em `transform`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Callable

import numpy as np
import pandas as pd

TASK_KINDS = ("binary", "multiclass", "regression")


@dataclass
class Task:
    """Descreve a base para o metodo (o LLM usa isto para entender as colunas)."""

    name: str
    target: str
    kind: str  # binary | multiclass | regression
    description: str = ""
    column_descriptions: dict = field(default_factory=dict)
    n_classes: int = 0

    def __post_init__(self):
        if self.kind not in TASK_KINDS:
            raise ValueError(f"kind deve ser um de {TASK_KINDS}")

    @property
    def primary(self) -> str:
        """Metrica principal do placar."""
        return {"binary": "auc", "multiclass": "logloss", "regression": "rmse"}[self.kind]

    @property
    def higher_is_better(self) -> bool:
        return self.kind == "binary"

    def describe(self) -> str:
        """Texto de descricao pronto para prompts."""
        parts = []
        if self.description:
            parts.append(self.description.strip())
        if self.column_descriptions:
            parts.append("Columns:\n" + "\n".join(f"- {c}: {d}" for c, d in self.column_descriptions.items()))
        return "\n".join(parts)

    @property
    def kind_words(self) -> str:
        return {"binary": "binary classification", "multiclass": "multiclass classification",
                "regression": "regression"}[self.kind]


@dataclass
class Budget:
    """Limites iguais para todos os metodos (definidos pelo Protocol)."""

    max_llm_calls: int = 20
    max_new_features: int = 10
    time_limit_s: int = 3600
    max_total_tokens: int | None = None
    search_max_rows: int = 5000  # linhas usadas na busca (amostra do treino)

    def to_dict(self):
        return asdict(self)


class Context:
    """Tudo o que o metodo tem direito de usar durante o `fit`."""

    def __init__(self, llm, budget: Budget, seed: int, task: Task, log: Callable[[str], None] | None = None):
        self.llm = llm  # BudgetedLLM ou None
        self.budget = budget
        self.seed = seed
        self.task = task
        self._log = log or (lambda s: None)
        self.notes: dict[str, Any] = {}

    def log(self, msg: str):
        self._log(msg)

    # utilitarios de validacao interna (so sobre o treino)
    def inner_splits(self, y: pd.Series, n_splits: int = 5, seed: int | None = None):
        from .prep import inner_splits

        return inner_splits(y, self.task.kind, n_splits, self.seed if seed is None else seed)

    def sample_rows(self, y: pd.Series, n: int | None = None, seed: int | None = None, exclude=None):
        """Posicoes (iloc) de uma amostra estratificada do treino com ate `n` linhas."""
        from .prep import sample_positions

        n = self.budget.search_max_rows if n is None else n
        return sample_positions(y, self.task.kind, n, self.seed if seed is None else seed, exclude)

    def cv_scores(self, F: pd.DataFrame, y: pd.Series, splits, n_estimators: int = 100):
        from .prep import cv_scores

        return cv_scores(F, y, splits, self.task.kind, self.seed, n_estimators=n_estimators)

    def remaining_calls(self) -> int:
        return 0 if self.llm is None else self.llm.remaining_calls()


class FittedFE:
    """Resultado do `fit`. `transform` recebe features cruas e devolve features (cruas + novas)."""

    inductive: bool = True  # False se o metodo calcula estatisticas usando as proprias linhas transformadas

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover - interface
        raise NotImplementedError

    def report(self) -> dict:
        """Dicionario serializavel em JSON com o que o metodo quiser registrar (programas, motivos...)."""
        return {}


class AutoFEMethod:
    """Classe-base. Defina `name` e `version`; implemente `fit`."""

    name: str = "metodo"
    version: str = "0"
    needs_llm: bool = False

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, task: Task, ctx: Context) -> FittedFE:  # pragma: no cover
        raise NotImplementedError


class IdentityFitted(FittedFE):
    def transform(self, X):
        return X.copy()


def new_columns(original: pd.DataFrame, out: pd.DataFrame) -> list[str]:
    """Colunas de `out` que nao existiam em `original` (as features novas)."""
    orig = set(map(str, original.columns))
    return [c for c in out.columns if str(c) not in orig]


def to_jsonable(obj, max_len: int = 4000):
    """Converte para algo gravavel em JSON; textos muito longos sao truncados."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, max_len) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v, max_len) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, str):
        return obj if len(obj) <= max_len else obj[:max_len] + "...[truncado]"
    if obj is None or isinstance(obj, (int, bool)):
        return obj
    return str(obj)[:max_len]
