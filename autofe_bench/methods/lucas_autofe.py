"""Adaptacao do AutoFe-LLM (github.com/LucasBarros28/AutoFe-LLM) ao contrato do autofe-bench.

Desenho preservado do original:
  - a LLM recebe so ESTATISTICAS das colunas (tipo, % ausentes, media, desvio, min, max, assimetria), nunca linhas;
  - a LLM so pode sugerir operacoes de uma lista permitida: log1p, missing_indicator, interaction_product;
  - as sugestoes sao validadas (operacao permitida, colunas existentes) e executadas pelo nosso motor, nunca pela LLM.

Simplificacoes em relacao ao original (que tem ~1700 linhas e um detector de oportunidades proprio):
  - sem detector de oportunidades: a LLM ve as estatisticas e propoe diretamente;
  - sem circuit breaker por provedor (o orcamento do protocolo ja limita as chamadas);
  - a escolha entre as sugestoes e feita por validacao cruzada interna no treino, greedy (aceita se o ganho > min_delta).
Sem LLM disponivel, o metodo nao adiciona nenhuma feature (equivale ao sem_fe).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..contract import AutoFEMethod, FittedFE
from ..prep import Preprocessor
from ..sandbox import extract_code
from .legacy import columns_line

ALLOWED_OPS = ("log1p", "missing_indicator", "interaction_product")
_ARITY = {"log1p": 1, "missing_indicator": 1, "interaction_product": 2}


def _numeric_cols(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns
            if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])]


def _apply(op: str, cols: tuple, X: pd.DataFrame):
    """Devolve (nome_da_feature, serie). Todas as operacoes sao row-wise (nao usam estatisticas)."""
    if op == "log1p":
        c = cols[0]
        return f"fe_log1p_{c}", np.log1p(X[c].astype(float).clip(lower=0))
    if op == "missing_indicator":
        c = cols[0]
        return f"fe_missing_{c}", X[c].isna().astype(float)
    a, b = cols
    return f"fe_prod_{a}_{b}", X[a].astype(float) * X[b].astype(float)


def _describe_columns(X: pd.DataFrame) -> str:
    lines = []
    for c in X.columns:
        s = X[c]
        if _is_num(s):
            v = s.dropna().astype(float)
            skew = float(v.skew()) if len(v) > 2 else 0.0
            lines.append(f"- {c}: numeric, {s.isna().mean() * 100:.1f}% missing, mean={v.mean():.4g}, "
                         f"std={v.std():.4g}, min={v.min():.4g}, max={v.max():.4g}, skew={skew:.2f}")
        else:
            lines.append(f"- {c}: categorical/text, {s.isna().mean() * 100:.1f}% missing, "
                         f"{s.nunique()} distinct values")
    return "\n".join(lines)


def _is_num(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _parse_ops(reply: str, X: pd.DataFrame, max_n: int) -> list[tuple[str, tuple]]:
    """Linhas 'OP|col1|col2' validadas. Descarta o que nao for permitido ou nao existir."""
    code = extract_code(reply)
    out, seen = [], set()
    nums = set(_numeric_cols(X))
    for raw in code.splitlines():
        parts = [p.strip() for p in raw.strip().strip("`").split("|")]
        if not parts or parts[0] not in ALLOWED_OPS:
            continue
        op, cols = parts[0], tuple(parts[1:1 + _ARITY[parts[0]]])
        if len(cols) != _ARITY[op] or any(c not in X.columns for c in cols):
            continue
        if op in ("log1p", "interaction_product") and any(c not in nums for c in cols):
            continue
        key = (op, cols)
        if key in seen:
            continue
        seen.add(key)
        out.append((op, cols))
        if len(out) >= max_n:
            break
    return out


class LucasFitted(FittedFE):
    inductive = True  # operacoes row-wise: o valor de uma linha nao depende das outras

    def __init__(self, ops: list[tuple[str, tuple]], info: dict | None = None):
        self.ops = list(ops)
        self.info = info or {}

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for op, cols in self.ops:
            name, s = _apply(op, cols, X)
            out[name] = s.values
        return out

    def report(self) -> dict:
        return {"ops": [f"{op}({', '.join(c)})" for op, c in self.ops]}


class LucasAutoFE(AutoFEMethod):
    name = "lucas_autofe"
    version = "1.0"
    needs_llm = True

    def __init__(self, min_delta: float = 0.003, iterations: int = 2, folds: int = 5):
        self.min_delta, self.iterations, self.folds = min_delta, iterations, folds
        self.params = {"min_delta": min_delta, "iterations": iterations}

    def fit(self, X_train, y_train, task, ctx):
        if ctx.llm is None:
            return LucasFitted([])
        pos = ctx.sample_rows(y_train)
        dfw = X_train.iloc[pos].reset_index(drop=True)
        y = y_train.iloc[pos].reset_index(drop=True)
        splits = ctx.inner_splits(y, self.folds)
        max_new = ctx.budget.max_new_features

        def score(frame):
            F = Preprocessor().fit_transform(frame)
            return float(np.mean(ctx.cv_scores(F, y, splits)))

        base = score(dfw)
        cur_score = base
        candidates: list[tuple[str, tuple]] = []
        for _ in range(self.iterations):
            if ctx.remaining_calls() <= 0:
                break
            prompt = self._prompt(task, dfw, max_new)
            reply = ctx.llm.chat([{"role": "user", "content": prompt}])
            for cand in _parse_ops(reply, dfw, max_new * 2):
                if cand not in candidates:
                    candidates.append(cand)

        accepted: list[tuple[str, tuple]] = []
        log = []
        frame = dfw.copy()
        for op, cols in candidates:
            if len(accepted) >= max_new:
                break
            name, s = _apply(op, cols, dfw)
            if name in frame.columns:
                continue
            cand_frame = frame.copy()
            cand_frame[name] = s.values
            sc = score(cand_frame)
            gain = sc - cur_score
            if gain > self.min_delta:
                frame, cur_score = cand_frame, sc
                accepted.append((op, cols))
                log.append({"op": op, "cols": list(cols), "gain": gain, "accepted": True})
            else:
                log.append({"op": op, "cols": list(cols), "gain": gain, "accepted": False})

        fitted = LucasFitted(accepted, info={"score_base": base, "score_final": cur_score, "candidates": log})
        return fitted

    def _prompt(self, task, dfw, max_new):
        ops = "\n".join([
            "- log1p|COL : log(1+x) de uma coluna numerica com valores >= 0 (para assimetria)",
            "- missing_indicator|COL : 1 se a coluna estiver ausente, 0 caso contrario",
            "- interaction_product|COL1|COL2 : produto de duas colunas numericas",
        ])
        return (
            f"You are an expert data scientist doing feature engineering for a tabular {task.kind_words} task.\n\n"
            f"Task description:\n{task.describe()}\n\n"
            f"{columns_line(dfw)}\n"
            f"Column statistics (computed on the training data):\n{_describe_columns(dfw)}\n\n"
            f"Choose at most {max_new} operations from this list only:\n{ops}\n\n"
            "Answer with ONE ```text code block containing one operation per line, in the form OP|COL1|COL2 "
            "(omit COL2 for one-column operations). Do not explain. Do not use the target column."
        )
