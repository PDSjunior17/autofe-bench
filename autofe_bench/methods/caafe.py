"""CAAFE (Hollmann et al., 2023) reimplementado de forma fiel ao artigo e generalizado (binaria, multiclasse, regressao).

A cada iteracao o LLM escreve UM bloco de codigo (nova coluna ou remocao). O bloco e executado; se der erro, o erro volta
ao LLM. Se rodar, mede-se o desempenho em 10 divisoes aleatorias de validacao (RF, amostra do treino) e o bloco e
mantido se a media da melhora for > 0. Parametro extra (nao esta no artigo): `max_history` limita quantas trocas ficam
no contexto (None = historico inteiro, como no artigo); util para gastar menos tokens na cota gratuita."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit

from ..contract import AutoFEMethod
from ..llm import BudgetExceeded, QuotaExhausted
from ..prep import Preprocessor
from ..sandbox import extract_code, run_block
from .legacy import LegacyProgramFitted, columns_line


def _col_info(df: pd.DataFrame, samples: pd.DataFrame) -> str:
    lines = []
    for c in df.columns:
        vals = ", ".join(repr(v.item() if hasattr(v, "item") else v) for v in samples[c].tolist())
        lines.append(f"{c} ({df[c].dtype}): NaN-freq [{df[c].isna().mean() * 100:.1f}%], Samples [{vals}]")
    return "\n".join(lines)


def caafe_prompt(df: pd.DataFrame, task, n_rows: int, seed: int) -> str:
    samples = df.sample(min(10, len(df)), random_state=seed)
    clf = task.kind != "regression"
    algo = "classification" if clf else "regression"
    metric = "accuracy" if task.kind == "binary" else ("accuracy" if clf else "R2")
    return f"""The dataframe `df` is loaded and in memory. Columns are also named attributes.
Description of the dataset in `df` (column dtypes might be inaccurate):
"{task.describe()}"

{columns_line(df)}

Columns in `df` (true feature dtypes listed here; text columns are kept as strings):
{_col_info(df, samples)}

This code was written by an expert datascientist working to improve predictions. It is a snippet of code that adds new columns to the dataset.
Number of samples (rows) in training dataset: {n_rows}

This code generates additional columns that are useful for a downstream {algo} algorithm (such as XGBoost) predicting "{task.target}".
Additional columns add new semantic information, that is they use real world knowledge on the dataset. They can e.g. be feature combinations, transformations, aggregations where the new column is a function of the existing columns.
The scale of columns and offset does not matter. Make sure all used columns exist. Follow the above description of columns closely and consider the data types and meanings of classes.
This code also drops columns, if these may be redundant and hurt the predictive performance of the downstream {algo} (Feature selection). Dropping columns may help as the chance of overfitting is lower, especially if the dataset is small.
The {algo} model will be trained on the dataset with the generated columns and evaluated on a holdout set. The evaluation metric is {metric}. The best performing code will be selected.
Added columns can be used in other codeblocks, dropped columns are not available anymore.

Code formatting for each added column:
```python
# (Feature name and description)
# Usefulness: (Description why this adds useful real world knowledge to predict "{task.target}" according to dataset description and attributes.)
# Input samples: (Three samples of the columns used in the following code, e.g. 'Sex': ['male', 'female', 'male'], 'Age': [22.0, 38.0, 26.0], ...)
(Some pandas code using the columns above to add a new column for each row in df)
```end

Code formatting for dropping columns:
```python
# Explanation why the column XX is dropped
df.drop(columns=['XX'], inplace=True)
```end

Each codeblock generates exactly one useful column and can drop unused columns (Feature selection).
Each codeblock ends with ```end and starts with "```python"
Do not use imports (pandas is already available as `pd`, numpy as `np`) and do not print anything.
Codeblock:
"""


class CAAFE(AutoFEMethod):
    name = "caafe"
    version = "1.0"
    needs_llm = True

    def __init__(self, iterations: int | None = None, n_val_splits: int = 10, max_history: int | None = None,
                 verbose: bool = False):
        self.iterations, self.n_val_splits, self.max_history, self.verbose = iterations, n_val_splits, max_history, verbose
        self.params = {"iterations": iterations, "n_val_splits": n_val_splits, "max_history": max_history}

    def _scores(self, df, y, splits, ctx):
        F = Preprocessor().fit_transform(df)
        if F.shape[1] == 0:
            raise ValueError("nenhuma coluna numerica restou")
        return ctx.cv_scores(F, y, splits)

    def fit(self, X_train, y_train, task, ctx):
        pos = ctx.sample_rows(y_train)
        df_work = X_train.iloc[pos].reset_index(drop=True).copy()
        y = y_train.iloc[pos].reset_index(drop=True)
        splits = list(ShuffleSplit(n_splits=self.n_val_splits, test_size=0.25, random_state=ctx.seed).split(np.arange(len(df_work))))
        base = self._scores(df_work, y, splits, ctx)
        first = {"role": "user", "content": caafe_prompt(df_work, task, len(df_work), ctx.seed)}
        history: list[dict] = []
        accepted, log = [], []
        its = self.iterations or ctx.budget.max_llm_calls
        for it in range(1, its + 1):
            if ctx.remaining_calls() <= 0:
                break
            t0 = time.time()
            entry = {"iter": it}
            hist = history if self.max_history is None else history[-2 * self.max_history:]
            try:
                reply = ctx.llm.chat([first, *hist], stop=["```end"], seed=ctx.seed * 1000 + it)
            except QuotaExhausted:
                raise
            except BudgetExceeded:
                break
            except Exception as e:  # noqa: BLE001
                entry.update(status="llm_error", error=f"{type(e).__name__}: {e}"[:200])
                log.append(entry)
                break
            code = extract_code(reply)
            entry["code"] = code
            history.append({"role": "assistant", "content": f"```python\n{code}\n```end"})
            try:
                df_new = run_block(code, df_work, forbidden=(task.target,))
                new = self._scores(df_new, y, splits, ctx)
                delta = float(np.mean(new - base))
                keep = delta > 0
                entry.update(status="aceito" if keep else "rejeitado", delta=delta,
                             score_before=float(base.mean()), score_after=float(new.mean()))
                fb = (f"Performance before adding features {base.mean():.4f}. After adding features {new.mean():.4f}. "
                      f"Improvement {delta:+.4f}.\n")
                if keep:
                    df_work, base = df_new, new
                    accepted.append(code)
                    fb += "The code was executed and changes to `df` were kept.\n"
                else:
                    fb += "The last code changes to `df` were discarded. (Improvement must be > 0)\n"
                history.append({"role": "user", "content": fb + "Next codeblock:\n"})
            except QuotaExhausted:
                raise
            except Exception as e:  # noqa: BLE001
                entry.update(status="erro_codigo", error=f"{type(e).__name__}: {e}"[:200])
                history.append({"role": "user", "content": f"Feedback: failed with error: {type(e).__name__}: {e}. "
                                "Fix the problem (consider missing values and data types) and write the next codeblock:\n"})
            entry["seconds"] = round(time.time() - t0, 1)
            log.append(entry)
            if self.verbose:
                ctx.log(f"[caafe {it}] {entry['status']} {entry.get('delta', '')}")
        program = "\n\n".join(accepted)
        info = {"log": log, "n_accepted": len(accepted),
                "n_code_errors": sum(1 for e in log if e.get("status") == "erro_codigo")}
        return LegacyProgramFitted(program, "blocks", X_train, forbidden=(task.target,), info=info)
