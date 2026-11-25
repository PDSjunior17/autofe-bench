"""Adaptadores para bibliotecas externas. Precisam do pacote instalado; falham com mensagem clara se faltar.

ATENCAO: FeatCopilot com o motor 'llm' chama o LLM por conta propria (via litellm), FORA do gateway do harness:
as chamadas dele NAO sao contadas nem limitadas pelo protocolo. Use engines=('tabular',) para um baseline sem LLM
(comparavel e medido) ou registre a ressalva ao relatar. Estes adaptadores NAO foram testados no ambiente de
desenvolvimento do harness (sem acesso aos pacotes); teste-os no Colab com `check` e uma base pequena."""
from __future__ import annotations

import os

import pandas as pd

from ..contract import AutoFEMethod, FittedFE


class _FeatCopilotFitted(FittedFE):
    inductive = True

    def __init__(self, eng, info):
        self.eng, self.info = eng, info

    def transform(self, X):
        out = self.eng.transform(X)
        out = pd.DataFrame(out).reset_index(drop=True)
        out.index = X.index
        return out

    def report(self):
        return self.info


class FeatCopilot(AutoFEMethod):
    name = "featcopilot"
    version = "1.0"
    needs_llm = False  # o motor llm (se ativado) usa litellm por fora do gateway

    def __init__(self, engines=("tabular",), llm_model: str | None = None, api_base: str | None = None):
        self.engines, self.llm_model, self.api_base = tuple(engines), llm_model, api_base
        self.params = {"engines": list(self.engines), "llm_model": llm_model}

    def fit(self, X_train, y_train, task, ctx):
        try:
            from featcopilot import AutoFeatureEngineer
        except ImportError as e:
            raise ImportError("featcopilot nao instalado: pip install featcopilot") from e
        cfg = None
        if "llm" in self.engines:
            cfg = {"model": self.llm_model, "backend": "litellm", "max_suggestions": 10, "domain": task.name,
                   "validate_features": True, "temperature": 0.3}
            if self.api_base:
                cfg["api_base"] = self.api_base
        eng = AutoFeatureEngineer(engines=list(self.engines), max_features=ctx.budget.max_new_features + X_train.shape[1],
                                  llm_config=cfg, verbose=False)
        kw = {}
        if "llm" in self.engines:
            kw = dict(column_descriptions=task.column_descriptions, task_description=task.describe())
        try:
            eng.fit(X_train, y_train, **kw)
        except TypeError:
            eng.fit(X_train, y_train)
        info = {"engines": list(self.engines), "llm_calls_unmetered": "llm" in self.engines}
        try:
            info["explanations"] = {k: str(v) for k, v in list(eng.explain_features().items())[:40]}
        except Exception:  # noqa: BLE001
            pass
        return _FeatCopilotFitted(eng, info)


class _OpenFEFitted(FittedFE):
    inductive = True

    def __init__(self, features, n_jobs):
        self.features, self.n_jobs = features, n_jobs

    def transform(self, X):
        from openfe import transform

        # o OpenFE exige dois conjuntos; passamos X duas vezes e usamos so o primeiro
        out, _ = transform(X.copy(), X.copy(), self.features, n_jobs=self.n_jobs)
        out.index = X.index
        return out

    def report(self):
        return {"n_features_selected": len(self.features)}


class OpenFE(AutoFEMethod):
    name = "openfe"
    version = "1.0"
    needs_llm = False

    def __init__(self, n_jobs: int = 2, top_k: int | None = None):
        self.n_jobs, self.top_k = n_jobs, top_k
        self.params = {"n_jobs": n_jobs, "top_k": top_k}

    def fit(self, X_train, y_train, task, ctx):
        try:
            from openfe import OpenFE as _OFE
        except ImportError as e:
            raise ImportError("openfe nao instalado: pip install openfe") from e
        ofe = _OFE()
        feats = ofe.fit(data=X_train, label=y_train, n_jobs=self.n_jobs, task="classification" if task.kind != "regression" else "regression",
                        seed=ctx.seed, verbose=False)
        feats = feats[: (self.top_k or ctx.budget.max_new_features)]
        return _OpenFEFitted(feats, self.n_jobs)
