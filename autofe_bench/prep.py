"""Pre-processamento comum (ajustado so no treino), modelos finais e metricas. Igual para TODOS os metodos."""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, f1_score, log_loss, mean_absolute_error, mean_squared_error,
                             r2_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

MAX_ONEHOT_CARD = 15


def _clean(name) -> str:
    return re.sub(r"[^0-9a-zA-Z_]+", "_", str(name))


class Preprocessor:
    """Transforma o DataFrame de um metodo em matriz numerica. Ajustado SO no treino:
    numericas e booleanas ficam; texto/categorica com <= max_card valores vira one-hot (NaN e uma categoria);
    texto com mais valores e descartado (o metodo de FE precisa extrair a informacao dele). Niveis nao vistos viram zeros."""

    def __init__(self, max_card: int = MAX_ONEHOT_CARD):
        self.max_card = max_card

    def fit(self, df: pd.DataFrame):
        self.num_cols, self.cat_levels, self.dropped = [], {}, []
        for col in df.columns:
            s = df[col]
            if pd.api.types.is_bool_dtype(s) or (pd.api.types.is_numeric_dtype(s) and not isinstance(s.dtype, pd.CategoricalDtype)):
                self.num_cols.append(col)
            else:
                v = s.astype(object).where(s.notna(), "__nan__").astype(str)
                lv = sorted(v.unique())
                if len(lv) <= self.max_card:
                    self.cat_levels[col] = lv
                else:
                    self.dropped.append(col)
        names, seen = [], set()
        self._plan = []
        for col in self.num_cols:
            n = _clean(col)
            self._plan.append(("num", col, None, n))
            names.append(n)
        for col, lv in self.cat_levels.items():
            for l in lv:
                self._plan.append(("cat", col, l, _clean(f"{col}_{l}")))
        final = []
        for i, (k, c, l, n) in enumerate(self._plan):
            base, j = n, 1
            while n in seen:
                j += 1
                n = f"{base}_{j}"
            seen.add(n)
            self._plan[i] = (k, c, l, n)
            final.append(n)
        self.columns_out = final
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = {}
        cache = {}
        for kind, col, lvl, name in self._plan:
            if col not in df.columns:
                out[name] = np.zeros(len(df)) if kind == "cat" else np.full(len(df), np.nan)
                continue
            if kind == "num":
                s = pd.to_numeric(df[col].astype(float) if pd.api.types.is_bool_dtype(df[col]) else df[col], errors="coerce")
                out[name] = s.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
            else:
                if col not in cache:
                    cache[col] = df[col].astype(object).where(df[col].notna(), "__nan__").astype(str).to_numpy()
                out[name] = (cache[col] == lvl).astype(float)
        return pd.DataFrame(out, index=range(len(df)))

    def fit_transform(self, df):
        return self.fit(df).transform(df)


# ---------------------------------------------------------------- divisoes e amostragem
def inner_splits(y: pd.Series, kind: str, n_splits: int, seed: int):
    y = pd.Series(np.asarray(y))
    if kind != "regression" and y.value_counts().min() >= n_splits:
        return list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))
    return list(KFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y))))


def sample_positions(y: pd.Series, kind: str, n: int, seed: int, exclude=None) -> np.ndarray:
    """Posicoes (iloc) ordenadas de uma amostra de ate n linhas (estratificada na classificacao)."""
    pos = np.arange(len(y))
    if exclude is not None:
        pos = np.setdiff1d(pos, np.asarray(exclude))
    if n is None or len(pos) <= n:
        return pos
    yy = np.asarray(y)[pos]
    strat = yy if (kind != "regression" and pd.Series(yy).value_counts().min() >= 2) else None
    keep, _ = train_test_split(pos, train_size=n, random_state=seed, stratify=strat)
    return np.sort(keep)


# ---------------------------------------------------------------- modelos
def make_model(name: str, kind: str, seed: int = 0, rf_trees: int = 300, rf_depth: int = 6, hgb_iter: int = 200):
    clf = kind != "regression"
    if name == "lr":
        est = LogisticRegression(max_iter=3000, random_state=seed) if clf else Ridge(alpha=1.0)
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), est)
    if name == "rf":
        cls = RandomForestClassifier if clf else RandomForestRegressor
        return make_pipeline(SimpleImputer(strategy="median"),
                             cls(n_estimators=rf_trees, max_depth=rf_depth, min_samples_leaf=5, random_state=seed, n_jobs=-1))
    if name == "hgb":
        cls = HistGradientBoostingClassifier if clf else HistGradientBoostingRegressor
        return cls(max_iter=hgb_iter, learning_rate=0.05, early_stopping=False, random_state=seed)
    if name == "xgb":
        import xgboost as xgb  # opcional

        est = (xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                                 random_state=seed, n_jobs=-1, verbosity=0)
               if clf else xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9,
                                            colsample_bytree=0.9, random_state=seed, n_jobs=-1, verbosity=0))
        return make_pipeline(SimpleImputer(strategy="median"), est)
    raise ValueError(f"modelo desconhecido: {name}")


def metrics(kind: str, y_true, model, F, classes=None) -> dict:
    y_true = np.asarray(y_true)
    if kind == "regression":
        p = model.predict(F)
        return {"rmse": float(np.sqrt(mean_squared_error(y_true, p))), "mae": float(mean_absolute_error(y_true, p)),
                "r2": float(r2_score(y_true, p))}
    proba = model.predict_proba(F)
    cls = list(model.classes_) if hasattr(model, "classes_") else list(model[-1].classes_)
    pred = np.asarray(cls)[proba.argmax(1)]
    out = {"acc": float(accuracy_score(y_true, pred))}
    if kind == "binary":
        out["auc"] = float(roc_auc_score(y_true, proba[:, 1]))
        out["f1"] = float(f1_score(y_true, pred))
    else:
        out["f1_macro"] = float(f1_score(y_true, pred, average="macro"))
        try:
            out["auc"] = float(roc_auc_score(y_true, proba, multi_class="ovr", labels=cls))
        except Exception:  # noqa: BLE001
            out["auc"] = float("nan")
    try:
        out["logloss"] = float(log_loss(y_true, proba, labels=cls))
    except Exception:  # noqa: BLE001
        out["logloss"] = float("nan")
    return out


def evaluate_models(Ftr, ytr, Fte, yte, kind: str, models, seed: int, rf_trees=300, rf_depth=6, hgb_iter=200) -> dict:
    res = {}
    for m in models:
        try:
            mdl = make_model(m, kind, seed, rf_trees, rf_depth, hgb_iter)
            mdl.fit(Ftr, ytr)
            res[m] = metrics(kind, yte, mdl, Fte)
        except Exception as e:  # noqa: BLE001
            res[m] = {"error": f"{type(e).__name__}: {e}"[:300]}
    return res


def _fold_score(kind: str, y_true, model, F, mode: str = "legacy") -> float:
    """Pontuacao de validacao interna, 'maior e melhor'.
    legacy (CAAFE, LLM-FE): (acc+auc)/2 na classificacao, R2 na regressao.
    primary (GuidedFE): a metrica primaria do protocolo, que e continua e menos ruidosa que a acuracia:
    AUC (binaria), -logloss (multiclasse), R2 (regressao)."""
    m = metrics(kind, y_true, model, F)
    if kind == "regression":
        return m["r2"]
    if mode == "primary":
        if kind == "binary" and np.isfinite(m.get("auc", float("nan"))):
            return m["auc"]
        if kind == "multiclass" and np.isfinite(m.get("logloss", float("nan"))):
            return -m["logloss"]
    auc = m.get("auc", float("nan"))
    return (m["acc"] + (auc if np.isfinite(auc) else m["acc"])) / 2


def cv_scores(F: pd.DataFrame, y, splits, kind: str, seed: int, n_estimators: int = 100, depth: int = 6) -> np.ndarray:
    """Pontuacao por dobra com RF (o avaliador interno padrao). F deve ser numerica."""
    Fv = F.reset_index(drop=True)
    yv = pd.Series(np.asarray(y)).reset_index(drop=True)
    out = []
    for tr, va in splits:
        mdl = make_model("rf", kind, seed, rf_trees=n_estimators, rf_depth=depth)
        mdl.fit(Fv.iloc[tr], yv.iloc[tr])
        out.append(_fold_score(kind, yv.iloc[va], mdl, Fv.iloc[va]))
    return np.array(out)
