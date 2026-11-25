"""GuidedFE: AutoFE com LLM guiado por perfil de dados, com selecao validada e sem vazamento por construcao.

Cada melhoria abaixo responde a um problema observado nos experimentos do Titanic e na literatura:

 1. PERFIL DOS DADOS no prompt (tipos, cardinalidade, faixas, associacao com o alvo, amostra de linhas, dicionario de
    colunas), para o LLM nao 'chutar'.
 2. MENU DE FAMILIAS DE OPERACOES em rodizio (razoes, estatisticas por grupo, frequencia, texto, faixas, transformacoes,
    ausentes, formulas de dominio), para vencer o vies dos LLMs por operadores simples (Kuken et al., 2024).
 3. CONTRATO INDUTIVO `features(df, ref)`: o programa nao ve o alvo e calcula estatisticas SO em `ref` (o treino). O mesmo
    codigo roda dentro da validacao interna (ref = treino da dobra), entao a selecao ve o que o teste veria.
 4. ACEITE VALIDADO (pontuado pela metrica primaria: AUC / -logloss / R2, nao por acuracia; v1.1) por coluna: ganho medio por dobra > min_gain e > se_mult x erro-padrao (regra do 1 erro-padrao),
    em vez do 'melhorou > 0' do CAAFE, que aceita ruido.
 5. DEDUPLICACAO (codigo repetido, coluna constante, correlacao > 0,98 com coluna existente) e ORCAMENTO de features.
 6. CONFIRMACAO em amostra disjunta (ou, se a base e pequena, nas linhas restantes): eliminacao regressiva remove o que
    nao ajuda de novo; se o conjunto final nao supera o baseline, devolve identidade.
 7. FEEDBACK RICO: o prompt lista o que foi aceito (com ganho) e o que foi rejeitado (com motivo).
"""
from __future__ import annotations

import hashlib
import time
from collections import Counter

import numpy as np
import pandas as pd

from ..contract import AutoFEMethod, FittedFE
from ..llm import BudgetExceeded, QuotaExhausted
from ..prep import Preprocessor, _fold_score, inner_splits, make_model, sample_positions
from ..sandbox import extract_code, run_features_program
from .legacy import columns_line

FAMILIES = [
    ("ratio_diff", "ratios, differences, products or percentages between numeric columns that make real-world sense"),
    ("group_stats", "group statistics computed on `ref`: for a categorical (or low-cardinality) column, the mean/median/std "
                    "of a numeric column per group and the row's deviation or ratio to its group "
                    "(ref.groupby(...).agg, then map onto df; fill unseen groups with the global statistic of ref)"),
    ("frequency", "frequency / count encodings and rarity of categories, computed with ref.value_counts() mapped onto df"),
    ("text_parse", "information extracted from text or identifier columns: prefixes, suffixes, lengths, token counts, "
                   "titles, parsed numbers"),
    ("bins_thresholds", "binned or thresholded versions of numeric columns with domain knowledge (age groups, brackets, "
                        "'is child'), and indicators of special values"),
    ("transform_scale", "non-linear transforms of skewed numeric columns (log1p, sqrt, powers) and clipped versions"),
    ("missing_flags", "missing-value indicators and consistency checks between columns (implausible or inconsistent values)"),
    ("domain_formula", "domain-specific formulas from real-world knowledge of this dataset (per-person or per-unit "
                       "quantities, standard indices, known physical or financial relations)"),
]


# ------------------------------------------------------------------ perfil dos dados
def _is_cat(s: pd.Series) -> bool:
    return not (pd.api.types.is_numeric_dtype(s) and not isinstance(s.dtype, pd.CategoricalDtype)) or pd.api.types.is_bool_dtype(s)


def _eta2(values: pd.Series, groups: pd.Series) -> float:
    v = values.astype(float)
    tot = float(((v - v.mean()) ** 2).sum())
    if tot <= 0:
        return 0.0
    gm = v.groupby(groups).transform("mean")
    return float(((gm - v.mean()) ** 2).sum() / tot)


def _assoc(col: pd.Series, y: pd.Series, kind: str) -> str:
    try:
        if not _is_cat(col):
            if kind == "multiclass":
                return f"eta2={_eta2(col.fillna(col.median()), y):.2f}"
            r = col.corr(y.astype(float), method="spearman")
            return f"|spearman|={abs(r):.2f}" if np.isfinite(r) else "-"
        if kind == "multiclass":
            return "-"
        g = col.astype(object).where(col.notna(), "__nan__").astype(str)
        top = g.value_counts().index[:30]
        gg = g.where(g.isin(top), "__other__")
        return f"eta2={_eta2(y.astype(float), gg):.2f}"
    except Exception:  # noqa: BLE001
        return "-"


def build_profile(X: pd.DataFrame, y: pd.Series, task, max_cols: int = 40) -> tuple[str, dict]:
    """Texto compacto do perfil + metadados usados para escolher familias aplicaveis."""
    meta = {"numeric": [], "categorical": [], "text": [], "missing": []}
    lines = []
    for c in list(X.columns)[: max(max_cols, 1) * 3]:
        s = X[c]
        miss = s.isna().mean()
        if miss > 0:
            meta["missing"].append(c)
        a = _assoc(s, y, task.kind)
        desc = task.column_descriptions.get(c, "")
        d = f" -- {desc}" if desc else ""
        if not _is_cat(s):
            meta["numeric"].append(c)
            q = s.quantile([0, 0.5, 1]).tolist() if s.notna().any() else [np.nan] * 3
            lines.append(f"- {c} ({s.dtype}) [numeric]: missing {miss:.0%}, min {q[0]:.4g}, median {q[1]:.4g}, "
                         f"max {q[2]:.4g}, mean {s.mean():.4g}, std {s.std():.4g}, assoc {a}{d}")
        else:
            nun = s.nunique(dropna=True)
            vc = s.value_counts(normalize=True).head(4)
            tops = ", ".join(f"{str(k)[:18]!r} {v:.0%}" for k, v in vc.items())
            if nun > 15:
                meta["text"].append(c)
            meta["categorical"].append(c)
            lines.append(f"- {c} ({s.dtype}) [categorical]: {nun} distinct, missing {miss:.0%}, top: {tops}, assoc {a}{d}")
    return "\n".join(lines), meta


def applicable_families(meta: dict) -> list[str]:
    ok = []
    has_num, has_cat = bool(meta["numeric"]), bool(meta["categorical"])
    for key, _ in FAMILIES:
        if key == "ratio_diff" and len(meta["numeric"]) >= 2:
            ok.append(key)
        elif key == "group_stats" and has_num and has_cat:
            ok.append(key)
        elif key == "frequency" and has_cat:
            ok.append(key)
        elif key == "text_parse" and meta["text"]:
            ok.append(key)
        elif key in ("bins_thresholds", "transform_scale") and has_num:
            ok.append(key)
        elif key == "missing_flags" and meta["missing"]:
            ok.append(key)
        elif key == "domain_formula":
            ok.append(key)
    return ok or ["domain_formula"]


def build_prompt(task, X: pd.DataFrame, profile: str, family: tuple, accepted: list, rejected: list, seed: int,
                 n_sample: int = 5) -> str:
    target_desc = (f'binary label (0/1)' if task.kind == "binary" else
                   f"class label ({task.n_classes} classes)" if task.kind == "multiclass" else "numeric value")
    sample = X.sample(min(n_sample, len(X)), random_state=seed)
    srows = "\n".join("  " + ", ".join(f"{c}={_fmt(v)}" for c, v in r.items()) for _, r in sample.iterrows())
    acc = "\n".join(f"- {a['name']} (gain {a['delta']:+.4f})" for a in accepted[-12:]) or "(none yet)"
    rej = "\n".join(f"- {r['name']}: {r['reason']}" for r in rejected[-6:]) or "(none)"
    return f"""You are an expert data scientist doing feature engineering for a tabular {task.kind_words} task.

TASK
{task.describe()}
Target: "{task.target}" ({target_desc}).

{columns_line(X)}

DATA PROFILE (training rows: {len(X)}; 'assoc' = association with the target)
{profile}

SAMPLE ROWS
{srows}

ALREADY ACCEPTED FEATURES (do not repeat them or close variants)
{acc}

RECENT REJECTIONS OR FAILURES (learn from them)
{rej}

THIS ROUND'S FOCUS: {family[0]} -- {family[1]}.

IMPORTANT: the target column is NOT available in `df` or `ref` (no target encoding, no target-based group means).
Only use columns listed in "Columns" above, plus new columns you create inside the function.
Write ONE python function. Its output is validated by cross-validation with a random forest, a gradient-boosting
model and a linear model, so only features that really help are kept. Most simple ratios already learned by tree
models are NOT kept: prefer features that expose NEW information (group statistics, text parsing, domain formulas).
```python
def features(df, ref):
    # df  : rows to transform (pandas DataFrame with the columns above, never the target)
    # ref : the training rows (same columns). Compute EVERY statistic (group means, frequencies, quantiles) on `ref`
    #       and map it onto `df`; never compute statistics on `df` itself.
    out = pd.DataFrame(index=df.index)
    out['new_feature_name'] = ...   # up to 3 new columns, numeric (or bool), new names
    return out
```
Rules: no imports (`pd` and `np` are available), no printing, do not use the target, handle NaN and unseen categories,
the code must give the same value for a row whether it is alone or in a batch. Prefer features grounded in real-world
knowledge of this data. Answer with a single ```python block.
"""


def _fmt(v):
    if isinstance(v, float):
        return "NaN" if np.isnan(v) else f"{v:.4g}"
    return repr(v) if isinstance(v, str) else str(v)


# ------------------------------------------------------------------ o metodo
class GuidedFitted(FittedFE):
    inductive = True

    def __init__(self, ref: pd.DataFrame, features: list[dict], target: str, info: dict):
        self.ref = ref.reset_index(drop=True).copy()
        self.features = features  # [{'code','col','final','name'}]
        self.target = target
        self.info = info
        self.transform_errors = 0

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        by_code: dict[str, list[dict]] = {}
        for f in self.features:
            by_code.setdefault(f["code"], []).append(f)
        for code, feats in by_code.items():
            try:
                new = run_features_program(code, X, self.ref, forbidden=(self.target,), timeout=120)
            except QuotaExhausted:
                raise
            except Exception:  # noqa: BLE001
                new = None
                self.transform_errors += 1
            for f in feats:
                out[f["final"]] = new[f["col"]].to_numpy() if (new is not None and f["col"] in new.columns) else np.nan
        return out

    def report(self):
        return {**self.info, "transform_errors": self.transform_errors,
                "features": [{k: f[k] for k in ("final", "family", "delta", "se", "code")} for f in self.features]}


class GuidedFE(AutoFEMethod):
    name = "guided"
    version = "1.2"
    needs_llm = True

    def __init__(self, min_gain: float = 0.001, se_mult: float = 1.0, temperature: float = 0.7, inner_trees: int = 100,
                 overcollect: float = 1.5, confirm: bool = True, folds: int = 5, verbose: bool = False,
                 selection: str = "multi"):
        # selection: "rf" (so floresta, como v1.1) ou "multi" (media de rf, hgb e lr: os modelos finais do protocolo).
        # Na v1.1 os ganhos de busca (+0.002..0.007 AUC) nao apareciam no teste; features de razao/diferenca ajudam
        # modelos lineares e pouco as arvores, entao medir so com RF escolhia o modelo errado para elas.
        if selection not in ("rf", "multi"):
            raise ValueError("selection deve ser 'rf' ou 'multi'")
        self.selection = selection
        self.min_gain, self.se_mult, self.temperature = min_gain, se_mult, temperature
        self.inner_trees, self.overcollect, self.confirm, self.folds, self.verbose = inner_trees, overcollect, confirm, folds, verbose
        self.params = {"min_gain": min_gain, "se_mult": se_mult, "temperature": temperature, "overcollect": overcollect,
                       "confirm": confirm, "selection": selection}

    # ---- avaliacao por dobra, com o mesmo contrato do uso final
    def _score(self, Xtr: pd.DataFrame, Xva: pd.DataFrame, ytr, yva, kind, seed) -> float:
        prep = Preprocessor().fit(Xtr)
        Ftr, Fva = prep.transform(Xtr), prep.transform(Xva)
        if Ftr.shape[1] == 0:
            return float("nan")
        names = ("rf",) if self.selection == "rf" else ("rf", "hgb", "lr")
        sc = []
        for nm in names:
            m = make_model(nm, kind, seed, rf_trees=self.inner_trees, rf_depth=6, hgb_iter=100)
            m.fit(Ftr, ytr)
            sc.append(_fold_score(kind, yva, m, Fva, mode="primary"))
        return float(np.mean(sc))

    def _folds(self, X: pd.DataFrame, y: pd.Series, task, ctx, seed):
        out = []
        for tr, va in inner_splits(y, task.kind, self.folds, seed):
            out.append({"raw_tr": X.iloc[tr].reset_index(drop=True), "raw_va": X.iloc[va].reset_index(drop=True),
                        "y_tr": y.iloc[tr].reset_index(drop=True), "y_va": y.iloc[va].reset_index(drop=True),
                        "acc_tr": pd.DataFrame(index=range(len(tr))), "acc_va": pd.DataFrame(index=range(len(va)))})
        return out

    def _cur_scores(self, folds, task, seed, drop: str | None = None) -> np.ndarray:
        sc = []
        for f in folds:
            atr = f["acc_tr"].drop(columns=[drop]) if drop else f["acc_tr"]
            ava = f["acc_va"].drop(columns=[drop]) if drop else f["acc_va"]
            sc.append(self._score(pd.concat([f["raw_tr"], atr], axis=1), pd.concat([f["raw_va"], ava], axis=1),
                                  f["y_tr"], f["y_va"], task.kind, seed))
        return np.array(sc)

    def _run_on_folds(self, code, folds, target):
        res = []
        for f in folds:
            ntr = run_features_program(code, f["raw_tr"], f["raw_tr"], forbidden=(target,))
            nva = run_features_program(code, f["raw_va"], f["raw_tr"], forbidden=(target,))
            res.append((ntr.reset_index(drop=True), nva.reset_index(drop=True)))
        return res

    def _eval_column(self, col, folds, news, task, seed, cur: np.ndarray):
        sc = []
        for f, (ntr, nva) in zip(folds, news):
            if col not in ntr.columns or col not in nva.columns:
                return None, "column_missing_in_fold"
            xtr = pd.concat([f["raw_tr"], f["acc_tr"], ntr[[col]]], axis=1)
            xva = pd.concat([f["raw_va"], f["acc_va"], nva[[col]]], axis=1)
            sc.append(self._score(xtr, xva, f["y_tr"], f["y_va"], task.kind, seed))
        return np.array(sc), None

    @staticmethod
    def _col_ok(col, folds, news, existing_names):
        """Filtros baratos: constante, quase tudo ausente, infinito, texto de alta cardinalidade, nome repetido."""
        if col in existing_names:
            return "nome_repetido"
        ntr = news[0][0][col]
        if pd.api.types.is_numeric_dtype(ntr) or pd.api.types.is_bool_dtype(ntr):
            v = pd.to_numeric(ntr, errors="coerce").astype(float)
            if np.isinf(v).any():
                return "infinito"
            if v.isna().mean() > 0.9:
                return "quase_tudo_ausente"
            if v.nunique(dropna=True) <= 1:
                return "constante"
        else:
            if ntr.astype(str).nunique() > 15:
                return "texto_alta_cardinalidade"
            if ntr.astype(str).nunique() <= 1:
                return "constante"
        return None

    @staticmethod
    def _too_correlated(col, folds, news) -> str | None:
        f, (ntr, _) = folds[0], news[0]
        s = ntr[col]
        if not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)):
            return None
        sv = pd.to_numeric(s, errors="coerce").astype(float)
        base = pd.concat([f["raw_tr"], f["acc_tr"]], axis=1)
        for c in base.columns:
            b = base[c]
            if pd.api.types.is_numeric_dtype(b) or pd.api.types.is_bool_dtype(b):
                r = sv.corr(pd.to_numeric(b, errors="coerce").astype(float))
                if np.isfinite(r) and abs(r) > 0.98:
                    return f"correlacao_{abs(r):.2f}_com_{c}"
        return None

    def fit(self, X_train, y_train, task, ctx):
        t_start = time.time()
        n = len(X_train)
        seed = ctx.seed
        # amostras A (busca) e B (confirmacao), disjuntas quando ha linhas suficientes
        lim = ctx.budget.search_max_rows
        if n >= 2 * lim:
            posA = ctx.sample_rows(y_train, lim, seed)
            posB = ctx.sample_rows(y_train, lim, seed + 1, exclude=posA)
            disjoint = True
        else:
            posA = sample_positions(y_train, task.kind, max(int(0.6 * n), min(n, 50)), seed)
            rest = np.setdiff1d(np.arange(n), posA)
            posB = rest if len(rest) >= 150 else posA
            disjoint = len(rest) >= 150
        XA, yA = X_train.iloc[posA].reset_index(drop=True), y_train.iloc[posA].reset_index(drop=True)
        XB, yB = X_train.iloc[posB].reset_index(drop=True), y_train.iloc[posB].reset_index(drop=True)

        profile, meta = build_profile(XA, yA, task)
        fams = [f for f in FAMILIES if f[0] in applicable_families(meta)]
        folds = self._folds(XA, yA, task, ctx, seed)
        cur = self._cur_scores(folds, task, seed)
        base_scores = cur.copy()
        accepted, rejected, log = [], [], []
        used_codes = set()
        names = set(map(str, X_train.columns))
        fam_count = Counter()
        max_acc = int(np.ceil(ctx.budget.max_new_features * self.overcollect))

        rnd = 0
        while ctx.remaining_calls() > 0 and len(accepted) < max_acc:
            fam = fams[rnd % len(fams)]
            rnd += 1
            fam_count[fam[0]] += 1
            entry = {"round": rnd, "family": fam[0]}
            prompt = build_prompt(task, XA, profile, fam, accepted, rejected, seed * 1000 + rnd)
            try:
                reply = ctx.llm.chat([{"role": "user", "content": prompt}], temperature=self.temperature,
                                     seed=seed * 1000 + rnd)
            except QuotaExhausted:
                raise
            except BudgetExceeded:
                break
            except Exception as e:  # noqa: BLE001
                entry.update(status="llm_error", error=f"{type(e).__name__}: {e}"[:200])
                log.append(entry)
                break
            code = extract_code(reply)
            h = hashlib.sha1(" ".join(code.split()).encode()).hexdigest()
            entry["code_hash"] = h[:8]
            if h in used_codes:
                entry["status"] = "codigo_repetido"
                rejected.append({"name": f"(round {rnd})", "reason": "same code as an earlier attempt"})
                log.append(entry)
                continue
            used_codes.add(h)
            try:
                news = self._run_on_folds(code, folds, task.target)
            except QuotaExhausted:
                raise
            except Exception as e:  # noqa: BLE001
                msg = f"{type(e).__name__}: {e}"[:160]
                entry.update(status="erro_codigo", error=msg)
                rejected.append({"name": f"(round {rnd})", "reason": f"code failed: {msg}"})
                log.append(entry)
                continue
            cols = list(dict.fromkeys(news[0][0].columns))[:3]
            entry["columns"] = cols
            statuses = []
            for col in cols:
                existing = names | {a["final"] for a in accepted}
                why = self._col_ok(col, folds, news, existing)
                if not why:
                    why = self._too_correlated(col, folds, news)
                if why:
                    rejected.append({"name": col, "reason": why})
                    statuses.append((col, why))
                    continue
                sc, err = self._eval_column(col, folds, news, task, seed, cur)
                if sc is None or not np.isfinite(sc).all():
                    rejected.append({"name": col, "reason": err or "pontuacao invalida"})
                    statuses.append((col, err or "pontuacao_invalida"))
                    continue
                d = sc - cur
                delta, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("inf")
                if delta > self.min_gain and delta > self.se_mult * se:
                    final = col
                    for f, (ntr, nva) in zip(folds, news):
                        f["acc_tr"][final] = ntr[col].to_numpy()
                        f["acc_va"][final] = nva[col].to_numpy()
                    cur = sc
                    accepted.append({"name": col, "final": final, "col": col, "code": code, "delta": delta, "se": se,
                                     "family": fam[0], "round": rnd})
                    statuses.append((col, f"aceita {delta:+.4f}"))
                else:
                    rejected.append({"name": col, "reason": f"gain {delta:+.4f} not above {max(self.min_gain, self.se_mult * se):.4f} (1-SE rule)"})
                    statuses.append((col, f"rejeitada {delta:+.4f} (se {se:.4f})"))
            entry["status"] = "avaliado"
            entry["resultado"] = statuses
            log.append(entry)

        # ---- confirmacao em amostra B + corte para o orcamento
        confirm_info = {"disjoint": disjoint, "dropped": []}
        final_feats = list(accepted)
        if accepted and self.confirm:
            foldsB = self._folds(XB, yB, task, ctx, seed + 7)
            for a in final_feats:
                for fb in foldsB:
                    ntr = run_features_program(a["code"], fb["raw_tr"], fb["raw_tr"], forbidden=(task.target,)).reset_index(drop=True)
                    nva = run_features_program(a["code"], fb["raw_va"], fb["raw_tr"], forbidden=(task.target,)).reset_index(drop=True)
                    fb["acc_tr"][a["final"]] = ntr[a["col"]].to_numpy() if a["col"] in ntr.columns else np.nan
                    fb["acc_va"][a["final"]] = nva[a["col"]].to_numpy() if a["col"] in nva.columns else np.nan
            baseB = np.array([self._score(f["raw_tr"], f["raw_va"], f["y_tr"], f["y_va"], task.kind, seed) for f in foldsB])
            full = self._cur_scores(foldsB, task, seed)
            confirm_info.update(base_B=float(baseB.mean()), all_B=float(full.mean()))
            # eliminacao regressiva: tira a feature cuja remocao nao piora (mais fraca primeiro)
            improved = True
            while improved and final_feats:
                improved = False
                cands = []
                for a in final_feats:
                    s_wo = self._cur_scores(foldsB, task, seed, drop=a["final"]) if len(final_feats) > 1 else baseB
                    cands.append((float(s_wo.mean() - full.mean()), a, s_wo))
                cands.sort(key=lambda t: -t[0])
                gain_if_removed, a, s_wo = cands[0]
                if gain_if_removed >= 0:  # remover nao piora (ou melhora): remove
                    confirm_info["dropped"].append({"name": a["final"], "motivo": "confirmacao", "delta_remover": gain_if_removed})
                    final_feats = [x for x in final_feats if x is not a]
                    for fb in foldsB:
                        fb["acc_tr"] = fb["acc_tr"].drop(columns=[a["final"]])
                        fb["acc_va"] = fb["acc_va"].drop(columns=[a["final"]])
                    full = s_wo
                    improved = True
            if final_feats and float(full.mean()) <= float(baseB.mean()):
                confirm_info["dropped"].append({"name": "(todas)", "motivo": "conjunto final nao supera o baseline em B"})
                final_feats = []
            confirm_info["final_B"] = float(full.mean()) if final_feats else float(baseB.mean())
        # orcamento: mantem as de maior ganho
        if len(final_feats) > ctx.budget.max_new_features:
            keep = sorted(final_feats, key=lambda a: -a["delta"])[: ctx.budget.max_new_features]
            confirm_info["dropped"] += [{"name": a["final"], "motivo": "orcamento"} for a in final_feats if a not in keep]
            final_feats = [a for a in final_feats if a in keep]

        info = {"search_rows": int(len(XA)), "confirm_rows": int(len(XB)), "families": dict(fam_count),
                "n_accepted_search": len(accepted), "n_final": len(final_feats), "confirm": confirm_info,
                "rejections": dict(Counter(r["reason"].split(" ")[0].split(":")[0] for r in rejected)),
                "baseline_score_A": float(base_scores.mean()), "final_score_A": float(cur.mean()),
                "log": log, "search_seconds": round(time.time() - t_start, 1)}
        return GuidedFitted(X_train, final_feats, task.target, info)
