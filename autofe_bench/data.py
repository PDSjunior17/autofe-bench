"""Bases e divisoes. Cada base vira um `Dataset` (X cru, y numerico, `Task` com as descricoes das colunas).

As bases do benchmark (adult, bank_marketing, cdc_diabetes, california_housing, diamonds) sao baixadas na primeira
execucao (precisam de internet: Colab serve). As bases `titanic`, `breast_cancer`, `diabetes_reg`, `wine` e
`synthetic` funcionam offline e servem para testes. As descricoes das colunas foram escritas a partir da
documentacao publica de cada base; confira-as antes de publicar resultados."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .contract import Task


@dataclass
class Dataset:
    name: str
    X: pd.DataFrame
    y: pd.Series
    task: Task
    class_labels: list | None = None
    source: str = ""

    def data_hash(self) -> str:
        h = hashlib.sha1()
        h.update(pd.util.hash_pandas_object(self.X, index=False).to_numpy().tobytes())
        h.update(np.asarray(self.y).tobytes())
        return h.hexdigest()[:10]


def make_splits(y: pd.Series, kind: str, seeds, test_size: float):
    """Divisoes externas: [(seed, treino_idx, teste_idx)] (posicoes). Seed 42 reproduz os splits do Titanic."""
    idx = np.arange(len(y))
    out = []
    for s in seeds:
        strat = y if (kind != "regression" and pd.Series(np.asarray(y)).value_counts().min() >= 2) else None
        tr, te = train_test_split(idx, test_size=test_size, random_state=int(s), stratify=strat)
        out.append((int(s), np.sort(tr), np.sort(te)))
    return out


def split_hash(te_idx) -> str:
    return hashlib.sha1(np.asarray(te_idx, dtype=np.int64).tobytes()).hexdigest()[:10]


def _encode_target(y: pd.Series, kind: str):
    if kind == "regression":
        return pd.Series(pd.to_numeric(y), name=y.name, dtype=float).reset_index(drop=True), None
    labels = sorted(pd.Series(y).astype(str).unique())
    m = {l: i for i, l in enumerate(labels)}
    return pd.Series(pd.Series(y).astype(str).map(m).astype(int).to_numpy(), name=y.name), labels


def _finish(name, X, y, kind, desc, coldesc, source, target_name):
    X = X.reset_index(drop=True).copy()
    for c in X.columns:  # categorias do pandas -> texto simples (programas de LLM lidam melhor)
        if isinstance(X[c].dtype, pd.CategoricalDtype):
            X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    yy, labels = _encode_target(pd.Series(y).reset_index(drop=True), kind)
    k = 0 if kind == "regression" else len(labels)
    if kind != "regression":
        kind = "binary" if k == 2 else "multiclass"
    task = Task(name=name, target=target_name, kind=kind, description=desc, column_descriptions=coldesc, n_classes=k)
    return Dataset(name, X, yy, task, labels, source)


# ------------------------------------------------------------------------------ descricoes
_TITANIC = ("Titanic passenger survival. Each row is one passenger of the RMS Titanic (1912). Predict whether the "
            "passenger survived the sinking (Survived = 1). Use real-world knowledge about the disaster "
            "(women and children first, class, family, deck).")
_TITANIC_COLS = {
    "Pclass": "Ticket class (1 upper, 2 middle, 3 lower); proxy for socio-economic status",
    "Name": 'Passenger name formatted as "Surname, Title. Given names"',
    "Sex": "male or female",
    "Age": "Age in years (fractional for infants, missing for some passengers)",
    "SibSp": "Number of siblings or spouses aboard",
    "Parch": "Number of parents or children aboard",
    "Ticket": "Ticket number (shared by passengers travelling together)",
    "Fare": "Fare paid in pounds",
    "Cabin": "Cabin identifier; first letter is the deck (mostly missing)",
    "Embarked": "Port of embarkation (C Cherbourg, Q Queenstown, S Southampton)",
}
_ADULT = ("US Census 'Adult' data: predict whether a person's annual income exceeds 50K USD from demographic and "
          "employment attributes.")
_ADULT_COLS = {
    "age": "Age in years", "workclass": "Employer type (private, government, self-employed...)",
    "fnlwgt": "Census sampling weight (how many people the row represents)", "education": "Highest education level",
    "education-num": "Education level as a number of years", "marital-status": "Marital status",
    "occupation": "Occupation", "relationship": "Role in the family (husband, wife, own-child...)", "race": "Race",
    "sex": "Sex", "capital-gain": "Capital gains in USD", "capital-loss": "Capital losses in USD",
    "hours-per-week": "Hours worked per week", "native-country": "Country of origin",
}
_BANK = ("Portuguese bank direct-marketing campaigns: predict whether the client subscribes a term deposit. "
         "Columns describe the client, the last contact and the economic situation.")
_BANK_COLS = {
    "age": "Client age", "job": "Type of job", "marital": "Marital status", "education": "Education level",
    "default": "Has credit in default?", "balance": "Average yearly balance in euros", "housing": "Has a housing loan?",
    "loan": "Has a personal loan?", "contact": "Contact communication type", "day": "Day of the month of the last contact",
    "month": "Month of the last contact", "duration": "Duration of the last contact in seconds",
    "campaign": "Number of contacts in this campaign", "pdays": "Days since the client was last contacted in a previous campaign (-1 = never)",
    "previous": "Number of contacts before this campaign", "poutcome": "Outcome of the previous campaign",
}
_BANK_V = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month",
           "duration", "campaign", "pdays", "previous", "poutcome"]
_CDC = ("CDC BRFSS 2015 survey: predict whether a person has diabetes (or prediabetes) from health indicators and "
        "lifestyle answers. Binary columns are 0/1 answers.")
_CDC_COLS = {
    "HighBP": "High blood pressure", "HighChol": "High cholesterol", "CholCheck": "Cholesterol check in the last 5 years",
    "BMI": "Body mass index", "Smoker": "Smoked at least 100 cigarettes in life", "Stroke": "Ever had a stroke",
    "HeartDiseaseorAttack": "Coronary heart disease or heart attack", "PhysActivity": "Physical activity in the last 30 days",
    "Fruits": "Eats fruit daily", "Veggies": "Eats vegetables daily", "HvyAlcoholConsump": "Heavy alcohol consumption",
    "AnyHealthcare": "Has any health coverage", "NoDocbcCost": "Could not see a doctor because of cost",
    "GenHlth": "Self-rated general health, 1 excellent to 5 poor", "MentHlth": "Days of poor mental health in the last 30",
    "PhysHlth": "Days of poor physical health in the last 30", "DiffWalk": "Serious difficulty walking",
    "Sex": "0 female, 1 male", "Age": "Age bracket, 1 (18-24) to 13 (80 or older)",
    "Education": "Education level, 1 to 6", "Income": "Income bracket, 1 to 8",
}
_CAL = ("California housing (1990 census, one row per block group): predict the median house value of the block group, "
        "in units of 100,000 USD.")
_CAL_COLS = {
    "MedInc": "Median income in the block group (tens of thousands of USD)", "HouseAge": "Median house age in years",
    "AveRooms": "Average number of rooms per household", "AveBedrms": "Average number of bedrooms per household",
    "Population": "Block group population", "AveOccup": "Average number of household members",
    "Latitude": "Latitude", "Longitude": "Longitude",
}
_DIAM = "Diamonds: predict the price in US dollars of a diamond from its weight, quality grades and dimensions."
_DIAM_COLS = {
    "carat": "Weight of the diamond", "cut": "Cut quality (Fair, Good, Very Good, Premium, Ideal)",
    "color": "Colour from J (worst) to D (best)", "clarity": "Clarity (I1 worst ... IF best)",
    "depth": "Total depth percentage = z / mean(x, y)", "table": "Width of the top relative to the widest point",
    "x": "Length in mm", "y": "Width in mm", "z": "Depth in mm",
}


# ------------------------------------------------------------------------------ carregadores
def _find_titanic() -> Path:
    cands = [os.environ.get("AUTOFE_TITANIC_CSV", ""), "data/titanic.csv", "data_titanic/train.csv",
             "/home/claude/data_titanic/train.csv", "/content/AutoFE_Titanic/data/train.csv",
             "/content/drive/MyDrive/AutoFE_Titanic/data/train.csv"]
    for c in cands:
        if c and Path(c).exists():
            return Path(c)
    raise FileNotFoundError("titanic: defina AUTOFE_TITANIC_CSV com o caminho do train.csv")


def _load_titanic():
    df = pd.read_csv(_find_titanic())
    y = df["Survived"]
    X = df.drop(columns=["Survived"] + [c for c in ("PassengerId",) if c in df.columns])
    return _finish("titanic", X, y, "binary", _TITANIC, _TITANIC_COLS, "csv local", "Survived")


def _load_breast_cancer():
    from sklearn.datasets import load_breast_cancer

    d = load_breast_cancer(as_frame=True)
    return _finish("breast_cancer", d.data, d.target, "binary",
                   "Breast cancer diagnosis from cell-nucleus measurements of a fine needle aspirate image. "
                   "Features are mean, standard error and worst value of 10 nucleus measurements.", {}, "sklearn", "malignant")


def _load_diabetes_reg():
    from sklearn.datasets import load_diabetes

    d = load_diabetes(as_frame=True, scaled=False)
    return _finish("diabetes_reg", d.data, d.target, "regression",
                   "Diabetes progression one year after baseline (quantitative measure) from patient age, sex, bmi, "
                   "blood pressure and six blood serum measurements (s1 to s6).",
                   {"bmi": "Body mass index", "bp": "Average blood pressure", "s1": "Total serum cholesterol",
                    "s2": "LDL cholesterol", "s3": "HDL cholesterol", "s4": "Total cholesterol / HDL",
                    "s5": "Log of serum triglycerides", "s6": "Blood sugar level"}, "sklearn", "progression")


def _load_wine():
    from sklearn.datasets import load_wine

    d = load_wine(as_frame=True)
    return _finish("wine", d.data, d.target, "multiclass", "Wine cultivar (3 classes) from chemical analysis.", {}, "sklearn", "cultivar")


def _load_synthetic(n: int = 6000, seed: int = 0):
    """Base sintetica SO para testar o harness: o alvo depende de uma razao e de uma media de grupo (nao de colunas cruas)."""
    rng = np.random.RandomState(seed)
    a = rng.lognormal(3, 0.5, n)
    b = rng.lognormal(2, 0.5, n)
    g = rng.choice(["north", "south", "east", "west"], n)
    c = rng.randn(n)
    gm = pd.Series(a).groupby(g).transform("mean").to_numpy()
    z = 2.5 * np.log(a / b) + 1.5 * (a - gm) / a.std() + 0.2 * c + 0.5 * rng.randn(n)
    y = (z > np.median(z)).astype(int)
    X = pd.DataFrame({"income": a, "debt": b, "region": g, "noise": c})
    return _finish("synthetic", X, pd.Series(y), "binary",
                   "Synthetic loan default data (for harness tests only): default risk depends on the income/debt ratio "
                   "and on how income compares with the regional average.",
                   {"income": "Monthly income", "debt": "Monthly debt payments", "region": "Region of the client",
                    "noise": "Unrelated measurement"}, "gerada", "default")


def _openml(data_id=None, name=None, version=None):
    from sklearn.datasets import fetch_openml

    kw = dict(as_frame=True)
    d = fetch_openml(data_id=data_id, **kw) if data_id else fetch_openml(name=name, version=version, **kw)
    return d.data, d.target


def _load_adult():
    X, y = _openml(data_id=1590)
    return _finish("adult", X, y, "binary", _ADULT, _ADULT_COLS, "OpenML 1590", "income>50K")


def _load_bank():
    X, y = _openml(data_id=1461)
    if all(str(c).startswith("V") for c in X.columns) and X.shape[1] == len(_BANK_V):
        X = X.copy()
        X.columns = _BANK_V  # nomes anonimos no OpenML: restaurados pela ordem documentada (VERIFIQUE)
        print("[aviso] bank_marketing: colunas renomeadas V1..V16 -> nomes originais; confira a ordem.")
    return _finish("bank_marketing", X, y, "binary", _BANK, _BANK_COLS, "OpenML 1461", "subscribed")


def _load_cdc():
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as e:
        raise ImportError("cdc_diabetes precisa de `pip install ucimlrepo`") from e
    d = fetch_ucirepo(id=891)
    X, y = d.data.features, d.data.targets.iloc[:, 0]
    return _finish("cdc_diabetes", X, y, "binary", _CDC, _CDC_COLS, "UCI 891", "Diabetes_binary")


def _load_california():
    from sklearn.datasets import fetch_california_housing

    d = fetch_california_housing(as_frame=True)
    return _finish("california_housing", d.data, d.target, "regression", _CAL, _CAL_COLS, "scikit-learn", "MedHouseVal")


def _load_diamonds():
    X, y = _openml(data_id=42225)
    return _finish("diamonds", X.drop(columns=["price"], errors="ignore"), X["price"] if "price" in X else y,
                   "regression", _DIAM, _DIAM_COLS, "OpenML 42225", "price")


LOADERS = {
    "titanic": _load_titanic, "breast_cancer": _load_breast_cancer, "diabetes_reg": _load_diabetes_reg,
    "wine": _load_wine, "synthetic": _load_synthetic,
    "adult": _load_adult, "bank_marketing": _load_bank, "cdc_diabetes": _load_cdc,
    "california_housing": _load_california, "diamonds": _load_diamonds,
}
BENCHMARK_5 = ["adult", "bank_marketing", "cdc_diabetes", "california_housing", "diamonds"]
OFFLINE = ["titanic", "breast_cancer", "diabetes_reg", "wine", "synthetic"]


def register_dataset(name: str, loader):
    """Registra um carregador proprio: `loader()` devolve um `Dataset`."""
    LOADERS[name] = loader


def dataset_from_csv(path, target, kind, description="", column_descriptions=None, name=None, drop=()):
    df = pd.read_csv(path)
    y = df[target]
    X = df.drop(columns=[target, *drop])
    return _finish(name or Path(path).stem, X, y, kind, description, column_descriptions or {}, str(path), target)


def dataset_from_openml(data_id: int, target: str | None = None, kind: str | None = None, name: str | None = None,
                        description: str = "", column_descriptions=None, drop=()):
    """Qualquer base do OpenML (openml.org). `kind` ('binary'|'multiclass'|'regression') e inferido do alvo se omitido.
    DICA: escreva uma `description` boa; os metodos com LLM dependem dela."""
    from sklearn.datasets import fetch_openml

    d = fetch_openml(data_id=int(data_id), as_frame=True)
    X, y = d.data, d.target
    if target is not None:
        full = d.frame
        y, X = full[target], full.drop(columns=[target])
    X = X.drop(columns=list(drop), errors="ignore")
    if kind is None:
        kind = "regression" if pd.api.types.is_numeric_dtype(y) and not isinstance(y.dtype, pd.CategoricalDtype) and y.nunique() > 20 else "classification"
    nm = name or str(getattr(d, "details", {}).get("name", f"openml_{data_id}"))
    desc = description or f"OpenML dataset {nm} (id {data_id}). Predict '{getattr(y, 'name', 'target')}'."
    return _finish(nm, X, y, kind, desc, column_descriptions or {}, f"OpenML {data_id}", getattr(y, "name", None) or target or "target")


def load_dataset(name: str, max_rows: int | None = None, seed: int = 0) -> Dataset:
    """`name` pode ser um nome registrado, 'openml:ID' ou 'csv:CAMINHO:ALVO:TIPO' (TIPO = binary|multiclass|regression)."""
    if name.startswith("openml:"):
        ds = dataset_from_openml(int(name.split(":", 1)[1]))
    elif name.startswith("csv:"):
        _, path, target, kind = name.split(":", 3)
        ds = dataset_from_csv(path, target, kind)
    elif name not in LOADERS:
        raise KeyError(f"base desconhecida: {name}. Registradas: {sorted(LOADERS)}; ou use 'openml:ID' / 'csv:CAMINHO:ALVO:TIPO'")
    else:
        ds = LOADERS[name]()
    if max_rows and len(ds.X) > max_rows:
        strat = ds.y if ds.task.kind != "regression" else None
        keep, _ = train_test_split(np.arange(len(ds.X)), train_size=max_rows, random_state=seed, stratify=strat)
        keep = np.sort(keep)
        ds = Dataset(ds.name, ds.X.iloc[keep].reset_index(drop=True), ds.y.iloc[keep].reset_index(drop=True),
                     ds.task, ds.class_labels, ds.source)
    return ds
