"""Testes dos adaptadores lucas_autofe e gabriel_llmfe (sem rede: LLM simulado)."""
import warnings

import pytest

warnings.filterwarnings("ignore")
from autofe_bench.contract import Context
from autofe_bench.data import load_dataset, make_splits
from autofe_bench.llm import BudgetedLLM, MockClient
from autofe_bench.methods import available_methods
from autofe_bench.methods.gabriel_llmfe import GabrielLLMFE
from autofe_bench.methods.lucas_autofe import LucasAutoFE, _parse_ops
from autofe_bench.protocol import get_protocol

LUCAS = "```text\nlog1p|mean radius\ninteraction_product|mean area|mean texture\nmissing_indicator|mean radius\n```"
GAB_OK = ("```python\ndef add_features(df):\n    out = df.copy()\n"
          "    out['fe_radius_area'] = df['mean radius'] * df['mean area']\n    return out\n```")
GAB_LEAK = ("```python\ndef add_features(df):\n    out = df.copy()\n"
            "    out['fe_z'] = (df['mean radius'] - df['mean radius'].mean()) / df['mean radius'].std()\n    return out\n```")
GAB_IMPORT = "```python\nimport os\ndef add_features(df):\n    return df.copy()\n```"


@pytest.fixture(scope="module")
def data():
    p = get_protocol("smoke")
    ds = load_dataset("breast_cancer", p.max_rows)
    seed, tr, te = make_splits(ds.y, ds.task.kind, (42,), p.test_size)[0]
    return (p, ds, seed, ds.X.iloc[tr].reset_index(drop=True), ds.y.iloc[tr].reset_index(drop=True),
            ds.X.iloc[te].reset_index(drop=True))


def _fit(method, reply, data):
    p, ds, seed, Xtr, ytr, _ = data
    llm = BudgetedLLM(MockClient(lambda m, i: reply), max_calls=4)
    return method.fit(Xtr, ytr, ds.task, Context(llm, p.budget(), seed, ds.task))


def test_registrados():
    assert {"lucas_autofe", "gabriel_llmfe"} <= set(available_methods())


def test_lucas_parser_descarta_invalidos(data):
    X = data[3]
    txt = "exec|mean radius\nlog1p|nao_existe\ninteraction_product|mean radius\nlog1p|mean radius"
    assert _parse_ops(txt, X, 5) == [("log1p", ("mean radius",))]


def test_lucas_gera_features_e_e_row_wise(data):
    f = _fit(LucasAutoFE(min_delta=-1), LUCAS, data)
    T = f.transform(data[5])
    assert {"fe_log1p_mean radius", "fe_prod_mean area_mean texture", "fe_missing_mean radius"} <= set(T.columns)
    assert T.iloc[:3].equals(f.transform(data[5].iloc[:3]).reset_index(drop=True))


def test_lucas_sem_llm_nao_adiciona_nada(data):
    p, ds, seed, Xtr, ytr, Xte = data
    f = LucasAutoFE().fit(Xtr, ytr, ds.task, Context(None, p.budget(), seed, ds.task))
    assert list(f.transform(Xte).columns) == list(Xte.columns)


def test_gabriel_aceita_feature_valida(data):
    f = _fit(GabrielLLMFE(min_delta=-1), GAB_OK, data)
    assert "fe_radius_area" in f.transform(data[5]).columns


@pytest.mark.parametrize("reply,motivo", [(GAB_LEAK, "outras linhas"), (GAB_IMPORT, "imports")])
def test_gabriel_rejeita_vazamento_e_import(data, reply, motivo):
    f = _fit(GabrielLLMFE(min_delta=-1), reply, data)
    assert f.programs == []
    assert any(motivo in e["status"] for e in f.info["log"])
