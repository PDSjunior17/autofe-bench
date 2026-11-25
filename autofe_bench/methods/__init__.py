"""Registro de metodos. Um metodo proprio pode ser usado de tres jeitos:
  - instancia:  run_benchmark([MeuMetodo()], ...)
  - nome:       "caafe", "llmfe", "guided", "sem_fe", "featcopilot", "openfe"
  - arquivo:    "meu_metodo.py:MinhaClasse"   (ou "pacote.modulo:Classe")
Parametros por texto: "caafe:max_history=6,temperature=0.5" (o nome do metodo no placar passa a incluir os parametros)."""
from __future__ import annotations

import ast
import importlib
import importlib.util
import sys
from pathlib import Path

from ..contract import AutoFEMethod, FittedFE, IdentityFitted


class SemFE(AutoFEMethod):
    """Linha de base: nenhuma feature nova (so o pre-processamento comum do harness)."""

    name = "sem_fe"
    version = "1"

    def fit(self, X_train, y_train, task, ctx):
        return IdentityFitted()


def _registry():
    from .caafe import CAAFE
    from .external import FeatCopilot, OpenFE
    from .gabriel_llmfe import GabrielLLMFE
    from .guided import GuidedFE
    from .llmfe import LLMFE
    from .lucas_autofe import LucasAutoFE

    return {"sem_fe": SemFE, "caafe": CAAFE, "llmfe": LLMFE, "guided": GuidedFE,
            "featcopilot": FeatCopilot, "openfe": OpenFE,
            "lucas_autofe": LucasAutoFE, "gabriel_llmfe": GabrielLLMFE}


def available_methods() -> list[str]:
    return sorted(_registry())


def _parse_kwargs(s: str) -> dict:
    out = {}
    for part in filter(None, (p.strip() for p in s.split(","))):
        k, _, v = part.partition("=")
        try:
            out[k.strip()] = ast.literal_eval(v.strip())
        except (ValueError, SyntaxError):
            out[k.strip()] = v.strip()
    return out


def _load_class(spec: str):
    mod, _, cls = spec.partition(":")
    if mod.endswith(".py"):
        p = Path(mod).resolve()
        name = f"_autofe_user_{p.stem}"
        sp = importlib.util.spec_from_file_location(name, p)
        m = importlib.util.module_from_spec(sp)
        sys.modules[name] = m
        sp.loader.exec_module(m)
    else:
        m = importlib.import_module(mod)
    return getattr(m, cls)


def resolve_method(spec) -> AutoFEMethod:
    if isinstance(spec, AutoFEMethod):
        return spec
    if isinstance(spec, type) and issubclass(spec, AutoFEMethod):
        return spec()
    if not isinstance(spec, str):
        raise TypeError(f"metodo invalido: {spec!r}")
    kwargs = {}
    base, _, rest = spec.partition("|")  # "arquivo.py:Classe|k=v" tambem aceito
    if ":" in base and not base.endswith(".py") and "=" in base.split(":", 1)[1]:
        base, _, kw = base.partition(":")
        kwargs = _parse_kwargs(kw)
    elif rest:
        kwargs = _parse_kwargs(rest)
    reg = _registry()
    if base in reg:
        cls = reg[base]
    elif ":" in base:
        cls = _load_class(base)
    else:
        raise KeyError(f"metodo desconhecido: {base}. Registrados: {sorted(reg)}")
    inst = cls(**kwargs)
    if kwargs:
        inst.name = f"{inst.name}[" + ",".join(f"{k}={v}" for k, v in kwargs.items()) + "]"
    inst.params = {**getattr(inst, "params", {}), **kwargs}
    return inst
