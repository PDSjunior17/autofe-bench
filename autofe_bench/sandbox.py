"""Execucao de codigo gerado por LLM: validacao por AST, builtins restritos, limite de tempo e de memoria.

ATENCAO: isto reduz o risco, mas NAO e um isolamento completo (codigo malicioso determinado pode escapar).
Para uso fora de um grupo de confianca, rode cada job em um container sem rede."""
from __future__ import annotations

import ast
import builtins as _bi
import contextlib
import os
import re
import signal

import numpy as np
import pandas as pd


class UnsafeCodeError(Exception):
    pass


class CodeTimeout(Exception):
    pass


_BLOCKED_NAMES = {
    "exec", "eval", "compile", "open", "__import__", "input", "globals", "locals", "vars", "getattr",
    "setattr", "delattr", "breakpoint", "exit", "quit", "os", "sys", "subprocess", "shutil", "importlib",
    "builtins", "help", "memoryview", "type", "super", "classmethod", "staticmethod", "property",
}

_SAFE_BUILTINS = {
    n: getattr(_bi, n)
    for n in (
        "abs all any bool dict enumerate filter float int isinstance len list map max min range reversed "
        "round set sorted str sum tuple zip print divmod pow repr hash chr ord object slice iter next "
        "True False None Exception ValueError KeyError TypeError ZeroDivisionError IndexError AttributeError"
    ).split()
}


def _vm_size_bytes() -> int:
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[0]) * os.sysconf("SC_PAGE_SIZE")
    except Exception:  # noqa: BLE001
        return 0


@contextlib.contextmanager
def time_limit(seconds: int, mem_gb: float | None = 4.0):
    """Limita tempo (SIGALRM) e memoria adicional (RLIMIT_AS). Funciona no processo principal (Linux)."""

    def _handler(signum, frame):
        raise CodeTimeout(f"execucao passou de {seconds}s")

    use = hasattr(signal, "SIGALRM")
    old_handler = old_lim = None
    try:
        import resource

        old_lim = resource.getrlimit(resource.RLIMIT_AS)
        vm = _vm_size_bytes()
        if vm and mem_gb:
            cap = int(vm + mem_gb * 1024 ** 3)
            if old_lim[1] in (-1, resource.RLIM_INFINITY) or cap < old_lim[1]:
                resource.setrlimit(resource.RLIMIT_AS, (cap, old_lim[1]))
    except Exception:  # noqa: BLE001
        old_lim = None
    if use:
        try:
            old_handler = signal.signal(signal.SIGALRM, _handler)
            signal.alarm(int(max(1, seconds)))
        except ValueError:  # fora do thread principal
            use = False
    try:
        yield
    finally:
        if use:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        if old_lim is not None:
            try:
                import resource

                resource.setrlimit(resource.RLIMIT_AS, old_lim)
            except Exception:  # noqa: BLE001
                pass


_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)(?:```|\Z)", re.DOTALL)


def extract_code(text: str) -> str:
    """Extrai codigo de uma resposta de LLM (com ou sem cerca, remove blocos <think>)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = _FENCE.search(text)
    code = m.group(1) if m else text
    code = code.replace("```end", "").replace("```", "")
    return code.strip("\n")


def sanitize_code(code: str, allow_def: bool = False) -> str:
    """Sem imports (pandas/numpy ja existem), sem funcoes perigosas, sem atributos privados, sem `while`."""
    lines = []
    for ln in code.splitlines():
        s = ln.strip()
        if re.fullmatch(r"import\s+(pandas\s+as\s+pd|numpy\s+as\s+np|pandas|numpy)", s):
            continue
        if re.fullmatch(r"from\s+(pandas|numpy)\s+import\s+.*", s):
            continue
        if re.fullmatch(r"print\(.*\)", s) or re.fullmatch(r"display\(.*\)", s):
            continue
        lines.append(ln)
    code = "\n".join(lines).strip("\n")
    if not code.strip():
        raise UnsafeCodeError("codigo vazio")
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise UnsafeCodeError(f"erro de sintaxe: {e}") from e
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise UnsafeCodeError("imports nao sao permitidos")
        if isinstance(node, (ast.Global, ast.Nonlocal, ast.While, ast.AsyncFunctionDef, ast.ClassDef, ast.Await)):
            raise UnsafeCodeError(f"construcao nao permitida: {type(node).__name__}")
        if isinstance(node, ast.FunctionDef) and not allow_def:
            raise UnsafeCodeError("definicao de funcao nao permitida aqui")
        if isinstance(node, ast.Name) and node.id in _BLOCKED_NAMES:
            raise UnsafeCodeError(f"nome nao permitido: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise UnsafeCodeError(f"atributo privado nao permitido: {node.attr}")
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value and node.value != "__nan__":
            raise UnsafeCodeError("string suspeita com '__'")
    return code


def check_frame(out, ref: pd.DataFrame, forbidden=()):
    if not isinstance(out, pd.DataFrame):
        raise ValueError("o programa nao devolveu um DataFrame")
    if len(out) != len(ref) or not out.index.equals(ref.index):
        raise ValueError("o programa alterou as linhas/indice do DataFrame")
    for f in forbidden:
        if f in out.columns:
            raise ValueError(f"o programa usou/criou a coluna alvo {f}")


def run_block(code: str, df: pd.DataFrame, forbidden=(), timeout: int = 30, mem_gb: float = 4.0) -> pd.DataFrame:
    """Bloco no estilo CAAFE: codigo que altera `df`. Devolve o novo DataFrame."""
    code = sanitize_code(code)
    d = df.copy()
    env = {"df": d, "pd": pd, "np": np, "__builtins__": dict(_SAFE_BUILTINS)}
    with time_limit(timeout, mem_gb):
        exec(compile(code, "<llm_block>", "exec"), env)  # noqa: S102
    out = env["df"]
    check_frame(out, df, forbidden)
    return out


def _find_fn(env, fn_name):
    fn = env.get(fn_name)
    if fn is None:
        cands = [k for k, v in env.items() if callable(v) and k.startswith(fn_name)]
        if not cands:
            raise UnsafeCodeError(f"funcao {fn_name} nao definida")
        fn = env[cands[-1]]
    return fn


def run_function_program(code: str, df: pd.DataFrame, fn_name: str = "modify_features", forbidden=(),
                         timeout: int = 30, mem_gb: float = 4.0) -> pd.DataFrame:
    """Programa do LLM-FE: define `modify_features(df)` e devolve um DataFrame."""
    code = sanitize_code(code, allow_def=True)
    env = {"pd": pd, "np": np, "__builtins__": dict(_SAFE_BUILTINS)}
    with time_limit(timeout, mem_gb):
        exec(compile(code, "<llm_program>", "exec"), env)  # noqa: S102
        out = _find_fn(env, fn_name)(df.copy())
    check_frame(out, df, forbidden)
    return out


def run_features_program(code: str, df: pd.DataFrame, ref: pd.DataFrame, fn_name: str = "features",
                         forbidden=(), timeout: int = 30, mem_gb: float = 4.0) -> pd.DataFrame:
    """Contrato indutivo do GuidedFE: `features(df, ref)` devolve SO as colunas novas, com o indice de `df`.
    `ref` e o treino (somente features, nunca o alvo); estatisticas de grupo devem ser calculadas em `ref`."""
    code = sanitize_code(code, allow_def=True)
    env = {"pd": pd, "np": np, "__builtins__": dict(_SAFE_BUILTINS)}
    with time_limit(timeout, mem_gb):
        exec(compile(code, "<llm_features>", "exec"), env)  # noqa: S102
        out = _find_fn(env, fn_name)(df.copy(), ref.copy())
    if isinstance(out, pd.Series):
        out = out.to_frame()
    check_frame(out, df, forbidden)
    return out
