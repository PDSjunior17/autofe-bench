"""Armazem de resultados em JSON Lines, com retomada: uma linha por job (metodo, base, divisao)."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pandas as pd

KEY_FIELDS = ("method", "method_version", "llm", "dataset", "seed", "protocol_hash")


def job_key(row: dict) -> tuple:
    return tuple(str(row.get(k)) for k in KEY_FIELDS)


class ResultStore:
    def __init__(self, path: str | Path, mirror: str | Path | None = None):
        """`path` e local e rapido (fsync). `mirror` (ex.: arquivo no Drive) recebe uma COPIA a cada job: gravar
        direto no Drive montado do Colab pode travar. Se `path` nao existe mas o espelho sim, restaura dele."""
        self.path = Path(path)
        self.mirror = Path(mirror) if mirror else None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.mirror and not self.path.exists() and self.mirror.exists():
            shutil.copyfile(self.mirror, self.path)
        self.rows: list[dict] = []
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    try:
                        self.rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # linha cortada por uma queda: ignora
        self._done = {job_key(r) for r in self.rows}

    def done(self, key: tuple, retry_errors: bool = True) -> bool:
        if key not in self._done:
            return False
        if retry_errors:
            last = [r for r in self.rows if job_key(r) == key][-1]
            return last.get("status") == "ok"
        return True

    def add(self, row: dict):
        with self.path.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.rows.append(row)
        self._done.add(job_key(row))
        self.sync()

    def sync(self):
        if self.mirror:
            try:
                self.mirror.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(self.path, self.mirror)
            except Exception as e:  # noqa: BLE001
                print(f"aviso: nao consegui copiar para {self.mirror}: {e}")

    def dataframe(self, only_ok: bool = True) -> pd.DataFrame:
        """Ultima linha de cada job. Colunas planas: <modelo>_<metrica>."""
        latest: dict[tuple, dict] = {}
        for r in self.rows:
            latest[job_key(r)] = r
        recs = []
        for r in latest.values():
            if only_ok and r.get("status") != "ok":
                continue
            flat = {k: v for k, v in r.items() if k != "metrics"}
            for m, d in (r.get("metrics") or {}).items():
                for k, v in d.items():
                    flat[f"{m}_{k}"] = v
            recs.append(flat)
        return pd.DataFrame(recs)


def merge_stores(paths, out) -> int:
    """Junta varios results.jsonl (de pessoas/maquinas diferentes) em um so; em caso de chave repetida fica a ultima."""
    import json as _json

    latest: dict[tuple, dict] = {}
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                try:
                    r = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                latest[job_key(r)] = r
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in latest.values():
            f.write(_json.dumps(r, ensure_ascii=False) + "\n")
    return len(latest)
