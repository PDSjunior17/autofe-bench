"""O protocolo versionado: tudo o que precisa ser igual para os resultados serem comparaveis."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict

from .contract import Budget


@dataclass
class Protocol:
    name: str = "v1"
    seeds: tuple = (42, 1, 2, 3, 4)
    test_size: float = 0.2
    models: tuple = ("lr", "rf", "hgb")
    max_rows: int | None = None            # subamostra estratificada fixa da base inteira (None = tudo)
    # orcamento (vale para todos os metodos)
    max_llm_calls: int = 20
    max_new_features: int = 10
    time_limit_s: int = 3600
    search_max_rows: int = 5000
    max_total_tokens: int | None = None
    # modelos finais (fixos, sem ajuste)
    rf_trees: int = 300
    rf_depth: int = 6
    hgb_iter: int = 200
    inner_folds: int = 5

    def budget(self) -> Budget:
        return Budget(self.max_llm_calls, self.max_new_features, self.time_limit_s, self.max_total_tokens,
                      self.search_max_rows)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["seeds"] = list(self.seeds)
        d["models"] = list(self.models)
        return d

    def hash(self) -> str:
        """Resumo dos campos que afetam a comparabilidade (o nome nao entra)."""
        d = self.to_dict()
        d.pop("name")
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:10]


PROTOCOLS = {
    # protocolo completo do benchmark (5 divisoes, 20 chamadas de LLM por divisao)
    "v1": Protocol(name="v1"),
    # desenvolvimento rapido: 2 divisoes, 8 chamadas, bases subamostradas. NAO comparar com v1.
    "dev": Protocol(name="dev", seeds=(42, 1), max_llm_calls=8, max_new_features=6, max_rows=8000,
                    search_max_rows=3000, rf_trees=100, hgb_iter=100, time_limit_s=900),
    # teste de fumaca: 1 divisao, 4 chamadas
    "smoke": Protocol(name="smoke", seeds=(42,), max_llm_calls=4, max_new_features=4, max_rows=3000,
                      search_max_rows=1500, rf_trees=50, hgb_iter=50, models=("lr", "rf"), time_limit_s=300),
}


def get_protocol(name: str) -> Protocol:
    if name not in PROTOCOLS:
        raise KeyError(f"protocolo desconhecido: {name}. Disponiveis: {list(PROTOCOLS)}")
    return PROTOCOLS[name]
