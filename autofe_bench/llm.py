"""Camada de LLM: clientes (Groq, Ollama, qualquer API compativel com OpenAI, simulado) e o `BudgetedLLM`,
o unico ponto por onde os metodos falam com o modelo. Ele conta chamadas e tokens, aplica o orcamento,
registra prompts e respostas (para auditar ou reproduzir) e respeita limites de taxa."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path


class QuotaExhausted(Exception):
    """Cota diaria (ou de longo prazo) esgotada. O corredor para e voce retoma depois (resultados ja gravados)."""


class BudgetExceeded(Exception):
    """O metodo tentou passar do orcamento de chamadas/tokens do protocolo."""


class ReplayMismatch(Exception):
    """No modo replay, o prompt atual nao bate com o gravado: o metodo ou os dados mudaram."""


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMClient:
    model: str = "?"
    provider: str = "?"

    def complete(self, messages, temperature=None, seed=None, stop=None, max_tokens=None) -> LLMResult:  # pragma: no cover
        raise NotImplementedError

    def describe(self) -> dict:
        return {"provider": self.provider, "model": self.model}


# ----------------------------------------------------------------------------------------------
# Limitador de taxa (janela deslizante de 60 s) para RPM e TPM
# ----------------------------------------------------------------------------------------------
class Throttle:
    def __init__(self, rpm: int | None = None, tpm: int | None = None, sleep=time.sleep, clock=time.monotonic):
        self.rpm, self.tpm, self._sleep, self._clock = rpm, tpm, sleep, clock
        self._events: deque = deque()  # (instante, tokens)

    def _purge(self, now):
        while self._events and now - self._events[0][0] >= 60:
            self._events.popleft()

    def wait(self, est_tokens: int):
        """Dorme o necessario para nao passar de RPM/TPM. Devolve o tempo dormido."""
        slept = 0.0
        while True:
            now = self._clock()
            self._purge(now)
            need = 0.0
            if self.rpm and len(self._events) >= self.rpm:
                need = max(need, 60 - (now - self._events[0][0]))
            if self.tpm:
                used = sum(t for _, t in self._events)
                if used + est_tokens > self.tpm and self._events:
                    # espera ate liberar tokens suficientes
                    freed, target = 0, used + est_tokens - self.tpm
                    for ts, tk in self._events:
                        freed += tk
                        if freed >= target:
                            need = max(need, 60 - (now - ts))
                            break
                    else:
                        need = max(need, 60 - (now - self._events[0][0]))
            if need <= 0:
                return slept
            need = min(need + 0.05, 60)
            self._sleep(need)
            slept += need

    def record(self, tokens: int):
        self._events.append((self._clock(), int(tokens)))


# ----------------------------------------------------------------------------------------------
# Clientes
# ----------------------------------------------------------------------------------------------
class OpenAICompatClient(LLMClient):
    """Cliente para qualquer endpoint /chat/completions no formato OpenAI (Groq, OpenRouter, vLLM...)."""

    provider = "openai-compat"

    def __init__(self, model, base_url, api_key=None, temperature=0.5, max_tokens=1500, timeout=120,
                 rpm=None, tpm=None, max_retries=6, extra_body=None, sleep=time.sleep):
        self.model, self.base_url = model, base_url.rstrip("/")
        self.api_key, self.temperature, self.max_tokens = api_key, temperature, max_tokens
        self.timeout, self.max_retries, self.extra_body = timeout, max_retries, dict(extra_body or {})
        self._sleep = sleep
        self.throttle = Throttle(rpm=rpm, tpm=tpm, sleep=sleep)
        self.waited_seconds = 0.0

    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    @staticmethod
    def _is_daily(msg: str) -> bool:
        m = msg.lower()
        return ("per day" in m) or ("(tpd)" in m) or ("(rpd)" in m) or ("daily" in m)

    def complete(self, messages, temperature=None, seed=None, stop=None, max_tokens=None) -> LLMResult:
        import requests

        mt = self.max_tokens if max_tokens is None else max_tokens
        payload = {"model": self.model, "messages": messages, "max_tokens": mt,
                   "temperature": self.temperature if temperature is None else temperature}
        if seed is not None:
            payload["seed"] = int(seed) % (2 ** 31 - 1)
        if stop:
            payload["stop"] = list(stop)[:4]
        payload.update(self.extra_body)
        est = sum(len(m["content"]) for m in messages) // 3 + mt
        last = None
        for attempt in range(self.max_retries + 1):
            self.waited_seconds += self.throttle.wait(est)
            try:
                r = requests.post(f"{self.base_url}/chat/completions", headers=self._headers(),
                                  json=payload, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as e:
                last = f"{type(e).__name__}: {e}"
                self._sleep(min(2 ** attempt, 30))
                continue
            if r.status_code == 200:
                data = r.json()
                text = (data["choices"][0]["message"].get("content") or "")
                u = data.get("usage") or {}
                res = LLMResult(text, int(u.get("prompt_tokens", 0) or 0), int(u.get("completion_tokens", 0) or 0))
                self.throttle.record(res.prompt_tokens + res.completion_tokens or est)
                return res
            body = ""
            try:
                body = (r.json().get("error") or {}).get("message", "") or r.text
            except Exception:  # noqa: BLE001
                body = r.text[:300]
            if r.status_code == 429:
                if self._is_daily(body):
                    raise QuotaExhausted(f"cota diaria esgotada ({self.model}): {body[:300]}")
                ra = r.headers.get("retry-after")
                try:
                    wait = float(ra) if ra is not None else min(2 ** attempt * 2, 60)
                except ValueError:
                    wait = min(2 ** attempt * 2, 60)
                last = f"429: {body[:200]}"
                self._sleep(wait + 0.5)
                self.waited_seconds += wait + 0.5
                continue
            if r.status_code == 413:
                raise ValueError(f"prompt grande demais para o limite do modelo: {body[:300]}")
            if r.status_code in (500, 502, 503, 504):
                last = f"{r.status_code}: {body[:200]}"
                self._sleep(min(2 ** attempt * 2, 60))
                continue
            raise RuntimeError(f"HTTP {r.status_code} de {self.base_url}: {body[:400]}")
        raise RuntimeError(f"falhou depois de {self.max_retries} tentativas: {last}")

    def describe(self):
        return {"provider": self.provider, "model": self.model, "base_url": self.base_url}


class GroqClient(OpenAICompatClient):
    """Groq (camada gratuita). Chave em GROQ_API_KEY. Os limites valem por organizacao e por modelo;
    confira os seus em console.groq.com/settings/limits e informe `rpm`/`tpm` para o limitador."""

    provider = "groq"
    BASE = "https://api.groq.com/openai/v1"

    # llama-3.3-70b-versatile foi desativado na camada gratuita em 16/08/2026; a Groq recomenda gpt-oss-120b.
    # Modelos de raciocinio (gpt-oss) gastam parte de max_tokens pensando, por isso o padrao aqui e maior.
    DEFAULT_MODEL = "openai/gpt-oss-120b"

    def __init__(self, model=DEFAULT_MODEL, api_key=None, **kw):
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError("defina GROQ_API_KEY (crie a chave em console.groq.com/keys)")
        kw.setdefault("max_tokens", 4096)
        super().__init__(model, self.BASE, api_key=key, **kw)


def list_groq_models(api_key=None) -> list[str]:
    """Modelos que a SUA chave enxerga agora. Use para escolher um id valido."""
    import requests

    key = api_key or os.environ.get("GROQ_API_KEY")
    r = requests.get(GroqClient.BASE + "/models", headers={"Authorization": f"Bearer {key}"}, timeout=30)
    r.raise_for_status()
    return sorted(m["id"] for m in r.json().get("data", []))


class OllamaClient(LLMClient):
    """Ollama local (HTTP /api/chat)."""

    provider = "ollama"

    def __init__(self, model, host=None, temperature=0.5, num_ctx=8192, num_predict=1500,
                 timeout=900, think=None):
        host = host or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        if not host.startswith("http"):
            host = "http://" + host
        self.model, self.host, self.temperature = model, host.rstrip("/"), temperature
        self.num_ctx, self.num_predict, self.timeout, self.think = num_ctx, num_predict, timeout, think

    def complete(self, messages, temperature=None, seed=None, stop=None, max_tokens=None) -> LLMResult:
        import requests

        opts = {"temperature": self.temperature if temperature is None else temperature,
                "num_ctx": self.num_ctx, "num_predict": max_tokens or self.num_predict}
        if seed is not None:
            opts["seed"] = int(seed) % (2 ** 31 - 1)
        if stop:
            opts["stop"] = list(stop)
        payload = {"model": self.model, "messages": messages, "stream": False, "options": opts, "keep_alive": "30m"}
        if self.think is not None:
            payload["think"] = self.think
        try:
            r = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        except requests.ConnectionError as e:
            raise RuntimeError(f"nao consegui conectar ao Ollama em {self.host}. O servidor esta rodando (`ollama serve`)? ") from e
        if r.status_code >= 400:
            hint = f" Baixe o modelo antes: `ollama pull {self.model}`." if r.status_code == 404 else ""
            raise RuntimeError(f"HTTP {r.status_code} do Ollama ({self.host}): {r.text[:300]}.{hint}")
        d = r.json()
        return LLMResult(d["message"]["content"], int(d.get("prompt_eval_count", 0) or 0), int(d.get("eval_count", 0) or 0))


class MockClient(LLMClient):
    """Cliente simulado. `responses` = lista (em ciclo) ou funcao(messages, call_index) -> str."""

    provider = "mock"

    def __init__(self, responses, model="mock"):
        self.model = model
        self._responses = responses
        self.i = 0

    def complete(self, messages, temperature=None, seed=None, stop=None, max_tokens=None) -> LLMResult:
        r = self._responses(messages, self.i) if callable(self._responses) else self._responses[self.i % len(self._responses)]
        self.i += 1
        return LLMResult(r, sum(len(m["content"]) for m in messages) // 4, len(r) // 4)


def parse_llm_spec(spec: str, **kw) -> LLMClient | None:
    """'groq:openai/gpt-oss-120b' | 'ollama:qwen2.5-coder:14b' | 'openai:MODEL@https://host/v1' | 'none'."""
    if not spec or spec == "none":
        return None
    prov, _, rest = spec.partition(":")
    if prov == "groq":
        return GroqClient(rest or GroqClient.DEFAULT_MODEL, **kw)
    if prov == "ollama":
        return OllamaClient(rest, **{k: v for k, v in kw.items() if k in ("temperature", "think", "num_ctx", "num_predict")})
    if prov == "openai":
        model, _, base = rest.partition("@")
        return OpenAICompatClient(model, base, api_key=os.environ.get("OPENAI_API_KEY"), **kw)
    raise ValueError(f"provedor desconhecido: {prov}")


# ----------------------------------------------------------------------------------------------
# O objeto que os metodos recebem em ctx.llm
# ----------------------------------------------------------------------------------------------
def _key(messages, temperature, stop) -> str:
    raw = json.dumps([messages, temperature, list(stop or [])], sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


class BudgetedLLM:
    """Conta chamadas/tokens, impoe o orcamento e grava (modo 'record') ou reproduz (modo 'replay') as chamadas."""

    def __init__(self, client: LLMClient, max_calls: int | None = None, max_total_tokens: int | None = None,
                 log_path: str | Path | None = None, mode: str = "record", default_temperature: float | None = None):
        assert mode in ("off", "record", "replay")
        self.client, self.max_calls, self.max_total_tokens = client, max_calls, max_total_tokens
        self.mode, self.log_path = mode, (Path(log_path) if log_path else None)
        self.default_temperature = default_temperature
        self.calls = self.prompt_tokens = self.completion_tokens = 0
        self.errors, self.last_error = 0, None
        self.seconds = 0.0
        self._replay: list[dict] = []
        if mode == "replay":
            if not self.log_path or not self.log_path.exists():
                raise FileNotFoundError(f"replay sem log: {self.log_path}")
            self._replay = [json.loads(l) for l in self.log_path.read_text().splitlines() if l.strip()]
        elif mode == "record" and self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text("")  # recomeca o log do job

    @property
    def model(self):
        return self.client.model if self.client else "none"

    def remaining_calls(self) -> int:
        return 10 ** 9 if self.max_calls is None else max(0, self.max_calls - self.calls)

    def remaining_tokens(self):
        return None if self.max_total_tokens is None else max(0, self.max_total_tokens - self.prompt_tokens - self.completion_tokens)

    def chat(self, messages, stop=None, temperature=None, seed=None, max_tokens=None) -> str:
        if self.max_calls is not None and self.calls >= self.max_calls:
            raise BudgetExceeded(f"orcamento de {self.max_calls} chamadas esgotado")
        if self.max_total_tokens is not None and self.prompt_tokens + self.completion_tokens >= self.max_total_tokens:
            raise BudgetExceeded(f"orcamento de {self.max_total_tokens} tokens esgotado")
        temperature = self.default_temperature if temperature is None else temperature
        key = _key(messages, temperature, stop)
        t0 = time.time()
        if self.mode == "replay":
            if self.calls >= len(self._replay):
                raise ReplayMismatch("o log tem menos chamadas do que o metodo fez agora")
            rec = self._replay[self.calls]
            if rec["key"] != key:
                raise ReplayMismatch(f"chamada {self.calls}: prompt diferente do gravado")
            res = LLMResult(rec["reply"], rec.get("prompt_tokens", 0), rec.get("completion_tokens", 0))
        else:
            try:
                res = self.client.complete(messages, temperature=temperature, seed=seed, stop=stop, max_tokens=max_tokens)
            except QuotaExhausted:
                raise
            except Exception as e:  # noqa: BLE001
                self.errors += 1
                self.last_error = f"{type(e).__name__}: {e}"[:400]
                raise
            if self.mode == "record" and self.log_path:
                with self.log_path.open("a") as f:
                    f.write(json.dumps({"i": self.calls, "key": key, "messages": messages, "reply": res.text,
                                        "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
                                        "model": self.model}, ensure_ascii=False) + "\n")
        self.seconds += time.time() - t0
        self.calls += 1
        self.prompt_tokens += res.prompt_tokens
        self.completion_tokens += res.completion_tokens
        return res.text

    def stats(self) -> dict:
        return {"llm_calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "llm_seconds": round(self.seconds, 1),
                "llm_errors": self.errors, "llm_last_error": self.last_error,
                "llm_model": self.model, "llm_provider": getattr(self.client, "provider", "?")}


def list_ollama_models(host: str | None = None) -> list[str]:
    """Modelos JA BAIXADOS no servidor Ollama (equivale a `ollama list`)."""
    import requests

    host = host or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    if not host.startswith("http"):
        host = "http://" + host
    r = requests.get(host.rstrip("/") + "/api/tags", timeout=15)
    r.raise_for_status()
    return sorted(m["name"] for m in r.json().get("models", []))
