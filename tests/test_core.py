import json, threading, warnings
from http.server import BaseHTTPRequestHandler, HTTPServer
import pandas as pd, pytest
warnings.filterwarnings("ignore")
from autofe_bench.llm import (BudgetedLLM, MockClient, OpenAICompatClient, QuotaExhausted, BudgetExceeded, Throttle)
from autofe_bench.sandbox import run_features_program, run_block, UnsafeCodeError
from autofe_bench.runner import run_benchmark
from autofe_bench.store import ResultStore
from autofe_bench.mockllm import HeuristicMockLLM
from autofe_bench.contract import AutoFEMethod, IdentityFitted, FittedFE


def test_budget_enforced():
    b = BudgetedLLM(MockClient(["x"]), max_calls=2)
    b.chat([{"role": "user", "content": "a"}]); b.chat([{"role": "user", "content": "b"}])
    with pytest.raises(BudgetExceeded):
        b.chat([{"role": "user", "content": "c"}])


def test_throttle_waits():
    t = {"now": 0.0}; slept = []
    th = Throttle(rpm=2, tpm=None, sleep=lambda s: (slept.append(s), t.__setitem__("now", t["now"] + s)), clock=lambda: t["now"])
    for _ in range(3):
        th.wait(10); th.record(10)
    assert slept and sum(slept) > 0


class _H(BaseHTTPRequestHandler):
    mode = "ok"; n = 0
    def log_message(self, *a): pass
    def do_POST(self):
        _H.n += 1
        self.rfile.read(int(self.headers.get("content-length", 0)))
        if _H.mode == "daily":
            body = json.dumps({"error": {"message": "Rate limit reached for model in organization on tokens per day (TPD): Limit 200000"}}).encode()
            self.send_response(429); self.send_header("retry-after", "3600")
        elif _H.mode == "retry" and _H.n == 1:
            body = json.dumps({"error": {"message": "Rate limit on requests per minute (RPM)"}}).encode()
            self.send_response(429); self.send_header("retry-after", "0")
        else:
            body = json.dumps({"choices": [{"message": {"content": "ola"}}], "usage": {"prompt_tokens": 3, "completion_tokens": 1}}).encode()
            self.send_response(200)
        self.send_header("content-type", "application/json"); self.send_header("content-length", str(len(body))); self.end_headers(); self.wfile.write(body)


@pytest.fixture
def server():
    s = HTTPServer(("127.0.0.1", 0), _H); threading.Thread(target=s.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{s.server_port}/v1"; s.shutdown()


def test_http_retry_and_daily_quota(server):
    _H.mode, _H.n = "retry", 0
    c = OpenAICompatClient("m", server, api_key="k", sleep=lambda s: None)
    assert c.complete([{"role": "user", "content": "oi"}]).text == "ola"
    _H.mode = "daily"
    with pytest.raises(QuotaExhausted):
        c.complete([{"role": "user", "content": "oi"}])


def test_sandbox_blocks_unsafe_and_target():
    df = pd.DataFrame({"a": [1.0, 2.0], "t": [0, 1]})
    with pytest.raises(Exception):
        run_features_program("def features(df, ref):\n    import os\n    return pd.DataFrame({'x': df.a})", df, df)
    with pytest.raises(Exception):
        run_features_program("def features(df, ref):\n    return pd.DataFrame({'t': df['a']}, index=df.index)", df, df, forbidden=("t",))
    out = run_features_program("def features(df, ref):\n    return pd.DataFrame({'x': df['a']*2}, index=df.index)", df, df, forbidden=("t",))
    assert list(out.x) == [2.0, 4.0]


def test_resume_and_replay(tmp_path):
    store = tmp_path / "r.jsonl"
    kw = dict(protocol="smoke", client=HeuristicMockLLM(), store_path=store, log_dir=tmp_path / "l", verbose=False)
    _, s1 = run_benchmark(["sem_fe", "guided"], ["breast_cancer"], **kw)
    _, s2 = run_benchmark(["sem_fe", "guided"], ["breast_cancer"], **kw)
    assert s1["rodados"] == 2 and s2["rodados"] == 0 and s2["pulados"] == 2


class _Leaky(AutoFEMethod):
    name = "leaky"; version = "1"
    class F(FittedFE):
        inductive = True   # mente: usa estatistica do proprio lote
        def transform(self, X):
            X = X.copy(); X["z"] = X.iloc[:, 0] - X.iloc[:, 0].mean(); return X
    def fit(self, X, y, task, ctx): return self.F()


def test_probe_detects_transduction(tmp_path):
    st, _ = run_benchmark([_Leaky()], ["breast_cancer"], protocol="smoke", store_path=tmp_path / "r.jsonl", log_dir=tmp_path, verbose=False)
    df = st.dataframe()
    assert bool(df[df.method == 'leaky'].transductive_detected.iloc[0]) is True


def test_report_runs(tmp_path):
    run_benchmark(["sem_fe", "caafe", "guided"], ["breast_cancer", "wine"], protocol="smoke", client=HeuristicMockLLM(),
                  store_path=tmp_path / "r.jsonl", log_dir=tmp_path, verbose=False)
    from autofe_bench.report import render
    txt = render(tmp_path / "r.jsonl")
    assert "Media por base" in txt and "guided" in txt


class _O(BaseHTTPRequestHandler):
    last = None
    def log_message(self, *a): pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
        _O.last = body
        if body["model"] == "ausente":
            out, code = {"error": "model 'ausente' not found"}, 404
        else:
            out, code = {"message": {"role": "assistant", "content": "ok"}, "prompt_eval_count": 7, "eval_count": 2}, 200
        b = json.dumps(out).encode()
        self.send_response(code); self.send_header("content-length", str(len(b))); self.end_headers(); self.wfile.write(b)


def test_ollama_client():
    from autofe_bench.llm import OllamaClient, parse_llm_spec
    s = HTTPServer(("127.0.0.1", 0), _O); threading.Thread(target=s.serve_forever, daemon=True).start()
    host = f"http://127.0.0.1:{s.server_port}"
    c = OllamaClient("qwen2.5-coder:7b", host=host)
    r = c.complete([{"role": "user", "content": "oi"}], seed=3, stop=["```end"])
    assert (r.text, r.prompt_tokens, r.completion_tokens) == ("ok", 7, 2)
    assert _O.last["stream"] is False and _O.last["options"]["stop"] == ["```end"] and _O.last["options"]["seed"] == 3
    with pytest.raises(RuntimeError, match="ollama pull"):
        OllamaClient("ausente", host=host).complete([{"role": "user", "content": "oi"}])
    with pytest.raises(RuntimeError, match="ollama serve"):
        OllamaClient("x", host="http://127.0.0.1:1").complete([{"role": "user", "content": "oi"}])
    assert parse_llm_spec("ollama:qwen2.5-coder:7b").model == "qwen2.5-coder:7b"
    s.shutdown()


def test_legacy_columns_stable_with_get_dummies():
    import numpy as np
    from autofe_bench.methods.legacy import LegacyProgramFitted
    code = "def modify_features(df):\n    df = df.copy()\n    return pd.concat([df, pd.get_dummies(df['c'], prefix='c')], axis=1)\n"
    tr = pd.DataFrame({"a": [1.0, 2, 3, 4], "c": ["x", "y", "x", "y"]})
    te = pd.DataFrame({"a": [5.0, 6], "c": ["z", "x"]})   # categoria nova no teste
    f = LegacyProgramFitted(code, "function", tr)
    assert list(f.transform(tr).columns) == list(f.transform(te).columns)


def test_baseline_added_once(tmp_path):
    st, _ = run_benchmark(["sem_fe", "guided"], ["breast_cancer"], protocol="smoke", client=HeuristicMockLLM(),
                          store_path=tmp_path / "r.jsonl", log_dir=tmp_path, verbose=False)
    assert sorted(st.dataframe().method) == ["guided", "sem_fe"]
    st2, _ = run_benchmark(["guided"], ["breast_cancer"], protocol="smoke", client=HeuristicMockLLM(),
                           store_path=tmp_path / "r2.jsonl", log_dir=tmp_path, verbose=False)
    assert "sem_fe" in set(st2.dataframe().method)


def test_dataset_specs_and_merge(tmp_path):
    from autofe_bench.data import load_dataset
    from autofe_bench.store import merge_stores
    pd.DataFrame({"a": range(60), "b": [i % 3 for i in range(60)], "y": [i % 2 for i in range(60)]}).to_csv(tmp_path / "d.csv", index=False)
    ds = load_dataset(f"csv:{tmp_path / 'd.csv'}:y:binary")
    assert ds.task.kind == "binary" and ds.X.shape == (60, 2)
    (tmp_path / "x.jsonl").write_text(json.dumps({"method": "m", "method_version": "1", "llm": "none", "dataset": "d", "seed": 1, "protocol_hash": "h", "status": "ok"}) + "\n")
    (tmp_path / "y.jsonl").write_text(json.dumps({"method": "m", "method_version": "1", "llm": "none", "dataset": "d", "seed": 2, "protocol_hash": "h", "status": "ok"}) + "\n")
    assert merge_stores([tmp_path / "x.jsonl", tmp_path / "y.jsonl"], tmp_path / "m.jsonl") == 2
