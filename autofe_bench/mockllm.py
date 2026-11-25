"""LLM SIMULADO para testar o encanamento (sem rede). Le a linha 'Columns: a (int64), ...' do prompt e escreve codigo
plausivel (razoes, frequencias, estatisticas por grupo, e as vezes codigo quebrado ou usando o alvo). Seus resultados
NAO medem a qualidade de nenhum metodo com um LLM de verdade."""
from __future__ import annotations

import random
import re

from .llm import MockClient


def _cols(prompt: str):
    m = re.search(r"^Columns: (.*)$", prompt, re.M)
    out = []
    if m:
        for part in re.findall(r"([^,]+?) \(([^)]+)\)", m.group(1)):
            out.append((part[0].strip(), part[1].strip()))
    num = [c for c, d in out if re.match(r"(int|float|uint)", d)]
    cat = [c for c, d in out if c not in num]
    return num, cat


def _target(prompt):
    m = re.search(r'Target: "([^"]+)"|predicting "([^"]+)"', prompt)
    return (m.group(1) or m.group(2)) if m else "target"


def heuristic_fn(messages, i):
    prompt = messages[0]["content"]
    rnd = random.Random(i * 7919 + len(prompt))
    num, cat = _cols(prompt)
    guided = "def features(df, ref)" in prompt
    legacy_fn = "modify_features" in prompt
    r = rnd.random()
    tgt = _target(prompt)
    if r < 0.08:  # codigo quebrado
        body = "out['x'] = df['coluna_que_nao_existe'] * 2"
    elif r < 0.12 and not guided:  # tenta usar o alvo
        body = f"df['leak'] = df['{tgt}']"
    else:
        ops = []
        if len(num) >= 2:
            a, b = rnd.sample(num, 2)
            ops.append(f"{a}_over_{b}|(df['{a}'] / (df['{b}'].abs() + 1e-6)).clip(-1e6, 1e6)")
            ops.append(f"{a}_minus_{b}|df['{a}'] - df['{b}']")
        if num:
            a = rnd.choice(num)
            ops.append(f"log_{a}|np.log1p(df['{a}'].clip(lower=0))")
            ops.append(f"{a}_missing|df['{a}'].isna().astype(int)")
        if cat:
            c = rnd.choice(cat)
            ops.append(f"freq_{c}|df['{c}'].map(ref['{c}'].value_counts(normalize=True)).fillna(0)" if guided
                       else f"freq_{c}|df['{c}'].map(df['{c}'].value_counts(normalize=True)).fillna(0)")
            if num:
                a = rnd.choice(num)
                if guided:
                    ops.append(f"{a}_by_{c}|df['{c}'].map(ref.groupby('{c}')['{a}'].mean()).fillna(ref['{a}'].mean())")
                else:
                    ops.append(f"{a}_by_{c}|df.groupby('{c}')['{a}'].transform('mean')")
        if not ops:
            ops = ["const|0"]
        picks = rnd.sample(ops, min(len(ops), 1 if not guided else rnd.randint(1, 2)))
        if guided:
            body = "\n    ".join(f"out['{p.split('|')[0]}'] = {p.split('|')[1]}" for p in picks)
        else:
            body = "\n    ".join(f"df['{p.split('|')[0]}'] = {p.split('|')[1]}" for p in picks)
    if guided:
        return f"```python\ndef features(df, ref):\n    out = pd.DataFrame(index=df.index)\n    {body}\n    return out\n```"
    if legacy_fn:
        return f"```python\ndef modify_features(df):\n    df = df.copy()\n    {body}\n    return df\n```"
    return f"```python\n# feature\n{body.replace('    ', '')}\n```end"


def HeuristicMockLLM():
    return MockClient(heuristic_fn)
