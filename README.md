# autofe-bench

Harness em Python para **comparar métodos de AutoFE (engenharia automática de features) com LLM sob o mesmo protocolo**: mesmas divisões, mesmo pré-processamento, mesmos modelos finais, mesmo orçamento de chamadas. Pensado para rodar no **Google Colab** (sem GPU local) com **Ollama** (GPU do Colab, sem cota) ou **Groq** (API gratuita com limites), além de qualquer API compatível com OpenAI.

Inclui: `sem_fe` (baseline), **CAAFE** e **LLM-FE** (reimplementações), **GuidedFE** (método novo do grupo) e adaptadores opcionais para **FeatCopilot** e **OpenFE**.

Também inclui adaptações de dois repositórios de colegas, ao contrato do harness:
- `lucas_autofe`: [LucasBarros28/AutoFe-LLM](https://github.com/LucasBarros28/AutoFe-LLM). A LLM vê só estatísticas das colunas e escolhe entre `log1p`, `missing_indicator` e `interaction_product`. Versão simplificada (sem detector de oportunidades nem circuit breaker); a aceitação é por validação cruzada interna.
- `gabriel_llmfe`: [GabrielFrederico1/LLM_based_FE](https://github.com/GabrielFrederico1/LLM_based_FE). A LLM escreve `add_features(df)`, só com features `fe_*` por linha; rejeita dependência de outras linhas e aceita por validação cruzada interna. Roda no sandbox do harness, em vez do `exec` do original.

Use-os pelo nome: `AUTOFES = ['caafe', 'llmfe', 'guided', 'lucas_autofe', 'gabriel_llmfe']`. Os resultados **não são comparáveis** com os scripts originais: mudam o prompt, o LLM, as bases e o protocolo.

> **Estado atual.** O encanamento está testado (11 testes automáticos). Os resultados com LLM real são preliminares (protocolo `dev`, 2 divisões) e **ainda não permitem concluir qual método é melhor**: veja [docs/resultados_preliminares.md](docs/resultados_preliminares.md). FeatCopilot, OpenFE e as bases `california_housing` e `diamonds` ainda não foram testados.

## Sumário
1. [Instalação](#1-instalação)
2. [Início rápido no Colab](#2-início-rápido-no-colab)
3. [Escolhendo o LLM (Ollama, Groq)](#3-escolhendo-o-llm)
4. [Protocolos](#4-protocolos)
5. [Bases de dados](#5-bases-de-dados)
6. [Testar o método de outra pessoa](#6-testar-o-seu-método-colegas)
7. [Juntar e analisar resultados](#7-juntar-e-analisar-resultados)
8. [Como o harness funciona](#8-como-o-harness-funciona)
9. [Estrutura do repositório](#9-estrutura-do-repositório)
10. [Limitações](#10-limitações)

## 1. Instalação
```bash
git clone https://github.com/SEU_USUARIO/autofe-bench.git
cd autofe-bench
pip install -e ".[dev]"      # numpy, pandas, scikit-learn, scipy, requests (+ pytest)
pip install ucimlrepo        # só para a base cdc_diabetes
pytest                       # 11 testes, sem rede
```

## 2. Início rápido no Colab
Abra `notebooks/autofe_bench_colab.ipynb` no Colab (ou suba-o no Drive). A célula 2 concentra os parâmetros:
```python
PROVEDOR     = 'ollama'                 # 'ollama' ou 'groq'
OLLAMA_MODEL = 'qwen3-coder:30b'
GROQ_MODEL   = 'openai/gpt-oss-120b'
DATASETS     = ['titanic', 'adult', 'bank_marketing', 'cdc_diabetes']
AUTOFES      = ['caafe', 'llmfe', 'guided']     # 'sem_fe' entra automaticamente
PROTOCOLO    = 'dev'
```
Rode as células **uma de cada vez**. Os resultados vão para um `results.jsonl` local, copiado para o Drive a cada job. Se a sessão cair ou a cota acabar, rode a célula de execução de novo: o harness **retoma** de onde parou.

Em Python ou na linha de comando:
```python
from autofe_bench.llm import OllamaClient
from autofe_bench.runner import run_benchmark
from autofe_bench.report import render
client = OllamaClient("qwen3-coder:30b")
run_benchmark(["caafe", "llmfe", "guided"], ["titanic", "adult"], protocol="dev", client=client, store_path="results/r.jsonl")
print(render("results/r.jsonl", model="hgb"))   # model: lr | rf | hgb
```
```bash
python -m autofe_bench run --methods caafe llmfe guided --datasets titanic adult --protocol dev --llm ollama:qwen3-coder:30b
python -m autofe_bench report --store results/results.jsonl --model hgb
python -m autofe_bench estimate --methods caafe llmfe guided --datasets titanic adult --protocol v1 --tpd 200000
python -m autofe_bench check guided --llm ollama:qwen3-coder:30b     # teste de contrato
python -m autofe_bench methods | datasets | models --provider ollama
```
`sem_fe` é o baseline (sem LLM) e **é incluído automaticamente**; listar `sem_fe` em `AUTOFES` não dá erro nem duplica.

## 3. Escolhendo o LLM
| Provedor | Como usar | Observações |
|---|---|---|
| **Ollama** | `OllamaClient("modelo")` ou `--llm ollama:modelo` | Sem cota. Precisa de GPU (no Colab: *Ambiente de execução → GPU*). |
| **Groq** | `GroqClient("modelo", rpm=.., tpm=..)` ou `--llm groq:modelo`; chave em `GROQ_API_KEY` | Camada gratuita com limites por modelo. Se a cota diária acabar, o harness salva e para; rode de novo depois. |
| Outro (OpenAI-compatível) | `--llm openai:MODELO@https://host/v1` | Chave em `OPENAI_API_KEY`. |

**Quais modelos posso usar?**
- Ollama, já baixados: `ollama list`, ou `python -m autofe_bench models --provider ollama`.
- Ollama, disponíveis para baixar: [ollama.com/library](https://ollama.com/library); baixe com `ollama pull NOME`.
- Groq, os que a sua chave enxerga: `python -m autofe_bench models --provider groq`. A Groq desativa modelos com frequência (o `llama-3.3-70b-versatile` saiu da camada gratuita em 16/08/2026).

**Qual é o melhor?** Não há resposta universal; depende da VRAM e da tarefa. Regra prática para este harness: modelos de código que **não "pensam" em voz alta** (a resposta tem que ser o código) e que **cabem na GPU com folga para o contexto**. Na L4 (24 GB): `qwen3-coder:30b` (MoE, ~19 GB, rápido) ou `qwen2.5-coder:32b` (~20 GB, mais lento). Em 16 GB: `qwen2.5-coder:14b`. O próprio harness serve para decidir: rode o mesmo método e as mesmas bases com dois modelos; a coluna `llm` separa os resultados. **Use o mesmo modelo em todos os métodos que for comparar.**

## 4. Protocolos
| Protocolo | Divisões | Chamadas/job | Linhas por base | Uso |
|---|---|---|---|---|
| `smoke` | 1 | 4 | 3.000 | testar que roda |
| `dev` | 2 | 8 | 8.000 | desenvolver (não comparar com `v1`) |
| `v1` | 5 (seeds 42, 1, 2, 3, 4) | 20 | toda a base | **comparação oficial** |

O hash do protocolo vai em cada linha de resultado; resultados de protocolos diferentes não se misturam. Mudar `Protocol` (em `protocol.py`) muda o hash.

## 5. Bases de dados
**Já incluídas** (`python -m autofe_bench datasets`): `titanic` (precisa do `train.csv`: defina `AUTOFE_TITANIC_CSV` ou use `/content/drive/MyDrive/AutoFE_Titanic/data/train.csv`), `breast_cancer`, `diabetes_reg`, `wine`, `synthetic` (offline); `adult`, `bank_marketing` (OpenML), `cdc_diabetes` (UCI, precisa de `ucimlrepo`), `california_housing`, `diamonds`. Já rodaram com sucesso: titanic, breast_cancer, diabetes_reg, adult, bank_marketing, cdc_diabetes.

**Adicionar uma base sem escrever código** (use no lugar do nome em `DATASETS`):
- OpenML: `'openml:ID'` (o ID aparece na página da base em openml.org). O tipo (binária/regressão) é inferido do alvo.
- CSV: `'csv:/caminho/arquivo.csv:NOME_DO_ALVO:binary'` (tipo: `binary`, `multiclass` ou `regression`).

**Adicionar com descrição própria (recomendado para métodos com LLM):** a `description` e as `column_descriptions` entram nos prompts, e o conhecimento de domínio é o que o LLM mais explora.
```python
from autofe_bench.data import dataset_from_csv, dataset_from_openml, register_dataset
ds = dataset_from_csv("meus_dados.csv", target="inadimplente", kind="binary", name="credito",
                      description="Clientes de um banco. Prever inadimplência em 12 meses.",
                      column_descriptions={"renda": "renda mensal em R$", "idade": "anos"})
run_benchmark(["caafe", "guided"], [ds], protocol="dev", client=client)          # passe o objeto direto
register_dataset("credito", lambda: dataset_from_csv(...))                       # ou registre um nome
```
**Onde procurar bases:** [OpenML](https://www.openml.org) (inclui suítes de benchmark para classificação e regressão), [UCI Machine Learning Repository](https://archive.ics.uci.edu), [Kaggle Datasets](https://www.kaggle.com/datasets), [PMLB](https://epistasislab.github.io/pmlb/). Para AutoFE com LLM, prefira bases **com semântica nas colunas** (nomes e unidades que fazem sentido), com colunas categóricas e de texto, e de tamanho médio (milhares a centenas de milhares de linhas). Evite colunas anonimizadas (`V1..Vn`): o LLM não tem o que explorar. Reserve algumas bases só para a rodada final.

## 6. Testar o seu método (colegas)
1. Copie `examples/template_method.py` e renomeie a classe. Implemente `fit(X_train, y_train, task, ctx)` devolvendo um `FittedFE` com `transform(X)`.
   - `ctx.llm.chat([...])` chama o LLM, conta as chamadas e respeita o orçamento; `ctx.remaining_calls()` diz quantas restam; `ctx.budget` traz os limites; `ctx.cv_scores(...)` e `ctx.sample_rows(...)` ajudam na validação interna.
   - `name` aparece no relatório; **mude `version` ao alterar o método** (versões diferentes ficam separadas).
2. Teste o contrato (poucas chamadas):
   ```bash
   python -m autofe_bench check meu_metodo.py:MinhaClasse --llm ollama:qwen3-coder:30b
   ```
   Ele confere: não altera a entrada, respeita o orçamento de chamadas e de features, não copia o alvo, é invariante à ordem das linhas, linha sozinha = linha em lote, e o relatório é serializável.
3. Compare com os outros, no mesmo protocolo e com o **mesmo LLM**:
   ```python
   run_benchmark(["meu_metodo.py:MinhaClasse", "caafe", "llmfe", "guided"], ["titanic", "adult"], protocol="dev", client=client)
   ```
   Parâmetros por texto: `"guided:selection='rf'"` (o nome do método no relatório passa a incluir o parâmetro).
4. Para incluir o método no repositório: coloque o arquivo em `autofe_bench/methods/`, registre-o em `_registry()` (`methods/__init__.py`) e abra um pull request.

**Regra de ouro:** estatísticas (médias por grupo, frequências) são calculadas **no treino** e guardadas no `FittedFE`; nunca sobre as linhas que chegam em `transform`. Se o seu método fizer isso, declare `inductive = False`: o harness também detecta com uma sonda e avisa no relatório que a comparação com métodos indutivos é aproximada.

## 7. Juntar e analisar resultados
Cada pessoa gera um `results.jsonl`. Para juntar (mesmo protocolo!):
```bash
python -m autofe_bench merge ana.jsonl bruno.jsonl -o todos.jsonl
python -m autofe_bench report --store todos.jsonl --model hgb
```
O relatório traz média por base, **ganho pareado sobre o baseline com IC bootstrap**, rank médio, custo (chamadas, tokens, segundos) e avisos (métodos transdutivos, jobs com erro). Um job só é `ok` se o LLM respondeu de fato: se nenhuma chamada teve sucesso, o job sai como erro com a mensagem do provedor.

## 8. Como o harness funciona
- **Contrato** (`contract.py`): `Task`, `Budget`, `Context`, `AutoFEMethod`, `FittedFE`.
- **LLM** (`llm.py`): clientes Groq/Ollama/OpenAI-compatível, limitador de RPM/TPM, tratamento de 429 e cota diária, `BudgetedLLM` (conta e limita chamadas/tokens; grava as chamadas em log).
- **Sandbox** (`sandbox.py`): valida o código do LLM por AST (sem imports/atributos privados), builtins restritos, timeout e limite de memória. É uma barreira contra erro, **não** um isolamento de segurança.
- **Pré-processamento comum** (`prep.py`): ajustado só no treino (one-hot até 15 níveis, NaN como categoria). Modelos finais fixos: regressão logística/Ridge, RF (300 árvores, prof. 6), HistGradientBoosting.
- **Corredor** (`runner.py`) e **armazenamento** (`store.py`): um job por (método, base, divisão), JSON Lines com retomada.
- **Relatório** (`report.py`), **teste de contrato** (`check.py`), **LLM simulado** para testes sem rede (`mockllm.py`).

CAAFE e LLM-FE são **reimplementações** fiéis ao artigo, não o código original; as escolhas que o artigo não especifica estão documentadas nos docstrings. FeatCopilot com motor `llm` chama o LLM por fora do harness: essas chamadas **não são contadas**.

## 9. Estrutura do repositório
```
autofe-bench/
├── README.md
├── pyproject.toml
├── .gitignore
├── autofe_bench/
│   ├── __init__.py  __main__.py  cli.py
│   ├── contract.py  protocol.py  data.py  prep.py  sandbox.py
│   ├── llm.py  mockllm.py  runner.py  store.py  report.py  check.py
│   └── methods/
│       ├── __init__.py      # registro e resolve_method
│       ├── caafe.py  llmfe.py  guided.py  legacy.py  external.py
├── examples/template_method.py
├── notebooks/autofe_bench_colab.ipynb
├── docs/resultados_preliminares.md
└── tests/test_core.py
```

## 10. Limitações
- Resultados com LLM real são preliminares; nenhuma conclusão sobre qual método é melhor.
- Resultados de LLMs diferentes não são comparáveis entre si. Temperatura > 0 gera variação: use as 5 divisões e os intervalos de confiança.
- Adaptadores FeatCopilot e OpenFE, e as bases `california_housing` e `diamonds`, não foram testados.
- O sandbox não é isolamento de segurança: rode código de LLM em ambiente descartável (como o Colab).
- Defina uma licença antes de tornar o repositório público.
