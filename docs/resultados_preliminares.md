# Resultados preliminares (protocolo `dev`)

**Leia isto como diagnóstico, não como comparação.** Protocolo `dev`: 2 divisões por base, 8 chamadas de LLM por job, bases subamostradas em 8.000 linhas, um único LLM (`qwen3-coder:30b` via Ollama, GPU L4). Com 2 divisões, diferenças de ±0,3% estão dentro do ruído. A comparação oficial exige o protocolo `v1` (5 divisões, 20 chamadas), mais bases e teste estatístico pareado.

Ganho relativo médio sobre o baseline `sem_fe` (positivo = melhor; AUC nas binárias, RMSE nas regressões), por modelo final. Entre parênteses: vitórias/empates/derrotas por divisão.

| Base | Método | lr | rf | hgb | novas feat. |
|---|---|---|---|---|---|
| titanic | CAAFE | +1,52% (2/0/0) | −0,71% (0/0/2) | +0,26% (1/0/1) | 3,5 |
| titanic | LLM-FE | +2,18% (2/0/0) | −0,20% (0/0/2) | +0,49% (1/0/1) | 6,0 |
| titanic | GuidedFE 1.1 | +0,02% (2/0/0) | −0,43% (0/0/2) | +0,30% (2/0/0) | 1,0 |
| breast_cancer | CAAFE | +0,02% | +0,30% | +0,25% | 3,5 |
| breast_cancer | LLM-FE | +0,12% | −0,21% | +0,48% | 6,0 |
| breast_cancer | GuidedFE 1.1 | +0,02% | −0,08% | +0,05% | 0,5 |
| diabetes_reg | CAAFE | −0,47% | +0,59% | +2,06% | 4,5 |
| diabetes_reg | LLM-FE | +1,24% | −0,48% | +2,05% | 3,0 |
| diabetes_reg | GuidedFE 1.1 | 0,00% | 0,00% | 0,00% | 0,0 |
| adult | CAAFE | +0,01% | −0,08% | −0,04% | 2,5 |
| adult | LLM-FE | −0,40% | −0,04% | −0,31% | 6,0 |
| adult | GuidedFE 1.1 | 0,00% | −0,02% | −0,12% | 0,5 |
| bank_marketing | CAAFE | +0,17% | −0,08% | −0,32% | 2,0 |
| bank_marketing | LLM-FE | +0,29% | −0,27% | −0,22% | 6,0 |
| bank_marketing | GuidedFE 1.1 | +0,04% | −0,26% | −0,06% | 1,5 |
| cdc_diabetes | CAAFE | +0,06% | −0,40% | +0,04% | 3,0 |
| cdc_diabetes | LLM-FE | +0,13% | −0,40% | −0,13% | 6,0 |
| cdc_diabetes | GuidedFE 1.1 | −0,12% | −0,23% | +0,05% | 0,5 |

Média nas 6 bases (12 divisões por método e modelo): CAAFE lr +0,22% / rf −0,06% / hgb +0,38%; LLM-FE lr +0,59% / rf −0,27% / hgb +0,40%; GuidedFE 1.1 lr −0,01% / rf −0,17% / hgb +0,04%.

## O que isso indica
1. **Nas bases grandes (adult, bank_marketing, cdc_diabetes) nenhum método melhora de forma visível.** O baseline já é forte (HGB ≈ 0,92 AUC no adult e no bank) e os ganhos ficam em ±0,3%.
2. **Os ganhos aparecem em modelos lineares (lr) e em bases pequenas**, não em árvores (rf), onde razões e diferenças entre colunas são aproximadas pelos próprios cortes.
3. **GuidedFE 1.1 é conservador: 4 de 12 jobs aceitaram zero features** e, quando aceitou, o ganho na validação interna (+0,002 a +0,007 AUC) não apareceu no teste. Das 196 colunas candidatas avaliadas, o ganho médio foi −0,0017 e só 16% tiveram ganho positivo.
4. Parte das chamadas foi desperdiçada: 6 das 96 respostas tinham código quebrado, e algumas tentavam usar o alvo (`Column not found: Survived`), o que o harness barra.

A v1.2 do GuidedFE responde aos pontos 2 e 3: a seleção passa a medir rf, hgb e lr (`selection='multi'`) e o prompt avisa que o alvo não existe. **Não foi avaliada ainda com LLM real.**
