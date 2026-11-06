# data/

Resultados que os scripts de `run/` escrevem e de onde saem os números do artigo.

| pasta | conteúdo | gerado por |
|---|---|---|
| `resultados/<perfil>/jobs/*.json` | um arquivo por job (método, caso, ruído, seed): métricas, campo predito, histórico da loss | `run/rodar.py --perfil <perfil>` |
| `resultados/<perfil>/resumo.csv`, `RESULTADOS.md` | mediana, média ± desvio, pior seed, tempo por método e cenário | `run/relatorio.py --perfil <perfil>` |
| `resultados/<perfil>/figuras/*.pdf` | figuras de diagnóstico do perfil | `run/relatorio.py` |
| `resultados/espectral/espectral.csv` | erro do campo vs. frequência do pulso | `run/fig_espectral.py` |
| `resultados/v1_padrao`, `resultados/v1_ablacao` | resultados da versão 1 do código: colocação num grid fixo, B-spline de 40 funções, sem bordas rígidas nem warm start | versão anterior de `run/rodar.py` |
| `parte1/*.csv` | diagnóstico inicial (estado \|+⟩ vs \|0⟩, desenho experimental) | `run/parte1_classicos.py` |

- Mantenha aqui tudo o que as figuras precisam, para que `run/` regenere o artigo a partir de
  um clone limpo.
- O GitHub rejeita arquivos acima de 100 MB e fica lento acima de ~1 GB por repositório. Dados
  brutos muito grandes devem ir para um arquivo externo, como o Zenodo, que também dá um DOI
  para a declaração de disponibilidade de dados.
- `resultados/rapido/` (teste de fumaça) não é versionado.
