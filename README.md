# Recovering a time-dependent control field from qubit expectation values: PINNs vs. classical inversion

Repositório do artigo sobre o problema inverso de recuperar o campo externo $f(t)$ de

$$H(t) = \tfrac12\,\sigma_z + \tfrac12\, f(t)\,\sigma_x$$

a partir de medidas de ⟨σx⟩, ⟨σy⟩, ⟨σz⟩. São comparadas redes neurais informadas pela física
(PINNs) com quatro parametrizações do campo (rede neural, Fourier, B-spline e MixFunn, mais a série
de cossenos) contra métodos clássicos (inversão direta, mínimos quadrados lineares e *shooting*),
todos sob um **protocolo único**: mesmos dados, mesmo estado inicial, campo verdadeiro fora da
loss, 5 seeds e duas métricas.

Organizado a partir do modelo [GubioGL/paper-template](https://github.com/GubioGL/paper-template):
o código, os dados e o manuscrito ficam juntos, e `paper/` pode ser sincronizado com o Overleaf.

## Layout

```text
├── src/          biblioteca: física (bloch.py), métodos clássicos, PINN, otimizadores, protocolo
│   └── paths.py  caminhos do repositório + save_figure()
├── run/          scripts executáveis (benchmark, relatórios, figuras) + requirements.txt
├── data/         resultados do benchmark (JSON por job, resumos CSV/MD)
├── paper/        manuscrito LaTeX no formato APS/revtex (sincronizado com o Overleaf)
│   ├── main.tex            artigo (duas colunas)
│   ├── supplementary.tex   material suplementar (detalhes técnicos e todos os valores)
│   ├── references.bib
│   ├── tabelas/  tabelas .tex geradas por run/tabelas_artigo.py
│   └── figures/  PDFs gerados por run/fig_artigo.py
├── notes/        análises e notas de trabalho (em português)
│   ├── ANALISE.md     diagnóstico, justificativas (ex.: estado |0⟩) e resultados
│   ├── PROTOCOLO.md   regras do protocolo, métodos, como rodar
│   ├── METODOS.md     revisão de métodos de PINN e o que foi medido
│   └── TODO.md        próximos casos (vários estados iniciais; dados parciais)
├── notebooks/    experimentos originais (modelo_1 … Modelo_5, 5_niveis, qutip_*) e parte 1
└── .github/      sincronização com o Overleaf (workflow + script)
```

## Como rodar

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

```bash
pip install -r run/requirements.txt
```

Para conferir as dependências e a GPU, e medir a velocidade:

```bash
python run/verificar_ambiente.py
```

Teste rápido (~3 min):

```bash
python run/rodar.py --perfil rapido
```

Benchmark principal (~3 h em 4 CPUs). Na GPU use `--device cuda --workers 8`:

```bash
python run/rodar.py --perfil padrao --workers 4
```

Tabelas em `data/resultados/padrao/`:

```bash
python run/relatorio.py --perfil padrao
```

Tabelas e figuras do artigo, em `paper/tabelas/` e `paper/figures/`, sempre a partir dos dados:

```bash
python run/tabelas_artigo.py
```

```bash
python run/fig_artigo.py
```

Para compilar localmente, sem o Overleaf (precisa do pacote `revtex`, no TeX Live `texlive-publishers`):

```bash
cd paper && latexmk -pdf main.tex && latexmk -pdf supplementary.tex
```

Os detalhes (perfis, opções, GPU, retomada) estão em [`notes/PROTOCOLO.md`](notes/PROTOCOLO.md).
Os resultados e as conclusões estão em [`notes/ANALISE.md`](notes/ANALISE.md).

## Figuras e artigo

Toda figura do artigo é salva por `paths.save_figure(fig, nome)`, que grava
`paper/figures/<nome>.pdf` (e um `.png` local de pré-visualização, que o git ignora). Em
`paper/main.tex`, use `\includegraphics{<nome>.pdf}`. Cada script de `run/` começa com o bloco de
cabeçalho do template, então `python run/<script>.py` funciona a partir de qualquer pasta.

## Sincronização com o Overleaf

O workflow `.github/workflows/overleaf-sync.yml` sincroniza `paper/` com um projeto do Overleaf
nas duas direções. Ele só roda no branch `main`, depois de configurado:

1. Em *Settings → Secrets and variables → Actions*, aba **Variables**, crie `OVERLEAF_PROJECT` com
   o endereço do projeto (`https://www.overleaf.com/project/...`).
2. Na aba **Secrets**, crie `OVERLEAF_TOKEN` com um token Git do Overleaf (*Account Settings → Git
   integration*). Isso requer plano premium do Overleaf.
3. Rode *Actions → Overleaf sync → Run workflow*.

Enquanto `OVERLEAF_PROJECT` não estiver definido, o job é pulado e não gasta minutos de Actions.
Detalhes no [README do template](https://github.com/GubioGL/paper-template#how-the-overleaf-sync-works).
