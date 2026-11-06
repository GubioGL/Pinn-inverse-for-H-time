# Notes for Claude

Research-paper repository made from GubioGL/paper-template. Subject: inverse problem of recovering
the time-dependent control field f(t) of H(t) = 0.5 σz + 0.5 f(t) σx from qubit expectation values,
comparing PINNs (field as neural net / Fourier / cosine series / B-spline / MixFunn; Adam, L-BFGS,
BFGS, Levenberg-Marquardt) against classical inversion (direct, linear least squares, shooting)
under a single protocol. The author writes notes in Portuguese (`notes/`); code comments and the
paper are in English where new, Portuguese in the existing code.

## Layout rules
- `src/` = importable library code. `run/` = entry points (experiments, analysis, figures).
- Every script in `run/` starts with the repository header block (copy it from `run/fig_artigo.py`):
  it puts `run/`, `src/` and the script's folder on `sys.path` and `chdir`s to the repository root.
- Figures: always write to `paper/figures/` as PDF via `paths.save_figure(fig, name)`. Never commit PNGs.
  Diagnostic figures of a benchmark profile go to `data/resultados/<perfil>/figuras/*.pdf`.
- Data the figures need lives in `data/`; huge raw data goes to an archive (see `data/README.md`).
- Before stating a number in the paper or in `notes/`, check it against the data in `data/`.
- `notebooks/` holds the original experiments (modelo_1 … Modelo_5, 5_niveis, qutip_*) and the
  superseded part-1 scripts; do not change their behaviour.

## Protocol (src/protocolo.py) — keep it fair
- Same data for every method per (case, noise, seed); initial state |0⟩ (justified in notes/ANALISE.md);
  N = 100; the true field never enters a loss or a hyper-parameter choice; field edges are imposed
  only when known from the pulse design (`bloch.bordas_conhecidas`); always report `erro_campo`
  and `rmse_traj`; ≥ 5 seeds for the main profiles.
- Results: `python run/rodar.py --perfil <p>` (resumable; one JSON per job in data/resultados/<p>/jobs),
  then `python run/relatorio.py --perfil <p>` and `python run/fig_artigo.py`.
- `data/resultados/v1_*` were produced by an earlier version of the code (kept for the record).

## Paper / Overleaf
- `paper/` is synced two-way with the Overleaf project in the repository variable `OVERLEAF_PROJECT`
  by `.github/workflows/overleaf-sync.yml` (script `.github/scripts/overleaf_sync.sh`; bookkeeping in
  `.github/overleaf-sync-state`, do not edit by hand). It only runs on `main`.
- To change the paper: edit `paper/` and push to `main`; the workflow pushes to Overleaf.
- Branch `overleaf-conflict` = unresolved same-line edits; merge it into `main` and push.
- Workflow status without a GitHub login: `https://api.github.com/repos/<owner>/<repo>/actions/runs`
  (public repositories), and error messages in `.../check-runs/<job id>/annotations`.
