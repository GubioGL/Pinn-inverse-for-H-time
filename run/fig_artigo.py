"""Figuras do artigo a partir dos resultados do protocolo (data/resultados/).

    python run/fig_artigo.py

Escreve em paper/figures/:
  fig_benchmark.pdf   mapa de calor: mediana do erro do campo (5 seeds) por método e cenário
  fig_campos.pdf      campo recuperado no caso 3 (seed 0): treino original x melhorado x clássico
  fig_ablacao.pdf     efeito de cada novidade do PINN (perfil "ablacao")
"""

# --- repository layout: src/ and run/ go on the import path, and relative paths
# --- such as "data/..." are resolved from the repository root. Keep this block in every script.
import os as _os
import sys as _sys
from pathlib import Path as _Path

_HERE = _Path(__file__).resolve().parent if "__file__" in globals() else _Path.cwd()
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "src").is_dir() and (p / "run").is_dir())
for _p in (_ROOT / "run", _ROOT / "src", _HERE):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
_os.chdir(_ROOT)
# ---------------------------------------------------------------------------------

import glob
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from paths import DATA, save_figure
import executar

ROTULOS = {
    "direta": "inversão direta", "lsq_fourier10": "LSQ Fourier M=10", "lsq_cosseno40": "LSQ cossenos M=40",
    "lsq_bspline40": "LSQ B-spline n=40", "lsq_bspline80": "LSQ B-spline n=80",
    "shooting_bs40": "shooting B-spline n=40", "shooting_bs80": "shooting B-spline n=80",
}
CAMPOS = ["nn", "fourier", "cosseno", "bspline", "mixfunn"]
RECEITAS = ["original", "lbfgs", "lm"]


def carregar(perfil):
    out = []
    for f in glob.glob(str(DATA / "resultados" / perfil / "jobs" / "*.json")):
        with open(f) as fh:
            out.append(json.load(fh))
    return out


def mediana(res, metodo, caso, ruido):
    e = [r["erro_campo"] for r in res if r["metodo"] == metodo and r["caso"] == caso
         and r["ruido"] == ruido and r["status"] == "ok"]
    return np.median(e) if e else np.nan


def fig_benchmark(res):
    cen = sorted({(r["caso"], r["ruido"]) for r in res})
    classicos = [m for m in ROTULOS if any(r["metodo"] == m for r in res)]
    pinns = [f"pinn_{c}_{rc}" for rc in RECEITAS for c in CAMPOS if any(r["metodo"] == f"pinn_{c}_{rc}" for r in res)]
    metodos = classicos + pinns
    M = np.array([[mediana(res, m, c, r) for c, r in cen] for m in metodos])
    fig, ax = plt.subplots(figsize=(4.8, 0.24 * len(metodos) + 1.0))
    im = ax.imshow(np.clip(M, 1e-6, 10), norm=LogNorm(1e-5, 1), cmap="viridis_r", aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isfinite(v):
                txt = f"{100 * v:.2g}%" if v >= 1e-4 else f"{100 * v:.0e}%"
                ax.text(j, i, txt, ha="center", va="center", fontsize=5.5,
                        color="white" if v < 3e-3 else "black")
    ax.set_xticks(range(len(cen)))
    ax.set_xticklabels([f"caso {c}\n$\\sigma$={r:g}" for c, r in cen], fontsize=7)
    nomes = [ROTULOS.get(m, m.replace("pinn_", "PINN ").replace("_", " + ")) for m in metodos]
    ax.set_yticks(range(len(metodos)))
    ax.set_yticklabels(nomes, fontsize=6.5)
    ax.axhline(len(classicos) - 0.5, color="w", lw=2)
    for k in range(1, len(RECEITAS)):
        ax.axhline(len(classicos) + k * len(CAMPOS) - 0.5, color="w", lw=1)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label("mediana do erro L2 relativo do campo", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    return save_figure(fig, "fig_benchmark")


def fig_campos(res):
    escolhas = [("pinn_nn_original", "PINN (treino original)", "#d62728", ":"),
                ("pinn_nn_lm", "PINN campo-rede + LM", "#1f77b4", "--"),
                ("shooting_bs80", "shooting B-spline n=80", "#9467bd", "-.")]
    ruidos = sorted({r["ruido"] for r in res if r["caso"] == 3})
    fig, axes = plt.subplots(len(ruidos), 1, figsize=(3.4, 1.9 * len(ruidos)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, ruido in zip(axes, ruidos):
        t, data, clean, ffn, T = executar.gerar_dados(3, ruido, 0)
        ax.plot(t, ffn(t), "k-", lw=2.2, label="verdadeiro")
        for m, rot, cor, ls in escolhas:
            r = [x for x in res if x["metodo"] == m and x["caso"] == 3 and x["ruido"] == ruido and x["seed"] == 0]
            if r and r[0]["status"] == "ok":
                ax.plot(t, r[0]["f_pred"], ls=ls, color=cor, lw=1.3,
                        label=f"{rot} ({100 * r[0]['erro_campo']:.2g}%)")
        ax.set_ylabel("$f(t)$")
        ax.set_title(rf"caso 3, $\sigma$ = {ruido:g}", fontsize=8)
        ax.legend(fontsize=5.5, frameon=False, loc="lower left")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[-1].set_xlabel("$t$")
    plt.tight_layout()
    return save_figure(fig, "fig_campos")


def fig_ablacao(res):
    ordem = ["lm", "lm_sembc", "lm_seminit", "lm_semreamostra", "lm_soDados", "lm_varpro", "lm_pesos"]
    rot = {"lm": "receita completa (LM)", "lm_sembc": "sem borda rígida do campo",
           "lm_seminit": "sem warm start (LSQ)", "lm_semreamostra": "colocação fixa (sem resorteio)",
           "lm_soDados": "EDO só nos pontos dos dados", "lm_varpro": "+ variable projection",
           "lm_pesos": "+ pesos adaptativos"}
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    y = 0
    ticks, labels = [], []
    for campo, cor in (("nn", "#1f77b4"), ("bspline", "#d62728")):
        for rc in ordem:
            m = f"pinn_{campo}_{rc}"
            e = [r["erro_campo"] for r in res if r["metodo"] == m and r["status"] == "ok"]
            if not e:
                continue
            ax.scatter(e, np.full(len(e), y), s=10, color=cor, alpha=0.6)
            ax.plot([np.median(e)], [y], "|", color=cor, ms=10, mew=2)
            ticks.append(y)
            labels.append(f"{campo}: {rot[rc]}")
            y += 1
        y += 0.6
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=6)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("erro L2 relativo do campo (caso 3, sem ruído)", fontsize=7)
    ax.grid(axis="x", alpha=0.3, which="both", lw=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    plt.tight_layout()
    return save_figure(fig, "fig_ablacao")


if __name__ == "__main__":
    padrao = carregar("padrao")
    if padrao:
        print("ok:", fig_benchmark(padrao))
        print("ok:", fig_campos(padrao))
    abl = carregar("ablacao")
    if abl:
        print("ok:", fig_ablacao(abl))
