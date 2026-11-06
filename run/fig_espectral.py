"""Resolução espectral: erro do campo em função da frequência do pulso.

    python run/rodar.py --perfil espectral      # gera data/resultados/espectral/jobs/*.json
    python run/fig_espectral.py

Pulsos f(t) = sin(w t) sin^2(pi t/T), T = 2.52 pi, estado inicial |0>, N = 100 pontos.
Para cada método e frequência w: mediana do erro L2 relativo do campo sobre as seeds
(faixa = mínimo-máximo). Escreve paper/figures/fig_espectral.pdf e
data/resultados/espectral/espectral.csv.
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

import csv
import glob
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import DATA, save_figure

ESTILO = {
    "direta":           ("inversão direta", "#6b6b6b", "o", "--"),
    "lsq_fourier10":    ("LSQ Fourier (M=10)", "#bcbd22", "v", "--"),
    "lsq_cosseno40":    ("LSQ cossenos (M=40)", "#8c564b", "^", "--"),
    "lsq_bspline80":    ("LSQ B-spline (n=80)", "#2ca02c", "s", "--"),
    "shooting_bs80":    ("shooting B-spline (n=80)", "#9467bd", "D", "--"),
    "pinn_nn_lm":       ("PINN campo-rede + LM", "#1f77b4", "o", "-"),
    "pinn_bspline_lm":  ("PINN B-spline + LM", "#d62728", "s", "-"),
}


def main():
    pasta = DATA / "resultados" / "espectral"
    res = []
    for f in glob.glob(str(pasta / "jobs" / "*.json")):
        with open(f) as fh:
            r = json.load(fh)
        if r["status"] == "ok":
            res.append(r)
    if not res:
        print("sem resultados em", pasta, "-> rode: python run/rodar.py --perfil espectral")
        return
    ruidos = sorted({r["ruido"] for r in res})
    linhas = []
    fig, axes = plt.subplots(1, len(ruidos), figsize=(3.6 * len(ruidos), 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, ruido in zip(axes, ruidos):
        for metodo, (rot, cor, mk, ls) in ESTILO.items():
            ws, med, lo, hi = [], [], [], []
            for w in sorted({float(r["caso"][1:]) for r in res}):
                e = [r["erro_campo"] for r in res if r["metodo"] == metodo and r["ruido"] == ruido
                     and float(r["caso"][1:]) == w]
                if e:
                    ws.append(w)
                    med.append(np.median(e))
                    lo.append(np.min(e))
                    hi.append(np.max(e))
                    linhas.append(dict(metodo=metodo, ruido=ruido, omega=w, n=len(e),
                                       erro_mediana=np.median(e), erro_min=np.min(e), erro_max=np.max(e)))
            if ws:
                ax.plot(ws, med, ls=ls, marker=mk, ms=3.5, color=cor, lw=1.2, label=rot)
                ax.fill_between(ws, lo, hi, color=cor, alpha=0.12, lw=0)
        T = 2.52 * np.pi
        ax.axvline(2 * np.pi * 10 / T, color="#bcbd22", lw=0.8, ls=":")   # maior frequência da Fourier M=10
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"frequência do pulso $\omega$")
        ax.set_title(rf"ruído $\sigma$ = {ruido:g}", fontsize=9)
        ax.grid(alpha=0.3, which="both", lw=0.4)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("erro L2 relativo do campo")
    axes[-1].legend(fontsize=6.5, frameon=False, loc="lower right")
    plt.tight_layout()
    path = save_figure(fig, "fig_espectral")
    with open(pasta / "espectral.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)
    print("ok:", path)


if __name__ == "__main__":
    main()
