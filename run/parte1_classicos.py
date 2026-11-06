"""Parte 1: comparação dos métodos sem PINN (estados |+> e |0>) e estudo de desenho experimental.

    python run/parte1_classicos.py

Escreve data/parte1/*.csv e paper/figures/parte1_metodos_sem_pinn.pdf.
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

import os
import csv
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bloch as cm
from classicos import direct_inversion, linear_lsq, BlochShooting, shooting_fit, lam_for
from paths import save_figure


# ---------------------------------------------------------------------------
# Experimento 1: comparação dos métodos nos 3 casos
# ---------------------------------------------------------------------------
def comparacao_metodos():
    print("\n=== Comparação dos métodos sem PINN (N=100 pontos) ===")
    rows = []
    fig_data = {}
    for case in [1, 2, 3]:
        for r0name, r0 in [("|+>", cm.R0_PLUS), ("|0>", cm.R0_ZERO)]:
            for noise in [0.0, 0.01, 0.03]:
                t, data, clean, ffn, T = cm.make_data(case, N=100, noise=noise, seed=1, r0=r0)
                ftrue = ffn(t)
                smooth = noise > 0
                lam = lam_for(noise)

                t0 = time.time()
                fA = direct_inversion(t, data, smooth=smooth)
                tA = time.time() - t0

                t0 = time.time()
                fB, cB = linear_lsq(t, data, T, "bspline", 40, lam=lam, smooth=smooth)
                tB = time.time() - t0

                t0 = time.time()
                fBf, _ = linear_lsq(t, data, T, "fourier", 10, lam=0.0, smooth=smooth)
                tBf = time.time() - t0

                sh = BlochShooting(t, T, n_basis=40, r0=r0)
                t0 = time.time()
                fC0, _ = sh.fit(data, lam=lam)                 # chute inicial c=0
                tC0 = time.time() - t0

                t0 = time.time()
                fC, _ = sh.fit(data, lam=lam, c0=cB)           # warm start vindo de (B)
                tC = time.time() - t0

                for name, fp, tt in [("A) inversão direta", fA, tA),
                                     ("B) LSQ linear B-spline40", fB, tB),
                                     ("B) LSQ linear Fourier10", fBf, tBf),
                                     ("C) shooting (c0=0)", fC0, tC0),
                                     ("C) shooting (warm start)", fC, tC)]:
                    e = cm.rel_l2(fp, ftrue)
                    rs = cm.resim_rmse(fp, t, clean, r0)
                    rows.append((case, r0name, noise, name, e, rs, tt))
                    print(f"caso {case} | r0={r0name} | ruído={noise:4.2f} | {name:26s} | "
                          f"erro campo={e:7.4f} | RMSE re-sim={rs:.2e} | {tt:6.2f}s", flush=True)
                if r0name == "|+>":
                    fig_data[(case, noise)] = (t, ftrue, fA, fB, fC)

    with open(os.path.join("data", "parte1", "resultados_sem_pinn.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["caso", "r0", "ruido", "metodo", "erro_L2_rel_campo", "rmse_resim", "tempo_s"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], r[3], f"{r[4]:.4f}", f"{r[5]:.2e}", f"{r[6]:.2f}"])

    fig, axes = plt.subplots(2, 3, figsize=(14, 6.5), sharex="col")
    for j, case in enumerate([1, 2, 3]):
        for i, noise in enumerate([0.0, 0.03]):
            t, ftrue, fA, fB, fC = fig_data[(case, noise)]
            ax = axes[i, j]
            ax.plot(t, ftrue, "k-", lw=2.5, label="verdadeiro")
            ax.plot(t, fA, ".", ms=3, color="tab:gray", label="(A) inversão direta")
            ax.plot(t, fB, "--", color="tab:blue", label="(B) LSQ linear B-spline")
            ax.plot(t, fC, "-", color="tab:red", lw=1.2, label="(C) shooting")
            ax.set_ylim(ftrue.min() - 0.8, ftrue.max() + 0.8)
            ax.set_title(f"caso {case}, ruído σ={noise}, r0=|+>")
    axes[0, 0].legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("t")
    plt.tight_layout()
    save_figure(fig, "parte1_metodos_sem_pinn")
    plt.close()
    return rows


# ---------------------------------------------------------------------------
# Experimento 2: desenho experimental com o MESMO orçamento de medidas
# ---------------------------------------------------------------------------
def desenho_experimental(case=3, n_rep=5):
    """Mesmo número total de medições (shots) em todas as opções:
       ruído por ponto cresce com sqrt(nº de pontos) quando dividimos o orçamento."""
    print(f"\n=== Desenho experimental (caso {case}, orçamento fixo, média de {n_rep} ruídos) ===")
    R0_PLUS_I = np.array([0.0, 1.0, 0.0])
    designs = [
        ("1 estado |+>, N=100", [cm.R0_PLUS], 100),
        ("1 estado |0>, N=100", [cm.R0_ZERO], 100),
        ("1 estado |+>, N=300", [cm.R0_PLUS], 300),
        ("3 estados |+>,|0>,|+i>, N=100 cada", [cm.R0_PLUS, cm.R0_ZERO, R0_PLUS_I], 100),
    ]
    out = []
    for sigma in [0.01, 0.03]:
        for label, r0s, N in designs:
            n_total = len(r0s) * N
            s_eff = sigma * np.sqrt(n_total / 100)   # mesmo orçamento que 1 estado x 100 pontos
            eB, eC = [], []
            for rep in range(n_rep):
                datas, cleans = [], []
                for j, r0 in enumerate(r0s):
                    t, d, c, ffn, T = cm.make_data(case, N=N, noise=s_eff, seed=100 * rep + j, r0=r0)
                    datas.append(d)
                    cleans.append(c)
                ftrue = ffn(t)
                lam = lam_for(sigma)
                fB, cB = linear_lsq(t, datas, T, "bspline", 40, lam=lam, smooth=True)
                shs = [BlochShooting(t, T, 40, substeps=max(1, 1000 // N), r0=r0) for r0 in r0s]
                fC, _ = shooting_fit(shs, datas, lam=lam, c0=cB)
                eB.append(cm.rel_l2(fB, ftrue))
                eC.append(cm.rel_l2(fC, ftrue))
            print(f"σ(1x100)={sigma:4.2f} | {label:36s} | σ/ponto={s_eff:.3f} | "
                  f"LSQ linear: {np.mean(eB):.3f}±{np.std(eB):.3f} | "
                  f"shooting: {np.mean(eC):.3f}±{np.std(eC):.3f}", flush=True)
            out.append((sigma, label, s_eff, np.mean(eB), np.std(eB), np.mean(eC), np.std(eC)))
    with open(os.path.join("data", "parte1", "resultados_desenho_experimental.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sigma_ref", "desenho", "sigma_por_ponto", "lsq_media", "lsq_std",
                    "shooting_media", "shooting_std"])
        for r in out:
            w.writerow([r[0], r[1], f"{r[2]:.4f}"] + [f"{x:.4f}" for x in r[3:]])
    return out


if __name__ == "__main__":
    comparacao_metodos()
    desenho_experimental()
