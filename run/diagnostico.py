"""
Diagnóstico do problema inverso (sem treinar nada).

1) Observabilidade: f(t) só aparece multiplicado por Y e Z:
        f Z = X - dY/dt ,   f Y = dZ/dt
   => a "informação" sobre f em cada instante é proporcional a Y^2 + Z^2 = 1 - X^2.
   Quando o vetor de Bloch está alinhado com o eixo do controle (sx), f é invisível.
   O estado inicial |+> dos notebooks tem X=1 => f(t~0) não é identificável.

2) Capacidade de representação de cada base (ajuste supervisionado de mínimos quadrados).

3) Condicionamento: quanto a trajetória muda quando mudamos o campo?
   (valores singulares do Jacobiano trajetória/coeficientes).

Uso:  python run/diagnostico.py   (figuras em paper/figures/)
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
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bloch as cm

from paths import save_figure


def observabilidade():
    print("\n=== 1) Observabilidade  (Y^2+Z^2 = 1-X^2) ===")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
    for ax, case in zip(axes, [1, 2, 3]):
        f, T = cm.make_case(case)
        t = np.linspace(0, T, 400)
        for name, r0 in [("|+> (notebooks)", cm.R0_PLUS), ("|0>", cm.R0_ZERO),
                         ("genérico (1,1,1)/√3", cm.R0_GENERIC)]:
            r = cm.simulate(f, t, r0)
            w = r[:, 1] ** 2 + r[:, 2] ** 2
            ax.plot(t, w, label=name)
            frac = np.mean(w < 0.05)
            print(f"caso {case} | r0={name:22s} | fração do tempo com Y²+Z² < 0.05: {frac:5.1%}"
                  f" | min = {w.min():.2e}")
        ax.set_title(f"caso {case}")
        ax.set_xlabel("t")
        ax.axhline(0.05, color="k", lw=0.5, ls=":")
    axes[0].set_ylabel(r"$Y^2+Z^2$  (sensibilidade a f)")
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    save_figure(fig, "observabilidade")
    plt.close()


def capacidade_bases():
    print("\n=== 2) Capacidade das bases (ajuste supervisionado, erro L2 relativo) ===")
    configs = [
        ("fourier", 10, None, "Fourier M=10 (modelo_3)"),
        ("fourier", 20, None, "Fourier M=20"),
        ("cosine", 20, None, "Cossenos M=20"),
        ("cosine", 30, None, "Cossenos M=30"),
        ("bspline", 30, 10, "B-spline n=30 k=10 (modelo_4)"),
        ("bspline", 30, 3, "B-spline n=30 k=3"),
        ("bspline", 40, 3, "B-spline n=40 k=3"),
    ]
    for case in [1, 2, 3]:
        f, T = cm.make_case(case)
        t = np.linspace(0, T, 2000)
        y = f(t)
        for kind, size, k, label in configs:
            B = cm.basis_matrix(kind, t, T, size, k or 3)
            c, *_ = np.linalg.lstsq(B, y, rcond=None)
            err = cm.rel_l2(B @ c, y)
            edge = np.abs(B @ c - y)[:50].max()
            print(f"caso {case} | {label:30s} | nparams={B.shape[1]:3d} | erro={err:.2e}"
                  f" | erro máx perto de t=0: {edge:.2e}")


def condicionamento():
    print("\n=== 3) Sensibilidade da trajetória ao campo (caso 3) ===")
    f, T = cm.make_case(3)
    t = np.linspace(0, T, 200)
    base = cm.simulate(f, t)
    env = lambda s: np.sin(np.pi * s / T) ** 2
    for name, df in [
        ("remover 0.4 sin(7t) * env", lambda s: -0.4 * np.sin(7 * s) * env(s)),
        ("remover 0.3 cos(3t-0.5) * env", lambda s: 0.3 * np.cos(3 * s - 0.5) * env(s)),
        ("remover 1.5 sin(2t) * env", lambda s: -1.5 * np.sin(2 * s) * env(s)),
    ]:
        g = lambda s, df=df: f(s) + df(s)
        traj = cm.simulate(g, t)
        sd = np.linalg.norm(df(t)) / np.linalg.norm(f(t))
        print(f"{name:32s} | mudança no campo (L2 rel) = {sd:.3f} | "
              f"mudança máx em <σ> = {np.abs(traj - base).max():.4f}")

    # Jacobiano numérico trajetória/coeficientes de uma base de senos sin(n pi t/T)
    print("\nValores singulares do Jacobiano  d<σ>(t)/d a_n,   f = sum a_n sin(n π t / T):")
    nmax = 30
    eps = 1e-4
    J = []
    for n in range(1, nmax + 1):
        g = lambda s, n=n: f(s) + eps * np.sin(n * np.pi * s / T)
        J.append(((cm.simulate(g, t) - base) / eps).ravel())
    J = np.array(J).T
    sv = np.linalg.svd(J, compute_uv=False)
    print("  freq. angular  ω_n = n π / T  e norma da coluna ||∂<σ>/∂a_n||:")
    for n in [1, 3, 5, 8, 10, 15, 20, 25, 30]:
        print(f"   n={n:2d}  ω={n*np.pi/T:5.2f}  ||coluna||={np.linalg.norm(J[:, n-1]):8.3f}")
    print(f"  número de condição (30 modos) = {sv[0]/sv[-1]:.1e}")

    plt.figure(figsize=(5, 3.4))
    plt.semilogy(np.arange(1, nmax + 1) * np.pi / T,
                 [np.linalg.norm(J[:, n]) for n in range(nmax)], "o-")
    plt.xlabel(r"frequência $\omega$ do modo de campo")
    plt.ylabel(r"$\|\partial \langle\sigma\rangle / \partial a_\omega\|$")
    plt.title("Sensibilidade dos dados por frequência")
    plt.tight_layout()
    save_figure(plt.gcf(), "sensibilidade_frequencia")
    plt.close()


if __name__ == "__main__":
    observabilidade()
    capacidade_bases()
    condicionamento()
