"""Figures of the paper and of the supplemental material (vector PDF in paper/figures/).

    python run/fig_artigo.py

Main text
  fig1_problem.pdf        test fields, Bloch trajectories (|0>) and identifiability (|+> vs |0>)
  fig2_field1.pdf         results for field 1 (cos t)
  fig3_field2.pdf         results for field 2 (multi-harmonic, non-periodic)
  fig4_field3.pdf         results for field 3 (smooth pulse), without and with noise
  fig5_optimizers.pdf     effect of the second-order optimization stage
Supplemental material
  figS1_sensitivity.pdf   sensitivity of the data to each frequency of the field
  figS2_bases.pdf         best-approximation error of each field basis vs pulse frequency
  figS3_overview.pdf      median field error, all methods x scenarios
  figS4..S7_detail_*.pdf  both metrics, every seed, every method, for each scenario

Numbers come from data/resultados/<perfil>/jobs/*.json (profile "padrao"; methods that are not
there yet fall back to the earlier protocol "v1_padrao" and are marked with a dagger).
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
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

import estilo
import bloch as cm
import executar
from paths import DATA, save_figure

estilo.aplicar()
OI = estilo.OI
FAM = estilo.FAMILIA

CAMPO_NOME = {"nn": "NN field", "fourier": "Fourier", "cosseno": "cosine", "bspline": "B-spline",
              "mixfunn": "MixFunn"}
RECEITA_NOME = {"original": "Adam", "lbfgs": "Adam+L-BFGS", "bfgs": "Adam+BFGS", "ssbfgs": "Adam+SSBFGS",
                "lm": "Adam+LM", "lm_causal": "Adam (causal)+LM"}
MARCADOR = {"nn": "o", "fourier": "v", "cosseno": "^", "bspline": "s", "mixfunn": "D"}
CLASSICO_NOME = {"direta": "direct inversion", "lsq_fourier10": r"LSQ, Fourier ($M=10$)",
                 "lsq_cosseno40": r"LSQ, cosine ($M=40$)", "lsq_bspline40": r"LSQ, B-spline ($n=40$)",
                 "lsq_bspline80": r"LSQ, B-spline ($n=80$)", "shooting_bs40": r"shooting, B-spline ($n=40$)",
                 "shooting_bs80": r"shooting, B-spline ($n=80$)"}
CLASSICO_MARC = {"direta": "x", "lsq_fourier10": "1", "lsq_cosseno40": "2", "lsq_bspline40": "+",
                 "lsq_bspline80": "P", "shooting_bs40": "*", "shooting_bs80": "X"}


def nome(m):
    if m in CLASSICO_NOME:
        return CLASSICO_NOME[m]
    _, campo, rec = m.split("_", 2)
    return f"PINN, {CAMPO_NOME[campo]}, {RECEITA_NOME.get(rec, rec)}"


def familia(m):
    return "classico" if not m.startswith("pinn_") else m.split("_", 2)[2]


def marcador(m):
    return CLASSICO_MARC.get(m) or MARCADOR[m.split("_")[1]]


def carregar(perfil):
    out = []
    for f in glob.glob(str(DATA / "resultados" / perfil / "jobs" / "*.json")):
        with open(f) as fh:
            r = json.load(fh)
        if r.get("status") == "ok":
            out.append(r)
    return out


def selecao(res, metodo, caso, ruido):
    return [r for r in res if r["metodo"] == metodo and r["caso"] == caso and r["ruido"] == ruido]


def obter(metodo, caso, ruido, v2, v1):
    """Runs of a method in the current protocol, or in protocol v1 as a fallback (dagger)."""
    rs = selecao(v2, metodo, caso, ruido)
    if rs:
        return rs, False
    return selecao(v1, metodo, caso, ruido), True


def limpar(ax):
    ax.tick_params(which="both", top=False, right=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# ---------------------------------------------------------------------------------------
def fig1_problema():
    fig, axes = plt.subplots(3, 3, figsize=(estilo.DUPLA, 4.6), sharex="col",
                             gridspec_kw=dict(hspace=0.12, wspace=0.28))
    letras = iter("abcdefghi")
    for j, caso in enumerate((1, 2, 3)):
        f, T = cm.make_case(caso)
        t = np.linspace(0, T, 600)
        ax = axes[0, j]
        ax.plot(t / T, f(t), color=OI["preto"], lw=1.2)
        ax.axhline(0, color=OI["cinza"], lw=0.4)
        ax.set_title(["field 1: $\\cos t$", "field 2: three harmonics", "field 3: enveloped pulse"][j])
        if j == 0:
            ax.set_ylabel("$f(t)$")
        r = cm.simulate(f, t, cm.R0_ZERO)
        ax = axes[1, j]
        for k, (lab, cor, ls) in enumerate([(r"$\langle\sigma_x\rangle$", OI["azul"], "-"),
                                            (r"$\langle\sigma_y\rangle$", OI["vermelho"], "--"),
                                            (r"$\langle\sigma_z\rangle$", OI["verde"], "-.")]):
            ax.plot(t / T, r[:, k], color=cor, ls=ls, label=lab)
        ax.set_ylim(-1.1, 1.1)
        if j == 0:
            ax.set_ylabel(r"$|0\rangle$: $\langle\sigma_\alpha\rangle(t)$")
            ax.legend(loc="lower left", ncol=3, columnspacing=0.8, handlelength=1.4,
                      frameon=True, framealpha=0.9, edgecolor="none", borderpad=0.2)
        ax = axes[2, j]
        for lab, r0, cor, ls in [(r"$|+\rangle$", cm.R0_PLUS, OI["vermelho"], "-"),
                                 (r"$|0\rangle$", cm.R0_ZERO, OI["azul"], "-"),
                                 (r"$(1,1,1)/\sqrt{3}$", cm.R0_GENERIC, OI["verde"], ":")]:
            rr = cm.simulate(f, t, r0)
            ax.plot(t / T, rr[:, 1] ** 2 + rr[:, 2] ** 2, color=cor, ls=ls, label=lab)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlabel("$t/T$")
        if j == 0:
            ax.set_ylabel(r"$Y^2+Z^2$")
            ax.legend(loc="lower left", fontsize=6, frameon=True, framealpha=0.9, edgecolor="none",
                      borderpad=0.2)
    for ax in axes.flat:
        ax.set_xlim(0, 1)
        limpar(ax)
        estilo.rotulo_painel(ax, next(letras), x=-0.2, y=0.98)
    return save_figure(fig, "fig1_problem")


# ---------------------------------------------------------------------------------------
CURVAS = [("pinn_nn_original", "PINN, Adam only", FAM["original"], ":"),
          ("pinn_nn_lm", "PINN (NN field), Adam+LM", FAM["lm"], "--"),
          ("pinn_bspline_lm", "PINN (B-spline), Adam+LM", OI["verde"], "-."),
          ("shooting_bs80", "shooting (B-spline)", OI["preto"], (0, (1, 1)))]


def _linha_resultados(axes, caso, ruido, v2, v1, metodos, legenda=True):
    t, data, clean, ffn, T = executar.gerar_dados(caso, ruido, 0)
    ftrue = ffn(t)
    ax0, ax1, ax2 = axes
    ax0.plot(t, ftrue, color="0.75", lw=3.2, label="true field", zorder=1)
    dag_any = False
    for m, lab, cor, ls in CURVAS:
        rs, dag = obter(m, caso, ruido, v2, v1)
        r0 = [r for r in rs if r["seed"] == 0]
        if not r0:
            continue
        dag_any |= dag
        fp = np.array(r0[0]["f_pred"])
        lab = lab + (r"$^\dagger$" if dag else "")
        ax0.plot(t, fp, color=cor, ls=ls, lw=1.1, label=lab, zorder=2)
        ax1.semilogy(t, np.abs(fp - ftrue) + 1e-12, color=cor, ls=ls, lw=0.9)
    ax0.set_xlabel("$t$")
    ax0.set_ylabel("$f(t)$")
    ax1.set_xlabel("$t$")
    ax1.set_ylabel(r"$|f_{\rm pred}-f|$")
    ax1.set_ylim(1e-7, 3)
    # both metrics, median over seeds, every method of the profile
    for m in metodos:
        rs = selecao(v2, m, caso, ruido)
        if not rs:
            continue
        e = np.median([r["erro_campo"] for r in rs])
        d = np.median([r["rmse_traj"] for r in rs])
        ax2.scatter(e, d, marker=marcador(m), s=22 if marcador(m) not in "x1+2*" else 30,
                    color=FAM[familia(m)], linewidths=0.9,
                    facecolors="none" if marcador(m) in "osv^D" and familia(m) == "original" else None)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.xaxis.set_minor_formatter(NullFormatter())
    ax2.yaxis.set_minor_formatter(NullFormatter())
    ax2.set_xlabel(r"median field error $\epsilon_f$")
    ax2.set_ylabel(r"median trajectory error $\epsilon_r$")
    for ax in axes:
        limpar(ax)
    return dag_any


def _legenda_metodos(fig, metodos, y):
    hs = []
    fams = []
    for m in metodos:
        f = familia(m)
        if f not in fams:
            fams.append(f)
    for f in fams:
        rot = "classical" if f == "classico" else RECEITA_NOME.get(f, f)
        hs.append(Line2D([], [], ls="", marker="s", color=FAM[f], label=rot, ms=4))
    for c, mk in MARCADOR.items():
        if any(m.startswith(f"pinn_{c}_") for m in metodos):
            hs.append(Line2D([], [], ls="", marker=mk, color="0.3", label=CAMPO_NOME[c], ms=4,
                             markerfacecolor="none"))
    for m in CLASSICO_NOME:
        if m in metodos:
            hs.append(Line2D([], [], ls="", marker=CLASSICO_MARC[m], color=FAM["classico"],
                             label=CLASSICO_NOME[m], ms=4))
    fig.legend(handles=hs, loc="lower center", ncol=6, bbox_to_anchor=(0.5, y), fontsize=5.8,
               columnspacing=0.9, handletextpad=0.2)


def fig_caso(caso, v2, v1, nomearq):
    ruidos = sorted({r["ruido"] for r in v2 if r["caso"] == caso}) or [0.0]
    metodos = sorted({r["metodo"] for r in v2 if r["caso"] == caso})
    nl = len(ruidos)
    fig, axes = plt.subplots(nl, 3, figsize=(estilo.DUPLA, 2.15 * nl + 0.55),
                             gridspec_kw=dict(wspace=0.33, hspace=0.5, width_ratios=[1.25, 1.0, 1.0]))
    axes = np.atleast_2d(axes)
    letras = iter("abcdefghi")
    for i, ruido in enumerate(ruidos):
        _linha_resultados(axes[i], caso, ruido, v2, v1, metodos, legenda=(i == 0))
        for ax in axes[i]:
            estilo.rotulo_painel(ax, next(letras), x=-0.22, y=1.0)
        if nl > 1:
            axes[i, 0].set_title(rf"$\sigma={ruido:g}$", loc="left", x=0.08)
    _legenda_metodos(fig, metodos, y=-0.02 - 0.06 / nl)
    hs, ls = axes[0, 0].get_legend_handles_labels()
    fig.legend(hs, ls, loc="upper center", ncol=len(hs), bbox_to_anchor=(0.5, 1.0 + 0.1 / nl),
               fontsize=6.3, handlelength=2.4, columnspacing=1.2)
    plt.subplots_adjust(bottom=0.2 / nl + 0.08)
    return save_figure(fig, nomearq)


# ---------------------------------------------------------------------------------------
def fig5_otimizadores():
    """Effect of the second-order stage (field 3, no noise). Uses the profile 'otimizadores'
    when it exists, otherwise protocol v1 (v1_ablacao + Adam-only runs of v1_padrao)."""
    otim = carregar("otimizadores")
    fonte = "otimizadores"
    if not otim:
        otim = carregar("v1_ablacao") + [r for r in carregar("v1_padrao") if r["metodo"].endswith("_original")]
        fonte = "v1"
    ordem = ["original", "lbfgs", "ssbfgs", "bfgs", "lm_causal", "lm"]
    fig, ax = plt.subplots(figsize=(estilo.COL, 2.25))
    for k, (campo, dx) in enumerate((("nn", -0.13), ("bspline", 0.13))):
        for i, rec in enumerate(ordem):
            e = [r["erro_campo"] for r in otim if r["metodo"] == f"pinn_{campo}_{rec}"
                 and r["caso"] == 3 and r["ruido"] == 0.0]
            if not e:
                continue
            x = i + dx
            ax.scatter(np.full(len(e), x), e, s=9, marker=MARCADOR[campo], color=FAM[rec],
                       alpha=0.75, linewidths=0.5, facecolors="none" if campo == "bspline" else None)
            ax.plot([x - 0.1, x + 0.1], [np.median(e)] * 2, color="k", lw=1.0)
    ax.set_yscale("log")
    ax.set_xticks(range(len(ordem)))
    ax.set_xticklabels(["Adam\nonly", "L-BFGS", "BFGS\n(self-sc.)", "BFGS", "LM\n(causal)", "LM"])
    ax.set_ylabel(r"field error $\epsilon_f$")
    hs = [Line2D([], [], ls="", marker="o", color="0.3", label="NN field", ms=4),
          Line2D([], [], ls="", marker="s", color="0.3", markerfacecolor="none", label="B-spline field", ms=4),
          Line2D([], [], ls="-", color="k", label="median of 5 seeds")]
    ax.legend(handles=hs, loc="upper right", fontsize=6)
    limpar(ax)
    print("fig5 data source:", fonte)
    return save_figure(fig, "fig5_optimizers"), fonte


# ---------------------------------------------------------------------------------------
def figS1_sensibilidade():
    f, T = cm.make_case(3)
    t = np.linspace(0, T, 200)
    base = cm.simulate(f, t, cm.R0_ZERO)
    nmax, eps = 40, 1e-4
    w, col = [], []
    for n in range(1, nmax + 1):
        g = lambda s, n=n: f(s) + eps * np.sin(n * np.pi * s / T) * np.sin(np.pi * s / T) ** 2
        J = (cm.simulate(g, t, cm.R0_ZERO) - base) / eps
        w.append(n * np.pi / T)
        col.append(np.sqrt(np.mean(J ** 2)))
    w, col = np.array(w), np.array(col)
    fig, ax = plt.subplots(figsize=(estilo.COL, 2.1))
    ax.loglog(w, col, "o-", color=OI["azul"], ms=2.5)
    c = col[5] * w[5]
    ax.loglog(w, c / w, "--", color=OI["cinza"], lw=0.8, label=r"$\propto 1/\omega$")
    ax.set_xlabel(r"frequency $\omega$ of the field perturbation")
    ax.set_ylabel(r"RMS $\partial\langle\boldsymbol{\sigma}\rangle/\partial a_\omega$")
    ax.legend()
    limpar(ax)
    return save_figure(fig, "figS1_sensitivity")


def figS2_bases():
    T = 2.52 * np.pi
    t = np.linspace(0, T, 3000)
    ws = np.linspace(0.5, 25, 60)
    confs = [("fourier", 10, None, r"Fourier, $M=10$", OI["laranja"], "-"),
             ("fourier", 20, None, r"Fourier, $M=20$", OI["laranja"], ":"),
             ("cosine", 40, None, r"cosine, $M=40$", OI["rosa"], "-"),
             ("bspline", 40, 3, r"cubic B-spline, $n=40$", OI["verde"], ":"),
             ("bspline", 80, 3, r"cubic B-spline, $n=80$", OI["verde"], "-")]
    fig, ax = plt.subplots(figsize=(estilo.COL, 2.2))
    for kind, size, k, lab, cor, ls in confs:
        B = cm.basis_matrix(kind, t, T, size, k or 3)
        P = np.linalg.pinv(B)
        e = []
        for w in ws:
            y = np.sin(w * t) * np.sin(np.pi * t / T) ** 2
            e.append(cm.rel_l2(B @ (P @ y), y))
        ax.semilogy(ws, np.clip(e, 1e-8, None), color=cor, ls=ls, label=lab)
    ax.set_xlabel(r"pulse frequency $\omega$")
    ax.set_ylabel("best-approximation error")
    ax.set_ylim(1e-6, 2)
    ax.legend(fontsize=5.8, loc="lower right")
    limpar(ax)
    return save_figure(fig, "figS2_bases")


def figS3_overview(v2):
    cen = sorted({(r["caso"], r["ruido"]) for r in v2})
    metodos = [m for m in CLASSICO_NOME if any(r["metodo"] == m for r in v2)]
    for rec in ("original", "lbfgs", "bfgs", "lm"):
        for c in MARCADOR:
            m = f"pinn_{c}_{rec}"
            if any(r["metodo"] == m for r in v2):
                metodos.append(m)
    M = np.array([[np.median([r["erro_campo"] for r in selecao(v2, m, c, s)]) if selecao(v2, m, c, s) else np.nan
                   for c, s in cen] for m in metodos])
    fig, ax = plt.subplots(figsize=(estilo.COL + 0.6, 0.2 * len(metodos) + 0.9))
    im = ax.imshow(np.clip(M, 1e-6, 10), norm=LogNorm(1e-5, 1), cmap="viridis_r", aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isfinite(v):
                txt = f"{v:.1e}".replace("e-0", "e-").replace("e+0", "e")
                ax.text(j, i, txt, ha="center", va="center", fontsize=4.8,
                        color="white" if v < 3e-3 else "black")
    ax.set_xticks(range(len(cen)))
    ax.set_xticklabels([f"field {c}\n$\\sigma={s:g}$" for c, s in cen], fontsize=6)
    ax.set_yticks(range(len(metodos)))
    ax.set_yticklabels([nome(m) for m in metodos], fontsize=5.5)
    ax.set_ylim(len(metodos) - 0.5, -0.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.03)
    cb.set_label(r"median $\epsilon_f$", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    ax.tick_params(which="both", top=False, right=False, length=0)
    return save_figure(fig, "figS3_overview")


def figS_detalhe(v2, caso, ruido, nomearq):
    metodos = sorted({r["metodo"] for r in v2 if r["caso"] == caso and r["ruido"] == ruido},
                     key=lambda m: np.median([r["erro_campo"] for r in selecao(v2, m, caso, ruido)]))
    fig, axes = plt.subplots(1, 2, figsize=(estilo.DUPLA, 0.17 * len(metodos) + 0.8), sharey=True,
                             gridspec_kw=dict(wspace=0.05))
    for ax, chave, rot in ((axes[0], "erro_campo", r"field error $\epsilon_f$"),
                           (axes[1], "rmse_traj", r"trajectory error $\epsilon_r$")):
        for i, m in enumerate(metodos):
            v = [r[chave] for r in selecao(v2, m, caso, ruido)]
            ax.scatter(np.clip(v, 1e-9, None), np.full(len(v), i), s=8, marker=marcador(m),
                       color=FAM[familia(m)], linewidths=0.6)
            ax.plot([np.median(v)] * 2, [i - 0.35, i + 0.35], color="k", lw=0.9)
        ax.set_xscale("log")
        ax.set_xlabel(rot)
        ax.grid(axis="x", which="major", lw=0.3, alpha=0.5)
        limpar(ax)
    axes[0].set_yticks(range(len(metodos)))
    axes[0].set_yticklabels([nome(m) for m in metodos], fontsize=6)
    axes[0].set_ylim(len(metodos) - 0.5, -0.5)
    axes[0].set_title(rf"field {caso}, $\sigma={ruido:g}$", loc="left")
    return save_figure(fig, nomearq)


if __name__ == "__main__":
    v2 = carregar("padrao")
    v1 = carregar("v1_padrao")
    print("ok:", fig1_problema())
    for caso, nomearq in ((1, "fig2_field1"), (2, "fig3_field2"), (3, "fig4_field3")):
        print("ok:", fig_caso(caso, v2, v1, nomearq))
    print("ok:", fig5_otimizadores()[0])
    print("ok:", figS1_sensibilidade())
    print("ok:", figS2_bases())
    if v2:
        print("ok:", figS3_overview(v2))
        k = 4
        for caso, ruido in sorted({(r["caso"], r["ruido"]) for r in v2}):
            tag = f"{ruido:g}".replace(".", "p")
            print("ok:", figS_detalhe(v2, caso, ruido, f"figS{k}_detail_field{caso}_noise{tag}"))
            k += 1
