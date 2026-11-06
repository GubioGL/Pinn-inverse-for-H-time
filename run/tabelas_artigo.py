"""LaTeX tables of the paper and of the supplemental material, written from the data.

    python run/tabelas_artigo.py

Writes paper/tabelas/*.tex (included with \\input in main.tex and supplementary.tex):
  tab_main.tex          median field error per method and scenario (main text)
  tab_S_<scenario>.tex  both metrics, median [min, max] over seeds, run time (supplement)
  tab_S_optim.tex       optimizer comparison, field 3 (supplement)
  tab_S_initial.tex     initial state |+> vs |0> with a fixed measurement budget (supplement)
  tab_S_hyper.tex       training recipes and hyper-parameters, read from src/protocolo.py
Results not yet available in the current protocol ("padrao") are taken from the earlier
protocol "v1_padrao" and marked with a dagger.
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
import math
import numpy as np

import protocolo as C
from paths import ROOT, DATA

OUT = ROOT / "paper" / "tabelas"

CAMPO = {"nn": "NN field", "fourier": "Fourier ($M{=}10$)", "cosseno": "cosine ($M{=}40$)",
         "bspline": "B-spline ($n{=}80$)", "mixfunn": "MixFunn"}
CLASSICO = {"direta": "direct inversion", "lsq_fourier10": "LSQ, Fourier ($M{=}10$)",
            "lsq_cosseno40": "LSQ, cosine ($M{=}40$)", "lsq_bspline40": "LSQ, B-spline ($n{=}40$)",
            "lsq_bspline80": "LSQ, B-spline ($n{=}80$)", "shooting_bs40": "shooting, B-spline ($n{=}40$)",
            "shooting_bs80": "shooting, B-spline ($n{=}80$)"}
RECEITA = {"original": "Adam only", "lbfgs": "Adam + L-BFGS", "bfgs": "Adam + BFGS",
           "ssbfgs": "Adam + self-scaled BFGS", "lm": "Adam + LM", "lm_causal": "Adam (causal) + LM"}


def carregar(perfil):
    out = []
    for f in glob.glob(str(DATA / "resultados" / perfil / "jobs" / "*.json")):
        with open(f) as fh:
            r = json.load(fh)
        if r.get("status") == "ok":
            out.append(r)
    return out


def sel(res, m, c, s):
    return [r for r in res if r["metodo"] == m and r["caso"] == c and r["ruido"] == s]


def num(x, sig=2):
    """1.23e-04 -> $1.2\\times10^{-4}$ ; 0.0123 -> $1.2\\times10^{-2}$."""
    if x is None or not math.isfinite(x):
        return "--"
    if x == 0:
        return "$0$"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    if round(m, sig - 1) >= 10:
        m /= 10
        e += 1
    if -1 <= e <= 1:
        return f"${x:.{max(0, sig - 1 - e)}f}$"
    return f"${m:.{sig - 1}f}\\times10^{{{e}}}$"


def nome(m):
    if m in CLASSICO:
        return CLASSICO[m]
    _, c, r = m.split("_", 2)
    return f"{CAMPO[c]}, {RECEITA[r]}"


def escrever(nomearq, linhas):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / nomearq).write_text("\n".join(linhas) + "\n")
    print("ok:", OUT / nomearq)


# ---------------------------------------------------------------------------------------
def tab_main(v2, v1):
    cen = [(1, 0.0), (2, 0.0), (3, 0.0), (3, 0.01)]
    linhas_cl = ["direta", "lsq_fourier10", "lsq_cosseno40", "lsq_bspline80", "shooting_bs80"]
    linhas_pinn = [("nn", "original"), ("nn", "lbfgs"), ("nn", "lm"), ("fourier", "lm"), ("cosseno", "lm"),
                   ("bspline", "lm"), ("mixfunn", "original"), ("mixfunn", "lm")]
    L = [r"\begin{tabular}{llcccc}", r"\toprule",
         r" & method & field 1 & field 2 & field 3 & field 3, $\sigma=0.01$ \\", r"\midrule"]

    def celula(m, c, s):
        rs = sel(v2, m, c, s)
        dag = ""
        if not rs:
            rs = sel(v1, m, c, s)
            dag = r"$^\dagger$"
        if not rs:
            return "--"
        return num(np.median([r["erro_campo"] for r in rs])) + dag

    for i, m in enumerate(linhas_cl):
        pre = r"\multirow{5}{*}{\rotatebox{90}{classical}}" if i == 0 else ""
        L.append(f"{pre} & {CLASSICO[m]} & " + " & ".join(celula(m, c, s) for c, s in cen) + r" \\")
    L.append(r"\midrule")
    for i, (c_, r_) in enumerate(linhas_pinn):
        m = f"pinn_{c_}_{r_}"
        pre = r"\multirow{8}{*}{\rotatebox{90}{PINN}}" if i == 0 else ""
        L.append(f"{pre} & {CAMPO[c_]}, {RECEITA[r_]} & " + " & ".join(celula(m, c, s) for c, s in cen) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    escrever("tab_main.tex", L)


def tab_detalhe(v2, caso, ruido):
    metodos = sorted({r["metodo"] for r in v2 if r["caso"] == caso and r["ruido"] == ruido},
                     key=lambda m: np.median([r["erro_campo"] for r in sel(v2, m, caso, ruido)]))
    L = [r"\begin{tabular}{lcccc}", r"\toprule",
         r"method & $\epsilon_f$ median & $\epsilon_f$ [min, max] & $\epsilon_r$ median & time (s) \\",
         r"\midrule"]
    for m in metodos:
        rs = sel(v2, m, caso, ruido)
        e = [r["erro_campo"] for r in rs]
        d = [r["rmse_traj"] for r in rs]
        tm = np.mean([r["tempo_s"] for r in rs])
        L.append(f"{nome(m)} & {num(np.median(e))} & [{num(min(e))}, {num(max(e))}] & {num(np.median(d))} & "
                 f"{tm:.0f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    escrever(f"tab_S_field{caso}_noise{str(ruido).replace('.', 'p')}.tex", L)


def tab_otim():
    otim = carregar("otimizadores")
    fonte = "current protocol"
    if not otim:
        otim = carregar("v1_ablacao") + [r for r in carregar("v1_padrao") if r["metodo"].endswith("_original")]
        fonte = "v1"
    L = [r"\begin{tabular}{lcccc}", r"\toprule",
         r"second stage & \multicolumn{2}{c}{NN field} & \multicolumn{2}{c}{B-spline field} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r" & $\epsilon_f$ median & [min, max] & $\epsilon_f$ median & [min, max] \\", r"\midrule"]
    for rec in ("original", "lbfgs", "ssbfgs", "bfgs", "lm_causal", "lm"):
        cel = []
        for campo in ("nn", "bspline"):
            e = [r["erro_campo"] for r in otim if r["metodo"] == f"pinn_{campo}_{rec}" and r["caso"] == 3
                 and r["ruido"] == 0.0]
            cel += [num(np.median(e)), f"[{num(min(e))}, {num(max(e))}]"] if e else ["--", "--"]
        rot = {"original": "none (Adam only)", "lbfgs": "L-BFGS", "ssbfgs": "self-scaled BFGS",
               "bfgs": "dense BFGS", "lm_causal": "LM, causal weights in Adam", "lm": "LM"}[rec]
        L.append(f"{rot} & " + " & ".join(cel) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    escrever("tab_S_optim.tex", L)
    return fonte


def tab_estado_inicial():
    arq = DATA / "parte1" / "resultados_desenho_experimental.csv"
    rows = list(csv.DictReader(open(arq)))
    L = [r"\begin{tabular}{lccccc}", r"\toprule",
         r"design & $\sigma$ per point & \multicolumn{2}{c}{$\sigma_{\rm ref}=0.01$} & \multicolumn{2}{c}{$\sigma_{\rm ref}=0.03$} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
         r" & & LSQ & shooting & LSQ & shooting \\", r"\midrule"]
    nomes = {"1 estado |+>, N=100": r"$|+\rangle$, $N=100$", "1 estado |0>, N=100": r"$|0\rangle$, $N=100$",
             "1 estado |+>, N=300": r"$|+\rangle$, $N=300$",
             "3 estados |+>,|0>,|+i>, N=100 cada": r"$|+\rangle,|0\rangle,|{+i}\rangle$, $N=100$ each"}
    fator = {"1 estado |+>, N=100": "$\\sigma$", "1 estado |0>, N=100": "$\\sigma$",
             "1 estado |+>, N=300": "$\\sqrt{3}\\,\\sigma$", "3 estados |+>,|0>,|+i>, N=100 cada": "$\\sqrt{3}\\,\\sigma$"}
    for d in nomes:
        a = [r for r in rows if r["desenho"] == d and float(r["sigma_ref"]) == 0.01][0]
        b = [r for r in rows if r["desenho"] == d and float(r["sigma_ref"]) == 0.03][0]
        f = lambda r, k: f"${float(r[k + '_media']):.3f}\\pm{float(r[k + '_std']):.3f}$"
        L.append(f"{nomes[d]} & {fator[d]} & {f(a, 'lsq')} & {f(a, 'shooting')} & {f(b, 'lsq')} & {f(b, 'shooting')} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    escrever("tab_S_initial.tex", L)


def tab_hiper():
    A0, A1 = C.ARQ_ORIGINAL, C.ARQ_MELHORADA
    L = [r"\begin{tabular}{p{0.2\linewidth}p{0.34\linewidth}p{0.38\linewidth}}", r"\toprule",
         r"item & Adam-only recipe (as in common practice) & improved recipes \\", r"\midrule",
         f"state network & ${A0['depth']}$ hidden layers $\\times$ ${A0['width']}$, $\\sin$ & "
         f"${A1['depth']}$ hidden layers $\\times$ ${A1['width']}$, $\\sin$ \\\\",
         r"network input & $t$ & $s=2t/T-1$ \\",
         r"initial condition & soft (data term) & hard, $\boldsymbol r(t)=\boldsymbol r_0+(t/T)\,\mathrm{NN}$ \\",
         r"field edges & free & hard when known ($f(0)=f(T)=0$ for field 3) \\",
         f"collocation points & the $N={C.N_PONTOS}$ data points & $N$ data points $+$ ${A1['n_rand']}$ "
         r"uniform random points, redrawn every Adam epoch \\",
         r"field coefficients at start & zero & linear least squares on the data \\",
         r"floating point & single (float32) & double (float64) \\"]
    adam0 = C.RECEITAS["original"]["estagios"][0]
    adam1 = C.ADAM
    L.append(f"Adam & {adam0['epochs']} epochs, learning rate {num(adam0['lr'], 1)}, constant & "
             f"{adam1['epochs']} epochs, learning rate {num(adam1['lr'], 1)} $\\to$ {num(adam1['lr'] * 1e-2, 1)} "
             r"(cosine schedule) \\")
    lb = C.RECEITAS["lbfgs"]["estagios"][1]
    bf = C.RECEITAS["bfgs"]["estagios"][1]
    lm = C.LM
    L.append(f"second stage & none & L-BFGS {lb['iters']} it.\\ (history 50, strong Wolfe); "
             f"BFGS {bf['iters']} it.; LM {lm['iters']} it. \\\\")
    lams = "; ".join(f"$\\sigma={k:g}$: " + (num(v, 1) if v else "$0$") for k, v in C.LAM_F.items())
    L.append(f"smoothness weight $\\lambda_f$ & \\multicolumn{{2}}{{l}}{{{lams}}} \\\\")
    L.append(f"time limit per run & \\multicolumn{{2}}{{l}}{{{C.PERFIS['padrao']['t_max_job']} s "
             r"(single CPU thread)} \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    escrever("tab_S_hyper.tex", L)


if __name__ == "__main__":
    v2 = carregar("padrao")
    v1 = carregar("v1_padrao")
    tab_main(v2, v1)
    for c, s in sorted({(r["caso"], r["ruido"]) for r in v2}):
        tab_detalhe(v2, c, s)
    print("optimizer table source:", tab_otim())
    tab_estado_inicial()
    tab_hiper()
