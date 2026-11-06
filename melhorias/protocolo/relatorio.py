"""
Agrega os resultados de um perfil em tabelas e figuras.

  python relatorio.py --perfil padrao

Gera em resultados/<perfil>/:
  resumo.csv        (uma linha por método x cenário: média, desvio, mediana, mín, máx)
  RESULTADOS.md     (tabelas em markdown + figuras)
  fig_erro_*.png    (erro do campo por método, cada ponto é uma seed)
  fig_campos_*.png  (campo recuperado, seed 0, melhor método de cada família)
"""
import os
import sys
import csv
import glob
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

CORES = {"classico": "#6b6b6b", "original": "#d62728", "lbfgs": "#ff7f0e", "bfgs": "#9467bd",
         "ssbfgs": "#8c564b", "lm": "#1f77b4", "lm_causal": "#17becf"}


def familia(r):
    return "classico" if r["tipo"] == "classico" else r["receita"]


def carregar(pasta):
    res = []
    for f in glob.glob(os.path.join(pasta, "jobs", "*.json")):
        with open(f) as fh:
            res.append(json.load(fh))
    return res


def resumir(res):
    grupos = {}
    for r in res:
        grupos.setdefault((r["caso"], r["ruido"], r["metodo"]), []).append(r)
    linhas = []
    for (caso, ruido, metodo), rs in sorted(grupos.items()):
        ok = [r for r in rs if r["status"] == "ok"]
        e = np.array([r["erro_campo"] for r in ok]) if ok else np.array([np.nan])
        tr = np.array([r["rmse_traj"] for r in ok]) if ok else np.array([np.nan])
        tm = np.array([r["tempo_s"] for r in rs])
        linhas.append(dict(caso=caso, ruido=ruido, metodo=metodo, familia=familia(rs[0]),
                           n_ok=len(ok), n=len(rs),
                           erro_media=np.mean(e), erro_std=np.std(e), erro_mediana=np.median(e),
                           erro_min=np.min(e), erro_max=np.max(e),
                           traj_media=np.mean(tr), traj_mediana=np.median(tr),
                           tempo_medio=np.mean(tm)))
    return linhas


def fmt_pct(x):
    if not np.isfinite(x):
        return "–"
    p = 100 * x
    if p >= 100:
        return f"{p:.0f} %"
    if p >= 10:
        return f"{p:.1f} %"
    if p >= 0.01:
        return f"{p:.2f} %"
    return f"{p:.1e} %"


def tabela_md(linhas, caso, ruido):
    ls = [l for l in linhas if l["caso"] == caso and l["ruido"] == ruido]
    ls.sort(key=lambda l: l["erro_mediana"] if np.isfinite(l["erro_mediana"]) else 1e99)
    out = [f"### Caso {caso}, ruído σ = {ruido:g}\n",
           "| # | método | erro do campo (mediana) | média ± desvio | pior seed | RMSE trajetória (mediana) | tempo médio | seeds ok |",
           "|---|---|---|---|---|---|---|---|"]
    for i, l in enumerate(ls, 1):
        out.append(f"| {i} | `{l['metodo']}` | **{fmt_pct(l['erro_mediana'])}** | "
                   f"{fmt_pct(l['erro_media'])} ± {fmt_pct(l['erro_std'])} | {fmt_pct(l['erro_max'])} | "
                   f"{l['traj_mediana']:.1e} | {l['tempo_medio']:.0f} s | {l['n_ok']}/{l['n']} |")
    return "\n".join(out) + "\n"


def tabela_cruzada(linhas, cenarios):
    metodos = sorted({l["metodo"] for l in linhas},
                     key=lambda m: np.nanmean([l["erro_mediana"] for l in linhas if l["metodo"] == m]))
    cab = "| método | " + " | ".join(f"caso {c}, σ={r:g}" for c, r in cenarios) + " |"
    out = [cab, "|---|" + "---|" * len(cenarios)]
    for m in metodos:
        cel = []
        for c, r in cenarios:
            l = [x for x in linhas if x["metodo"] == m and x["caso"] == c and x["ruido"] == r]
            cel.append(fmt_pct(l[0]["erro_mediana"]) if l else "–")
        out.append(f"| `{m}` | " + " | ".join(cel) + " |")
    return "\n".join(out) + "\n"


def fig_erros(res, linhas, caso, ruido, arq):
    ls = [l for l in linhas if l["caso"] == caso and l["ruido"] == ruido]
    ls.sort(key=lambda l: l["erro_mediana"] if np.isfinite(l["erro_mediana"]) else 1e99, reverse=True)
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(ls) + 1.2))
    for i, l in enumerate(ls):
        rs = [r for r in res if r["metodo"] == l["metodo"] and r["caso"] == caso
              and r["ruido"] == ruido and r["status"] == "ok"]
        e = np.array([r["erro_campo"] for r in rs])
        cor = CORES.get(l["familia"], "k")
        ax.scatter(np.clip(e, 1e-5, None), np.full(len(e), i), s=14, color=cor, alpha=0.6, zorder=3)
        ax.plot([l["erro_mediana"]], [i], "|", color=cor, ms=14, mew=2.5, zorder=4)
    ax.set_yticks(range(len(ls)))
    ax.set_yticklabels([l["metodo"] for l in ls], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("erro L2 relativo do campo (pontos = seeds, traço = mediana)")
    ax.set_title(f"caso {caso}, ruído σ = {ruido:g}, estado inicial |0⟩")
    ax.grid(axis="x", which="both", alpha=0.3)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=k) for k, c in CORES.items()
               if any(l["familia"] == k for l in ls)]
    ax.legend(handles=handles, fontsize=7, loc="lower right")
    plt.tight_layout()
    plt.savefig(arq, dpi=120)
    plt.close()


def fig_campos(res, linhas, caso, ruido, arq):
    import executar
    t, data, clean, ffn, T = executar.gerar_dados(caso, ruido, 0)
    ls = [l for l in linhas if l["caso"] == caso and l["ruido"] == ruido and np.isfinite(l["erro_mediana"])]
    melhores = {}
    for l in ls:
        f = l["familia"]
        if f not in melhores or l["erro_mediana"] < melhores[f]["erro_mediana"]:
            melhores[f] = l
    fams = [f for f in CORES if f in melhores]
    fig, axes = plt.subplots(len(fams), 1, figsize=(7, 1.8 * len(fams) + 0.4), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, f in zip(axes, fams):
        m = melhores[f]["metodo"]
        r = [x for x in res if x["metodo"] == m and x["caso"] == caso and x["ruido"] == ruido
             and x["seed"] == 0 and x["status"] == "ok"]
        ax.plot(t, ffn(t), "k-", lw=2, label="verdadeiro")
        if r:
            ax.plot(t, r[0]["f_pred"], "--", color=CORES[f], lw=1.6, label=m)
            ax.set_title(f"{m} (seed 0): erro = {fmt_pct(r[0]['erro_campo'])}", fontsize=9)
        lo, hi = ffn(t).min(), ffn(t).max()
        ax.set_ylim(lo - 0.4 * (hi - lo), hi + 0.4 * (hi - lo))
    axes[-1].set_xlabel("t")
    axes[0].legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    plt.savefig(arq, dpi=120)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perfil", default="padrao")
    ap.add_argument("--saida", default=os.path.join(HERE, "resultados"))
    a = ap.parse_args()
    pasta = os.path.join(a.saida, a.perfil)
    res = carregar(pasta)
    if not res:
        print("nenhum resultado em", pasta)
        return
    linhas = resumir(res)
    with open(os.path.join(pasta, "resumo.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)
    cenarios = sorted({(l["caso"], l["ruido"]) for l in linhas})
    md = [f"# Resultados do perfil `{a.perfil}`\n",
          f"{len(res)} jobs, {sum(r['status'] == 'ok' for r in res)} ok. Estado inicial |0⟩, "
          "N = 100, campo verdadeiro fora da loss. Erro = ||f_pred − f_true|| / ||f_true||.\n",
          "## Mediana do erro do campo por cenário\n", tabela_cruzada(linhas, cenarios), ""]
    for c, r in cenarios:
        nome = f"c{c}_r{r:g}"
        fig_erros(res, linhas, c, r, os.path.join(pasta, f"fig_erro_{nome}.png"))
        fig_campos(res, linhas, c, r, os.path.join(pasta, f"fig_campos_{nome}.png"))
        md += [tabela_md(linhas, c, r), f"![erros](fig_erro_{nome}.png)\n",
               f"![campos](fig_campos_{nome}.png)\n"]
    erros = [r for r in res if r["status"] != "ok"]
    if erros:
        md.append("## Jobs com erro\n")
        md += [f"- `{r['metodo']}` caso {r['caso']} σ={r['ruido']:g} seed {r['seed']}: {r.get('msg')}"
               for r in erros]
    with open(os.path.join(pasta, "RESULTADOS.md"), "w") as fh:
        fh.write("\n".join(md))
    print(f"ok: {os.path.join(pasta, 'RESULTADOS.md')}")


if __name__ == "__main__":
    main()
