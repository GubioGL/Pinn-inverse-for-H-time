"""
Roda o ablation do PINN (todos os presets x várias seeds) em paralelo e salva
resultados_pinn/ablation.json + figura com os campos recuperados.

Uso:  python rodar_ablation.py [--seeds 3] [--procs 4]
"""
import os
import json
import argparse
import multiprocessing as mp
import numpy as np

import pinn_melhorado as pm

HERE = os.path.dirname(os.path.abspath(__file__))


def _job(args):
    preset, seed, over = args
    cfg = pm.make_cfg(preset, seed=seed, **over)
    if over.get("tag"):
        cfg["name"] = f"{preset}_{over['tag']}"
    cfg.pop("tag", None)
    return pm.run(cfg, verbose=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--presets", nargs="*", default=None, help="padrão: todos")
    ap.add_argument("--sem-extras", action="store_true", help="não roda r0=|0> e ruído")
    ap.add_argument("--out", default="ablation.json")
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, "resultados_pinn"), exist_ok=True)

    jobs = []
    for s in range(a.seeds):
        for p in (a.presets or list(pm.PRESETS)):
            jobs.append((p, s, {}))
        if not a.sem_extras:
            # efeito do estado inicial e do ruído no PINN completo
            jobs.append(("completo", s, {"r0": "zero", "tag": "r0zero"}))
            jobs.append(("completo", s, {"noise": 0.01, "tag": "ruido0.01"}))

    with mp.get_context("spawn").Pool(a.procs) as pool:
        res = []
        for r in pool.imap_unordered(_job, jobs):
            print(f"{r['name']:24s} seed={r['seed']}  erro campo={r['erro_campo']:.4f}  "
                  f"RMSE re-sim={r['rmse_resim']:.2e}  ({r['tempo_s']:.0f}s)", flush=True)
            res.append(r)

    with open(os.path.join(HERE, "resultados_pinn", a.out), "w") as fh:
        json.dump(res, fh, indent=1)

    names = sorted({r["name"] for r in res}, key=lambda n: [r["name"] for r in res].index(n))
    print("\nResumo (média ± desvio sobre seeds):")
    for n in names:
        e = np.array([r["erro_campo"] for r in res if r["name"] == n])
        rs = np.array([r["rmse_resim"] for r in res if r["name"] == n])
        print(f"  {n:24s} erro campo = {e.mean():.4f} ± {e.std():.4f}   "
              f"RMSE re-sim = {rs.mean():.2e}")
