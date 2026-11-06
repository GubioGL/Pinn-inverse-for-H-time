"""
Roda o protocolo em paralelo, de forma retomável.

Cada job (método, caso, ruído, seed) grava um JSON em  data/resultados/<perfil>/jobs/.
Se a execução for interrompida, rodar de novo pula os jobs já concluídos.

Exemplos
--------
  # teste rápido (~5-10 min em CPU)
  python run/rodar.py --perfil rapido

  # protocolo padrão usando todas as CPUs
  python run/rodar.py --perfil padrao --workers 8

  # GPU: vários processos compartilham a(s) GPU(s); os jobs PINN vão para cuda:0, cuda:1, ...
  python run/rodar.py --perfil completo --device cuda --workers 6

  # só alguns métodos (filtro por substring)
  python run/rodar.py --perfil padrao --metodos lm lsq

Depois:  python run/relatorio.py --perfil padrao
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
import sys
import json
import time
import argparse
import multiprocessing as mp

import protocolo as C

_DEVICE = "cpu"


def _init_worker(device, threads, n_gpu):
    global _DEVICE
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    import torch as tc
    tc.set_num_threads(threads)
    if device.startswith("cuda") and n_gpu > 0:
        idx = (mp.current_process()._identity[0] - 1) if mp.current_process()._identity else 0
        _DEVICE = f"cuda:{idx % n_gpu}" if device == "cuda" else device
    else:
        _DEVICE = "cpu"
    import warnings
    warnings.filterwarnings("ignore")


def _rodar_job(args):
    nome, spec, caso, ruido, seed, t_max, arquivo = args
    from executar import executar
    res = executar(nome, spec, caso, ruido, seed, device=_DEVICE, t_max=t_max)
    tmp = arquivo + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(res, fh)
    os.replace(tmp, arquivo)
    return res


def nome_arquivo(pasta, nome, caso, ruido, seed):
    return os.path.join(pasta, f"{nome}__c{caso}__r{ruido:g}__s{seed}.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perfil", default="padrao", choices=list(C.PERFIS))
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    ap.add_argument("--threads", type=int, default=None, help="threads de CPU por worker")
    ap.add_argument("--saida", default=os.path.join("data", "resultados"))
    ap.add_argument("--metodos", nargs="*", default=None, help="filtra métodos por substring")
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="sobrescreve as seeds do perfil")
    ap.add_argument("--refazer", action="store_true", help="refaz jobs já concluídos")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    import torch as tc
    if a.device == "auto":
        a.device = "cuda" if tc.cuda.is_available() else "cpu"
    n_gpu = tc.cuda.device_count() if a.device.startswith("cuda") else 0
    if a.device.startswith("cuda") and n_gpu == 0:
        print("CUDA pedida mas não disponível -> usando CPU")
        a.device = "cpu"
    threads = a.threads or max(1, (os.cpu_count() or 1) // a.workers)

    P = C.PERFIS[a.perfil]
    pasta = os.path.join(a.saida, a.perfil, "jobs")
    os.makedirs(pasta, exist_ok=True)
    seeds = a.seeds if a.seeds is not None else P["seeds"]
    metodos = {k: v for k, v in P["metodos"].items()
               if not a.metodos or any(s in k for s in a.metodos)}

    jobs = []
    for (caso, ruido) in P["cenarios"]:
        for seed in seeds:
            for nome, spec in metodos.items():
                arq = nome_arquivo(pasta, nome, caso, ruido, seed)
                if os.path.exists(arq) and not a.refazer:
                    continue
                jobs.append((nome, spec, caso, ruido, seed, P["t_max_job"], arq))
    jobs.sort(key=lambda j: -C.custo_estimado(j[1]))

    print(f"perfil={a.perfil}  jobs pendentes={len(jobs)}  workers={a.workers}  "
          f"device={a.device} (GPUs={n_gpu})  threads/worker={threads}")
    print(f"resultados em {pasta}")
    if a.dry_run or not jobs:
        for j in jobs[:20]:
            print("  ", j[0], "caso", j[2], "ruido", j[3], "seed", j[4])
        return

    # os workers (spawn) herdam estas variáveis antes de importar numpy/torch
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(threads)

    t0 = time.time()
    feitos = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, initializer=_init_worker, initargs=(a.device, threads, n_gpu)) as pool:
        for res in pool.imap_unordered(_rodar_job, jobs):
            feitos += 1
            dt = time.time() - t0
            eta = dt / feitos * (len(jobs) - feitos)
            st = "ok " if res["status"] == "ok" else "ERRO"
            print(f"[{feitos:4d}/{len(jobs)}] {st} {res['metodo']:26s} c{res['caso']} r{res['ruido']:g} "
                  f"s{res['seed']} erro_campo={res['erro_campo']:.3e} rmse_traj={res['rmse_traj']:.2e} "
                  f"({res['tempo_s']:.0f}s, {res['device']})  ETA {eta/60:.0f} min", flush=True)
            if res["status"] != "ok":
                print("      ", res.get("msg"), flush=True)
    print(f"concluído em {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
