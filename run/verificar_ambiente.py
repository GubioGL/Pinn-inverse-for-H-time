"""Confere dependências, GPU e mede a velocidade de um passo de treino do PINN.

  python run/verificar_ambiente.py
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
import time
import platform


print("python  ", platform.python_version(), "|", platform.platform())
for mod in ("numpy", "scipy", "matplotlib", "torch"):
    try:
        m = __import__(mod)
        print(f"{mod:10s}", m.__version__)
    except ImportError:
        print(f"{mod:10s} NÃO INSTALADO  ->  pip install -r requirements.txt")
        sys.exit(1)

import numpy as np   # noqa: E402
import torch as tc   # noqa: E402

print("CPUs    ", os.cpu_count())
print("CUDA    ", tc.cuda.is_available(), "| GPUs:", tc.cuda.device_count())
for i in range(tc.cuda.device_count()):
    p = tc.cuda.get_device_properties(i)
    print(f"   cuda:{i}  {p.name}  {p.total_memory / 2**30:.1f} GB")

import bloch as cm
from pinn import ProblemaPINN       # noqa: E402
import otimizadores as ot           # noqa: E402

t, data, clean, ffn, T = cm.make_data(3, N=100, noise=0.0, seed=0, r0=cm.R0_ZERO)
devs = ["cpu"] + [f"cuda:{i}" for i in range(tc.cuda.device_count())]
for dev in devs:
    cfg = dict(field="bspline", field_size=40, width=32, depth=3, act="sin", norm_time=True,
               hard_ic=True, n_col=200, lam_f=1e-10)
    pr = ProblemaPINN(t, data, cm.R0_ZERO, T, cfg, device=dev, dtype=tc.float64)
    th = pr.init_theta(0)
    ot.adam(pr, th, 20)                       # aquecimento
    if dev.startswith("cuda"):
        tc.cuda.synchronize()
    t0 = time.time()
    ot.adam(pr, th, 200)
    if dev.startswith("cuda"):
        tc.cuda.synchronize()
    ta = (time.time() - t0) / 200
    pr.jacobiano(th)
    t0 = time.time()
    for _ in range(3):
        r, J = pr.jacobiano(th)
        K = J @ J.T
        tc.linalg.cholesky(K + 1e-6 * tc.eye(K.shape[0], dtype=K.dtype, device=K.device))
    if dev.startswith("cuda"):
        tc.cuda.synchronize()
    tl = (time.time() - t0) / 3
    print(f"{dev:7s}: passo Adam = {ta * 1e3:.1f} ms | iteração LM = {tl:.2f} s  "
          f"(P = {pr.n_params}, R = {J.shape[0]})")
print("ok")
