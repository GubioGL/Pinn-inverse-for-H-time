"""Executa UM job do protocolo: (método, caso, ruído, seed) -> dicionário de resultados."""
import time
import math
import zlib
import traceback
import numpy as np

import bloch as cm
import protocolo as C


def gerar_dados(caso, ruido, seed):
    """Mesmos dados para todos os métodos: depende só de (caso, ruído, seed)."""
    base = caso if isinstance(caso, int) else zlib.crc32(str(caso).encode()) % 1000
    data_seed = 100_000 * base + 1_000 * int(round(ruido * 1000)) + seed
    t, data, clean, ffn, T = cm.make_data(caso, N=C.N_PONTOS, noise=ruido, seed=data_seed,
                                          r0=np.array(C.R0))
    return t, data, clean, ffn, T


def _metricas(f_pred, t, data, clean, ffn):
    f_true = ffn(t)
    from scipy.interpolate import CubicSpline
    traj = cm.simulate(CubicSpline(t, f_pred), t, np.array(C.R0))
    return dict(
        erro_campo=float(cm.rel_l2(f_pred, f_true)),
        rmse_traj=float(np.sqrt(np.mean((traj - clean) ** 2))),
        rmse_dados=float(np.sqrt(np.mean((traj - data) ** 2))),
    )


def _classico(spec, t, data, T, ruido):
    import classicos as msp
    smooth = ruido > 0
    alg = spec["alg"]
    lam = C.lam_d2(ruido, spec.get("tam", 40))
    if alg == "direta":
        return msp.direct_inversion(t, data, smooth=smooth), {}
    if alg == "lsq":
        f, _ = msp.linear_lsq(t, data, T, spec["base"], spec["tam"],
                              lam=(lam if spec["base"] == "bspline" else 0.0), smooth=smooth)
        return f, {}
    if alg == "shooting":
        _, c0 = msp.linear_lsq(t, data, T, "bspline", spec["tam"], lam=lam, smooth=smooth)
        sh = msp.BlochShooting(t, T, n_basis=spec["tam"], r0=np.array(C.R0))
        f, sol = sh.fit(data, lam=lam, c0=c0)
        return f, dict(nfev=int(sol.nfev))
    raise ValueError(alg)


def _pinn(spec, t, data, T, caso, ruido, seed, device, t_max):
    import torch as tc
    from pinn import ProblemaPINN
    import otimizadores as ot
    rec = C.RECEITAS[spec["receita"]]
    arq = dict(rec["arq"])
    dtype = tc.float64 if arq.pop("dtype", "float64") == "float64" else tc.float32
    bc = arq.pop("bc", "nenhuma")
    init_lsq = arq.pop("init_lsq", False)
    cfg = dict(C.CAMPOS[spec["campo"]])
    cfg.update(arq)
    cfg["lam_f"] = C.lam_f(ruido)
    cfg["w_data"] = 1.0
    cfg["seed"] = seed
    cfg["bc_campo"] = cm.bordas_conhecidas(caso) if bc == "conhecida" else None
    tc.manual_seed(seed)
    prob = ProblemaPINN(t, data, np.array(C.R0), T, cfg, device=device, dtype=dtype)
    theta = prob.init_theta(seed, init_lsq=init_lsq, suavizar=ruido > 0)
    fator = spec.get("fator", 1.0)
    estagios = []
    for st in rec["estagios"]:
        st = dict(st)
        for k in ("epochs", "iters"):
            if k in st:
                st[k] = max(1, int(st[k] * fator))
        estagios.append(st)
    log = []
    theta = ot.treinar(prob, theta, estagios, log=log, t_max_total=t_max)
    L, parts = prob.loss(theta)
    f = prob.campo_dados(theta)
    extra = dict(n_params=int(prob.n_params), bc_campo=cfg["bc_campo"], loss=float(L), l_ode=float(parts[0]),
                 l_data=float(parts[1]), l_reg=float(parts[2]),
                 historico=[(a, int(b), float(c)) for a, b, c in log[-50:]])
    return f, extra


def executar(nome, spec, caso, ruido, seed, device="cpu", t_max=None):
    t0 = time.time()
    out = dict(metodo=nome, tipo=spec["tipo"], campo=spec.get("campo"), receita=spec.get("receita"),
               caso=caso, ruido=ruido, seed=seed, device=str(device) if spec["tipo"] == "pinn" else "cpu")
    try:
        t, data, clean, ffn, T = gerar_dados(caso, ruido, seed)
        if spec["tipo"] == "classico":
            f, extra = _classico(spec, t, data, T, ruido)
        else:
            f, extra = _pinn(spec, t, data, T, caso, ruido, seed, device, t_max)
        f = np.asarray(f, dtype=float)
        out["tempo_s"] = time.time() - t0
        if not np.all(np.isfinite(f)):
            raise FloatingPointError("campo predito com NaN/Inf")
        out.update(_metricas(f, t, data, clean, ffn))
        out.update(extra)
        out["f_pred"] = [float(x) for x in f]
        out["status"] = "ok"
    except Exception as e:                       # noqa: BLE001
        out["tempo_s"] = time.time() - t0
        out["status"] = "erro"
        out["msg"] = f"{type(e).__name__}: {e}"
        out["trace"] = traceback.format_exc()[-2000:]
        out["erro_campo"] = math.nan
        out["rmse_traj"] = math.nan
    return out
