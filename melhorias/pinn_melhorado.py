"""
PINN para o campo f(t) com as melhorias sugeridas, ligáveis uma a uma para
medir o efeito de cada uma (ablation).

Sistema: H = 0.5 sz + 0.5 f(t) sx   ->   dX=-Y, dY=X-fZ, dZ=fY.

Melhorias disponíveis (ver PRESETS no final):
  norm_time  : entrada da rede em [-1,1] (s = 2t/T-1) em vez de t em [0, 7.9]
  hard_ic    : r(t) = r0 + (t/T) * NN(t)  -> condição inicial exata
  width/depth: rede maior (o estado tem estrutura fina vinda do termo sin(7t))
  n_col      : pontos de colocação densos para a EDO (independentes dos dados)
  varpro     : "variable projection" - como a EDO é LINEAR nos coeficientes c do
               campo (f = B c), c é resolvido em forma fechada a cada iteração
               (mínimos quadrados com Tikhonov). Remove o acoplamento difícil
               estado <-> campo que o Adam tem de resolver.
  lbfgs      : refinamento final com L-BFGS (padrão em PINNs)
  bc_true    : penalidade f(0)=f_true(0), f(T)=f_true(T) (usa o campo verdadeiro,
               como nos scripts originais; deixe False para um problema inverso honesto)

Uso:
  python pinn_melhorado.py --preset original
  python pinn_melhorado.py --preset completo --seed 0
  python pinn_melhorado.py --all            # roda todos os presets (ablation)
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch as tc
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as cm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


class Sin(nn.Module):
    def forward(self, x):
        return tc.sin(x)


class MLP(nn.Module):
    def __init__(self, width, depth, out=3, act="sin"):
        super().__init__()
        A = Sin if act == "sin" else nn.Tanh
        layers = [nn.Linear(1, width), A()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), A()]
        layers += [nn.Linear(width, out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def field_basis(kind, t, T):
    if kind.startswith("fourier"):
        return cm.fourier_matrix(t, T, int(kind[len("fourier"):]))
    if kind.startswith("bspline"):              # "bspline40" (k=3) ou "bspline30k10"
        spec = kind[len("bspline"):]
        n, k = (spec.split("k") + ["3"])[:2]
        return cm.bspline_matrix(t, T, int(n), int(k))
    raise ValueError(kind)


def run(cfg, verbose=True):
    tc.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    tc.set_num_threads(1)

    r0 = {"plus": cm.R0_PLUS, "zero": cm.R0_ZERO}[cfg["r0"]]
    t_np, data_np, clean_np, ffn, T = cm.make_data(cfg["case"], N=cfg["N"], noise=cfg["noise"],
                                                    seed=1234, r0=r0)
    f_true = ffn(t_np)

    # pontos de colocação da EDO
    t_col_np = t_np if cfg["n_col"] is None else np.linspace(0, T, cfg["n_col"])

    dt = tc.float64 if cfg["float64"] else tc.float32
    tc.set_default_dtype(dt)
    t_data = tc.tensor(t_np, dtype=dt).reshape(-1, 1)
    t_col = tc.tensor(t_col_np, dtype=dt).reshape(-1, 1).requires_grad_(True)
    data = tc.tensor(data_np, dtype=dt)
    r0_t = tc.tensor(r0, dtype=dt)

    nn_field = cfg["field"] == "nn"              # campo = 4a saída da rede (modelo_2)
    if nn_field:
        B_col = B_data = None
        nb = 1
        Dm = None
    else:
        B_col = tc.tensor(field_basis(cfg["field"], t_col_np, T), dtype=dt)
        B_data = tc.tensor(field_basis(cfg["field"], t_np, T), dtype=dt)
        nb = B_col.shape[1]
        Dm = tc.tensor(cm.second_diff(nb), dtype=dt) if cfg["field"].startswith("bspline") else None

    net = MLP(cfg["width"], cfg["depth"], 4 if nn_field else 3, cfg["act"])
    if cfg["coef_init"] == "rand":
        c = nn.Parameter(tc.rand(nb))            # como no modelo_3 original
    else:
        c = nn.Parameter(tc.zeros(nb))

    def state(t, with_field=False):
        x = 2 * t / T - 1 if cfg["norm_time"] else t
        out = net(x)
        y = out[:, :3]
        if cfg["hard_ic"]:
            y = r0_t + (t / T) * y
        if with_field:
            return y, (out[:, 3] if nn_field else None)
        return y

    def field_at_data():
        if nn_field:
            return state(t_data, True)[1]
        return B_data @ c

    def solve_c(Xc, Yc, Zc, dY, dZ):
        """mínimos quadrados fechados para c (variable projection)."""
        w = (Yc ** 2 + Zc ** 2)
        A = B_col.T @ (w[:, None] * B_col)
        rhs = B_col.T @ (Zc * (Xc - dY) + Yc * dZ)
        if Dm is not None:
            A = A + cfg["lam_varpro"] * Dm.T @ Dm
        else:
            A = A + cfg["lam_varpro"] * tc.eye(nb)
        return tc.linalg.solve(A, rhs)

    def losses():
        y_col, f_nn = state(t_col, True)
        d = [tc.autograd.grad(y_col[:, i].sum(), t_col, create_graph=True)[0][:, 0] for i in range(3)]
        Xc, Yc, Zc = y_col[:, 0], y_col[:, 1], y_col[:, 2]
        if cfg["varpro"]:
            with tc.no_grad():
                c.data = solve_c(Xc.detach(), Yc.detach(), Zc.detach(), d[1].detach(), d[2].detach())
        f = f_nn if nn_field else B_col @ c
        l_ode = ((d[0] + Yc) ** 2 + (d[1] - Xc + f * Zc) ** 2 + (d[2] - f * Yc) ** 2).mean()
        y_dat = state(t_data)
        l_data = ((y_dat - data) ** 2).sum(1).mean()
        loss = cfg["w_ode"] * l_ode + cfg["w_data"] * l_data
        if cfg["bc_true"]:
            fd = field_at_data()
            loss = loss + (fd[0] - ffn(0.0)) ** 2 + (fd[-1] - ffn(T)) ** 2
        if Dm is not None and cfg["lam_smooth"] > 0 and not cfg["varpro"]:
            loss = loss + cfg["lam_smooth"] * ((Dm @ c) ** 2).mean()
        if cfg["varpro"]:
            # mesmo termo de Tikhonov que solve_c minimiza: assim c* é o minimizador EXATO
            # da parte da loss que depende de c, e o gradiente em relação à rede (com c
            # "congelado") é exato (teorema do envelope). Sem isso a busca de linha do
            # L-BFGS falha e ele para na 1a iteração.
            reg = (Dm @ c) if Dm is not None else c
            loss = loss + cfg["w_ode"] * cfg["lam_varpro"] / t_col.shape[0] * (reg ** 2).sum()
        return loss, l_ode, l_data

    params = list(net.parameters()) + ([] if (cfg["varpro"] or nn_field) else [c])
    opt = tc.optim.Adam(params, lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = (tc.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"], eta_min=cfg["lr"] * 1e-2)
             if cfg["sched"] else None)

    t0 = time.time()
    for ep in range(cfg["epochs"]):
        loss, l_ode, l_data = losses()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sched:
            sched.step()
        if verbose and (ep % max(1, cfg["epochs"] // 5) == 0):
            with tc.no_grad():
                e = cm.rel_l2(field_at_data().numpy(), f_true)
            print(f"  [{cfg['name']}] ep {ep:6d} loss={loss.item():.2e} ode={l_ode.item():.2e} "
                  f"data={l_data.item():.2e} erro_campo={e:.4f}", flush=True)

    if cfg["lbfgs"] > 0:
        opt2 = tc.optim.LBFGS(params, lr=1.0, max_iter=cfg["lbfgs"], history_size=50,
                              tolerance_grad=1e-12, tolerance_change=1e-15,
                              line_search_fn="strong_wolfe")

        def closure():
            opt2.zero_grad()
            loss, _, _ = losses()
            loss.backward()
            return loss
        opt2.step(closure)

    loss, l_ode, l_data = losses()
    with tc.no_grad():
        f_pred = field_at_data().numpy()
    err = cm.rel_l2(f_pred, f_true)
    rs = cm.resim_rmse(f_pred, t_np, clean_np, r0)
    out = dict(name=cfg["name"], seed=cfg["seed"], erro_campo=float(err), rmse_resim=float(rs),
               l_ode=float(l_ode.detach()), l_data=float(l_data.detach()), tempo_s=time.time() - t0)
    if verbose:
        print(f"==> {cfg['name']} seed={cfg['seed']}: erro campo={err:.4f}  RMSE re-sim={rs:.2e}  "
              f"tempo={out['tempo_s']:.0f}s", flush=True)
    np.save(os.path.join(HERE, "resultados_pinn", f"campo_{cfg['name']}_s{cfg['seed']}.npy"),
            np.stack([t_np, f_true, f_pred]))
    return out


BASE = dict(case=3, N=100, noise=0.0, r0="plus", seed=0, float64=False,
            field="fourier10", coef_init="rand", norm_time=False, hard_ic=False,
            width=15, depth=2, act="sin", n_col=None, varpro=False, lam_varpro=1e-6,
            lam_smooth=0.0, w_ode=1.0, w_data=1.0, bc_true=True,
            epochs=100_000, lr=1e-3, wd=1e-5, sched=False, lbfgs=0)

PRESETS = {
    # réplica do modelo_3/fourier_pinn_field_3.py
    "original": dict(),
    # réplica do modelo_2 (PINN_field_3.ipynb): campo = 4a saída da rede
    "original_nn": dict(field="nn", epochs=50_000, wd=0.0),
    # réplica do modelo_4 (Bspliner_PINN_field_3.ipynb): B-spline grau 10, 30 funções
    "original_bspline": dict(field="bspline30k10", coef_init="zeros", bc_true=False,
                             lam_smooth=1e-4),
    # 1) entrada normalizada + IC exata + rede maior + scheduler (sem usar f_true nas bordas)
    "rede": dict(norm_time=True, hard_ic=True, width=64, depth=3, coef_init="zeros",
                 wd=0.0, sched=True, epochs=30_000, bc_true=False),
    # 2) + colocação densa da EDO (1000 pontos)
    "rede+col": dict(norm_time=True, hard_ic=True, width=64, depth=3, coef_init="zeros",
                     wd=0.0, sched=True, epochs=30_000, bc_true=False, n_col=1000),
    # 3) + variable projection (campo resolvido em forma fechada) com B-spline
    "rede+col+varpro": dict(norm_time=True, hard_ic=True, width=64, depth=3, wd=0.0, sched=True,
                            epochs=30_000, bc_true=False, n_col=1000, varpro=True,
                            field="bspline40", lam_varpro=1e-4),
    # 4) + L-BFGS no final
    "completo": dict(norm_time=True, hard_ic=True, width=64, depth=3, wd=0.0, sched=True,
                     epochs=30_000, bc_true=False, n_col=1000, varpro=True,
                     field="bspline40", lam_varpro=1e-4, lbfgs=2000),
    # 5) completo em float64 e com L-BFGS mais longo (o Adam sozinho estaciona num platô)
    "completo_f64": dict(norm_time=True, hard_ic=True, width=64, depth=3, wd=0.0, sched=True,
                         epochs=30_000, bc_true=False, n_col=1000, varpro=True,
                         field="bspline40", lam_varpro=1e-4, lbfgs=8000, float64=True),
}


def make_cfg(preset, **over):
    cfg = dict(BASE)
    cfg.update(PRESETS[preset])
    cfg.update(over)
    cfg["name"] = preset
    return cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="completo", choices=list(PRESETS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--r0", default="plus", choices=["plus", "zero"])
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, "resultados_pinn"), exist_ok=True)
    presets = list(PRESETS) if a.all else [a.preset]
    res = []
    for p in presets:
        over = dict(seed=a.seed, noise=a.noise, r0=a.r0)
        if a.epochs:
            over["epochs"] = a.epochs
        res.append(run(make_cfg(p, **over)))
    print(json.dumps(res, indent=1))
