"""
Três baselines que NÃO usam PINN, para comparar com os 4 métodos do repositório.

(A) Inversão direta ponto-a-ponto
    Das equações de Bloch:  f Z = X - dY/dt ,  f Y = dZ/dt.
    Mínimos quadrados em cada instante:
        f(t) = [ Z (X - dY/dt) + Y dZ/dt ] / (Y^2 + Z^2)
    Derivadas obtidas por spline (cúbica sem ruído, smoothing spline com ruído).

(B) Mínimos quadrados LINEARES nos coeficientes da base (Fourier / B-spline)
    Com o estado fixo (dados), o resíduo da EDO é LINEAR em f = B c:
        r1 = Z * (B c) - (X - dY/dt),   r2 = Y * (B c) - dZ/dt
    => c = argmin ||W c - b||^2 + lam ||D2 c||^2   (solução fechada, sem Adam).
    É exatamente a "física" que os modelos 3 e 4 tentam aprender com Adam.

(C) Shooting / simulação diferenciável (estilo GRAPE / neural-ODE)
    f = B c (B-spline cúbica). Integra-se a equação de Bloch (RK4 + equações de
    sensibilidade) e ajusta-se c por Levenberg-Marquardt (scipy.least_squares)
    minimizando || <σ>_sim(t_i; c) - dados_i ||^2 + lam ||D2 c||^2.
    A dinâmica é satisfeita EXATAMENTE (não é um termo de penalidade como no PINN).
    O problema é não-convexo: use o resultado de (B) como chute inicial (warm start).

Todos os métodos aceitam VÁRIOS experimentos (estados iniciais diferentes) com o
mesmo campo, o que melhora muito a identificabilidade.

Uso:  python metodos_sem_pinn.py
"""
import os
import csv
import time
import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as cm

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figuras")


# ---------------------------------------------------------------------------
# (A) inversão direta
# ---------------------------------------------------------------------------
def direct_inversion(t, data, smooth=False, eps=1e-3):
    vals, ders = cm.data_derivative(t, data, smooth=smooth)
    X, Y, Z = vals.T
    dY, dZ = ders[:, 1], ders[:, 2]
    w = Y ** 2 + Z ** 2
    return (Z * (X - dY) + Y * dZ) / (w + eps)


# ---------------------------------------------------------------------------
# (B) mínimos quadrados lineares (aceita lista de experimentos)
# ---------------------------------------------------------------------------
def linear_lsq(t, datas, T, kind="bspline", size=40, lam=1e-6, smooth=False, k=3):
    if isinstance(datas, np.ndarray):
        datas = [datas]
    B = cm.basis_matrix(kind, t, T, size, k)
    Ws, bs = [], []
    for data in datas:
        vals, ders = cm.data_derivative(t, data, smooth=smooth)
        X, Y, Z = vals.T
        dY, dZ = ders[:, 1], ders[:, 2]
        Ws += [Z[:, None] * B, Y[:, None] * B]
        bs += [X - dY, dZ]
    W = np.vstack(Ws)
    b = np.concatenate(bs)
    D = cm.second_diff(B.shape[1])
    A = np.vstack([W, np.sqrt(lam) * D])
    rhs = np.concatenate([b, np.zeros(D.shape[0])])
    c, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    return B @ c, c


# ---------------------------------------------------------------------------
# (C) shooting com sensibilidades (RK4)
# ---------------------------------------------------------------------------
class BlochShooting:
    def __init__(self, t_data, T, n_basis=40, k=3, substeps=10, r0=cm.R0_PLUS):
        self.t_data = t_data
        self.T = T
        N = len(t_data)
        self.m = substeps
        self.h = (t_data[1] - t_data[0]) / substeps
        n_steps = (N - 1) * substeps
        # grade com meio-passo para os estágios do RK4
        self.t_half = np.linspace(t_data[0], t_data[-1], 2 * n_steps + 1)
        self.Bh = cm.bspline_matrix(self.t_half, T, n_basis, k)
        self.B_data = cm.bspline_matrix(t_data, T, n_basis, k)
        self.n_steps = n_steps
        self.nb = n_basis
        self.r0 = np.asarray(r0, float)

    @staticmethod
    def _rhs(r, S, f, b):
        X, Y, Z = r
        dr = np.array([-Y, X - f * Z, f * Y])
        A = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, -f], [0.0, f, 0.0]])
        g = np.array([0.0, -Z, Y])
        dS = A @ S + np.outer(g, b)
        return dr, dS

    def simulate(self, c):
        """Retorna trajetória r(t_i) (N,3) e sensibilidades dr/dc (N,3,nb)."""
        fh = self.Bh @ c
        r = self.r0.copy()
        S = np.zeros((3, self.nb))
        out_r = [r.copy()]
        out_S = [S.copy()]
        h = self.h
        for s in range(self.n_steps):
            i = 2 * s
            k1r, k1S = self._rhs(r, S, fh[i], self.Bh[i])
            k2r, k2S = self._rhs(r + 0.5 * h * k1r, S + 0.5 * h * k1S, fh[i + 1], self.Bh[i + 1])
            k3r, k3S = self._rhs(r + 0.5 * h * k2r, S + 0.5 * h * k2S, fh[i + 1], self.Bh[i + 1])
            k4r, k4S = self._rhs(r + h * k3r, S + h * k3S, fh[i + 2], self.Bh[i + 2])
            r = r + h / 6 * (k1r + 2 * k2r + 2 * k3r + k4r)
            S = S + h / 6 * (k1S + 2 * k2S + 2 * k3S + k4S)
            if (s + 1) % self.m == 0:
                out_r.append(r.copy())
                out_S.append(S.copy())
        return np.array(out_r), np.array(out_S)

    def fit(self, data, lam=1e-6, c0=None, max_nfev=200):
        return shooting_fit([self], [data], lam=lam, c0=c0, max_nfev=max_nfev)


def shooting_fit(shooters, datas, lam=1e-6, c0=None, max_nfev=200):
    """Ajusta UM campo a vários experimentos (um BlochShooting por estado inicial)."""
    nb = shooters[0].nb
    D = cm.second_diff(nb)
    sl = np.sqrt(lam)
    cache = {}

    def run(c):
        key = c.tobytes()
        if key not in cache:
            cache.clear()
            cache[key] = [sh.simulate(c) for sh in shooters]
        return cache[key]

    def res(c):
        parts = [(r - d).ravel() for (r, _), d in zip(run(c), datas)]
        return np.concatenate(parts + [sl * D @ c])

    def jac(c):
        parts = [S.reshape(-1, nb) for _, S in run(c)]
        return np.vstack(parts + [sl * D])

    c0 = np.zeros(nb) if c0 is None else c0
    sol = least_squares(res, c0, jac=jac, method="lm", max_nfev=max_nfev,
                        xtol=1e-12, ftol=1e-12, gtol=1e-12)
    return shooters[0].B_data @ sol.x, sol


def lam_for(noise):
    return 1e-6 if noise == 0 else (1e-3 if noise <= 0.01 else 1e-2)


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

    with open(os.path.join(HERE, "resultados_sem_pinn.csv"), "w", newline="") as fh:
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
    plt.savefig(os.path.join(FIG, "metodos_sem_pinn.png"), dpi=130)
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
    with open(os.path.join(HERE, "resultados_desenho_experimental.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sigma_ref", "desenho", "sigma_por_ponto", "lsq_media", "lsq_std",
                    "shooting_media", "shooting_std"])
        for r in out:
            w.writerow([r[0], r[1], f"{r[2]:.4f}"] + [f"{x:.4f}" for x in r[3:]])
    return out


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    comparacao_metodos()
    desenho_experimental()
