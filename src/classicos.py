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

Experimentos que usam este módulo: run/parte1_classicos.py
"""
import numpy as np
from scipy.optimize import least_squares

import bloch as cm


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
