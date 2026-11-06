"""
Utilitários comuns para o problema inverso  H(t) = 0.5*sz + 0.5*f(t)*sx.

Para 1 qubit puro, <sx>,<sy>,<sz> = r = (X, Y, Z) satisfazem as equações de Bloch

    dr/dt = Omega(t) x r ,   Omega(t) = (f(t), 0, 1)

    dX/dt = -Y
    dY/dt =  X - f Z
    dZ/dt =  f Y

(são exatamente as equações usadas nas losses dos notebooks).

Aqui só usamos numpy/scipy, para que tudo rode rápido e sem qutip.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import BSpline, CubicSpline, make_smoothing_spline


# ---------------------------------------------------------------------------
# Campos usados nos notebooks
# ---------------------------------------------------------------------------
def make_case(case):
    """Retorna (field_fn, tfinal).

    1, 2, 3 : campos dos notebooks
    "w<omega>" (ex. "w6"): pulso de frequência única para o estudo de resolução espectral,
               f(t) = sin(omega t) sin^2(pi t / T), T = 2.52 pi
    """
    if isinstance(case, str) and case.startswith("w"):
        omega = float(case[1:])
        tfinal = 2.52 * np.pi
        return (lambda t: np.sin(omega * t) * np.sin(np.pi * t / tfinal) ** 2), tfinal
    case = int(case)
    if case == 1:
        tfinal = 2 * np.pi
        return (lambda t: np.cos(t)), tfinal
    if case == 2:
        tfinal = 2.52 * np.pi
        return (lambda t: 1.5 * np.sin(2.0 * t) + 0.8 * np.cos(3.5 * t + 0.5)
                + 0.4 * np.sin(7.0 * t)), tfinal
    if case == 3:
        tfinal = 2.52 * np.pi
        return (lambda t: (1.5 * np.sin(2.0 * t) - 0.3 * np.cos(3 * t - 0.5)
                           + 0.4 * np.sin(7.0 * t)) * np.sin(np.pi * t / tfinal) ** 2), tfinal
    raise ValueError(case)


def bordas_conhecidas(case):
    """Valores (f(0), f(T)) conhecidos A PRIORI pelo desenho do pulso, ou None.

    Caso 3 e os pulsos "w<omega>" têm envelope sin^2(pi t/T): o pulso liga e desliga em zero,
    o que é uma informação do experimento (não do campo a ser descoberto). Nos casos 1 e 2
    não há essa informação, então nenhuma restrição de borda é imposta ao campo."""
    if (isinstance(case, str) and case.startswith("w")) or str(case) == "3":
        return (0.0, 0.0)
    return None


R0_PLUS = np.array([1.0, 0.0, 0.0])      # |+>  (estado usado nos notebooks)
R0_ZERO = np.array([0.0, 0.0, 1.0])      # |0>
R0_GENERIC = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)


def bloch_rhs(t, r, f):
    X, Y, Z = r
    ft = f(t)
    return [-Y, X - ft * Z, ft * Y]


def simulate(field_fn, tlist, r0=R0_PLUS):
    """Integra as equações de Bloch com tolerância alta. Retorna (N,3)."""
    sol = solve_ivp(bloch_rhs, (tlist[0], tlist[-1]), r0, t_eval=tlist,
                    args=(field_fn,), rtol=1e-11, atol=1e-12, method="DOP853")
    return sol.y.T


def make_data(case, N=100, noise=0.0, seed=0, r0=R0_PLUS):
    field_fn, tfinal = make_case(case)
    t = np.linspace(0.0, tfinal, N)
    clean = simulate(field_fn, t, r0)
    rng = np.random.default_rng(seed)
    data = clean + noise * rng.standard_normal(clean.shape)
    return t, data, clean, field_fn, tfinal


# ---------------------------------------------------------------------------
# Bases para o campo
# ---------------------------------------------------------------------------
def fourier_matrix(t, T, M):
    """Base de Fourier periódica com período T (igual ao modelo_3)."""
    n = np.arange(1, M + 1)[None, :]
    ang = 2 * np.pi / T * t[:, None] * n
    return np.hstack([np.ones((len(t), 1)), np.cos(ang), np.sin(ang)])


def cosine_matrix(t, T, M):
    """Série de cossenos em [0,T] (extensão par, período 2T).
    Não tem fenômeno de Gibbs quando f(0) != f(T)."""
    n = np.arange(0, M + 1)[None, :]
    return np.cos(np.pi / T * t[:, None] * n)


def open_uniform_knots(t0, t1, n_basis, k):
    n_internal = n_basis - k - 1
    internal = np.linspace(t0, t1, n_internal + 2)[1:-1] if n_internal > 0 else np.array([])
    return np.r_[np.full(k + 1, t0), internal, np.full(k + 1, t1)]


def bspline_matrix(t, T, n_basis, k=3):
    knots = open_uniform_knots(0.0, T, n_basis, k)
    tt = np.clip(t, 0.0, T * (1 - 1e-12))
    return BSpline.design_matrix(tt, knots, k).toarray()


def basis_matrix(kind, t, T, size, k=3):
    if kind == "fourier":
        return fourier_matrix(t, T, size)
    if kind == "cosine":
        return cosine_matrix(t, T, size)
    if kind == "bspline":
        return bspline_matrix(t, T, size, k)
    raise ValueError(kind)


def second_diff(n):
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i:i + 3] = [1.0, -2.0, 1.0]
    return D


# ---------------------------------------------------------------------------
# Derivadas dos dados (interpolação / suavização)
# ---------------------------------------------------------------------------
def data_derivative(t, data, smooth=False, lam=None):
    """Retorna (valores, derivadas) de cada coluna dos dados.
    smooth=False -> CubicSpline (dados sem ruído)
    smooth=True  -> smoothing spline (lam=None escolhe lambda por GCV)."""
    vals, ders = [], []
    for j in range(data.shape[1]):
        if smooth:
            s = make_smoothing_spline(t, data[:, j], lam=lam)
        else:
            s = CubicSpline(t, data[:, j])
        vals.append(s(t))
        ders.append(s.derivative()(t))
    return np.array(vals).T, np.array(ders).T


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def rel_l2(pred, true):
    return np.linalg.norm(pred - true) / np.linalg.norm(true)


def resim_rmse(field_values, t, data_clean, r0=R0_PLUS):
    """Re-simula com o campo predito (interpolado por spline cúbica, como o qutip faz)
    e mede o RMSE em relação à trajetória verdadeira."""
    fi = CubicSpline(t, field_values)
    traj = simulate(fi, t, r0)
    return np.sqrt(np.mean((traj - data_clean) ** 2))
