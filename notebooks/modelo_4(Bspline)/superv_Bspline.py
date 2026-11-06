"""
Aprendizado supervisionado de omega_x(t) usando s.
"""

import torch
import torch as tc
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import BSpline

def open_uniform_knots(t0, t1, n_basis, k):
    n_internal = n_basis - k - 1
    if n_internal > 0:
        internal = np.linspace(t0, t1, n_internal + 2)[1:-1]
    else:
        internal = np.array([])
    knots = np.r_[np.full(k+1, t0), internal, np.full(k+1, t1)]
    return knots

def bspline_design_matrix(t_eval, knots, k):
    n_basis = len(knots) - k - 1
    B = np.zeros((len(t_eval), n_basis), dtype=np.float64)

    for j in range(n_basis):
        c = np.zeros(n_basis)
        c[j] = 1.0
        spl = BSpline(knots, c, k, extrapolate=True)
        B[:, j] = spl(t_eval)

    return B

class BSplineField(tc.nn.Module):
    def __init__(self, t_eval, t0, t1, n_basis=10, k=3, init_fun=None):
        super().__init__()

        knots = open_uniform_knots(t0, t1, n_basis, k)
        B = bspline_design_matrix(np.asarray(t_eval, dtype=np.float64), knots, k)

        self.register_buffer("B", tc.tensor(B, dtype=tc.float32))
        self.register_buffer("knots", tc.tensor(knots, dtype=tc.float32))
        self.k = k

        coeff0 = np.zeros(n_basis, dtype=np.float32)

        # opcional: inicializar os coeficientes perto de uma função conhecida
        if init_fun is not None:
            y0 = init_fun(np.asarray(t_eval, dtype=np.float64))
            coeff0, *_ = np.linalg.lstsq(B, y0, rcond=None)

        self.coeffs = tc.nn.Parameter(
            tc.tensor(coeff0.reshape(-1, 1), dtype=tc.float32)
        )

    def forward(self, time):
        # aqui assumimos que o grid de treino é o mesmo usado para construir B
        return self.B @ self.coeffs

    def smoothness_penalty(self):
        c = self.coeffs[:, 0]
        return ((c[2:] - 2*c[1:-1] + c[:-2])**2).mean()


# 0. Reprodutibilidade

# 1. Parâmetros
T_START = 0.0
T_END   = 2*np.pi
N_DATA  = 100

def omega_x_true(t):
    return ( 1.5* np.sin(2.0 * t) - 0.3*np.cos(3* t - 0.5) + 0.4*np.sin(7.0 * t)
    )*np.sin(np.pi*t/T_END)**2
    
# 2. Dados supervisionados
t_data_np = np.linspace(T_START, T_END, N_DATA)
omega_data_np = omega_x_true(t_data_np)

t_data = torch.tensor(t_data_np, dtype=torch.float32).unsqueeze(1)
omega_data = torch.tensor(omega_data_np, dtype=torch.float32).unsqueeze(1)

J_spline = BSplineField(
    t_eval = t_data.detach().cpu().numpy().flatten(),
    t0  = T_START,
    t1  = T_END,
    n_basis = 30,
    k   = 10,
)

# 4. Treinamento
optimizer = torch.optim.Adam(J_spline.parameters(), lr=1e-2)

print("Treinando B-spline (supervisionado)...")
for epoch in range(10000):
    optimizer.zero_grad()
    pred = J_spline(t_data)
    loss = torch.mean((pred - omega_data) ** 2)
    loss.backward()
    optimizer.step()
    if epoch % 200 == 0:
        print(f"Época {epoch} | Loss: {loss.item():.4e}")

# 5. Avaliação e plot
t_test_np = np.linspace(T_START, T_END, 100)
t_test = torch.tensor(t_test_np, dtype=torch.float32).unsqueeze(1)
with torch.no_grad():
    ox_pred = J_spline(t_test).numpy().flatten()
ox_true = omega_x_true(t_test_np)

plt.figure(figsize=(8, 5))
plt.plot(t_test_np, ox_true, label='omega_x Verdadeiro', color='blue')
plt.plot(t_test_np, ox_pred, label='omega_x Predito (B-spline)', color='red', linestyle='--')
plt.xlabel('t')
plt.ylabel('omega_x(t)')
plt.legend()
plt.title('Comparação: omega_x verdadeiro vs predito (B-spline, supervisionado)')
plt.tight_layout()
plt.savefig('comparison_bspline_supervised.png')
print("Plot salvo em comparison_bspline_supervised.png")
