"""
Aprendizado supervisionado de omega_x(t) usando série de Fourier.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

# 3. Expansão de Fourier
M_HARMONICS = 10
T_PERIOD = T_END - T_START
n_range = torch.arange(1, M_HARMONICS + 1, dtype=torch.float32).view(1, -1)

a0 = nn.Parameter(torch.zeros(1))
an = nn.Parameter(torch.zeros(1, M_HARMONICS))
bn = nn.Parameter(torch.zeros(1, M_HARMONICS))

def get_fourier_field(t):
    angles = (2.0 * np.pi / T_PERIOD) * t * n_range
    return a0 + torch.sum(an * torch.cos(angles) + bn * torch.sin(angles), dim=1, keepdim=True)

# 4. Treinamento
optimizer = torch.optim.Adam([a0, an, bn], lr=1e-2)

print("Treinando Fourier (supervisionado)...")
for epoch in range(10000):
    optimizer.zero_grad()
    pred = get_fourier_field(t_data)
    loss = torch.mean((pred - omega_data) ** 2)
    loss.backward()
    optimizer.step()
    if epoch % 200 == 0:
        print(f"Época {epoch} | Loss: {loss.item():.4e}")

# 5. Avaliação e plot
t_test_np = np.linspace(T_START, T_END, 1000)
t_test = torch.tensor(t_test_np, dtype=torch.float32).unsqueeze(1)
with torch.no_grad():
    ox_pred = get_fourier_field(t_test).numpy().flatten()
ox_true = omega_x_true(t_test_np)

plt.figure(figsize=(8, 5))
plt.plot(t_test_np, ox_true, label='omega_x Verdadeiro', color='blue')
plt.plot(t_test_np, ox_pred, label='omega_x Predito (Fourier)', color='red', linestyle='--')
plt.xlabel('t')
plt.ylabel('omega_x(t)')
plt.legend()
plt.title('Comparação: omega_x verdadeiro vs predito (Fourier, supervisionado)')
plt.tight_layout()
plt.savefig('comparison_fourier_supervised.png')
print("Plot salvo em comparison_fourier_supervised.png")
