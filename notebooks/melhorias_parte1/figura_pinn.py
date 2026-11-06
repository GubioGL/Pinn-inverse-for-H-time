"""Figura com o campo recuperado por cada preset do PINN (seed 0)."""
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "resultados_pinn")

ordem = ["original_nn", "original", "original_bspline", "rede", "rede+col",
         "rede+col+varpro", "completo", "completo_f64", "completo_r0zero"]
titulos = {
    "original_nn": "modelo_2 (campo = saída da rede)",
    "original": "modelo_3 (Fourier, como no repo)",
    "original_bspline": "modelo_4 (B-spline k=10, como no repo)",
    "rede": "+ tempo normalizado, IC exata, rede 64x3",
    "rede+col": "+ 1000 pontos de colocação",
    "rede+col+varpro": "+ campo por mínimos quadrados (varpro)",
    "completo": "+ L-BFGS (PINN melhorado)",
    "completo_f64": "+ float64 e L-BFGS longo",
    "completo_r0zero": "PINN melhorado com estado inicial |0>",
}
files = [(n, os.path.join(R, f"campo_{n}_s0.npy")) for n in ordem]
files = [(n, f) for n, f in files if os.path.exists(f)]
fig, axes = plt.subplots(len(files), 1, figsize=(7, 1.9 * len(files)), sharex=True)
for ax, (n, f) in zip(np.atleast_1d(axes), files):
    t, ftrue, fpred = np.load(f)
    e = np.linalg.norm(fpred - ftrue) / np.linalg.norm(ftrue)
    ax.plot(t, ftrue, "k-", lw=2, label="verdadeiro")
    ax.plot(t, fpred, "r--", lw=1.5, label="PINN")
    ax.set_title(f"{titulos.get(n, n)}   —   erro L2 rel = {e:.3f}", fontsize=9)
    ax.set_ylim(-2.2, 2.4)
np.atleast_1d(axes)[0].legend(fontsize=8, loc="upper right")
np.atleast_1d(axes)[-1].set_xlabel("t")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "figuras", "pinn_ablation.png"), dpi=120)
print("figura salva")
