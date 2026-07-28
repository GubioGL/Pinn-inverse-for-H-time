import jax
import jax.numpy as jnp
import jax.flatten_util
import matplotlib.pyplot as plt
import qutip_jax as qj
import optax
from flax import linen as nn
from jax import jit, value_and_grad
from qutip import CoreOptions, basis, mesolve, sigmax, sigmay, sigmaz, sigmam
from typing import Sequence, Callable

qj.set_as_default()

# ============================================================
# 1. Parâmetros físicos
# ============================================================

gamma = 0.0
omega_fixed = 1.0
theta_true = 1.35

tfinal = 2.0 * jnp.pi
N = 200
tlist = jnp.linspace(0.0, tfinal, N)


# ============================================================
# 2. Rede neural (Flax) — mesmo estilo de rede_jax.py
#    Modela o campo externo drive_coeff(t)
# ============================================================

class Rede(nn.Module):
    neuronio: Sequence[int]
    activation: Sequence[Callable]
    output_: int = 1

    @nn.compact
    def __call__(self, x):
        for n, act in zip(self.neuronio, self.activation):
            x = nn.Dense(n)(x)
            x = act(x)
        return nn.Dense(self.output_)(x)


model = Rede(
    neuronio=[50, 50],
    activation=[jnp.tanh, jnp.tanh],
    output_=1,
)

key_init = jax.random.PRNGKey(42)
nn_params_init = model.init(key_init, jnp.zeros((1,)))["params"]

# ravel_pytree: parâmetros Flax <-> vetor flat 1-D (diferenciável via QuTiP args)
nn_flat_init, unravel_nn = jax.flatten_util.ravel_pytree(nn_params_init)


def apply_model(nn_flat, t):
    x = jnp.atleast_1d(jnp.asarray(t))
    return model.apply({"params": unravel_nn(nn_flat)}, x).squeeze()


# ============================================================
# 3. Sistema quântico
# ============================================================

with CoreOptions(default_dtype="jax"):
    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()
    sm = sigmam()

    theta0_state = jnp.pi / 4.0
    psi0 = (
        jnp.cos(theta0_state) * basis(2, 0)
        + jnp.sin(theta0_state) * basis(2, 1)
    )

    e_ops = [sx, sy, sz]
    c_ops = [] if gamma == 0.0 else [jnp.sqrt(gamma) * sm]


# ============================================================
# 4. Coeficientes do Hamiltoniano
# ============================================================

def theta_coeff(t, theta, **kwargs):
    return 0.5 * theta


def drive_coeff(t, nn_flat, **kwargs):
    return apply_model(nn_flat, t)


H = [[sz, theta_coeff], [sx, drive_coeff]]

solver_options = {
    "method": "diffrax",
    "normalize_output": False,
    "progress_bar": False,
    "rtol": 1e-5,
    "atol": 1e-6,
}


# ============================================================
# 5. Dados sintéticos com o drive verdadeiro 0.5·cos(t)
# ============================================================

def drive_coeff_true(t, omega, **kwargs):
    return 0.5 * jnp.cos(omega * t)


H_true = [[sz, theta_coeff], [sx, drive_coeff_true]]


def simulate_expect_true(theta, omega):
    result = mesolve(
        H_true, psi0, tlist, c_ops=c_ops, e_ops=e_ops,
        args={"theta": theta, "omega": omega},
        options=solver_options,
    )
    return jnp.stack([jnp.asarray(obs).real for obs in result.expect], axis=1)


data_clean = simulate_expect_true(theta_true, omega_fixed)
key_noise = jax.random.PRNGKey(123)
noise_level = 0.0
data = data_clean + noise_level * jax.random.normal(key_noise, shape=data_clean.shape)


# ============================================================
# 6. Simulação e loss
# ============================================================

def simulate_expect(theta, nn_flat):
    result = mesolve(
        H, psi0, tlist, c_ops=c_ops, e_ops=e_ops,
        args={"theta": theta, "nn_flat": nn_flat},
        options=solver_options,
    )
    return jnp.stack([jnp.asarray(obs).real for obs in result.expect], axis=1)


def loss_fn(params, data):
    pred = simulate_expect(params["theta"], params["nn_flat"])
    return jnp.mean((pred - data) ** 2)


# ============================================================
# 7. Otimizador Optax Adam — análogo ao optim.Adam do PyTorch
#    Combina theta + parâmetros da NN em um único pytree
# ============================================================

params = {
    "theta": jnp.array(0.4),
    "nn_flat": nn_flat_init,
}

tx = optax.adam(learning_rate=1e-3)
opt_state = tx.init(params)

grad_fn = jit(value_and_grad(loss_fn))


# ============================================================
# 8. Passo de treino — análogo a opt.zero_grad() / backward() / opt.step()
# ============================================================

def train_step(params, opt_state, data):
    loss, grads = grad_fn(params, data)
    updates, opt_state = tx.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


# ============================================================
# 9. Treinamento
# ============================================================

epochs = 500
loss_history = []
theta_history = []

for epoch in range(epochs):
    params, opt_state, loss_value = train_step(params, opt_state, data)

    loss_history.append(float(loss_value))
    theta_history.append(float(params["theta"]))

    if epoch % 50 == 0:
        print(
            f"epoch={epoch:04d} | "
            f"loss={float(loss_value):.6e} | "
            f"theta={float(params['theta']):.6f}"
        )

theta_learned = params["theta"]
nn_flat_learned = params["nn_flat"]

print("\nResultado final:")
print("theta_true    =", float(theta_true))
print("theta_learned =", float(theta_learned))


# ============================================================
# 10. Comparação — campo externo real vs. NN aprendida
# ============================================================

pred_final = simulate_expect(theta_learned, nn_flat_learned)

drive_true_vals = jax.vmap(lambda t: 0.5 * jnp.cos(omega_fixed * t))(tlist)
drive_nn_vals = jax.vmap(lambda t: apply_model(nn_flat_learned, t))(tlist)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(loss_history)
axes[0, 0].set_yscale("log")
axes[0, 0].set_xlabel("Época")
axes[0, 0].set_ylabel("MSE")
axes[0, 0].set_title("Loss de treinamento")

axes[0, 1].plot(tlist, drive_true_vals, label="0.5·cos(t) [verdadeiro]")
axes[0, 1].plot(tlist, drive_nn_vals, "--", label="NN aprendida")
axes[0, 1].set_xlabel("t")
axes[0, 1].set_ylabel("drive_coeff(t)")
axes[0, 1].legend()
axes[0, 1].set_title("Campo externo: verdadeiro vs. NN")

axes[1, 0].plot(theta_history)
axes[1, 0].axhline(float(theta_true), linestyle="--", label=r"$\theta_{\rm true}$")
axes[1, 0].set_xlabel("Época")
axes[1, 0].set_ylabel(r"$\theta$")
axes[1, 0].legend()
axes[1, 0].set_title(r"Histórico de $\theta$")

labels = [r"$\langle\sigma_x\rangle$", r"$\langle\sigma_y\rangle$", r"$\langle\sigma_z\rangle$"]
for i, label in enumerate(labels):
    axes[1, 1].plot(tlist, data[:, i], "o", ms=2, label=f"dados {label}")
    axes[1, 1].plot(tlist, pred_final[:, i], "-", label=f"NN {label}")
axes[1, 1].set_xlabel("t")
axes[1, 1].legend(fontsize=7)
axes[1, 1].set_title("Valores esperados")

plt.tight_layout()
plt.savefig("resultado_flax.png", dpi=150)
plt.show()
