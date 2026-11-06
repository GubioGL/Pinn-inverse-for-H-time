import jax
import jax.numpy as jnp
import jax.flatten_util
import matplotlib.pyplot as plt
import qutip_jax as qj
from jax import jit, value_and_grad
from qutip import (CoreOptions, basis, mesolve, sigmax, sigmay, sigmaz, sigmam)
from qutip import Options

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
# 2. Rede neural: drive_coeff(t) = NN(t ; params)
#    Arquitetura: 1 -> 50 -> 50 -> 1, ativação tanh
# ============================================================

def init_nn_params(key):
    k1, k2, k3 = jax.random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 1)
    scale2 = jnp.sqrt(2.0 / 50)
    return {
        "W1": jax.random.normal(k1, (1, 50)) * scale1,
        "b1": jnp.zeros(50),
        "W2": jax.random.normal(k2, (50, 50)) * scale2,
        "b2": jnp.zeros(50),
        "W3": jax.random.normal(k3, (50, 1)) * 0.01,
        "b3": jnp.zeros(1),
    }


@jit
def nn_forward(nn_params, t):
    x = jnp.atleast_1d(jnp.asarray(t))
    x = jnp.tanh(x @ nn_params["W1"] + nn_params["b1"])
    x = jnp.tanh(x @ nn_params["W2"] + nn_params["b2"])
    return (x @ nn_params["W3"] + nn_params["b3"]).squeeze()


# ============================================================
# 3. Coeficiente do Hamiltoniano fixo
# ============================================================

@jit
def theta_coeff(t, theta, **kwargs):
    return 0.5 * theta


# ============================================================
# 4. Sistema quântico
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
# 5. Inicializar NN, criar drive_coeff e Hamiltoniano
#    Os parâmetros da NN são achatados em um vetor 1-D para
#    passar pelo sistema de args do qutip de forma transparente
# ============================================================

key_init = jax.random.PRNGKey(42)
nn_params_init = init_nn_params(key_init)

# ravel_pytree: NN params <-> vetor flat 1-D (diferenciável)
nn_flat_init, unravel_nn = jax.flatten_util.ravel_pytree(nn_params_init)


def drive_coeff(t, nn_flat, **kwargs):
    return nn_forward(unravel_nn(nn_flat), t)


H = [
    [sz, theta_coeff],
    [sx, drive_coeff],
]

solver_options = {
    "method": "diffrax",
    "normalize_output": False,
    "progress_bar": False,
    "rtol": 1e-5,
    "atol": 1e-6,
}


# ============================================================
# 6. Função diferenciável: (theta, nn_flat) -> valores esperados
# ============================================================

def simulate_expect(theta, nn_flat):
    result = mesolve(
        H, psi0, tlist, c_ops=c_ops, e_ops=e_ops,
        args={"theta": theta, "nn_flat": nn_flat},
        options=solver_options,
    )
    return jnp.stack(
        [jnp.asarray(obs).real for obs in result.expect],
        axis=1,
    )


# ============================================================
# 7. Dados sintéticos gerados com o drive verdadeiro cos(t)
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
    return jnp.stack(
        [jnp.asarray(obs).real for obs in result.expect],
        axis=1,
    )


data_clean = simulate_expect_true(theta_true, omega_fixed)

key_noise = jax.random.PRNGKey(123)
noise_level = 0.0
data = data_clean + noise_level * jax.random.normal(key_noise, shape=data_clean.shape)


# ============================================================
# 8. Loss
# ============================================================

def loss_fn(params, data):
    pred = simulate_expect(params["theta"], params["nn_flat"])
    return jnp.mean((pred - data) ** 2)


# ============================================================
# 9. Otimizador Adam (implementado em JAX puro)
# ============================================================

def adam_init(params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": m, "v": v, "t": 0, "lr": lr, "beta1": beta1, "beta2": beta2, "eps": eps}


def adam_step(params, grads, state):
    t = state["t"] + 1
    b1, b2, eps, lr = state["beta1"], state["beta2"], state["eps"], state["lr"]

    m = jax.tree_util.tree_map(lambda mi, gi: b1 * mi + (1 - b1) * gi, state["m"], grads)
    v = jax.tree_util.tree_map(lambda vi, gi: b2 * vi + (1 - b2) * gi**2, state["v"], grads)

    m_hat = jax.tree_util.tree_map(lambda mi: mi / (1 - b1**t), m)
    v_hat = jax.tree_util.tree_map(lambda vi: vi / (1 - b2**t), v)

    params = jax.tree_util.tree_map(
        lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps),
        params, m_hat, v_hat,
    )
    return params, {**state, "m": m, "v": v, "t": t}


# ============================================================
# 10. Treinamento
# ============================================================

params = {
    "theta": jnp.array(0.4),
    "nn_flat": nn_flat_init,
}

epochs = 500
opt_state = adam_init(params, lr=1e-3)

# Compilar UMA VEZ — evita retrace a cada época
grad_fn = jit(value_and_grad(loss_fn))

loss_history = []
theta_history = []

for epoch in range(epochs):
    loss_value, grads = grad_fn(params, data)
    params, opt_state = adam_step(params, grads, opt_state)

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
# 11. Comparação final
# ============================================================

pred_final = simulate_expect(theta_learned, nn_flat_learned)

# Campo externo: NN aprendida vs. cos verdadeiro
drive_true = jax.vmap(lambda t: 0.5 * jnp.cos(omega_fixed * t))(tlist)
nn_params_learned = unravel_nn(nn_flat_learned)
drive_nn = jax.vmap(lambda t: nn_forward(nn_params_learned, t))(tlist)

plt.figure()
plt.plot(loss_history)
plt.yscale("log")
plt.xlabel("Época")
plt.ylabel("MSE")
plt.title("Treinamento")
plt.show()

plt.figure()
plt.plot(tlist, drive_true, label="drive verdadeiro: 0.5·cos(t)")
plt.plot(tlist, drive_nn, "--", label="NN aprendida")
plt.xlabel("t")
plt.ylabel("drive_coeff(t)")
plt.legend()
plt.title("Campo externo: verdadeiro vs. NN")
plt.show()

labels = [r"$\langle\sigma_x\rangle$", r"$\langle\sigma_y\rangle$", r"$\langle\sigma_z\rangle$"]
for i, label in enumerate(labels):
    plt.figure()
    plt.plot(tlist, data[:, i], "o", ms=2, label="dados")
    plt.plot(tlist, pred_final[:, i], "-", label="NN treinada")
    plt.xlabel("t")
    plt.ylabel(label)
    plt.legend()
    plt.title(f"Comparação {label}")
    plt.show()

plt.figure()
plt.plot(theta_history)
plt.axhline(float(theta_true), linestyle="--", label=r"$\theta_{\rm true}$")
plt.xlabel("Época")
plt.ylabel(r"$\theta$")
plt.legend()
plt.title("Histórico de θ")
plt.show()
