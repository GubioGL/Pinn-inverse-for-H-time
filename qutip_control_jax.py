import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import qutip_jax as qj
from jax import jit, value_and_grad
from qutip import ( CoreOptions, basis, mesolve, sigmax, sigmay, sigmaz, sigmam,)

qj.set_as_default()

# ============================================================
# 1. Parâmetros físicos
# ============================================================

gamma = 0.0          # comece fechado; depois pode colocar gamma = 0.1
omega_fixed = 1.0    # omega fixo durante o treinamento
theta_true = 1.35    # parâmetro verdadeiro usado para gerar os dados

tfinal = 2.0 * jnp.pi
N = 100
tlist = jnp.linspace(0.0, tfinal, N)


# ============================================================
# 2. Coeficientes JAX do Hamiltoniano
#    H(t) = 0.5 * theta * sigma_z + 0.5 * cos(omega t) * sigma_x
# ============================================================

@jit
def theta_coeff(t, theta, **kwargs):
    return 0.5 * theta


@jit
def drive_coeff(t, omega, **kwargs):
    return 0.5 * jnp.cos(omega * t)


# ============================================================
# 3. Sistema quântico
# ============================================================

with CoreOptions(default_dtype="jax"):
    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()
    sm = sigmam()

    # estado inicial: cos(pi/4)|0> + sin(pi/4)|1>
    theta0_state = jnp.pi / 4.0
    phi0_state = 0.0

    psi0 = (
        jnp.cos(theta0_state) * basis(2, 0)
        + jnp.sin(theta0_state) * jnp.exp(1j * phi0_state) * basis(2, 1)
    )

    # observáveis usados na loss
    e_ops = [sx, sy, sz]

    # colapsos opcionais
    c_ops = [] if gamma == 0.0 else [jnp.sqrt(gamma) * sm]

    # Hamiltoniano parametrizado
    H = [
        0.0 * sz,
        [sz, theta_coeff],
        [sx, drive_coeff],
    ]


solver_options = {
    "method": "diffrax",
    "normalize_output": False,
    "progress_bar": False,
}


# ============================================================
# 4. Função diferenciável: theta -> valores esperados
# ============================================================

def simulate_expect(theta, omega):
    result = mesolve( H, psi0, tlist, c_ops=c_ops,e_ops=e_ops,
        args={
            "theta": theta,
            "omega": omega,
        },
        options=solver_options,
    )

    # shape: (N_tempos, N_observaveis)
    expect = jnp.stack(
        [jnp.asarray(obs).real for obs in result.expect],
        axis=1,
    )

    return expect


# ============================================================
# 5. Geração dos dados sintéticos
# ============================================================

data_clean = simulate_expect(theta_true, omega_fixed)

# opcional: adicionar ruído
key = jax.random.PRNGKey(123)
noise_level = 0.0

data = data_clean + noise_level * jax.random.normal(
    key,
    shape=data_clean.shape,
)


# ============================================================
# 6. Loss supervisionada
# ============================================================

def loss_fn(params, data):
    theta = params["theta"]
    pred = simulate_expect(theta, omega_fixed)
    loss = jnp.mean((pred - data) ** 2)

    return loss


# ============================================================
# 7. Treinamento de theta
# ============================================================

params = {"theta": jnp.array(0.4),}

learning_rate = 1e-2
epochs = 300

loss_history = []
theta_history = []

for epoch in range(epochs):
    loss_value, grads = value_and_grad(loss_fn)(params, data)

    params = jax.tree_util.tree_map(
        lambda p, g: p - learning_rate * g,
        params,
        grads,
    )

    loss_history.append(float(loss_value))
    theta_history.append(float(params["theta"]))

    if epoch % 25 == 0:
        print(
            f"epoch={epoch:04d} | "
            f"loss={float(loss_value):.6e} | "
            f"theta={float(params['theta']):.8f}"
        )

theta_learned = params["theta"]

print("\nResultado final:")
print("theta_true    =", float(theta_true))
print("theta_learned =", float(theta_learned))


# ============================================================
# 8. Comparação final
# ============================================================

pred_final = simulate_expect(theta_learned, omega_fixed)

plt.figure()
plt.plot(loss_history)
plt.yscale("log")
plt.xlabel("Época")
plt.ylabel("MSE")
plt.title("Treinamento de theta")
plt.show()


labels = [
    r"$\langle \sigma_x \rangle$",
    r"$\langle \sigma_y \rangle$",
    r"$\langle \sigma_z \rangle$",
]

for i, label in enumerate(labels):
    plt.figure()
    plt.plot(tlist, data[:, i], "o", label="dados")
    plt.plot(tlist, pred_final[:, i], "-", label="solver treinado")
    plt.xlabel("t")
    plt.ylabel(label)
    plt.legend()
    plt.title(f"Comparação para {label}")
    plt.show()


plt.figure()
plt.plot(theta_history)
plt.axhline(float(theta_true), linestyle="--", label=r"$\theta_{\rm true}$")
plt.xlabel("Época")
plt.ylabel(r"$\theta$")
plt.legend()
plt.title("Histórico do parâmetro aprendido")
plt.show()