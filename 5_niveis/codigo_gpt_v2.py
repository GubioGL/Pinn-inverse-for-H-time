import math
import numpy as np
import torch as tc
from fun import Rede,SIN
import matplotlib.pyplot as plt
from tqdm import tqdm
from qutip import destroy, basis, mesolve

# ============================================================
# DEVICE
# ============================================================
device = "cpu"  # tc.device("cuda" if tc.cuda.is_available() else "cpu")
print("device =", device)


# ============================================================
# QUTIP: MESMO MODELO EFETIVO DAS EQUAÇÕES USADAS NO PINN
# ============================================================
def simulate_transmon(t_points):
    a = destroy(N_levels)
    adag = a.dag()
    Xop = adag + a

    # Mantendo a mesma convenção do seu caso:
    # H = fq a†a + alfa (a+a†)^4 + Omega (a+a†)
    H0 = fq * adag * a + alfa * (Xop ** 4)
    H_drive = Omega * Xop
    H = H0 + H_drive

    psi0 = basis(N_levels, 0)
    proj_ops = [basis(N_levels, i) * basis(N_levels, i).dag() for i in range(N_levels)]

    result = mesolve(H, psi0, t_points, [], proj_ops)
    expect = np.array(result.expect).T
    return tc.tensor(expect, dtype=tc.float32, device=device)

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================
def get_time_chunks(n_points, n_chunks):
    """
    Divide indices [0, ..., n_points-1] em blocos temporais contíguos.
    """
    edges = np.linspace(0, n_points, n_chunks + 1, dtype=int)
    chunks = []
    for i in range(n_chunks):
        a, b = edges[i], edges[i + 1]
        if b > a:
            chunks.append((a, b))
    return chunks

def gradients_all_outputs(y, x):
    """
    dy_i/dt para todas as saídas da rede.
    """
    grads = []
    for i in range(y.shape[1]):
        gi = tc.autograd.grad(
            outputs=y[:, i],
            inputs=x,
            grad_outputs=tc.ones_like(y[:, i]),
            create_graph=True
        )[0]
        grads.append(gi)
    return tc.cat(grads, dim=1)

def compute_pointwise_residuals(y, dy_dt, wt):
    """
    Retorna o residual quadrático ponto a ponto, somando todas as 10 EDOs.
    """
    R0 = y[:, 0:1]
    R1 = y[:, 1:2]
    R2 = y[:, 2:3]
    R3 = y[:, 3:4]
    R4 = y[:, 4:5]

    I0 = y[:, 5:6]
    I1 = y[:, 6:7]
    I2 = y[:, 7:8]
    I3 = y[:, 8:9]
    I4 = y[:, 9:10]

    dR0 = dy_dt[:, 0:1]
    dR1 = dy_dt[:, 1:2]
    dR2 = dy_dt[:, 2:3]
    dR3 = dy_dt[:, 3:4]
    dR4 = dy_dt[:, 4:5]

    dI0 = dy_dt[:, 5:6]
    dI1 = dy_dt[:, 6:7]
    dI2 = dy_dt[:, 7:8]
    dI3 = dy_dt[:, 8:9]
    dI4 = dy_dt[:, 9:10]

    # EDOs do potencial quartico
    fR0 = -3.0 * alfa * I0 - 6.0 * sqrt2 * alfa * I2 - 2.0 * sqrt6 * alfa * I4 + wt * I1
    fI0 =  3.0 * alfa * R0 + 6.0 * sqrt2 * alfa * R2 + 2.0 * sqrt6 * alfa * R4 - wt * R1

    fR1 = -15.0 * alfa * I1 - 10.0 * sqrt6 * alfa * I3 + E1 * I1 + wt * I0 + sqrt2 * wt * I2
    fI1 =  15.0 * alfa * R1 + 10.0 * sqrt6 * alfa * R3 - E1 * R1 - wt * R0 - sqrt2 * wt * R2

    fR2 = -6.0 * sqrt2 * alfa * I0 - 39.0 * alfa * I2 - 18.0 * sqrt3 * alfa * I4 + E2 * I2 + sqrt2 * wt * I1 + sqrt3 * wt * I3
    fI2 =  6.0 * sqrt2 * alfa * R0 + 39.0 * alfa * R2 + 18.0 * sqrt3 * alfa * R4 - E2 * R2 - sqrt2 * wt * R1 - sqrt3 * wt * R3

    fR3 = -10.0 * sqrt6 * alfa * I1 - 55.0 * alfa * I3 + E3 * I3 + sqrt3 * wt * I2 + 2.0 * wt * I4
    fI3 =  10.0 * sqrt6 * alfa * R1 + 55.0 * alfa * R3 - E3 * R3 - sqrt3 * wt * R2 - 2.0 * wt * R4

    fR4 = -2.0 * sqrt6 * alfa * I0 - 18.0 * sqrt3 * alfa * I2 - 28.0 * alfa * I4 + E4 * I4 + 2.0 * wt * I3
    fI4 =  2.0 * sqrt6 * alfa * R0 + 18.0 * sqrt3 * alfa * R2 + 28.0 * alfa * R4 - E4 * R4 - 2.0 * wt * R3

    res = (dR0 - fR0) ** 2
    res += (dR1 - fR1) ** 2
    res += (dR2 - fR2) ** 2
    res += (dR3 - fR3) ** 2
    res += (dR4 - fR4) ** 2

    res += (dI0 - fI0) ** 2
    res += (dI1 - fI1) ** 2
    res += (dI2 - fI2) ** 2
    res += (dI3 - fI3) ** 2
    res += (dI4 - fI4) ** 2

    return res  # shape [N_time, 1]

def compute_ic_loss(y):
    """
    psi(0)=|0>  =>  c0(0)=1, demais 0
    """
    R0 = y[0, 0]
    R1 = y[0, 1]
    R2 = y[0, 2]
    R3 = y[0, 3]
    R4 = y[0, 4]

    I0 = y[0, 5]
    I1 = y[0, 6]
    I2 = y[0, 7]
    I3 = y[0, 8]
    I4 = y[0, 9]

    loss_ic = (R0 - 1.0) ** 2
    loss_ic += (R1 - 0.0) ** 2
    loss_ic += (R2 - 0.0) ** 2
    loss_ic += (R3 - 0.0) ** 2
    loss_ic += (R4 - 0.0) ** 2

    loss_ic += (I0 - 0.0) ** 2
    loss_ic += (I1 - 0.0) ** 2
    loss_ic += (I2 - 0.0) ** 2
    loss_ic += (I3 - 0.0) ** 2
    loss_ic += (I4 - 0.0) ** 2
    return loss_ic

def compute_data_loss(y, data):
    """
    Supervisionado opcional com populações do QuTiP.
    """
    R = y[:, 0:5]
    I = y[:, 5:10]
    pop = R**2 + I**2

    loss_data = (pop[:, 0:1] - data[:, 0:1])**2
    loss_data += (pop[:, 1:2] - data[:, 1:2])**2
    loss_data += 10.0 * (pop[:, 2:3] - data[:, 2:3])**2
    loss_data += 30.0 * (pop[:, 3:4] - data[:, 3:4])**2
    loss_data += 50.0 * (pop[:, 4:5] - data[:, 4:5])**2
    return loss_data.mean()

def compute_norm_loss(y):
    R = y[:, 0:5]
    I = y[:, 5:10]
    pop = R**2 + I**2
    norm = pop.sum(dim=1, keepdim=True)
    return ((norm - 1.0)**2).mean()

def compute_temporal_losses(y, dy_dt, wt, chunks, lambda_ic=1e3):
    """
    Constrói o vetor:
      L_vec = [L(t0), L(t1), ..., L(t_Nt)]
    onde:
      L(t0) = lambda_ic * IC
      L(ti) = residual médio no bloco temporal i
    """
    pointwise_res = compute_pointwise_residuals(y, dy_dt, wt)
    ic_loss = lambda_ic * compute_ic_loss(y)

    L_list = [ic_loss.reshape(1)]

    for a, b in chunks:
        Li = pointwise_res[a:b].mean()
        L_list.append(Li.reshape(1))

    L_vec = tc.cat(L_list, dim=0)  # shape [n_chunks+1]
    return L_vec, pointwise_res, ic_loss

def compute_causal_weights(L_vec, eps):
    """
    Pesos causais:
      w0 = 1
      wi = exp(-eps * sum_{k < i} L_k)
    sem backprop pelos pesos.
    """
    with tc.no_grad():
        prefix = tc.cumsum(L_vec.detach(), dim=0)
        w = tc.ones_like(L_vec)
        if len(L_vec) > 1:
            w[1:] = tc.exp(-eps * prefix[:-1])
    return w

if "__main__" == __name__:
    # ============================================================
    # PARÂMETROS FÍSICOS
    # ============================================================
    N_levels = 5
    fq = 5.0
    alfa = -0.3
    Omega = 1.0

    tfinal = 2.5
    N_time = 500
    lr = 1e-3

    sqrt2 = math.sqrt(2.0)
    sqrt3 = math.sqrt(3.0)
    sqrt4 = math.sqrt(4.0)
    sqrt6 = math.sqrt(6.0)

    E0 = 0.0
    E1 = fq
    E2 = 2.0 * fq
    E3 = 3.0 * fq
    E4 = 4.0 * fq

    # ============================================================
    # TEMPO
    # ============================================================
    time_np = np.linspace(0.0, tfinal, N_time)

    time = tc.linspace(
        0.0,
        tfinal,
        N_time,
        dtype=tc.float32,
        requires_grad=True,
        device=device
    ).reshape((-1, 1))

    # ============================================================
    # SUA REDE
    # Saída:
    # y[:,0:5] = R0..R4
    # y[:,5:10] = I0..I4
    # ============================================================
    
    neuronio = [20, 20]
    X_vector = Rede(
        neuronio=neuronio,
        input_=1,
        output_=10,
        activation=[SIN()] * len(neuronio)
    ).to(device)
    
    neuronio = [5, 5]    
    p_vector = Rede(
        neuronio=neuronio,
        input_=1,
        output_=1,
        activation=[tc.nn.Tanh()] * len(neuronio)
    ).to(device)
    
    opt = tc.optim.Adam(list(X_vector.parameters()) + list(p_vector.parameters()), lr=lr)
    data = simulate_transmon(time_np)
    # ============================================================
    # HIPERPARÂMETROS DO CAUSAL TRAINING
    # Inspirados no artigo:
    # eps_list = [1e-2, 1e-1, 1e0, 1e1, 1e2]
    # delta ~ 0.99
    # ============================================================
    eps_list = [1e-2, 1e-1, 1e0, 1e1, 1e2]
    delta = 0.99

    # Número de blocos causais no tempo
    # Pode testar 16, 32, 64. Para seu caso, 32 costuma ser um bom início.
    
    Nt_causal = 32
    chunks = get_time_chunks(N_time, Nt_causal)
    
    # iterações máximas por estágio epsilon
    steps_per_eps = 10000

    # pesos globais
    lambda_ic = 1e3
    lambda_data = 1.0
    lambda_norm = 1.0
    
    # ============================================================
    # HISTÓRICO
    # ============================================================
    LOSS_total_hist = []
    LOSS_phys_hist = []
    LOSS_ic_hist = []
    LOSS_data_hist = []
    LOSS_norm_hist = []
    MIN_W_hist = []
    EPS_hist = []

    # ============================================================
    # TREINAMENTO CAUSAL
    # ============================================================
    global_step = 0

    for eps in eps_list:
        print(f"\n=== Iniciando estágio causal com eps = {eps:.1e} ===")

        for step in tqdm(range(steps_per_eps), desc=f"eps={eps:.1e}"):
            wt = p_vector(time)
            y = X_vector(time)
            dy_dt = gradients_all_outputs(y, time)

            # perdas temporais por bloco
            L_vec, pointwise_res, ic_loss_unweighted = compute_temporal_losses(
                y, dy_dt, wt, chunks, lambda_ic=lambda_ic
            )

            # pesos causais (detach!)
            w = compute_causal_weights(L_vec, eps)

            # perda física causal: soma ponderada dos blocos
            loss_phys = (w * L_vec).mean()

            # perdas auxiliares
            loss_data = lambda_data * compute_data_loss(y, data)
            loss_norm = lambda_norm * compute_norm_loss(y)

            # total
            loss = loss_phys + loss_data + loss_norm

            opt.zero_grad()
            loss.backward()
            opt.step()

            # logs
            min_w = w[1:].min().item() if len(w) > 1 else 1.0

            LOSS_total_hist.append(loss.detach().cpu().item())
            LOSS_phys_hist.append(loss_phys.detach().cpu().item())
            LOSS_ic_hist.append((ic_loss_unweighted.detach() / lambda_ic).cpu().item())
            LOSS_data_hist.append(loss_data.detach().cpu().item())
            LOSS_norm_hist.append(loss_norm.detach().cpu().item())
            MIN_W_hist.append(min_w)
            EPS_hist.append(eps)

            global_step += 1

            # stopping criterion do artigo
            if min_w > delta:
                print(f"Stopping criterion atingido em eps={eps:.1e}: min(w)={min_w:.4f}")
                break

    # ============================================================
    # PLOTS DE TREINO
    # ============================================================
    plt.figure(figsize=(9, 5))
    plt.plot(LOSS_phys_hist, label="loss_phys_causal")
    plt.plot(LOSS_ic_hist, label="loss_ic")
    plt.plot(LOSS_data_hist, label="loss_data")
    plt.plot(LOSS_norm_hist, label="loss_norm")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 4))
    plt.plot(LOSS_total_hist, "k-", label="loss_total")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 4))
    plt.plot(MIN_W_hist, label="min temporal weight")
    plt.axhline(delta, color="r", linestyle="--", label=f"delta={delta}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ============================================================
    # COMPARAÇÃO FINAL COM QUTIP
    # ============================================================
    with tc.no_grad():
        y_pred = X_vector(time)

    R_pred = y_pred[:, 0:5]
    I_pred = y_pred[:, 5:10]
    P_pred = (R_pred**2 + I_pred**2).cpu().numpy()
    P_qutip = data.cpu().numpy()

    plt.figure(figsize=(10, 6))
    for i in range(5):
        plt.plot(time_np, P_qutip[:, i], label=f"P{i} qutip")
        plt.plot(time_np, P_pred[:, i], "--", label=f"P{i} pinn")

    plt.xlabel("t")
    plt.ylabel("População")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.show()