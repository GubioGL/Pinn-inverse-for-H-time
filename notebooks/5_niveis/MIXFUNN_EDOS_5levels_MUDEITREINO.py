import torch as tc
import numpy as np
import matplotlib.pyplot as plt
import mixfunn as my

import math

from fun import Rede,SIN
from tqdm import tqdm
from torch.nn.utils import prune
from qutip import destroy, basis, mesolve

def simulate_transmon(t_points, fq, alfa, field_fn, N_levels):
    a = destroy(N_levels)
    adag = a.dag()
    Xop = adag + a

    H0 = fq * adag * a + (alfa / 2.0)*(a.dag() +a)*(a.dag() +a)*(a.dag() +a)*(a.dag() +a)
    #H0 = fq * a.dag() * a + (alfa/2) * a.dag() * a.dag() * a * a
    
    # Time-dependent drive: H = [H0, [Xop, field_array]]
    field_array = field_fn(t_points)
    H = [H0, [Xop, field_array]]

    psi0 = basis(N_levels, 0)
    proj_ops = [basis(N_levels, i) * basis(N_levels, i).dag() for i in range(N_levels)]

    result = mesolve(H, psi0, t_points, [], proj_ops)
    expect = np.array(result.expect).T
    return tc.tensor(expect, dtype=tc.float32, device=device)

def compute_term1_batched(y, time):
    n_out = y.shape[1]
    grad_outputs = tc.eye(n_out, device=y.device, dtype=y.dtype)[:, None, :].expand(-1, y.shape[0], -1)
    grads = tc.autograd.grad(outputs=y, inputs=time, grad_outputs=grad_outputs,
                              create_graph=True, is_grads_batched=True)[0]
    return grads.squeeze(-1).transpose(0, 1)

def compute_expects(y):
    return [y[:, i:i+1]**2 + y[:, i+5:i+6]**2 for i in range(5)]

def compute_loss_edo(term1, y, wt, E_levels, alfa):
    s2, s3, s6 = math.sqrt(2), math.sqrt(3), math.sqrt(6)
    E0, E1, E2, E3, E4 = E_levels
    I = [y[:, i:i+1] for i in range(5)]
    R = [y[:, i:i+1] for i in range(5, 10)]

    loss  = (term1[:, 0:1] + (E0*R[0] + 1.5*alfa*R[0] + 3*s2*alfa*R[2] + s6*alfa*R[4] + wt*R[1]))**2
    loss += (term1[:, 1:2] + (E1*R[1] + 7.5*alfa*R[1] + 5*s6*alfa*R[3] + wt*(R[0] + s2*R[2])))**2
    loss += (term1[:, 2:3] + (E2*R[2] + 3*s2*alfa*R[0] + 19.5*alfa*R[2] + 9*s3*alfa*R[4] + wt*(s2*R[1] + s3*R[3])))**2
    loss += (term1[:, 3:4] + (E3*R[3] + 5*s6*alfa*R[1] + 27.5*alfa*R[3] + wt*(s3*R[2] + 2*R[4])))**2
    loss += (term1[:, 4:5] + (E4*R[4] + s6*alfa*R[0] + 9*s3*alfa*R[2] + 14*alfa*R[4] + wt*(2*R[3])))**2

    loss += (term1[:, 5:6] - (E0*I[0] + 1.5*alfa*I[0] + 3*s2*alfa*I[2] + s6*alfa*I[4] + wt*I[1]))**2
    loss += (term1[:, 6:7] - (E1*I[1] + 7.5*alfa*I[1] + 5*s6*alfa*I[3] + wt*(I[0] + s2*I[2])))**2
    loss += (term1[:, 7:8] - (E2*I[2] + 3*s2*alfa*I[0] + 19.5*alfa*I[2] + 9*s3*alfa*I[4] + wt*(s2*I[1] + s3*I[3])))**2
    loss += (term1[:, 8:9] - (E3*I[3] + 5*s6*alfa*I[1] + 27.5*alfa*I[3] + wt*(s3*I[2] + 2*I[4])))**2
    loss += (term1[:, 9:10] - (E4*I[4] + s6*alfa*I[0] + 9*s3*alfa*I[2] + 14*alfa*I[4] + wt*(2*I[3])))**2

    return loss.mean()

def compute_loss_cc(y):
    target = tc.zeros(10, device=y.device)
    target[5] = 1.0  # psi(t=0) = |0>  =>  R0=1, resto=0
    return tc.abs(y[0] - target).sum()

def compute_loss_data(expects, data, weights, squared=False):
    loss = tc.zeros(1, device=data.device)
    for i, (e, w) in enumerate(zip(expects, weights)):
        diff = e - data[:, i:i+1]
        loss = loss + w * (diff**2 if squared else diff.abs())
    return loss.mean()

def plot_losses(LOSS1, LOSS2, LOSS3):
    plt.plot(LOSS1, 'r-', label="loss_edo")
    plt.plot(LOSS2, label="LOSS_cc")
    plt.plot(LOSS3, label="LOSS_data")
    plt.yscale("log")
    plt.legend()
    plt.show()

def plot_populations(time_np, data, X_,save=False):
    colors = ["b", "k", "g", "r", "m"]
    plt.figure(figsize=(10, 6))
    for i, c in enumerate(colors):
        plt.plot(time_np, data[:, i].cpu(), c, label=f'Nível |{i}>')
        plt.plot(time_np, X_[:, i]**2 + X_[:, i+5]**2, c + '--')
    plt.title('Evolução das Populações do Transmon (5 níveis)')
    plt.xlabel('Tempo (ns)')
    plt.ylabel('População')
    plt.legend()
    plt.grid(True)
    if save == True:
        plt.savefig("populations.png")
    plt.show()

def prune_layer(layer_p, linear_module, prune_amount=0.15, threshold=0.02, keep_fraction=0.8):
    prune.l1_unstructured(linear_module, 'weight', amount=prune_amount)
    prune.l1_unstructured(linear_module, 'bias', amount=prune_amount)

    p_norm = tc.abs(layer_p.data) / tc.sum(tc.abs(layer_p.data))
    layer_p.data = tc.where(p_norm < threshold, tc.zeros_like(layer_p.data), layer_p.data)

    p_norm = tc.abs(layer_p.data) / tc.sum(tc.abs(layer_p.data))
    _, indices = tc.sort(p_norm)
    n = indices.shape[1]
    cutoff = int((1 - keep_fraction) * n)
    mask = tc.zeros(n, device=layer_p.device)
    for k in range(n):
        if k > cutoff:
            mask[indices[0][k]] = 1.0
    layer_p.data = mask * layer_p.data
    print(layer_p.data)
    return mask

def make_time_tensor(tfinal, N, device):
    return tc.linspace(0, tfinal, N, dtype=tc.float32, device=device,
                       requires_grad=True).reshape(-1, 1)

def train(model, opt, time, data, E_levels, alfa, epocas, device,wt,
          warmup=0, data_weights_warmup=None, data_weights=None,
          squared=False, masks=None):
    
    LOSS, LOSS1, LOSS2, LOSS3 = [], [], [], []

    for epoch in tqdm(range(epocas)):
        if masks is not None:
            model.layers1.p.data = masks[0] * model.layers1.p.data
            model.layers2.p.data = masks[1] * model.layers2.p.data

        y = model(time)
        _wt = wt(time)
        term1 = compute_term1_batched(y, time)
        expects = compute_expects(y)

        loss_edo = compute_loss_edo(term1, y, _wt, E_levels, alfa)
        loss_cc = compute_loss_cc(y)

        in_warmup = warmup > 0 and epoch < warmup
        weights = data_weights_warmup if in_warmup else data_weights
        loss_data = compute_loss_data(expects, data, weights, squared=squared and not in_warmup)

        loss_i = loss_data if in_warmup else loss_edo + loss_cc + loss_data

        opt.zero_grad()
        loss_i.backward()
        opt.step()

        LOSS1.append(loss_edo.detach().cpu().numpy())
        LOSS2.append(loss_cc.detach().cpu().numpy())
        LOSS.append(loss_i.detach().cpu().numpy())
        LOSS3.append(loss_data.detach().cpu().numpy())

    return LOSS, LOSS1, LOSS2, LOSS3

def Field(t,tfinal):
    return (
        1.5 * np.sin(2.0 * t) - 0.3 * np.cos(3 * t - 0.5) + 0.4 * np.sin(7.0 * t)
    ) * np.sin(np.pi * t / tfinal) ** 2

if __name__ == "__main__":

    #device = "cpu" 
    device =tc.device("cuda" if tc.cuda.is_available() else "cpu")
    print(f"Dispositivo em uso: {device}")
    
    
    tfinal, N, Nedo = 1*np.pi, 1000, 10
    f_q, alfa = 5.0, -0.3
    E_levels = [i * f_q for i in range(5)]


    time_np = np.linspace(0, tfinal, N)
    data = simulate_transmon(time_np, f_q, alfa, lambda t: Field(t, tfinal), 5).to(device)
    
    neuronio = [50, 50]    
    p_vector = Rede(
        neuronio=neuronio,
        input_=1,
        output_=1,
        activation=[tc.nn.Tanh()] * len(neuronio)
    ).to(device)
    
    #wt = tc.ones((N, 1), device=device)

    X_vector = my.MixFunn(N_layers=2, N_neuro=5, N_output=Nedo,
                          second_order_function=True).to(device)
    time = make_time_tensor(tfinal, N, device)

    # Fase 1: warm-up nos dados, depois treino completo com física
    # opt = tc.optim.Adam(X_vector.parameters(), lr=0.01, betas=(0.95, 0.95))

    opt = tc.optim.Adam(list(X_vector.parameters()) + list(p_vector.parameters()), lr=0.01)
    _, LOSS1, LOSS2, LOSS3 = train(
        X_vector, opt, time, data, E_levels, alfa, epocas=50000, device=device,
        warmup=10000,
        data_weights_warmup=[1, 1,   1,   1,   1],
        data_weights=        [1, 1, 100, 100, 100],
        wt=p_vector
    )
    #plot_losses(LOSS1, LOSS2, LOSS3)

    # with tc.no_grad():
    #     X_ = X_vector(time).detach().cpu().numpy()
    #plot_populations(time_np, data, X_)

    # Pruning
    mask1 = prune_layer(X_vector.layers1.p, X_vector.layers1.project1.linear)
    mask2 = prune_layer(X_vector.layers2.p, X_vector.layers2.project1.linear)

    # Fase 2: refinamento pós-pruning com MSE e pesos progressivos
    time = make_time_tensor(tfinal, N, device)
    opt = tc.optim.Adam(list(X_vector.parameters()) + list(p_vector.parameters()), lr=1e-3)
    _, LOSS1, LOSS2, LOSS3 = train(
        X_vector, opt, time, data, E_levels, alfa, epocas=100000, device=device,
        warmup=20000,
        data_weights_warmup=[1, 1,   1,   1,   1],
        data_weights=        [1, 1, 50, 50, 50],
        squared=True,
        masks=(mask1, mask2),
        wt=p_vector
    )
    plot_losses(LOSS1, LOSS2, LOSS3)

    with tc.no_grad():
        X_ = X_vector(time).detach().cpu().numpy()
    plot_populations(time_np, data, X_, save=True)
    
    # ============================================================
    # COMPARAÇÃO: CAMPO REAL vs CAMPO APRENDIDO PELA REDE
    # ============================================================
    with tc.no_grad():
        field_pred = p_vector(time).cpu().numpy().flatten()

    field_true = Field(time_np, tfinal)

    plt.figure(figsize=(10, 4))
    plt.plot(time_np, field_true, label="Field(t) real", color="blue")
    plt.plot(time_np, field_pred, "--", label="p_vector(t) aprendido", color="red")
    plt.xlabel("t")
    plt.ylabel("Campo")
    plt.legend()
    plt.tight_layout()
    plt.savefig("campo_pinn_vs_real.png", dpi=150)
    plt.show()
