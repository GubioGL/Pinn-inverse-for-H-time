import torch as tc
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
# print(sm * sm.dag()*sm)

# Define the loss function
def mse_loss(y_pred, y_true):
    return tc.mean((y_pred - y_true)**2)

# Define the loss function
def msa_loss(y_pred, y_true):
    return tc.mean(abs(y_pred - y_true))

def diagonal(M):
    traco_ = M.diagonal(offset=0,dim1=1, dim2=2)
    return traco_

def expected(A,B):
    return diagonal(A@B)

def commutator(A, B):
    return tc.matmul(A,B) - tc.matmul(B,A)

def Loss_EDO(H_,rho_rvetor,rho_ivetor,tempo,base_rho):
    rho_  = rho_rvetor.reshape((rho_rvetor.shape[0],base_rho,base_rho)) + 1j*rho_ivetor.reshape((rho_rvetor.shape[0],base_rho,base_rho))
    #  Calculando o Comutador do Hamiltoniano com a matriz densidade. 
    #   O resultado deve ser um tensor de 100 linhas e (2,2).
    H_rho_R = (commutator(H_.real,rho_.imag)- commutator(H_.imag,rho_.real))
    H_rho_I = (commutator(H_.imag,rho_.imag)- commutator(H_.real,rho_.real))
    #converter para vetor
    H_rho_R = H_rho_R.reshape((len(tempo),base_rho**2))
    H_rho_I = H_rho_I.reshape((len(tempo),base_rho**2))
    # Calculando o gradiente de drho_dt separando a parte real e imagina
    # Em seguida, iremos  calcular o Erro quadrático médio da equaçao de Von Neumann
    loss_edo = 0
    for i in range(rho_rvetor.shape[1]):
        drho_dt_real = tc.autograd.grad(outputs = rho_rvetor[:,i], 
                            inputs = tempo,
                            grad_outputs = tc.ones_like(rho_rvetor[:,i]),
                            retain_graph = True,
                            create_graph = True
                            )[0][:,0]

        drho_dt_imag = tc.autograd.grad(outputs = rho_ivetor[:,i], 
                            inputs = tempo,
                            grad_outputs = tc.ones_like(rho_ivetor[:,i]),
                            retain_graph = True,
                            create_graph = True
                            )[0][:,0]
        
        # Von Neuman equation
        loss_edo += tc.mean( (drho_dt_real - H_rho_R[:,i])**2 + (drho_dt_imag - H_rho_I[:,i])**2 )
    return loss_edo

def expected_plot(rho_, O_, expected_data, time_, select_observables=None, save_plot=None):
    """
    Grafica los valores esperados ⟨O⟩ para cada observable, calculados a partir de rho_.

    rho_:              [N, d, d]  matriz densidad compleja
    O_:                [n_obs, d, d] operadores observables
    expected_data:     [N, n_obs] datos reales
    time_:             [N, 1] tiempos
    select_observables: lista de índices de observables (int), default = todos
    save_plot:         ruta opcional para guardar la figura
    """
    #print("este")

    N, d, _ = rho_.shape
    n_obs = O_.shape[0]

    if select_observables is None:
        select_observables = list(range(n_obs))

    n_plots = len(select_observables)
    fig, axes = plt.subplots(n_plots, 1, figsize=(8, 3 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    for i, obs_idx in enumerate(select_observables):
        Oi = O_[obs_idx]                     # Operador observable [d,d]
        #print(rho_.shape, Oi.shape)
        v_pred = expected(rho_, Oi).real     # ⟨O⟩ = Tr[ρ O], toma parte real
        v_true = expected_data[:, obs_idx].real
        #print(v_pred.shape)
        #print(v_true.shape)

        axes[i].plot(time_.detach().cpu().numpy(),
                     v_pred.detach().cpu().numpy(),
                     "r.", label="Neural Network")
        axes[i].plot(time_.detach().cpu().numpy(),
                     v_true.detach().cpu().numpy(),
                     "k-", label="Data")

        axes[i].set_ylabel(f"⟨O[{obs_idx}]⟩")
        axes[i].legend()
        axes[i].set_title(f"Expected Value ⟨O[{obs_idx}]⟩")

    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    if save_plot:
        fig.savefig(save_plot, dpi=300)
    plt.show()

def plots_rho(rho_NNR=0,rho_NNI=0,rho_data=0 ):
    fig, axs = plt.subplots(nrows=3, ncols=2 , figsize=(12,4), sharex=True)

    im =axs[0,0].imshow(rho_NNR.detach().numpy().T,cmap="jet")
    axs[0,0].set_title(r"$\mathcal{R}(\rho_{NN})$")
    axs[0,0].set_aspect("auto")
    fig.colorbar(im, orientation='vertical')

    im =axs[1,0].imshow(rho_data.real.T.detach().numpy(),cmap="jet")
    axs[1,0].set_title(r"$\mathcal{R}(\hat{\rho}_t)$")
    axs[1,0].set_aspect("auto")
    fig.colorbar(im, orientation='vertical')

    im =axs[2,0].imshow(abs(rho_data.real-rho_NNR).T.detach().numpy(),cmap="jet")
    axs[2,0].set_title(r"$|\mathcal{R}(\hat{\rho}_t) - \mathcal{R}(\rho_t)|$")
    axs[2,0].set_xlabel(r"$t$")
    axs[2,0].set_aspect("auto")
    fig.colorbar(im, orientation='vertical')

    im =axs[0,1].imshow(rho_NNI.detach().numpy().T,cmap="jet")
    axs[0,1].set_title(r"$\mathcal{I}(\rho_{NN})$")
    axs[0,1].set_aspect("auto")
    fig.colorbar(im, orientation='vertical')

    im =axs[1,1].imshow(rho_data.imag.T.detach().numpy(),cmap="jet")
    axs[1,1].set_title(r"$\mathcal{I}(\hat{\rho}_t)$")
    axs[1,1].set_aspect("auto")
    fig.colorbar(im, orientation='vertical')

    im = axs[2,1].imshow(abs(rho_data.imag-rho_NNI).T.detach().numpy() ,cmap="jet")
    axs[2,1].set_title(r"$|\mathcal{I}(\hat{\rho}_{NN}) - \mathcal{I}(\rho_t)|$")
    axs[2,1].set_aspect("auto")
    fig.colorbar(im, orientation='vertical')

    plt.tight_layout()
    plt.show()

def data(lista_J, tfinal, N, device="cpu", state=True):
    """
    Hamiltoniana dependente do tempo:
        H(t) = - B0 * sigma_y  -  B1 * cos(omega * t) * sigma_x

    lista_J = [B0, B1, omega]  (em unidades de frequência; multiplicadas por 2*pi no código)
    """

    # Operadores de Pauli
    sx = qt.sigmax()
    sy = qt.sigmay()
    sz = qt.sigmaz()

    # Non-driving Hamiltonian
    B0 = lista_J[0] * np.pi
    H0 = B0 * sz

    # Driving Hamiltonian
    B1 = lista_J[1] 
    w  = lista_J[2] * np.pi
    H1 = B1 * sx
    args = {"w": w}

    # Forma compatível com sesolve/mesolve
    H_qutip = [H0, [H1, 'cos(w * t)']]

    # Collapse operators
    c_op_list = []
    # Operadores para medição <O>  ->  <σx>, <σy>, <σz>
    O_qobj = [sx, sy, sz]

    # Solução da equação de Schrödinger / mestre
    options = qt.Options(store_states=True)

    # Obtain Time Evolution
    tlist = np.linspace(0,tfinal, N)

    psi0 = qt.basis(2, 0)
    if state:
        print("Using Schrodinger equation solver...")
        #result = qt.sesolve(H_qutip, psi0, tlist, e_ops=O_qobj, args=args, options=options)
    else:
        result = qt.mesolve(H_qutip, psi0, tlist,
                            c_ops=c_op_list, e_ops=O_qobj, args=args, options=options)

    # -----------------------
    # Construir H(t) em cada instante como matriz densa 2x2
    # -----------------------
    H_t_np = []
    for t in tlist:
        H_t = B0 * sz + B1 * np.cos(w * t) * sx
        H_t_np.append(H_t.full())

    H_t_np = np.array(H_t_np, dtype=np.complex64)   # shape (N, 2, 2)

    # Operadores de observável em forma de matriz
    O_op_np = np.array([op.full() for op in O_qobj], dtype=np.complex64)

    # Estados ao longo do tempo, achatados
    y_train_np = np.array(
        [result.states[i].full().flatten() for i in range(N)],
        dtype=np.complex64
    )

    # Valores médios <O>
    expect_np = np.array(result.expect)   # shape (n_ops, N)

    # -----------------------
    # Conversão para tensores do PyTorch
    # -----------------------
    H = tc.tensor(H_t_np, dtype=tc.complex64, device=device)      # (N, 2, 2)
    O_op = tc.tensor(O_op_np, device=device)                      # (3, 2, 2)
    y_train = tc.tensor(y_train_np, device=device)                # (N, 4)
    expect = tc.tensor(expect_np, device=device).transpose(0, 1)  # (N, 3)

    return y_train, expect, H, O_op
