import torch as tc
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from main.function import expected, msa_loss, mse_loss, diagonal, Loss_EDO, plots_rho,data_jc,expected_plot, lerning_parameter_hamiltonina
from main.neuralnetwork import Rede, SIN
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

class Train_jc:
    def __init__(self,fockbase,device="cpu"):        
        self.device = device
        self.base_rho    = 2*fockbase
    def _creat_data(self,lista_J,tfinal,N):

        self.rho_train, self.expected_CA,self.hamiltonian,self.Observavel_data= data_jc(
            lista_J=lista_J, tfinal=tfinal, N=N, device=self.device,state=False)
        self.Observavel0 = self.Observavel_data[0]
        self.Observavel1 = self.Observavel_data[1]

    def U_evolution(self, hamiltonian, rho_inicial, tlist, n_field, n_atom, device="cpu"):
        if not tc.is_tensor(tlist):
            tlist = tc.tensor(tlist, device=device, dtype=tc.float64)
        else:
            tlist = tlist.to(device)

        # 1. Preparar as dimensões para Broadcasting
        t_reshaped = tlist.view(-1, 1, 1).to(device)
        
        # Hamiltonian shape: (D, D) -> (1, D, D)
        h_batched = hamiltonian.unsqueeze(0)
        
        # 2. Calcular o expoente para todos os tempos de uma vez
        exponent = -1j * h_batched * t_reshaped
        
        # 3. Exponencial Matricial em Batch
        U_t = tc.matrix_exp(exponent)  # Shape: (N, D, D)
        
        # 4. Calcular U_dagger (transposta conjugada)
        U_dag = U_t.mH 
        
        # 5. Evoluir rho: U @ rho @ U_dag
        rho_expanded = rho_inicial.unsqueeze(0)
        
        # Matmul (bmm) automático pelo PyTorch
        rho_t = U_t @ rho_expanded @ U_dag # Shape: (N, D, D)
        
        ef = tc.einsum('bij,ji->b', rho_t, n_field).real
        ea = tc.einsum('bij,ji->b', rho_t, n_atom).real
        
        return ef, ea, tlist

    def _prepare_input(self,lista_J,tfinal,N):
        self.tfinal_    = tfinal
        self.N          = N
        self.lista_J    = lista_J
        self.t_train    = tc.linspace(
                        0, 
                        self.tfinal_, 
                        self.N,
                        dtype=tc.float32, 
                        requires_grad=True, 
                        device=self.device).reshape((-1, 1))
        
        self._creat_data(lista_J, self.tfinal_, self.N)

    def _initialize_networks(self, neuro_,funçao_de_ativa=None, creat_p=False, N_parametr=1,path=".../", load_net=False):           
        self.real_net = Rede(
            input_      = 1,
            neuronio    = neuro_,
            output_     = self.base_rho**2,
            activation  = funçao_de_ativa,
            creat_p     = creat_p,
            N_of_paramater = N_parametr
            ).to(self.device)
        
        self.imag_net = Rede(
            input_      = 1,
            neuronio    = neuro_,
            output_     = self.base_rho**2,
            activation  = funçao_de_ativa
            ).to(self.device) 
    
        if load_net == True:
            self.real_net = tc.load(path)['real_net']
            self.imag_net = tc.load(path)['imag_net']
            self.real_net.to(device=self.device)
            self.imag_net.to(device=self.device)

    def _initialize_optimizer(self,lr=0.01, step_size=500, gamma=0.9, epocas=5000,path=".../", load_net=False):
        self.lr     = lr
        self.gamma  = gamma
        self.epocas = epocas
        self.LOSS   = []
        self.LOSS_parametr = []
        self.step_size  = step_size
        self.opt = tc.optim.Adam(
            list(self.real_net.parameters()) + list(self.imag_net.parameters()),
            lr=self.lr,
            betas=(0.9, 0.99),
            amsgrad=True,
        )
        self.scheduler = StepLR(self.opt, step_size=self.step_size, gamma=self.gamma)
        if load_net == True:
            self.opt.load_state_dict(tc.load(path)['otimization'])
            self.scheduler.load_state_dict(tc.load(path)['step_lr'])
            self.LOSS = tc.load(path)['loss']
            
    def plot_loss(self):
        plt.subplots(figsize=(4, 4))
        plt.plot(self.LOSS)
        plt.yscale("log")
        plt.show()

    def plot_evaluate(self,lista_J, path=None, load_net=False):
        if load_net == True:
            real_net = self.real_net_load
            imag_net = self.imag_net_load
            real_net.load_state_dict(tc.load(path)['real_net'])
            imag_net.load_state_dict(tc.load(path)['imag_net'])
            real_net.eval()
            y_pred_real = real_net(self.t_train)
            imag_net.eval()
            y_pred_imag = imag_net(self.t_train)
            
            plots_rho(y_pred_real, y_pred_imag, self.rho_train)
            
        else:
            self.real_net.eval()
            self.imag_net.eval()            
            
            ####### Data #######
            self._creat_data(lista_J, self.tfinal_,self.N)
            
            ####### Forward pass #######
            y_pred_real = self.real_net(self.t_train)
            y_pred_imag = self.imag_net(self.t_train)
            
            ####### Criando rho no farmato de matriz #######
            rho  = (y_pred_real + 1j * y_pred_imag).reshape((self.N, self.base_rho, self.base_rho))
            
            plots_rho(y_pred_real, y_pred_imag, self.rho_train)
            expected_plot(
                rho_ = rho,
                O_   = self.Observavel_data,
                expected_data = self.expected_CA,
                time_ = self.t_train)

    def train_parametr(self):

        for _ in tqdm(range(self.epocas)):

            y_pred_real = self.real_net(self.t_train)
            y_pred_imag = self.imag_net(self.t_train)
            rho = (y_pred_real + 1j * y_pred_imag).reshape(
                (len(self.t_train), self.base_rho, self.base_rho)
            )
            Tr_rho_O0 = expected(rho, self.Observavel0).sum(dim=-1).real
            Tr_rho_O1 = expected(rho, self.Observavel1).sum(dim=-1).real
            loss_data = msa_loss(Tr_rho_O0, self.expected_CA[:, 0])
            loss_data += msa_loss(Tr_rho_O1, self.expected_CA[:, 1])

            loss_ic = mse_loss(y_pred_real[:], self.rho_train[:].real)
            loss_ic += mse_loss(y_pred_imag[:], self.rho_train[:].imag)


            # diagonals per sample, shape (N, base_rho)
            Tr_rho = (rho.diagonal(offset=0, dim1=-2, dim2=-1)).sum(-1)
            loss_norma = tc.mean((Tr_rho.real - 1) ** 2 + abs(Tr_rho.imag))

            loss_edo = Loss_EDO(
                H_= lerning_parameter_hamiltonina(
                    self.real_net.parametro,
                    device=self.device),
                rho_rvetor=y_pred_real,
                rho_ivetor=y_pred_imag,
                tempo=self.t_train,
                base_rho=self.base_rho,
            )
            field_vals, atom_vals, _ = self.U_evolution(
                lerning_parameter_hamiltonina(
                    self.real_net.parametro,
                    device=self.device), 
                self.rho_train[0].reshape((self.base_rho, self.base_rho)), 
                self.t_train, 
                self.Observavel_data[0], 
                self.Observavel_data[1],
                device=self.device)

            loss_extra = mse_loss(field_vals,self.expected_CA[:, 0]) + mse_loss(atom_vals,self.expected_CA[:, 1])

            loss = loss_ic  + loss_edo + loss_norma +loss_extra + loss_data #+ tc.mean(abs(self.real_net.parametro)**2)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.scheduler.step()

            self.LOSS.append(loss.cpu().detach().numpy())
            self.LOSS_parametr.append(self.real_net.parametro.cpu().detach().numpy())
