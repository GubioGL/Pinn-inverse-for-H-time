import torch as tc
import numpy as np
import qutip as qt
import function as my
import mixfunn as mf
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.utils.prune as prune

O_op =[ qt.sigmax(),
         qt.sigmay(),
         qt.sigmaz()
                ]

def Field(t):
    return ( 1.5* np.sin(2.0 * t) - 0.3*np.cos(3* t - 0.5) + 0.4*np.sin(7.0 * t)
        )*np.sin(np.pi*t/tfinal)**2


def data_qubit(O_op,tlist,field,device="cpu"):
    # Lista de tempos para a evolução
    H0 = 0.5*qt.sigmaz()
    H1 = 0.5*qt.sigmax()
    H  = [H0,[H1,field(tlist)]]
    
    # Hamiltonian Lindbladian
    c_ops = []
    
    # Estado inicial (cada qubit na superposição de |0> e |1>)
    # |+> = (|0> + |1>)/sqrt(2)
    theta1  = np.pi/4
    phi1    = 0.0 #np.pi/3
    ket_plus1 = (np.cos(theta1)*qt.basis(2, 0)+np.sin(theta1)*np.exp(1j*phi1)*qt.basis(2, 1))
    psi0 = ket_plus1
    
    # Solução da equação de Schrödinger
    options = qt.Options(nsteps = 100000, atol = 1e-14, rtol = 1e-14)
    result  = qt.mesolve(H, psi0, tlist, c_ops=c_ops, e_ops=O_op,options=options)

    expect  = tc.tensor( np.array( result.expect),device = device).transpose(0, 1)
    return expect


if __name__ == "__main__":
    for seed in range(5):
        np.random.seed(seed)
        tc.manual_seed(seed)

        tfinal      = 2.52*np.pi
        N           = 200
        Nedo        = 3

        tlist = np.linspace(0.0, tfinal, N)
        valor_esperado_data  = data_qubit(O_op,tlist,Field,device="cpu")

        time = tc.linspace(
            0, tfinal, N,
            dtype = tc.float32,
            requires_grad = True).reshape((-1, 1))

        X_vector = my.Rede(
            neuronio   = [15,15],
            input_     = 1,
            output_    = Nedo,
            activation =[mf.SIN(), mf.SIN()])

        mixfunn_field = mf.MixFunn(
            N_layers = 2,
            N_neuro = 4,
            second_order_function = True)

        opt = tc.optim.Adam(
            list(mixfunn_field.parameters())+list(X_vector.parameters()),
            lr=0.01, betas=(0.9,0.9))

        JZ = 1e0
        JY = 0e0
        epocas  = 50000
        
        LOSS    = []
        LOSS1   = []
        LOSS2   = []
        for _ in tqdm(range(epocas)):
            ####### Forward pass #######
            y   = X_vector(time)    # [N,3]
            JX  = mixfunn_field(time)   # [N,1]

            ######## Loss edo #######
            dX_dt = []
            for i in range(y.shape[1]):
                dX_dt.append(
                    tc.autograd.grad(
                        outputs = y[:, i],
                        inputs = time,
                        grad_outputs = tc.ones_like(y[:, i]),
                        #retain_graph = True,
                        create_graph = True)[0])
            dX_dt   = tc.cat(dX_dt, dim=1)
            
            X,Y,Z = y[:, 0:1], y[:, 1:2], y[:, 2:3]
            LOSS_edo  = 0
            
            LOSS_edo += (dX_dt[:,0:1] - (-1.0*Y))**2
            LOSS_edo += (dX_dt[:,1:2] - ( 1.0*X - 1.0*JX*Z))**2
            LOSS_edo += (dX_dt[:,2:3] - ( 1.0*JX*Y))**2
            LOSS_edo = LOSS_edo.mean()

            ####### loss data(expected values) #######
            LOSS_data=0    
            LOSS_data +=(y[:,0:1]  - valor_esperado_data[:,0:1])**2 
            LOSS_data +=(y[:,1:2]  - valor_esperado_data[:,1:2])**2 
            LOSS_data +=(y[:,2:3]  - valor_esperado_data[:,2:3])**2 
            LOSS_data = LOSS_data.mean()
            
            ####### Loss total #######
            loss_i = LOSS_edo + LOSS_data + (JX[0,0] - Field(0.0))**2 + (JX[-1,0] - Field(tfinal))**2


            ####### Backpropagation #######
            opt.zero_grad()
            loss_i.backward()
            opt.step()
            LOSS1.append(LOSS_edo.cpu().detach().numpy())
            LOSS2.append(LOSS_data.cpu().detach().numpy())
            LOSS.append(loss_i.cpu().detach().numpy())
            
        ##### Prunning #####
        p = mixfunn_field.layers1.p.data
        L = tc.sum(tc.abs(p))
        p = tc.abs(p)/L
            
        prune.l1_unstructured(mixfunn_field.layers1.project1.linear, 'weight', amount=0.1)
        prune.l1_unstructured(mixfunn_field.layers1.project1.linear, 'bias', amount=0.1)

        prune.l1_unstructured(mixfunn_field.layers2.project1.linear, 'weight', amount=0.1)
        prune.l1_unstructured(mixfunn_field.layers2.project1.linear, 'bias', amount=0.1)

        p = mixfunn_field.layers1.p.data
        L = tc.sum(tc.abs(p))
        p = tc.abs(p)/L
        mixfunn_field.layers1.p.data = tc.where(p < 0.01, 0.0, mixfunn_field.layers1.p.data)

        p = mixfunn_field.layers1.p.data
        L = tc.sum(abs(p))
        p = abs(p)/L

        values, indices = tc.sort(p)
        mask = tc.zeros(indices.shape[1])
        for k in range(indices.shape[1]):
            if k > int(0.2*indices.shape[1]):
                mask[indices[0][k]] = 1.0
        mixfunn_field.layers1.p.data = mask*mixfunn_field.layers1.p.data

        p = mixfunn_field.layers2.p.data
        L = tc.sum(tc.abs(p))
        p = tc.abs(p)/L
        mixfunn_field.layers2.p.data = tc.where(p < 0.01, 0.0, mixfunn_field.layers2.p.data)

        p = mixfunn_field.layers2.p.data
        L = tc.sum(abs(p))
        p = abs(p)/L

        values, indices = tc.sort(p)
        mask2 = tc.zeros(indices.shape[1])
        for k in range(indices.shape[1]):
            if k > int(0.2*indices.shape[1]):
                mask2[indices[0][k]] = 1.0
        mixfunn_field.layers2.p.data = mask2*mixfunn_field.layers2.p.data

        opt = tc.optim.Adam(
            list(mixfunn_field.parameters())+list(X_vector.parameters()),
            lr=0.001)

        LOSS    = []
        LOSS1   = []
        LOSS2   = []
        for _ in tqdm(range(epocas)):
            # aplicaçao da mascara para os pesos novos
            mixfunn_field.layers1.p.data = mask*mixfunn_field.layers1.p.data
            mixfunn_field.layers2.p.data = mask2*mixfunn_field.layers2.p.data

            ####### Forward pass #######
            y   = X_vector(time)    # [N,3]
            JX  = mixfunn_field(time)   # [N,1]

            ######## Loss edo #######
            dX_dt = []
            for i in range(y.shape[1]):
                dX_dt.append(
                    tc.autograd.grad(
                        outputs = y[:, i],
                        inputs = time,
                        grad_outputs = tc.ones_like(y[:, i]),
                        #retain_graph = True,
                        create_graph = True)[0])
            dX_dt   = tc.cat(dX_dt, dim=1)
            
            X,Y,Z = y[:, 0:1], y[:, 1:2], y[:, 2:3]
            LOSS_edo  = 0
            
            LOSS_edo += (dX_dt[:,0:1] - (-1.0*Y))**2
            LOSS_edo += (dX_dt[:,1:2] - ( 1.0*X - 1.0*JX*Z))**2
            LOSS_edo += (dX_dt[:,2:3] - ( 1.0*JX*Y))**2
            LOSS_edo = LOSS_edo.mean()


            ####### loss data(expected values) #######
            LOSS_data=0    
            LOSS_data +=(y[:,0:1]  - valor_esperado_data[:,0:1])**2 
            LOSS_data +=(y[:,1:2]  - valor_esperado_data[:,1:2])**2 
            LOSS_data +=(y[:,2:3]  - valor_esperado_data[:,2:3])**2 
            LOSS_data = LOSS_data.mean()
            

            ####### Loss total #######
            loss_i = LOSS_edo + LOSS_data + (JX[0,0] - Field(0.0))**2 + (JX[-1,0] - Field(tfinal))**2


            ####### Backpropagation #######
            opt.zero_grad()
            loss_i.backward()
            opt.step()
            LOSS1.append(LOSS_edo.cpu().detach().numpy())
            LOSS2.append(LOSS_data.cpu().detach().numpy())
            LOSS.append(loss_i.cpu().detach().numpy())

        ######
        t_points = np.linspace(0,tfinal, N)
        JX =  mixfunn_field(time).detach().numpy()

        field_original = Field(t_points)
        
        X_vector.eval()
        X_  = X_vector(time).detach().numpy()   
        valor_esperado_data[:,i].cpu()
        X_[:,i]
        
        tc.save(X_vector.state_dict(), f"Modelo 5(Mixfunn)/save/X_vector_{seed}.pth")
        tc.save(mixfunn_field.state_dict(), f"Modelo 5(Mixfunn)/save/mixfunn_field_{seed}.pth")
        np.save(f"Modelo 5(Mixfunn)/save/field_original_{seed}.npy", field_original)
        np.save(f"Modelo 5(Mixfunn)/save/field_predito_{seed}.npy", JX)
        np.save(f"Modelo 5(Mixfunn)/save/X_predito_{seed}.npy", X_)
        np.save(f"Modelo 5(Mixfunn)/save/X_esperado_{seed}.npy", valor_esperado_data.cpu().numpy())
        
        
        ###### plote save ########
        t_points = np.linspace(0,tfinal, N)
        JX =  mixfunn_field(time).detach().numpy()

        field_original=Field(t_points)
        plt.figure()
        plt.plot(JX,"r--",label="NN - Field")
        plt.plot(field_original,"b-",label="Original Field")
        plt.legend()
        plt.savefig(f"Modelo 5(Mixfunn)/save/field_comparison_{seed}.png")
        #plt.show()

        plt.figure()
        X_vector.eval()
        X_  = X_vector(time).detach().numpy()

        colors = ['r', 'g', 'b']
        for i in range(len(O_op)):
            plt.plot(valor_esperado_data[:,i].cpu() ,label=fr"$\hat{{O}}_{{{i+1}}}$", color=colors[i])
            plt.plot(X_[:,i],"." ,label=fr"$NN - O_{{{i}}}$", color=colors[i])
        plt.xlabel('Time')
        plt.ylabel('Expectation Value')
        plt.legend()
        plt.title('Time Evolution from PINN')
        plt.savefig(f"Modelo 5(Mixfunn)/save/expectation_comparison_{seed}.png")
        #plt.show()
