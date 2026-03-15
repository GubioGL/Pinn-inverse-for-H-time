import numpy as np
import torch as tc
import torch.nn as nn

class SIN(nn.Module):
    def __init__(self): 
        super(SIN, self).__init__() 
    def forward(self, x):
        return tc.sin(x)
        
class Rede(nn.Module):
    def __init__(self, neuronio, activation, input_=1, output_=1, creat_p=False, N_of_paramater=1, dropout_prob=0.0):
        super().__init__()
        self.neuronio = neuronio
        self.output = output_
        self.creat_p = creat_p
        self.N_of_paramater = N_of_paramater
        self.dropout_prob = dropout_prob
        
        # input camada linear
        self.hidden_layers = nn.ModuleList([nn.Linear(input_, neuronio[0])])
        # camadas do meio
        self.hidden_layers.extend([nn.Linear(neuronio[_], neuronio[_+1]) for _ in range(len(self.neuronio)-1)])
        # Última camada linear
        self.output_layer = nn.Linear(neuronio[-1], output_)
        
        # Função de ativação
        self.activation_ = activation
        
        # Camadas de Dropout
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_prob) for _ in range(len(self.neuronio))])
        
        # Criar o parâmetro
        if creat_p:
            self.parametro = nn.Parameter(tc.rand(N_of_paramater), requires_grad=True)#* 2 * np.pi)
            
    def forward(self, x):
        for layer, activation, dropout in zip(self.hidden_layers, self.activation_, self.dropouts):
            x = activation(layer(x))
            x = dropout(x)
        x = self.output_layer(x)
        return x