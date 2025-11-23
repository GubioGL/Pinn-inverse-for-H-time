import torch as tc
import numpy as np
import qutip as qt
import torch.nn as nn

from tqdm import tqdm


    
class Rede(nn.Module):
    def __init__(self, neuronio, activation, input_=1, output_=1, creat_p=False, N_of_paramater=1):
        super().__init__()       
        # input camada linear
        self.first_layer = nn.Linear(input_, neuronio[0])
        self.second_layer = nn.Linear(neuronio[0], output_)
        
        # Função de ativação
        self.activation_ = activation
        
        # Criar o parâmetro
        if creat_p:
            self.parametro = nn.Parameter(tc.rand(N_of_paramater))
            
    def forward(self, x):
        x = self.first_layer(x)
        x = self.activation_(x)
        x = self.second_layer(x)
        return x
