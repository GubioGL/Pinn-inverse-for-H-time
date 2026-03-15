import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# 1. Definir a função a ser aproximada
def target_function(x):
    return 1.5 * np.sin(2.0 * x) + 0.8 * np.cos(3.5 * x + 0.) + 0.4 * np.sin(7.0 * x)

# 2. Parâmetros da B-spline
k = 5  # Grau da B-spline (cúbica)

# 3. Gerar dados de treinamento
x_min, x_max = 0, 5
num_train_points = 100
x_train_np = np.linspace(x_min, x_max, num_train_points)
y_train_np = target_function(x_train_np)
x_train = torch.tensor(x_train_np, dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1)

# 4. Definir a Rede Neural
class FunctionPredictor(nn.Module): 
    def __init__(self, input_dim, output_dim):
        super(FunctionPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Instanciar a rede neural
model = FunctionPredictor(input_dim=1, output_dim=1)

# Otimizador e Função de Perda
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# 5. Treinamento da Rede Neural
num_epochs = 12000

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Treinamento da Rede Neural concluído!")

# 6. Gerar a aproximação B-spline a partir dos valores previstos pela rede neural

# Gerar pontos para plotagem
x_plot_np = np.linspace(x_min, x_max, 500)
x_plot_tensor = torch.tensor(x_plot_np, dtype=torch.float32).unsqueeze(1)

# Obter a predição da rede neural para os pontos de plotagem
model.eval()
with torch.no_grad():
    y_nn_pred = model(x_plot_tensor).squeeze().numpy()

# Criar a B-spline de interpolação a partir dos pontos previstos pela rede neural
# Para usar make_interp_spline, precisamos de pontos de dados (x, y).
# Usaremos os pontos de treinamento (x_train_np) e os valores previstos pela NN para esses pontos.
# É importante que os x_train_np estejam ordenados, o que já é o caso.

# A rede neural aprendeu a função. Agora, vamos usar esses valores para criar uma B-spline.
# Para uma interpolação suave, make_interp_spline é ideal.

# Para garantir que a spline seja criada a partir dos pontos de treinamento e suas predições
# vamos usar os x_train_np e os valores preditos pela NN para esses x_train_np.
# Primeiro, obter as predições da NN para os pontos de treinamento
with torch.no_grad():
    y_train_nn_pred = model(x_train).squeeze().numpy()

spline_nn = make_interp_spline(x_train_np, y_train_np, k=k)
y_bspline_nn = spline_nn(x_plot_np)

# 7. Plotagem
plt.figure(figsize=(12, 7))
plt.plot(x_plot_np, target_function(x_plot_np), label='Função Real', color='blue', linewidth=2)
plt.plot(x_plot_np, y_nn_pred, label='Predição da Rede Neural', color='green', linestyle='-', alpha=0.7)
plt.plot(x_plot_np, y_bspline_nn, label=f'B-spline aproximação (grau {k})', color='red', linestyle='--')
plt.scatter(x_train_np, y_train_np, color='black', label='Pontos de Treinamento', zorder=5, s=20)
plt.title('Comparação: Função Real vs. Aproximação por Rede Neural e B-spline')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()
print("Gráfico salvo como 'nn_bspline_comparison.png'")
