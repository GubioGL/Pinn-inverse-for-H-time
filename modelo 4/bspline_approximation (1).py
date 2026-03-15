import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# 1. Definir a função a ser aproximada
def target_function(x):
    # Exemplo: Uma função suave com variações
    return 1.5 * np.sin(2.0 * x) + 0.8 * np.cos(3.5 * x + 0.) + 0.4 * np.sin(7.0 * x)

# 2. Definir o intervalo e os pontos para a aproximação
x_min, x_max = 0, 15
num_points = 100  # Poucos pontos para mostrar a interpolação
x_data = np.linspace(x_min, x_max, num_points)
y_data = target_function(x_data)

# 3. Criar a B-spline de interpolação (grau k=3 para cúbica)
k = 5
spline = make_interp_spline(x_data, y_data, k=k)

# 4. Gerar pontos para plotar a função original e a aproximação
x_plot = np.linspace(x_min, x_max, 500)
y_real = target_function(x_plot)
y_bspline = spline(x_plot)

# 5. Plotagem
plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_real, label='Função Real', color='blue', linewidth=2, alpha=0.6)
plt.plot(x_plot, y_bspline, label=f'Aproximação B-spline (grau {k})', color='red', linestyle='--')
plt.scatter(x_data, y_data, color='black', label='Pontos de Amostragem', zorder=5)
plt.title('Comparação: Função Real vs. Aproximação B-spline')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()
print("Gráfico salvo como 'bspline_comparison.png'")
