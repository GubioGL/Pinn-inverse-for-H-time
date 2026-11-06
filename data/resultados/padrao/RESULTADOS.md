# Resultados do perfil `padrao`

320 jobs, 320 ok. Estado inicial |0⟩, N = 100, campo verdadeiro fora da loss. Erro = ||f_pred − f_true|| / ||f_true||.

## Mediana do erro do campo por cenário

| método | caso 1, σ=0 | caso 2, σ=0 | caso 3, σ=0 | caso 3, σ=0.01 |
|---|---|---|---|---|
| `pinn_nn_lm` | 3.4e-03 % | 0.48 % | 0.10 % | 5.45 % |
| `lsq_bspline40` | 8.4e-04 % | 0.24 % | 0.28 % | 5.93 % |
| `shooting_bs40` | 1.4e-03 % | 0.22 % | 0.28 % | 6.18 % |
| `pinn_bspline_lm` | 3.3e-03 % | 0.76 % | 0.76 % | 5.47 % |
| `direta` | 0.13 % | 0.30 % | 0.16 % | 6.88 % |
| `pinn_bspline_lbfgs` | 0.70 % | 10.8 % | 2.45 % | 6.21 % |
| `lsq_fourier10` | 4.1e-04 % | 16.2 % | 0.49 % | 3.90 % |
| `pinn_fourier_lm` | 8.1e-04 % | 17.8 % | 0.51 % | 4.81 % |
| `pinn_nn_lbfgs` | 0.73 % | 13.8 % | 3.32 % | 5.97 % |
| `pinn_fourier_lbfgs` | 0.29 % | 22.7 % | 2.15 % | 5.67 % |
| `pinn_mixfunn_lm` | 7.5e-03 % | 3.18 % | 16.3 % | 21.3 % |
| `pinn_bspline_original` | 0.74 % | 21.7 % | 25.3 % | 25.3 % |
| `pinn_nn_original` | 0.55 % | 22.2 % | 25.9 % | 25.8 % |
| `pinn_mixfunn_lbfgs` | 1.23 % | 21.9 % | 26.0 % | 26.1 % |
| `pinn_fourier_original` | 0.37 % | 29.7 % | 25.3 % | 25.4 % |
| `pinn_mixfunn_original` | 5.34 % | 84.6 % | 65.7 % | 68.5 % |


### Caso 1, ruído σ = 0

| # | método | erro do campo (mediana) | média ± desvio | pior seed | RMSE trajetória (mediana) | tempo médio | seeds ok |
|---|---|---|---|---|---|---|---|
| 1 | `lsq_fourier10` | **4.1e-04 %** | 4.1e-04 % ± 0.0e+00 % | 4.1e-04 % | 5.5e-07 | 0 s | 5/5 |
| 2 | `pinn_fourier_lm` | **8.1e-04 %** | 8.3e-04 % ± 4.5e-04 % | 1.6e-03 % | 5.1e-07 | 163 s | 5/5 |
| 3 | `lsq_bspline40` | **8.4e-04 %** | 8.4e-04 % ± 0.0e+00 % | 8.4e-04 % | 6.0e-07 | 0 s | 5/5 |
| 4 | `shooting_bs40` | **1.4e-03 %** | 1.4e-03 % ± 1.7e-19 % | 1.4e-03 % | 1.4e-07 | 0 s | 5/5 |
| 5 | `pinn_bspline_lm` | **3.3e-03 %** | 2.9e-03 % ± 1.5e-03 % | 5.2e-03 % | 6.6e-07 | 165 s | 5/5 |
| 6 | `pinn_nn_lm` | **3.4e-03 %** | 3.8e-03 % ± 2.0e-03 % | 7.7e-03 % | 7.2e-07 | 151 s | 5/5 |
| 7 | `pinn_mixfunn_lm` | **7.5e-03 %** | 9.5e-03 % ± 4.5e-03 % | 0.02 % | 3.8e-06 | 326 s | 5/5 |
| 8 | `direta` | **0.13 %** | 0.13 % ± 0.0e+00 % | 0.13 % | 1.2e-03 | 0 s | 5/5 |
| 9 | `pinn_fourier_lbfgs` | **0.29 %** | 0.30 % ± 0.16 % | 0.58 % | 2.8e-04 | 22 s | 5/5 |
| 10 | `pinn_fourier_original` | **0.37 %** | 0.40 % ± 0.13 % | 0.63 % | 6.7e-04 | 25 s | 5/5 |
| 11 | `pinn_nn_original` | **0.55 %** | 0.49 % ± 0.11 % | 0.61 % | 9.2e-04 | 25 s | 5/5 |
| 12 | `pinn_bspline_lbfgs` | **0.70 %** | 0.62 % ± 0.22 % | 0.95 % | 3.0e-04 | 23 s | 5/5 |
| 13 | `pinn_nn_lbfgs` | **0.73 %** | 0.68 % ± 0.13 % | 0.84 % | 4.1e-04 | 21 s | 5/5 |
| 14 | `pinn_bspline_original` | **0.74 %** | 0.77 % ± 0.36 % | 1.34 % | 1.0e-03 | 26 s | 5/5 |
| 15 | `pinn_mixfunn_lbfgs` | **1.23 %** | 1.71 % ± 0.87 % | 3.17 % | 8.4e-04 | 62 s | 5/5 |
| 16 | `pinn_mixfunn_original` | **5.34 %** | 13.1 % ± 12.5 % | 35.6 % | 1.4e-02 | 106 s | 5/5 |

![erros](fig_erro_c1_r0.png)

![campos](fig_campos_c1_r0.png)

### Caso 2, ruído σ = 0

| # | método | erro do campo (mediana) | média ± desvio | pior seed | RMSE trajetória (mediana) | tempo médio | seeds ok |
|---|---|---|---|---|---|---|---|
| 1 | `shooting_bs40` | **0.22 %** | 0.22 % ± 0.0e+00 % | 0.22 % | 5.9e-05 | 0 s | 5/5 |
| 2 | `lsq_bspline40` | **0.24 %** | 0.24 % ± 0.0e+00 % | 0.24 % | 1.6e-04 | 0 s | 5/5 |
| 3 | `direta` | **0.30 %** | 0.30 % ± 0.0e+00 % | 0.30 % | 1.7e-03 | 0 s | 5/5 |
| 4 | `pinn_nn_lm` | **0.48 %** | 0.39 % ± 0.17 % | 0.58 % | 1.9e-04 | 169 s | 5/5 |
| 5 | `pinn_bspline_lm` | **0.76 %** | 0.77 % ± 0.06 % | 0.88 % | 3.0e-04 | 186 s | 5/5 |
| 6 | `pinn_mixfunn_lm` | **3.18 %** | 6.07 % ± 7.52 % | 20.9 % | 1.8e-03 | 301 s | 5/5 |
| 7 | `pinn_bspline_lbfgs` | **10.8 %** | 11.1 % ± 1.89 % | 13.8 % | 7.6e-03 | 21 s | 5/5 |
| 8 | `pinn_nn_lbfgs` | **13.8 %** | 13.2 % ± 2.27 % | 16.0 % | 1.0e-02 | 21 s | 5/5 |
| 9 | `lsq_fourier10` | **16.2 %** | 16.2 % ± 0.0e+00 % | 16.2 % | 5.5e-02 | 0 s | 5/5 |
| 10 | `pinn_fourier_lm` | **17.8 %** | 17.8 % ± 1.8e-03 % | 17.8 % | 1.0e-02 | 182 s | 5/5 |
| 11 | `pinn_bspline_original` | **21.7 %** | 21.1 % ± 1.90 % | 23.7 % | 2.1e-02 | 26 s | 5/5 |
| 12 | `pinn_mixfunn_lbfgs` | **21.9 %** | 23.5 % ± 3.31 % | 29.9 % | 1.9e-02 | 59 s | 5/5 |
| 13 | `pinn_nn_original` | **22.2 %** | 22.4 % ± 0.76 % | 23.8 % | 2.2e-02 | 25 s | 5/5 |
| 14 | `pinn_fourier_lbfgs` | **22.7 %** | 23.1 % ± 2.48 % | 27.0 % | 1.7e-02 | 21 s | 5/5 |
| 15 | `pinn_fourier_original` | **29.7 %** | 28.5 % ± 2.79 % | 31.1 % | 6.0e-02 | 25 s | 5/5 |
| 16 | `pinn_mixfunn_original` | **84.6 %** | 1512 % ± 2772 % | 7051 % | 2.7e-01 | 106 s | 5/5 |

![erros](fig_erro_c2_r0.png)

![campos](fig_campos_c2_r0.png)

### Caso 3, ruído σ = 0

| # | método | erro do campo (mediana) | média ± desvio | pior seed | RMSE trajetória (mediana) | tempo médio | seeds ok |
|---|---|---|---|---|---|---|---|
| 1 | `pinn_nn_lm` | **0.10 %** | 0.13 % ± 0.04 % | 0.21 % | 2.4e-05 | 174 s | 5/5 |
| 2 | `direta` | **0.16 %** | 0.16 % ± 0.0e+00 % | 0.16 % | 4.7e-04 | 0 s | 5/5 |
| 3 | `shooting_bs40` | **0.28 %** | 0.28 % ± 0.0e+00 % | 0.28 % | 4.2e-05 | 0 s | 5/5 |
| 4 | `lsq_bspline40` | **0.28 %** | 0.28 % ± 0.0e+00 % | 0.28 % | 4.6e-05 | 0 s | 5/5 |
| 5 | `lsq_fourier10` | **0.49 %** | 0.49 % ± 0.0e+00 % | 0.49 % | 4.6e-04 | 0 s | 5/5 |
| 6 | `pinn_fourier_lm` | **0.51 %** | 0.51 % ± 1.1e-04 % | 0.51 % | 2.0e-04 | 183 s | 5/5 |
| 7 | `pinn_bspline_lm` | **0.76 %** | 0.74 % ± 0.13 % | 0.93 % | 2.1e-04 | 183 s | 5/5 |
| 8 | `pinn_fourier_lbfgs` | **2.15 %** | 2.57 % ± 1.74 % | 5.84 % | 1.0e-03 | 21 s | 5/5 |
| 9 | `pinn_bspline_lbfgs` | **2.45 %** | 3.01 % ± 1.53 % | 5.88 % | 9.2e-04 | 21 s | 5/5 |
| 10 | `pinn_nn_lbfgs` | **3.32 %** | 3.50 % ± 0.46 % | 4.41 % | 1.3e-03 | 21 s | 5/5 |
| 11 | `pinn_mixfunn_lm` | **16.3 %** | 14.6 % ± 7.98 % | 24.9 % | 7.7e-03 | 329 s | 5/5 |
| 12 | `pinn_bspline_original` | **25.3 %** | 25.7 % ± 0.79 % | 26.8 % | 1.3e-02 | 26 s | 5/5 |
| 13 | `pinn_fourier_original` | **25.3 %** | 23.8 % ± 4.07 % | 26.7 % | 1.3e-02 | 26 s | 5/5 |
| 14 | `pinn_nn_original` | **25.9 %** | 25.7 % ± 0.46 % | 26.2 % | 1.3e-02 | 25 s | 5/5 |
| 15 | `pinn_mixfunn_lbfgs` | **26.0 %** | 26.2 % ± 0.28 % | 26.6 % | 1.3e-02 | 58 s | 5/5 |
| 16 | `pinn_mixfunn_original` | **65.7 %** | 1168 % ± 2222 % | 5612 % | 2.0e-01 | 104 s | 5/5 |

![erros](fig_erro_c3_r0.png)

![campos](fig_campos_c3_r0.png)

### Caso 3, ruído σ = 0.01

| # | método | erro do campo (mediana) | média ± desvio | pior seed | RMSE trajetória (mediana) | tempo médio | seeds ok |
|---|---|---|---|---|---|---|---|
| 1 | `lsq_fourier10` | **3.90 %** | 3.94 % ± 0.80 % | 5.31 % | 4.6e-03 | 0 s | 5/5 |
| 2 | `pinn_fourier_lm` | **4.81 %** | 4.49 % ± 0.71 % | 4.96 % | 3.1e-03 | 162 s | 5/5 |
| 3 | `pinn_nn_lm` | **5.45 %** | 5.70 % ± 0.74 % | 7.15 % | 3.2e-03 | 153 s | 5/5 |
| 4 | `pinn_bspline_lm` | **5.47 %** | 5.66 % ± 0.74 % | 7.09 % | 3.2e-03 | 161 s | 5/5 |
| 5 | `pinn_fourier_lbfgs` | **5.67 %** | 5.19 % ± 1.08 % | 6.35 % | 3.4e-03 | 21 s | 5/5 |
| 6 | `lsq_bspline40` | **5.93 %** | 6.11 % ± 0.38 % | 6.71 % | 5.0e-03 | 0 s | 5/5 |
| 7 | `pinn_nn_lbfgs` | **5.97 %** | 5.91 % ± 0.89 % | 7.05 % | 3.3e-03 | 21 s | 5/5 |
| 8 | `shooting_bs40` | **6.18 %** | 6.88 % ± 1.87 % | 10.6 % | 3.1e-03 | 0 s | 5/5 |
| 9 | `pinn_bspline_lbfgs` | **6.21 %** | 6.18 % ± 1.14 % | 8.07 % | 3.6e-03 | 22 s | 5/5 |
| 10 | `direta` | **6.88 %** | 7.03 % ± 0.24 % | 7.46 % | 4.9e-03 | 0 s | 5/5 |
| 11 | `pinn_mixfunn_lm` | **21.3 %** | 17.0 % ± 9.10 % | 26.1 % | 1.1e-02 | 319 s | 5/5 |
| 12 | `pinn_bspline_original` | **25.3 %** | 25.7 % ± 0.77 % | 26.8 % | 1.4e-02 | 26 s | 5/5 |
| 13 | `pinn_fourier_original` | **25.4 %** | 25.3 % ± 1.31 % | 26.6 % | 1.3e-02 | 26 s | 5/5 |
| 14 | `pinn_nn_original` | **25.8 %** | 25.7 % ± 0.44 % | 26.2 % | 1.3e-02 | 25 s | 5/5 |
| 15 | `pinn_mixfunn_lbfgs` | **26.1 %** | 26.2 % ± 0.15 % | 26.5 % | 1.3e-02 | 59 s | 5/5 |
| 16 | `pinn_mixfunn_original` | **68.5 %** | 68.6 % ± 33.4 % | 113 % | 2.2e-01 | 106 s | 5/5 |

![erros](fig_erro_c3_r0.01.png)

![campos](fig_campos_c3_r0.01.png)
