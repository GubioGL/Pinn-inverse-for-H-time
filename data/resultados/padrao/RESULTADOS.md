# Resultados do perfil `padrao`

260 jobs, 260 ok. Estado inicial |0⟩, N = 100, campo verdadeiro fora da loss. Erro = ||f_pred − f_true|| / ||f_true||.

## Mediana do erro do campo por cenário

| método | caso 1, σ=0 | caso 2, σ=0 | caso 3, σ=0 | caso 3, σ=0.01 |
|---|---|---|---|---|
| `pinn_nn_lm` | 3.9e-03 % | 0.31 % | 0.20 % | – |
| `pinn_bspline_lm` | 3.6e-03 % | 0.51 % | 0.18 % | – |
| `pinn_cosseno_lm` | 1.3e-03 % | 2.38 % | 0.19 % | – |
| `shooting_bs80` | 2.4e-03 % | 0.37 % | 0.01 % | 5.66 % |
| `lsq_bspline40` | 8.4e-04 % | 0.24 % | 0.28 % | 5.93 % |
| `shooting_bs40` | 1.4e-03 % | 0.22 % | 0.28 % | 6.18 % |
| `lsq_bspline80` | 8.6e-04 % | 0.12 % | 0.03 % | 6.73 % |
| `direta` | 0.13 % | 0.30 % | 0.16 % | 6.88 % |
| `lsq_cosseno40` | 5.6e-04 % | 1.59 % | 0.02 % | 6.53 % |
| `lsq_fourier10` | 4.1e-04 % | 16.2 % | 0.49 % | 3.90 % |
| `pinn_fourier_lm` | 7.8e-04 % | 17.9 % | 0.54 % | – |
| `pinn_mixfunn_lm` | 0.03 % | 13.0 % | 25.0 % | 25.7 % |
| `pinn_mixfunn_lbfgs` | 1.24 % | 22.3 % | 26.0 % | 26.0 % |
| `pinn_mixfunn_original` | 5.27 % | 89.0 % | 65.2 % | 57.6 % |


### Caso 1, ruído σ = 0

| # | método | erro do campo (mediana) | média ± desvio | pior seed | RMSE trajetória (mediana) | tempo médio | seeds ok |
|---|---|---|---|---|---|---|---|
| 1 | `lsq_fourier10` | **4.1e-04 %** | 4.1e-04 % ± 0.0e+00 % | 4.1e-04 % | 5.5e-07 | 0 s | 5/5 |
| 2 | `lsq_cosseno40` | **5.6e-04 %** | 5.6e-04 % ± 0.0e+00 % | 5.6e-04 % | 5.1e-07 | 0 s | 5/5 |
| 3 | `pinn_fourier_lm` | **7.8e-04 %** | 7.9e-04 % ± 4.6e-04 % | 1.6e-03 % | 5.0e-07 | 311 s | 5/5 |
| 4 | `lsq_bspline40` | **8.4e-04 %** | 8.4e-04 % ± 0.0e+00 % | 8.4e-04 % | 6.0e-07 | 0 s | 5/5 |
| 5 | `lsq_bspline80` | **8.6e-04 %** | 8.6e-04 % ± 0.0e+00 % | 8.6e-04 % | 6.9e-07 | 0 s | 5/5 |
| 6 | `pinn_cosseno_lm` | **1.3e-03 %** | 1.5e-03 % ± 4.9e-04 % | 2.3e-03 % | 6.0e-07 | 314 s | 5/5 |
| 7 | `shooting_bs40` | **1.4e-03 %** | 1.4e-03 % ± 1.7e-19 % | 1.4e-03 % | 1.4e-07 | 0 s | 5/5 |
| 8 | `shooting_bs80` | **2.4e-03 %** | 2.4e-03 % ± 0.0e+00 % | 2.4e-03 % | 9.4e-07 | 0 s | 5/5 |
| 9 | `pinn_bspline_lm` | **3.6e-03 %** | 3.4e-03 % ± 1.6e-03 % | 5.9e-03 % | 6.7e-07 | 318 s | 5/5 |
| 10 | `pinn_nn_lm` | **3.9e-03 %** | 3.6e-03 % ± 1.1e-03 % | 4.8e-03 % | 8.9e-07 | 288 s | 5/5 |
| 11 | `pinn_mixfunn_lm` | **0.03 %** | 0.04 % ± 0.02 % | 0.08 % | 1.5e-05 | 422 s | 5/5 |
| 12 | `direta` | **0.13 %** | 0.13 % ± 0.0e+00 % | 0.13 % | 1.2e-03 | 0 s | 5/5 |
| 13 | `pinn_mixfunn_lbfgs` | **1.24 %** | 1.72 % ± 0.68 % | 2.65 % | 6.9e-04 | 55 s | 5/5 |
| 14 | `pinn_mixfunn_original` | **5.27 %** | 10.5 % ± 12.4 % | 35.2 % | 8.8e-03 | 80 s | 5/5 |

Figuras: [erro por seed](figuras/fig_erro_c1_r0.pdf) · [campo recuperado](figuras/fig_campos_c1_r0.pdf)

### Caso 2, ruído σ = 0

| # | método | erro do campo (mediana) | média ± desvio | pior seed | RMSE trajetória (mediana) | tempo médio | seeds ok |
|---|---|---|---|---|---|---|---|
| 1 | `lsq_bspline80` | **0.12 %** | 0.12 % ± 0.0e+00 % | 0.12 % | 6.2e-05 | 0 s | 5/5 |
| 2 | `shooting_bs40` | **0.22 %** | 0.22 % ± 0.0e+00 % | 0.22 % | 5.9e-05 | 0 s | 5/5 |
| 3 | `lsq_bspline40` | **0.24 %** | 0.24 % ± 0.0e+00 % | 0.24 % | 1.6e-04 | 0 s | 5/5 |
| 4 | `direta` | **0.30 %** | 0.30 % ± 0.0e+00 % | 0.30 % | 1.7e-03 | 0 s | 5/5 |
| 5 | `pinn_nn_lm` | **0.31 %** | 0.35 % ± 0.13 % | 0.51 % | 1.1e-04 | 290 s | 5/5 |
| 6 | `shooting_bs80` | **0.37 %** | 0.37 % ± 4.3e-17 % | 0.37 % | 1.5e-04 | 0 s | 5/5 |
| 7 | `pinn_bspline_lm` | **0.51 %** | 0.71 % ± 0.40 % | 1.50 % | 2.2e-04 | 319 s | 5/5 |
| 8 | `lsq_cosseno40` | **1.59 %** | 1.59 % ± 0.0e+00 % | 1.59 % | 4.4e-03 | 0 s | 5/5 |
| 9 | `pinn_cosseno_lm` | **2.38 %** | 2.47 % ± 0.24 % | 2.80 % | 6.7e-04 | 313 s | 5/5 |
| 10 | `pinn_mixfunn_lm` | **13.0 %** | 11.6 % ± 5.38 % | 18.3 % | 1.0e-02 | 422 s | 5/5 |
| 11 | `lsq_fourier10` | **16.2 %** | 16.2 % ± 0.0e+00 % | 16.2 % | 5.5e-02 | 0 s | 5/5 |
| 12 | `pinn_fourier_lm` | **17.9 %** | 17.8 % ± 0.02 % | 17.9 % | 1.0e-02 | 309 s | 5/5 |
| 13 | `pinn_mixfunn_lbfgs` | **22.3 %** | 25.0 % ± 4.39 % | 33.0 % | 1.8e-02 | 51 s | 5/5 |
| 14 | `pinn_mixfunn_original` | **89.0 %** | 1544 % ± 2831 % | 7202 % | 2.9e-01 | 77 s | 5/5 |

Figuras: [erro por seed](figuras/fig_erro_c2_r0.pdf) · [campo recuperado](figuras/fig_campos_c2_r0.pdf)

### Caso 3, ruído σ = 0

| # | método | erro do campo (mediana) | média ± desvio | pior seed | RMSE trajetória (mediana) | tempo médio | seeds ok |
|---|---|---|---|---|---|---|---|
| 1 | `shooting_bs80` | **0.01 %** | 0.01 % ± 0.0e+00 % | 0.01 % | 4.1e-06 | 0 s | 5/5 |
| 2 | `lsq_cosseno40` | **0.02 %** | 0.02 % ± 0.0e+00 % | 0.02 % | 2.6e-05 | 0 s | 5/5 |
| 3 | `lsq_bspline80` | **0.03 %** | 0.03 % ± 0.0e+00 % | 0.03 % | 1.7e-05 | 0 s | 5/5 |
| 4 | `direta` | **0.16 %** | 0.16 % ± 0.0e+00 % | 0.16 % | 4.7e-04 | 0 s | 5/5 |
| 5 | `pinn_bspline_lm` | **0.18 %** | 0.17 % ± 0.03 % | 0.20 % | 4.2e-05 | 313 s | 5/5 |
| 6 | `pinn_cosseno_lm` | **0.19 %** | 0.17 % ± 0.04 % | 0.22 % | 4.9e-05 | 307 s | 5/5 |
| 7 | `pinn_nn_lm` | **0.20 %** | 0.17 % ± 0.05 % | 0.22 % | 5.8e-05 | 287 s | 5/5 |
| 8 | `shooting_bs40` | **0.28 %** | 0.28 % ± 0.0e+00 % | 0.28 % | 4.2e-05 | 0 s | 5/5 |
| 9 | `lsq_bspline40` | **0.28 %** | 0.28 % ± 0.0e+00 % | 0.28 % | 4.6e-05 | 0 s | 5/5 |
| 10 | `lsq_fourier10` | **0.49 %** | 0.49 % ± 0.0e+00 % | 0.49 % | 4.6e-04 | 0 s | 5/5 |
| 11 | `pinn_fourier_lm` | **0.54 %** | 0.54 % ± 9.1e-04 % | 0.55 % | 2.6e-04 | 299 s | 5/5 |
| 12 | `pinn_mixfunn_lm` | **25.0 %** | 22.7 % ± 3.10 % | 25.2 % | 1.2e-02 | 422 s | 5/5 |
| 13 | `pinn_mixfunn_lbfgs` | **26.0 %** | 25.8 % ± 0.52 % | 26.3 % | 1.3e-02 | 73 s | 5/5 |
| 14 | `pinn_mixfunn_original` | **65.2 %** | 1138 % ± 2157 % | 5451 % | 1.6e-01 | 82 s | 5/5 |

Figuras: [erro por seed](figuras/fig_erro_c3_r0.pdf) · [campo recuperado](figuras/fig_campos_c3_r0.pdf)

### Caso 3, ruído σ = 0.01

| # | método | erro do campo (mediana) | média ± desvio | pior seed | RMSE trajetória (mediana) | tempo médio | seeds ok |
|---|---|---|---|---|---|---|---|
| 1 | `lsq_fourier10` | **3.90 %** | 3.94 % ± 0.80 % | 5.31 % | 4.6e-03 | 0 s | 5/5 |
| 2 | `shooting_bs80` | **5.66 %** | 6.24 % ± 1.71 % | 9.59 % | 2.9e-03 | 1 s | 5/5 |
| 3 | `lsq_bspline40` | **5.93 %** | 6.11 % ± 0.38 % | 6.71 % | 5.0e-03 | 0 s | 5/5 |
| 4 | `shooting_bs40` | **6.18 %** | 6.88 % ± 1.87 % | 10.6 % | 3.1e-03 | 0 s | 5/5 |
| 5 | `lsq_cosseno40` | **6.53 %** | 6.50 % ± 0.39 % | 7.18 % | 4.9e-03 | 0 s | 5/5 |
| 6 | `lsq_bspline80` | **6.73 %** | 6.76 % ± 0.27 % | 7.25 % | 5.0e-03 | 0 s | 5/5 |
| 7 | `direta` | **6.88 %** | 7.03 % ± 0.24 % | 7.46 % | 4.9e-03 | 0 s | 5/5 |
| 8 | `pinn_mixfunn_lm` | **25.7 %** | 17.9 % ± 9.66 % | 26.1 % | 1.3e-02 | 423 s | 5/5 |
| 9 | `pinn_mixfunn_lbfgs` | **26.0 %** | 25.9 % ± 0.79 % | 26.8 % | 1.3e-02 | 119 s | 5/5 |
| 10 | `pinn_mixfunn_original` | **57.6 %** | 68.2 % ± 36.1 % | 122 % | 1.3e-01 | 140 s | 5/5 |

Figuras: [erro por seed](figuras/fig_erro_c3_r0.01.pdf) · [campo recuperado](figuras/fig_campos_c3_r0.01.pdf)
