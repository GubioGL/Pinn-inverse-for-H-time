# Resultados do perfil `ablacao`

50 jobs, 50 ok. Estado inicial |0⟩, N = 100, campo verdadeiro fora da loss. Erro = ||f_pred − f_true|| / ||f_true||.

## Mediana do erro do campo por cenário

| método | caso 3, σ=0 |
|---|---|
| `pinn_nn_lm` | 0.10 % |
| `pinn_nn_lm_causal` | 0.17 % |
| `pinn_bspline_lm_causal` | 0.72 % |
| `pinn_bspline_lm` | 0.76 % |
| `pinn_bspline_bfgs` | 1.10 % |
| `pinn_nn_bfgs` | 1.34 % |
| `pinn_bspline_lbfgs` | 2.45 % |
| `pinn_nn_lbfgs` | 3.32 % |
| `pinn_bspline_ssbfgs` | 9.86 % |
| `pinn_nn_ssbfgs` | 10.0 % |


### Caso 3, ruído σ = 0

| # | método | erro do campo (mediana) | média ± desvio | pior seed | RMSE trajetória (mediana) | tempo médio | seeds ok |
|---|---|---|---|---|---|---|---|
| 1 | `pinn_nn_lm` | **0.10 %** | 0.13 % ± 0.04 % | 0.21 % | 2.4e-05 | 174 s | 5/5 |
| 2 | `pinn_nn_lm_causal` | **0.17 %** | 0.15 % ± 0.04 % | 0.20 % | 3.7e-05 | 158 s | 5/5 |
| 3 | `pinn_bspline_lm_causal` | **0.72 %** | 0.67 % ± 0.12 % | 0.84 % | 2.1e-04 | 175 s | 5/5 |
| 4 | `pinn_bspline_lm` | **0.76 %** | 0.74 % ± 0.13 % | 0.93 % | 2.1e-04 | 183 s | 5/5 |
| 5 | `pinn_bspline_bfgs` | **1.10 %** | 1.24 % ± 0.51 % | 2.25 % | 3.7e-04 | 108 s | 5/5 |
| 6 | `pinn_nn_bfgs` | **1.34 %** | 1.16 % ± 0.39 % | 1.67 % | 3.8e-04 | 106 s | 5/5 |
| 7 | `pinn_bspline_lbfgs` | **2.45 %** | 3.01 % ± 1.53 % | 5.88 % | 9.2e-04 | 21 s | 5/5 |
| 8 | `pinn_nn_lbfgs` | **3.32 %** | 3.50 % ± 0.46 % | 4.41 % | 1.3e-03 | 21 s | 5/5 |
| 9 | `pinn_bspline_ssbfgs` | **9.86 %** | 9.92 % ± 3.64 % | 16.1 % | 4.6e-03 | 106 s | 5/5 |
| 10 | `pinn_nn_ssbfgs` | **10.0 %** | 10.2 % ± 2.61 % | 14.8 % | 4.6e-03 | 105 s | 5/5 |

![erros](fig_erro_c3_r0.png)

![campos](fig_campos_c3_r0.png)
