# Métodos para melhorar PINNs: o que a literatura recente diz e o que foi testado aqui

Resumo da pesquisa e do que foi implementado em `src/otimizadores.py` / `src/pinn.py`. Os números
medidos estão em `data/resultados/*/RESULTADOS.md` e no resumo de [`ANALISE.md`](ANALISE.md).

**Resultado medido na v1** (perfil `v1_ablacao`: caso 3, sem ruído, estado |0⟩, 5 seeds; mesma rede 3×32,
mesmos 3000 passos de Adam antes do 2º estágio; mediana do erro do campo):

| 2º estágio | campo = rede (`nn`) | campo = B-spline 40 | tempo/seed (CPU, 1 thread) |
|---|---|---|---|
| nenhum (receita `original`, só Adam 20k) | 25,9 % | 25,3 % | 25 s |
| L-BFGS 5000 it. | 3,3 % | 2,5 % | 21 s |
| BFGS denso auto-escalonado 8000 it. | 10,0 % | 9,9 % | 105 s |
| BFGS denso (τ = 1) 8000 it. | 1,3 % | 1,1 % | 107 s |
| **Levenberg-Marquardt 600 it.** | **0,10 %** | **0,76 %** | 175 s |
| Levenberg-Marquardt + pesos causais no Adam | 0,17 % | 0,72 % | 160 s |
| *referência: mínimos quadrados lineares B-spline 40 (sem PINN)* | | *0,28 %* | *< 1 s* |

A B-spline com 40 funções tem um piso de representação de ~0,3 %, então o LM com B-spline
para em ~0,7 %. O campo como rede não tem esse piso e chega a 0,10 %, melhor que todos os
métodos clássicos testados.

## 1. Otimização de segunda ordem: a mudança de maior impacto

O ablation da primeira etapa já mostrava que o gargalo deste problema é o otimizador: o Adam
para num platô em que a trajetória já está boa, mas o campo ainda não. A literatura de
2023–2025 chega à mesma conclusão para PINNs em geral.

| método | ideia | referência | status aqui |
|---|---|---|---|
| **Adam → L-BFGS** | quasi-Newton com memória limitada depois do Adam; padrão em PINNs | Rathore et al., *Challenges in Training PINNs: A Loss Landscape Perspective*, ICML 2024 ([arXiv:2402.01868](https://arxiv.org/abs/2402.01868)). Mostram que a loss do PINN é mal condicionada por causa do operador diferencial e que Adam+L-BFGS é melhor que cada um sozinho. Propõem também o NysNewton-CG. | receita `lbfgs` |
| **BFGS denso / auto-escalonado (SSBFGS, SSBroyden)** | Hessiana inversa completa, com auto-escalonamento de Oren–Luenberger/Al-Baali | Urbán, Stefanou & Pons, *Unveiling the optimization process of PINNs: How accurate and competitive can PINNs be?*, J. Comput. Phys. 2025 ([arXiv:2405.04230](https://arxiv.org/abs/2405.04230)); *Which Optimizer Works Best for PINNs and KANs?* ([arXiv:2501.16371](https://arxiv.org/abs/2501.16371)). Redes compactas (2–3 camadas de 20–30 neurônios) chegam a precisão de diferenças finitas. | receitas `bfgs` (τ=1) e `ssbfgs`. Medido: BFGS denso 1,1–1,3 %, auto-escalonado ~10 %. Minha implementação do auto-escalonamento (τ = min(1, sᵀy/yᵀHy)) é uma versão simplificada e não reproduziu os ganhos do artigo; vale testar a fórmula exata do SSBroyden deles |
| **Gauss-Newton / natural gradient (ENGD)** | usa a estrutura de mínimos quadrados da loss: passo de Gauss-Newton no espaço de funções | Müller & Zeinhofer, *Achieving High Accuracy with PINNs via Energy Natural Gradient Descent*, ICML 2023 ([arXiv:2302.13163](https://arxiv.org/abs/2302.13163)); Jnini, Vella & Zeinhofer, *Gauss-Newton Natural Gradient Descent for Physics-Informed CFD* ([arXiv:2402.10680](https://arxiv.org/abs/2402.10680)) | **receita `lm`** (Levenberg-Marquardt = Gauss-Newton amortecido) |
| **Dual natural gradient** | resolver o passo de Gauss-Newton no espaço dos resíduos, (JJᵀ+μI)a = r, quando há mais parâmetros que resíduos | Jnini & Vella, *Dual Natural Gradient Descent for Scalable Training of PINNs* ([arXiv:2505.21404](https://arxiv.org/abs/2505.21404)) | usado automaticamente no `lm` quando P > R |

Por que o LM funciona tão bem aqui: a loss é uma soma de quadrados de resíduos, e com
float64 e poucos milhares de parâmetros o Jacobiano completo cabe na memória (R ≈ 1100
resíduos × P ≈ 2300 parâmetros). Gauss-Newton converge muito mais rápido que métodos de
primeira ordem quando o resíduo no ótimo é quase zero, que é o caso sem ruído.

Detalhes da implementação (`otimizadores.py::lm`):
- O Jacobiano é montado a partir de Jacobianos **por ponto** (`vmap(jacrev)` de uma função que
  devolve [X, Y, Z, Ẋ, Ẏ, Ż, f] em um ponto). O custo é O(pontos × parâmetros). Depois a regra
  da cadeia monta as linhas da EDO, dos dados e da regularização.
- O amortecimento μ segue a regra de Nielsen (razão entre a redução real e a prevista).
- A loss do PINN é **exatamente** ‖r‖², então LM, BFGS e Adam minimizam a mesma função.

## 2. Outras técnicas da literatura

| técnica | ideia | referência | status aqui |
|---|---|---|---|
| **Pesos causais** | em problemas de evolução temporal, pesar o resíduo de cada bloco de tempo por exp(−ε · resíduo acumulado dos blocos anteriores), para o treino respeitar a ordem temporal | Wang, Sankaran & Perdikaris, *Respecting causality is all you need for training PINNs* ([arXiv:2203.07404](https://arxiv.org/abs/2203.07404)) | receita `lm_causal`. Medido: **sem ganho** (0,17 % vs 0,10 % com nn; 0,72 % vs 0,76 % com B-spline, dentro da variação entre seeds). Faz sentido: aqui há dados densos em todo o intervalo, e eles já "ancoram" todos os tempos |
| **Restrições rígidas** | impor condições por construção na saída da rede: r(t) = r0 + (t/T)·NN(t) e, quando f(0), f(T) são conhecidos pelo desenho do pulso, f = g − (1−τ)g(0) − τg(T) + (1−τ)a + τb | Lu et al., *PINNs with hard constraints for inverse design*, SIAM J. Sci. Comput. 2021 | **em todas as receitas melhoradas** (v2). Efeito da borda do campo: receita `lm_sembc`, perfil `ablacao` |
| **Normalização da entrada** | t → s = 2t/T − 1 | prática padrão | em todas as receitas melhoradas |
| **Multi-stage NN** | treinar uma 2ª rede no resíduo da 1ª, e assim por diante; chega perto da precisão de máquina | Wang & Lai, *Multi-stage neural networks: Function approximator of machine precision*, J. Comput. Phys. 2024 ([arXiv:2307.08934](https://arxiv.org/abs/2307.08934)) | não implementado. Candidato natural depois do LM, se for preciso ir abaixo de 1e-3 |
| **Pesos adaptativos** (gradient-norm) | ajustar o peso da loss de dados pela razão max\|∇L_EDO\| / mean\|∇L_dados\| | Wang, Teng & Perdikaris, SIAM J. Sci. Comput. 2021 (arXiv:2001.04536); variantes: SA-PINN (McClenny & Braga-Neto, arXiv:2009.04544), RBA (Anagnostopoulos et al., arXiv:2307.00379) | **implementado (v2)** no Adam (`pesos_adapt`, receita `lm_pesos`). Volta a 1 no estágio de 2ª ordem, para todos os métodos minimizarem a mesma função |
| **Colocação aleatória reamostrada** | pontos da EDO = dados + pontos uniformes aleatórios sorteados a cada época | Wu et al., *A comprehensive study of non-adaptive and residual-based adaptive sampling for PINNs*, CMAME 2023 | **implementado (v2)**, padrão nas receitas melhoradas. Controles: `lm_semreamostra` (sorteio fixo), `lm_soDados` (EDO só nos pontos dos dados) |
| **Variable projection** | coeficientes de campos lineares resolvidos em forma fechada (mínimos quadrados) a cada passo | Golub & Pereyra, *Separable nonlinear least squares: the variable projection method*, Inverse Problems 2003 | **implementado (v2)** no Adam (`varpro`, receita `lm_varpro`) |
| **Warm start por mínimos quadrados** | coeficientes iniciais do campo a partir dos dados (derivadas por spline) | — | **implementado (v2)**, padrão nas receitas melhoradas; controle `lm_seminit` |
| Fourier features / PirateNets | combater o viés espectral da rede | Tancik et al. 2020; Wang et al., *PirateNets* (arXiv:2402.00326) | não implementado. O campo tem componente sin 7t, então pode ajudar a rede do estado |
| NysNewton-CG | Newton-CG com precondicionador de Nyström | Rathore et al. 2024 | não implementado. Alternativa ao LM para redes grandes |

## 3. PINN em problemas quânticos inversos (contexto)

- Norambuena, Mattheakis, González & Coto, *Physics-Informed Neural Networks for Quantum Control*,
  Phys. Rev. Lett. 132, 010801 (2024) ([arXiv:2206.06287](https://arxiv.org/abs/2206.06287)):
  PINN para encontrar campos de controle em sistemas abertos. É o problema "direto" de controle,
  não a identificação de um campo desconhecido a partir de dados.
- *Hamiltonian Learning via Inverse Physics-Informed Neural Networks* (iPINN-HL),
  [arXiv:2506.10379](https://arxiv.org/abs/2506.10379), Phys. Rev. Research 7, 043137 (2025):
  identifica parâmetros constantes do Hamiltoniano com a equação de Schrödinger como
  penalidade.
- *PINNverse* para sistemas abertos (Lindblad), [arXiv:2507.12552](https://arxiv.org/abs/2507.12552):
  identifica parâmetros e taxas de decaimento a partir de dados ruidosos.

Não encontrei, nessa busca, um trabalho que faça exatamente o que este repositório faz:
identificar um **campo dependente do tempo** a partir de valores esperados, comparando
parametrizações do campo e otimizadores sob um protocolo único. Isso sugere que a comparação
deste repositório tem espaço como contribuição. Vale uma busca bibliográfica mais completa antes
de afirmar isso num artigo.
