# Protocolo único de comparação

Compara, **com os mesmos dados e as mesmas regras**, os 4 métodos do repositório (campo como rede
neural, Fourier, B-spline e MixFunn), mais a série de cossenos sugerida na análise. Cada um é
treinado de várias formas e comparado com métodos clássicos sem PINN.

$$H(t)=\tfrac12\sigma_z+\tfrac12 f(t)\sigma_x,\qquad \dot X=-Y,\ \dot Y=X-fZ,\ \dot Z=fY$$

## Regras (`src/protocolo.py`)

| item | valor |
|---|---|
| estado inicial | **\|0⟩**. Justificativa em [`ANALISE.md`](ANALISE.md#por-que-o-estado-inicial-0-e-não-) |
| dados | N = 100 pontos de ⟨σx⟩, ⟨σy⟩, ⟨σz⟩, iguais para todos os métodos (gerados por `(caso, ruído, seed)`) |
| campo verdadeiro | **nunca** entra em loss nem em escolha de hiperparâmetro |
| bordas do campo | impostas de forma rígida **só quando conhecidas pelo desenho do pulso**: f(0) = f(T) = 0 no caso 3 e nos pulsos do estudo espectral; nenhuma nos casos 1 e 2 |
| regularização | fixada *a priori* pelo nível de ruído (`LAM_D2`, `LAM_F`), igual para todos; zero sem ruído |
| seeds | 5 nos perfis principais (inicialização, ruído e sorteio dos pontos de colocação) |
| métricas | **sempre duas:** `erro_campo` = ‖f_pred − f_true‖/‖f_true‖ e `rmse_traj` (re-simulação com f_pred vs. trajetória verdadeira). Também `rmse_dados` e o tempo |
| orçamento | a mesma rede do estado, o mesmo nº de épocas de Adam e o mesmo 2º estágio para todos os campos |

## Métodos

**Clássicos (sem PINN):**
- `direta`: inversão ponto a ponto.
- `lsq_fourier10`, `lsq_cosseno40`, `lsq_bspline40`, `lsq_bspline80`: mínimos quadrados lineares nos coeficientes.
- `shooting_bs40`, `shooting_bs80`: simulação + Levenberg-Marquardt, partindo do LSQ.

**PINN:** `pinn_<campo>_<receita>`.

Campos:

| campo | descrição |
|---|---|
| `nn` | 4ª saída da rede do estado (modelo_2) |
| `fourier` | Fourier periódica M=10, como no modelo_3 |
| `cosseno` | série de cossenos M=40, sem Gibbs quando f(0) ≠ f(T) |
| `bspline` | B-spline cúbica com 80 funções (o modelo_4 usava grau 10 e 30 funções) |
| `mixfunn` | MixFunn do Modelo_5 |

Receitas:

| receita | rede do estado e colocação | otimização |
|---|---|---|
| `original` | [15,15] sin, t cru, IC pela loss, EDO **só nos pontos dos dados**, float32 | Adam lr=1e-3 fixo, 20k épocas (como nos notebooks) |
| `lbfgs` | 3×32 sin, t normalizado, **restrições rígidas**, EDO nos **dados + 200 pontos aleatórios sorteados a cada época**, float64, coeficientes iniciais por mínimos quadrados | Adam 3k (cosseno) → **L-BFGS** 5k |
| `bfgs` / `ssbfgs` | idem | Adam 3k → BFGS denso / auto-escalonado 8k |
| `lm` | idem | Adam 3k → **Levenberg-Marquardt** 600 it. |
| `lm_causal` | idem | Adam 3k com pesos causais → LM |
| ablation | `lm_sembc`, `lm_seminit`, `lm_semreamostra`, `lm_soDados`, `lm_varpro`, `lm_pesos`, `lm_reamostraLM` | cada uma muda UMA coisa em relação a `lm` |

### Restrições rígidas (na saída da rede)

- Estado: $r(t) = r_0 + (t/T)\,\mathrm{NN}(t)$, então $r(0)=r_0$ exatamente.
- Campo, quando $(a,b) = (f(0), f(T))$ é conhecido:
  $f(t) = g(t) - (1-\tau)\,g(0) - \tau\,g(T) + (1-\tau)\,a + \tau\,b$, com $\tau=t/T$.
  - $g$ é a saída "livre" de qualquer parametrização (rede, Fourier, cossenos, B-spline,
    MixFunn).
  - A correção é linear em $g$, então as bases continuam lineares nos coeficientes, e
    mínimos quadrados, *variable projection* e LM continuam valendo.
- **No tempo final, o estado não tem condição de contorno:** é um problema de valor inicial.
  O estado em T só é conhecido pelos dados, que têm ruído. Impor $r(T)$ = dado seria injetar o
  ruído de um ponto como restrição exata.

### Pontos de colocação

A EDO é imposta nos N pontos dos dados **mais** `n_rand = 200` pontos sorteados uniformemente em
[0, T] **a cada época do Adam**. Ao longo do treino, a rede vê centenas de milhares de pontos
diferentes. O L-BFGS, o BFGS e o LM precisam de uma função objetivo fixa durante a busca de
linha, então usam o último sorteio. O LM tem a opção `reamostrar_cada=K`, que sorteia de novo a
cada K iterações. A regularização do campo usa um grid uniforme separado de 200 pontos.

### Sugestões implementadas para Fourier e B-spline

| sugestão | onde |
|---|---|
| Série de **cossenos** no lugar da Fourier periódica (sem Gibbs) | campo `cosseno`, `lsq_cosseno40` |
| B-spline **cúbica** e com **mais funções** (80) | campo `bspline`, `lsq_bspline80`, `shooting_bs80` |
| Coeficientes **iniciados por mínimos quadrados** a partir dos dados | `init_lsq` (padrão nas receitas melhoradas) |
| **Variable projection**: coeficientes resolvidos em forma fechada a cada passo | `varpro=True` no Adam (receita `lm_varpro`) |
| Sem `weight_decay` nos coeficientes; penalidade de 2ª diferença só com ruído | `LAM_F` |
| Bordas do campo impostas na saída quando conhecidas | `bc="conhecida"` |

### Pesos adaptativos

O Adam pode ajustar o peso da loss de dados pela razão das normas dos gradientes,
w ← (1−α) w + α · max|∇L_EDO| / mean|∇L_dados| (Wang, Teng & Perdikaris, 2021). Opção
`pesos_adapt=True`, receita `lm_pesos`. No fim do Adam o peso volta a 1. Assim o LM minimiza a
mesma função objetivo em todos os métodos.

## Instalação

```bash
pip install -r run/requirements.txt
```

Com GPU NVIDIA, instale também o PyTorch com CUDA (escolha a versão do seu driver em pytorch.org):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Para conferir o ambiente:

```bash
python run/verificar_ambiente.py
```

## Como rodar

Teste rápido (~3 min):

```bash
python run/rodar.py --perfil rapido
```

Protocolo padrão (~3 h em 4 CPUs):

```bash
python run/rodar.py --perfil padrao --workers 4
```

Na GPU, com vários processos por placa:

```bash
python run/rodar.py --perfil completo --device cuda --workers 8
```

Os outros perfis:

```bash
python run/rodar.py --perfil ablacao
```

```bash
python run/rodar.py --perfil espectral
```

Tabelas e figuras de diagnóstico de um perfil, em `data/resultados/<perfil>/`:

```bash
python run/relatorio.py --perfil padrao
```

Figuras do artigo (vão para `paper/figures/*.pdf`):

```bash
python run/fig_artigo.py
```

```bash
python run/fig_espectral.py
```

```bash
python run/diagnostico.py
```

O runner é **retomável**: cada job grava `data/resultados/<perfil>/jobs/<metodo>__c<caso>__r<ruido>__s<seed>.json`
e, se a execução for interrompida, rodar o mesmo comando de novo pula o que já terminou.

Opções úteis do `rodar.py`:

| opção | efeito |
|---|---|
| `--metodos lm lsq` | roda só os métodos cujo nome contém alguma das palavras |
| `--seeds 0 1 2 3 4 5 6 7 8 9` | troca as seeds do perfil |
| `--threads N` | threads de CPU por processo (padrão: CPUs / workers) |
| `--refazer` | refaz os jobs já concluídos |
| `--dry-run` | só lista os jobs |

| perfil | cenários (caso, σ) | métodos |
|---|---|---|
| `rapido` | (3, 0), 2 seeds, 10 % das épocas | clássicos + original/lbfgs/lm × 5 campos |
| `padrao` | (1,0) (2,0) (3,0) (3,0.01), 5 seeds | clássicos + original/lbfgs/lm × 5 campos |
| `completo` | casos 1–3 × σ ∈ {0, 0.01, 0.03}, 5 seeds | clássicos + original/lbfgs/bfgs/lm × 5 campos |
| `otimizadores` | (3, 0), 5 seeds | lbfgs/bfgs/ssbfgs/lm/lm_causal × campos nn e bspline |
| `ablacao` | (3, 0), 3 seeds | variantes do `lm` × campos nn e bspline |
| `espectral` | pulsos sin(ωt)sin²(πt/T), ω ∈ {1,…,20}, σ ∈ {0, 0.01}, 2 seeds (mais com `--seeds`) | clássicos + PINN-LM (nn, bspline) |

## Arquivos

| arquivo | conteúdo |
|---|---|
| `src/bloch.py` | equações de Bloch, casos (incluindo os pulsos do estudo espectral), bases numpy, métricas |
| `src/classicos.py` | inversão direta, mínimos quadrados lineares, shooting (multi-experimento) |
| `src/pinn.py` | PINN funcional: vetor plano de parâmetros, derivada temporal analítica, 5 campos, bases em torch, restrições rígidas, colocação aleatória, Jacobiano por ponto, varpro, warm start |
| `src/otimizadores.py` | Adam (+ reamostragem, pesos causais, pesos adaptativos, varpro), L-BFGS, BFGS denso/auto-escalonado, Levenberg-Marquardt |
| `src/protocolo.py` | regras, métodos, receitas e perfis |
| `src/executar.py` | executa um job e calcula as métricas |
| `run/rodar.py` | execução paralela e retomável (CPU/GPU) |
| `run/relatorio.py` | tabelas (mediana, média ± desvio, pior seed) e figuras de diagnóstico |
| `run/fig_artigo.py`, `run/fig_espectral.py`, `run/diagnostico.py` | figuras do artigo |
| `run/verificar_ambiente.py` | confere dependências/GPU e mede velocidade |
