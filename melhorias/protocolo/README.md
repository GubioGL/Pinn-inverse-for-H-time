# Protocolo único de comparação: campo externo f(t) a partir de ⟨σx⟩, ⟨σy⟩, ⟨σz⟩

Compara, **com os mesmos dados e as mesmas regras**, os 4 métodos do repositório (campo como
rede neural, Fourier, B-spline e MixFunn) treinados de várias formas, contra métodos clássicos
sem PINN.

$$H(t)=\tfrac12\sigma_z+\tfrac12 f(t)\sigma_x,\qquad \dot X=-Y,\ \dot Y=X-fZ,\ \dot Z=fY$$

## Regras do protocolo (`config.py`)

| item | valor |
|---|---|
| estado inicial | **\|0⟩** (vetor de Bloch (0,0,1)): o campo é observável em todo o intervalo |
| dados | N = 100 pontos de ⟨σx⟩,⟨σy⟩,⟨σz⟩, iguais para todos os métodos (gerados por `(caso, ruído, seed)`) |
| campo verdadeiro | **nunca** entra em loss nem em escolha de hiperparâmetro |
| regularização | fixada *a priori* pelo nível de ruído (`LAM_D2`, `LAM_F`), igual para todos |
| seeds | 5 (inicialização + realização do ruído) |
| métricas | `erro_campo` = ‖f_pred − f_true‖/‖f_true‖ · `rmse_traj` (re-simulação vs. trajetória verdadeira) · `rmse_dados` · tempo |

## Métodos

**Clássicos (sem PINN):** `direta` (inversão ponto a ponto), `lsq_bspline40` e `lsq_fourier10`
(mínimos quadrados lineares nos coeficientes) e `shooting_bs40` (simulação + Levenberg-Marquardt,
partindo do LSQ).

**PINN:** `pinn_<campo>_<receita>` com campo ∈ {`nn`, `fourier`, `bspline`, `mixfunn`} e receita:

| receita | rede do estado | otimização |
|---|---|---|
| `original` | [15,15] sin, t cru, IC pela loss, colocação nos pontos dos dados, float32 | Adam lr=1e-3 fixo, 20k épocas (como nos notebooks) |
| `lbfgs` | 3×32 sin, t normalizado, **IC exata**, 200 pontos de colocação, float64 | Adam 3k (cosseno) → **L-BFGS** 5k |
| `bfgs` | idem | Adam 3k → **BFGS denso** 8k |
| `ssbfgs` | idem | Adam 3k → BFGS **auto-escalonado** (Urbán et al.) 8k |
| `lm` | idem | Adam 3k → **Levenberg-Marquardt** (Gauss-Newton) 600 it. |
| `lm_causal` | idem | Adam 3k **com pesos causais** (Wang et al.) → LM 600 it. |

## Instalação

```bash
cd melhorias/protocolo
pip install -r requirements.txt
```

Com GPU NVIDIA, instale o PyTorch com CUDA (escolha a versão do seu driver em pytorch.org):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Confira o ambiente (mostra as GPUs e mede o tempo de um passo de Adam e de uma iteração de LM
em CPU e em cada GPU):

```bash
python verificar_ambiente.py
```

## Como rodar

Teste rápido, para conferir que tudo funciona (~1–5 min):

```bash
python rodar.py --perfil rapido
```

O protocolo padrão: casos 1, 2 e 3 sem ruído e caso 3 com σ = 0,01, 5 seeds, 16 métodos.
Leva ~1,5 h em 4 núcleos de CPU:

```bash
python rodar.py --perfil padrao --workers 4
```

Na GPU, vários processos dividem a(s) GPU(s). As redes são pequenas, então 4–8 processos por GPU
costumam aproveitar melhor a placa do que 1. Os jobs clássicos rodam na CPU:

```bash
python rodar.py --perfil completo --device cuda --workers 8
```

Depois de rodar, gere as tabelas e figuras:

```bash
python relatorio.py --perfil padrao
```

O resultado fica em `resultados/padrao/RESULTADOS.md`.

Opções úteis do `rodar.py`:

| opção | efeito |
|---|---|
| `--metodos lm lsq` | roda só os métodos cujo nome contém alguma das palavras |
| `--seeds 0 1 2 3 4 5 6 7 8 9` | troca as seeds do perfil |
| `--threads N` | threads de CPU por processo (padrão: CPUs / workers) |
| `--refazer` | refaz jobs já concluídos |
| `--dry-run` | só lista os jobs |

**O runner é retomável:** cada job grava `resultados/<perfil>/jobs/<metodo>__c<caso>__r<ruido>__s<seed>.json`.
Se a execução for interrompida, rodar o mesmo comando de novo pula o que já terminou.

Perfis (`config.py::PERFIS`):

| perfil | cenários (caso, σ) | métodos |
|---|---|---|
| `rapido` | (3, 0), 2 seeds, 10 % das épocas | clássicos + original/lbfgs/lm |
| `padrao` | (1,0) (2,0) (3,0) (3,0.01), 5 seeds | clássicos + original/lbfgs/lm × 4 campos |
| `completo` | casos 1–3 × σ ∈ {0, 0.01, 0.03}, 5 seeds | clássicos + original/lbfgs/bfgs/lm × 4 campos |
| `ablacao` | (3, 0), 5 seeds | lbfgs/bfgs/ssbfgs/lm/lm_causal × campos nn e bspline |

## Arquivos

| arquivo | conteúdo |
|---|---|
| `config.py` | protocolo, métodos, receitas, perfis |
| `pinn.py` | PINN funcional (vetor plano de parâmetros, derivada temporal analítica em modo forward, 4 campos, Jacobiano por ponto) |
| `otimizadores.py` | Adam (+ pesos causais), L-BFGS, BFGS denso / auto-escalonado, Levenberg-Marquardt |
| `executar.py` | executa um job e calcula as métricas |
| `rodar.py` | execução paralela e retomável (CPU/GPU) |
| `relatorio.py` | tabelas (mediana, média ± desvio, pior seed) e figuras |
| `verificar_ambiente.py` | confere dependências/GPU e mede velocidade |

Detalhes de implementação que importam:
- **Derivada temporal analítica:** cada camada propaga (h, dh/dt) numa única passada, então não
  é preciso `autograd.grad(create_graph=True)`. Isso fica conferido contra o autograd até 1e-17.
- **Loss = ‖r(θ)‖²** exatamente, onde r é o vetor de resíduos (EDO, dados, suavidade do campo).
  Por isso o LM e os outros otimizadores minimizam a mesma função.
- **Jacobiano do LM:** é montado a partir de Jacobianos por ponto (`vmap(jacrev)`), que custam
  O(pontos × parâmetros). Quando há mais parâmetros que resíduos, o passo é resolvido no espaço
  dos resíduos, (JJᵀ + μI)a = r, δ = −Jᵀa. O amortecimento μ segue a regra de Nielsen.
