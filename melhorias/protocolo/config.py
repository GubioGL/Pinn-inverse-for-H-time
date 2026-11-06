"""
Protocolo único de comparação (o mesmo para TODOS os métodos).

Regras
------
* Mesmos dados para todos os métodos: para cada (caso, ruído, seed) os dados são gerados
  uma única vez, de forma determinística (equações de Bloch, scipy, tolerância 1e-11).
* Estado inicial |0>  (vetor de Bloch (0,0,1)): o campo é observável em todo o intervalo.
* N = 100 pontos de <sx>, <sy>, <sz> igualmente espaçados em [0, T].
* O campo verdadeiro NÃO entra em nenhuma loss nem em nenhuma escolha de hiperparâmetro.
  A regularização depende só do nível de ruído (que no experimento é conhecido pelo
  número de shots), com valores fixados a priori.
* >= 5 seeds. A seed controla a inicialização e a realização do ruído.
* Métricas (calculadas no mesmo grid dos dados para todos):
    erro_campo  = ||f_pred - f_true|| / ||f_true||
    rmse_traj   = RMSE( simulação com f_pred  -  trajetória verdadeira sem ruído )
    rmse_dados  = RMSE( simulação com f_pred  -  dados medidos )
    tempo_s

Perfis
------
  rapido   : teste de fumaça (caso 3, sem ruído, 2 seeds, épocas reduzidas)
  padrao   : casos 1, 2 e 3 sem ruído + caso 3 com ruído 0.01; 5 seeds
  completo : casos 1, 2, 3 x ruído {0, 0.01, 0.03}; 5 seeds (para GPU / muitas CPUs)
  ablacao  : caso 3, sem ruído, 5 seeds: otimizadores e pesos causais no PINN
"""
import numpy as np

N_PONTOS = 100
R0 = (0.0, 0.0, 1.0)          # |0>

# regularização fixada a priori pelo nível de ruído (não usa o campo verdadeiro)
#   LSQ / shooting : lam * ||D2 c||^2  (2a diferença dos coeficientes da B-spline)
#   PINN           : lam_f * (1/T) * integral f''(t)^2 dt  (escala equivalente)
LAM_D2 = {0.0: 1e-6, 0.01: 1e-3, 0.03: 1e-2}
LAM_F = {0.0: 1e-10, 0.01: 1e-6, 0.03: 1e-5}


def lam_d2(ruido):
    return LAM_D2[min(LAM_D2, key=lambda k: abs(k - ruido))]


def lam_f(ruido):
    return LAM_F[min(LAM_F, key=lambda k: abs(k - ruido))]


# ---------------------------------------------------------------------------
# Métodos
# ---------------------------------------------------------------------------
# Métodos clássicos (sem PINN)
CLASSICOS = {
    "direta":        dict(tipo="classico", alg="direta"),
    "lsq_bspline40": dict(tipo="classico", alg="lsq", base="bspline", tam=40),
    "lsq_fourier10": dict(tipo="classico", alg="lsq", base="fourier", tam=10),
    "shooting_bs40": dict(tipo="classico", alg="shooting", tam=40),
}

# Parametrizações do campo (os 4 métodos do repositório)
CAMPOS = {
    "nn":      dict(field="nn"),
    "fourier": dict(field="fourier", field_size=10),
    "bspline": dict(field="bspline", field_size=40, spline_k=3),
    "mixfunn": dict(field="mixfunn", mixfunn=dict(N_layers=2, N_neuro=4, second_order_function=True)),
}

# Receitas de treino (iguais para todas as parametrizações)
ARQ_ORIGINAL = dict(width=15, depth=2, act="sin", norm_time=False, hard_ic=False, n_col=None,
                    mixfunn_input="t", dtype="float32")
ARQ_MELHORADA = dict(width=32, depth=3, act="sin", norm_time=True, hard_ic=True, n_col=200,
                     mixfunn_input="s", dtype="float64")

RECEITAS = {
    # como nos notebooks: só Adam, lr fixo, rede pequena, t cru, IC "mole"
    "original": dict(arq=ARQ_ORIGINAL,
                     estagios=[dict(tipo="adam", epochs=20000, lr=1e-3, sched="const")]),
    # arquitetura melhorada + Adam (cosseno) + L-BFGS
    "lbfgs":    dict(arq=ARQ_MELHORADA,
                     estagios=[dict(tipo="adam", epochs=3000, lr=1e-3, sched="cos"),
                               dict(tipo="lbfgs", iters=5000)]),
    # arquitetura melhorada + Adam + BFGS denso (Hessiana inversa completa)
    "bfgs":     dict(arq=ARQ_MELHORADA,
                     estagios=[dict(tipo="adam", epochs=3000, lr=1e-3, sched="cos"),
                               dict(tipo="ssbfgs", iters=8000, auto_escala=False)]),
    # arquitetura melhorada + Adam + Levenberg-Marquardt (Gauss-Newton amortecido)
    "lm":       dict(arq=ARQ_MELHORADA,
                     estagios=[dict(tipo="adam", epochs=3000, lr=1e-3, sched="cos"),
                               dict(tipo="lm", iters=600)]),
    # LM com Adam inicial usando pesos causais (Wang, Sankaran & Perdikaris 2022)
    "lm_causal": dict(arq=ARQ_MELHORADA,
                      estagios=[dict(tipo="adam", epochs=3000, lr=1e-3, sched="cos", causal_eps=1.0),
                                dict(tipo="lm", iters=600)]),
    # BFGS com auto-escalonamento de Oren-Luenberger/Al-Baali (Urbán et al. 2025)
    "ssbfgs":   dict(arq=ARQ_MELHORADA,
                     estagios=[dict(tipo="adam", epochs=3000, lr=1e-3, sched="cos"),
                               dict(tipo="ssbfgs", iters=8000, auto_escala=True)]),
}


def metodos_pinn(receitas, campos=tuple(CAMPOS)):
    out = {}
    for r in receitas:
        for c in campos:
            out[f"pinn_{c}_{r}"] = dict(tipo="pinn", campo=c, receita=r)
    return out


def _epocas(metodos, fator):
    """Cópia dos métodos com épocas/iterações reduzidas (perfil rápido)."""
    return {k: dict(v, fator=fator) for k, v in metodos.items()}


PERFIS = {
    "rapido": dict(
        cenarios=[(3, 0.0)], seeds=[0, 1],
        metodos={**CLASSICOS, **_epocas(metodos_pinn(["original", "lbfgs", "lm"]), 0.1)},
        t_max_job=300),
    "padrao": dict(
        cenarios=[(1, 0.0), (2, 0.0), (3, 0.0), (3, 0.01)], seeds=[0, 1, 2, 3, 4],
        metodos={**CLASSICOS, **metodos_pinn(["original", "lbfgs", "lm"])},
        t_max_job=900),
    "completo": dict(
        cenarios=[(c, r) for c in (1, 2, 3) for r in (0.0, 0.01, 0.03)], seeds=[0, 1, 2, 3, 4],
        metodos={**CLASSICOS, **metodos_pinn(["original", "lbfgs", "bfgs", "lm"])},
        t_max_job=1800),
    "ablacao": dict(
        cenarios=[(3, 0.0)], seeds=[0, 1, 2, 3, 4],
        metodos={**metodos_pinn(["lbfgs", "bfgs", "ssbfgs", "lm", "lm_causal"], ("nn", "bspline"))},
        t_max_job=900),
}


def custo_estimado(m):
    """Peso aproximado para ordenar os jobs (os mais caros primeiro)."""
    if m["tipo"] == "classico":
        return 0.1
    base = {"original": 1.0, "lbfgs": 1.0, "bfgs": 2.0, "ssbfgs": 2.0, "lm": 4.0, "lm_causal": 4.0}
    return base.get(m["receita"], 1.0) * (5.0 if m["campo"] == "mixfunn" else 1.0) * m.get("fator", 1.0)
