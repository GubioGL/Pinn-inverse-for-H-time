"""
Protocolo único de comparação (o mesmo para TODOS os métodos).

Regras
------
* Mesmos dados para todos os métodos: para cada (caso, ruído, seed) os dados são gerados
  uma única vez, de forma determinística (equações de Bloch, scipy, tolerância 1e-11).
* Estado inicial |0> (vetor de Bloch (0,0,1)). Justificativa em notes/ANALISE.md
  ("Por que |0> e não |+>"): com |+>, autoestado do operador de controle sx, o campo não tem
  efeito sobre o estado em t ~ 0 (Y^2 + Z^2 = 0) e não é identificável ali. Com |0>,
  Y^2 + Z^2 >= 0.13 em todo o intervalo nos três casos, e com o mesmo orçamento de medidas o
  erro do campo cai pela metade.
* N = 100 pontos de <sx>, <sy>, <sz> igualmente espaçados em [0, T].
* O campo verdadeiro NÃO entra em nenhuma loss nem em nenhuma escolha de hiperparâmetro.
  - A regularização depende só do nível de ruído (conhecido pelo nº de shots), fixada a priori.
  - As bordas do campo só são impostas quando são conhecidas pelo desenho do pulso
    (bloch.bordas_conhecidas): f(0) = f(T) = 0 no caso 3 e nos pulsos do estudo espectral.
* >= 5 seeds (perfis principais). A seed controla a inicialização, a realização do ruído e o
  sorteio dos pontos de colocação.
* Duas métricas sempre (calculadas no grid dos dados para todos os métodos):
    erro_campo  = ||f_pred - f_true|| / ||f_true||
    rmse_traj   = RMSE( simulação com f_pred  -  trajetória verdadeira sem ruído )
  e ainda rmse_dados (simulação vs. dados medidos) e tempo_s.

Perfis
------
  rapido      : teste de fumaça (caso 3, sem ruído, 2 seeds, 10 % das épocas)
  padrao      : casos 1, 2, 3 sem ruído + caso 3 com ruído 0.01; 5 seeds
  completo    : casos 1, 2, 3 x ruído {0, 0.01, 0.03}; 5 seeds (para GPU / muitas CPUs)
  otimizadores: caso 3, sem ruído, 5 seeds: L-BFGS, BFGS, SS-BFGS, LM, LM+causal
  ablacao     : caso 3, sem ruído, 2 seeds, metade das épocas/iterações: efeito de cada novidade (colocação aleatória,
                borda rígida, warm start, variable projection, pesos adaptativos)
  espectral   : pulsos sin(w t) sin^2(pi t/T) com w = 1 ... 20; ruído 0 e 0.01; 2 seeds
                (use --seeds 0 1 2 3 4 numa GPU para mais seeds)
"""
N_PONTOS = 100
R0 = (0.0, 0.0, 1.0)          # |0>

# regularização fixada a priori pelo nível de ruído (não usa o campo verdadeiro)
#   LSQ / shooting : lam * ||D2 c||^2  (2a diferença dos coeficientes da B-spline de 40 funções;
#                    para n funções usa-se lam * (n/40)^3, a mesma suavidade em termos de f'')
#   PINN           : lam_f * (1/T) * integral f''(t)^2 dt  (escala equivalente)
LAM_D2 = {0.0: 1e-6, 0.01: 1e-3, 0.03: 1e-2}
LAM_F = {0.0: 0.0, 0.01: 1e-6, 0.03: 1e-5}


def _mais_proximo(tab, ruido):
    return tab[min(tab, key=lambda k: abs(k - ruido))]


def lam_d2(ruido, n_basis=40):
    return _mais_proximo(LAM_D2, ruido) * (n_basis / 40.0) ** 3


def lam_f(ruido):
    return _mais_proximo(LAM_F, ruido)


# ---------------------------------------------------------------------------
# Métodos clássicos (sem PINN)
# ---------------------------------------------------------------------------
CLASSICOS = {
    "direta":        dict(tipo="classico", alg="direta"),
    "lsq_fourier10": dict(tipo="classico", alg="lsq", base="fourier", tam=10),
    "lsq_cosseno40": dict(tipo="classico", alg="lsq", base="cosine", tam=40),
    "lsq_bspline40": dict(tipo="classico", alg="lsq", base="bspline", tam=40),
    "lsq_bspline80": dict(tipo="classico", alg="lsq", base="bspline", tam=80),
    "shooting_bs40": dict(tipo="classico", alg="shooting", tam=40),
    "shooting_bs80": dict(tipo="classico", alg="shooting", tam=80),
}

# ---------------------------------------------------------------------------
# Parametrizações do campo (os 4 métodos do repositório + série de cossenos)
# ---------------------------------------------------------------------------
CAMPOS = {
    "nn":      dict(field="nn"),
    "fourier": dict(field="fourier", field_size=10),                 # periódica, como no modelo_3
    "cosseno": dict(field="cosseno", field_size=40),                 # sugestão: sem Gibbs
    "bspline": dict(field="bspline", field_size=80, spline_k=3),     # sugestão: cúbica, mais funções
    "mixfunn": dict(field="mixfunn", mixfunn=dict(N_layers=2, N_neuro=4, second_order_function=True)),
}

# ---------------------------------------------------------------------------
# Arquiteturas e receitas de treino (iguais para todas as parametrizações)
# ---------------------------------------------------------------------------
# como nos notebooks: rede pequena, t cru, IC pela loss, EDO só nos pontos dos dados, float32
ARQ_ORIGINAL = dict(width=15, depth=2, act="sin", norm_time=False, hard_ic=False, n_rand=0,
                    bc="nenhuma", init_lsq=False, mixfunn_input="t", dtype="float32")
# melhorada: t normalizado; restrições rígidas (r(0) = r0 e, quando conhecidas, f(0), f(T));
# EDO nos pontos dos dados + 200 pontos aleatórios sorteados a cada época;
# coeficientes dos campos lineares iniciados por mínimos quadrados a partir dos dados
ARQ_MELHORADA = dict(width=32, depth=3, act="sin", norm_time=True, hard_ic=True, n_rand=200,
                     n_reg=200, bc="conhecida", init_lsq=True, mixfunn_input="s", dtype="float64")

ADAM = dict(tipo="adam", epochs=3000, lr=1e-3, sched="cos", reamostrar=True)
LM = dict(tipo="lm", iters=600)


def _receita(arq=ARQ_MELHORADA, adam=None, segundo=None, arq_over=None):
    a = dict(arq)
    a.update(arq_over or {})
    est = [dict(ADAM, **(adam or {}))]
    if segundo is not None:
        est.append(segundo)
    return dict(arq=a, estagios=est)


RECEITAS = {
    "original":  dict(arq=ARQ_ORIGINAL,
                      estagios=[dict(tipo="adam", epochs=20000, lr=1e-3, sched="const", reamostrar=False)]),
    "lbfgs":     _receita(segundo=dict(tipo="lbfgs", iters=5000)),
    "bfgs":      _receita(segundo=dict(tipo="ssbfgs", iters=8000, auto_escala=False)),
    "ssbfgs":    _receita(segundo=dict(tipo="ssbfgs", iters=8000, auto_escala=True)),
    "lm":        _receita(segundo=LM),
    "lm_causal": _receita(adam=dict(causal_eps=1.0), segundo=LM),
    # --- ablation das novidades (cada uma tira ou acrescenta UMA coisa em relação a "lm")
    "lm_sembc":        _receita(segundo=LM, arq_over=dict(bc="nenhuma")),
    "lm_seminit":      _receita(segundo=LM, arq_over=dict(init_lsq=False)),
    "lm_semreamostra": _receita(adam=dict(reamostrar=False), segundo=LM),
    "lm_soDados":      _receita(adam=dict(reamostrar=False), segundo=LM, arq_over=dict(n_rand=0)),
    "lm_varpro":       _receita(adam=dict(varpro=True), segundo=LM),
    "lm_pesos":        _receita(adam=dict(pesos_adapt=True), segundo=LM),
    "lm_reamostraLM":  _receita(segundo=dict(LM, reamostrar_cada=20)),
}


def metodos_pinn(receitas, campos=tuple(CAMPOS)):
    return {f"pinn_{c}_{r}": dict(tipo="pinn", campo=c, receita=r) for r in receitas for c in campos}


def _epocas(metodos, fator):
    """Cópia dos métodos com épocas/iterações reduzidas (perfil rápido)."""
    return {k: dict(v, fator=fator) for k, v in metodos.items()}


OMEGAS = (1, 2, 4, 6, 8, 10, 12, 15, 20)

PERFIS = {
    "rapido": dict(
        cenarios=[(3, 0.0)], seeds=[0, 1],
        metodos={**CLASSICOS, **_epocas(metodos_pinn(["original", "lbfgs", "lm"]), 0.1)},
        t_max_job=300),
    "padrao": dict(
        cenarios=[(1, 0.0), (2, 0.0), (3, 0.0), (3, 0.01)], seeds=[0, 1, 2, 3, 4],
        metodos={**CLASSICOS, **metodos_pinn(["original", "lbfgs", "lm"])},
        t_max_job=420),
    "completo": dict(
        cenarios=[(c, r) for c in (1, 2, 3) for r in (0.0, 0.01, 0.03)], seeds=[0, 1, 2, 3, 4],
        metodos={**CLASSICOS, **metodos_pinn(["original", "lbfgs", "bfgs", "lm"])},
        t_max_job=1800),
    "otimizadores": dict(
        cenarios=[(3, 0.0)], seeds=[0, 1, 2, 3, 4],
        metodos=metodos_pinn(["lbfgs", "bfgs", "ssbfgs", "lm", "lm_causal"], ("nn", "bspline")),
        t_max_job=900),
    "ablacao": dict(
        cenarios=[(3, 0.0)], seeds=[0, 1],
        metodos=_epocas({**metodos_pinn(["lm", "lm_sembc", "lm_semreamostra", "lm_soDados", "lm_pesos"],
                                        ("nn", "bspline")),
                         **metodos_pinn(["lm_seminit", "lm_varpro"], ("bspline",))}, 0.5),
        t_max_job=240),
    "espectral": dict(
        cenarios=[(f"w{w}", r) for r in (0.0, 0.01) for w in OMEGAS], seeds=[0, 1],
        metodos={k: CLASSICOS[k] for k in ("direta", "lsq_fourier10", "lsq_cosseno40",
                                           "lsq_bspline80", "shooting_bs80")}
        | metodos_pinn(["lm"], ("nn", "bspline")),
        t_max_job=600),
}


def custo_estimado(m):
    """Peso aproximado para ordenar os jobs (os mais caros primeiro)."""
    if m["tipo"] == "classico":
        return 0.1
    r = m["receita"]
    base = 4.0 if r.startswith("lm") else {"original": 1.0, "lbfgs": 1.0}.get(r, 2.0)
    return base * (5.0 if m["campo"] == "mixfunn" else 1.0) * m.get("fator", 1.0)
