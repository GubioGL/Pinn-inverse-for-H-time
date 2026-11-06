# Lembretes / próximos passos

## [ ] Rodar o estudo de resolução espectral (código pronto, não rodado ainda)

Pulsos f(t) = sin(ωt)·sin²(πt/T), ω ∈ {1, 2, 4, 6, 8, 10, 12, 15, 20}, ruído 0 e 0,01. Métodos:
inversão direta, LSQ (Fourier 10, cossenos 40, B-spline 80), shooting B-spline 80, PINN-LM
(campo-rede e B-spline). Leva ~1 h em 4 CPUs com 2 seeds; numa GPU, use mais seeds:

```bash
python run/rodar.py --perfil espectral --device cuda --workers 8 --seeds 0 1 2 3 4
python run/fig_espectral.py        # -> paper/figures/fig_espectral.pdf
```

O que esperar (capacidade das bases, `run/diagnostico.py` e Parte 3 de ANALISE.md): a Fourier
M=10 não passa de ω ≈ 2π·10/T ≈ 7,9; os cossenos M=40 vão até ω ≈ 10–15; a B-spline de 80
funções até ω ≈ 15 (1 %); a sensibilidade dos dados cai ~1/ω, então com ruído o erro cresce com ω.

## [ ] Novo caso: vários estados iniciais com o mesmo campo (NÃO fazer agora)

**Por quê:** cada estado inicial "enxerga" o campo em instantes diferentes, porque a informação
sobre f em t é proporcional a Y² + Z² = 1 − X². Ajustar um único f(t) a experimentos com
estados iniciais diferentes cobre os pontos cegos de cada um. Na Parte 1
(`run/parte1_classicos.py`, `data/parte1/resultados_desenho_experimental.csv`) isso já apareceu.
Com o mesmo número total de medidas, 3 estados (|+⟩, |0⟩, |+i⟩) deram 0,07 ± 0,01 de erro
(LSQ, σ = 0,01), contra 0,15 ± 0,05 só com |+⟩. Isso também resolve a ambiguidade de sinal
f → −f do item abaixo.

**O que fazer:**
1. `src/protocolo.py`: cenário com uma lista de estados iniciais, ex.
   `r0s = [(0,0,1), (1,0,0), (0,1,0)]`, e orçamento de medidas fixo (ruído por ponto
   σ·√(nº de estados), como na Parte 1).
2. `src/pinn.py`: uma rede do estado por experimento (ou uma rede com o estado inicial como
   entrada) e **um único** campo. A loss soma os resíduos de todos os experimentos.
3. `src/classicos.py`: `linear_lsq` e `shooting_fit` já aceitam vários experimentos
   (listas de dados/shooters). Basta ligá-los no `src/executar.py`.
4. Comparar com o mesmo protocolo (5 seeds; ruído 0 e 0,01): 1 estado vs. 2 vs. 3.

## [ ] Dados parciais: medir só ⟨σz⟩ (fazer DEPOIS que o protocolo completo estiver fechado)

**Por quê:** com ⟨σx⟩, ⟨σy⟩ e ⟨σz⟩ medidos, o problema é quase linear, e a inversão direta ou
o mínimo quadrado linear já resolvem bem. Com **só ⟨σz⟩** (ou ⟨σx⟩ e ⟨σz⟩) esses métodos não se
aplicam diretamente, porque precisam do estado completo. É nesse cenário que o PINN pode ter
vantagem real, junto com o shooting.

**O que fazer:**
1. Em `src/protocolo.py`, adicionar ao cenário a lista de observáveis medidos, por exemplo
   `obs=(2,)` para só σz, `(0, 2)` para σx e σz.
2. Em `src/pinn.py::ProblemaPINN`, aplicar a máscara na loss de dados e nos resíduos de
   dados (`qd[:, obs] - data[:, obs]`). A EDO continua com X, Y, Z completos, que passam a ser
   variáveis latentes. Ajustar `jacobiano()` do mesmo jeito (só as linhas de `obs`).
3. Em `src/classicos.py::shooting_fit`, usar só as colunas `obs` no resíduo e no Jacobiano.
   A inversão direta e o LSQ linear não se aplicam: marcar como "n/a" no relatório.
4. **Identificabilidade (já verificado):** com só ⟨σz⟩ e estado inicial **|0⟩, f e −f geram
   exatamente o mesmo ⟨σz⟩(t)**. Isso vem da simetria σz H(f) σz = H(−f) com σz|0⟩ = |0⟩. A
   diferença medida foi 0,0 no caso 3. O sinal do campo não é identificável e nenhum método
   resolve isso. Com |+⟩ acontece o mesmo em ⟨σx⟩ (diferença 0,00). Opções:
   - estado inicial genérico, ex. (1,1,1)/√3: max|Z(f) − Z(−f)| = 1,5, então a simetria quebra;
   - medir dois observáveis (ex. ⟨σx⟩ e ⟨σz⟩);
   - aceitar f a menos de sinal e reportar |f|.
   Rodar `python -c` com `common.simulate(f, t, r0)` vs `common.simulate(lambda s: -f(s), t, r0)`
   para conferir cada combinação de estado inicial e observável antes de treinar.
5. Rodar com o mesmo protocolo (5 seeds, |0⟩, ruído 0 e 0,01) e comparar PINN-LM × shooting.
