# Lembretes / próximos passos

## [ ] Dados parciais: medir só ⟨σz⟩ (fazer DEPOIS que o protocolo completo estiver fechado)

**Por quê:** com ⟨σx⟩, ⟨σy⟩ e ⟨σz⟩ medidos, o problema é quase linear, e a inversão direta ou
o mínimo quadrado linear já resolvem bem. Com **só ⟨σz⟩** (ou ⟨σx⟩ e ⟨σz⟩) esses métodos não se
aplicam diretamente, porque precisam do estado completo. É nesse cenário que o PINN pode ter
vantagem real, junto com o shooting.

**O que fazer:**
1. Em `protocolo/config.py`, adicionar ao cenário a lista de observáveis medidos, por exemplo
   `obs=(2,)` para só σz, `(0, 2)` para σx e σz.
2. Em `protocolo/pinn.py::ProblemaPINN`, aplicar a máscara na loss de dados e nos resíduos de
   dados (`qd[:, obs] - data[:, obs]`). A EDO continua com X, Y, Z completos, que passam a ser
   variáveis latentes. Ajustar `jacobiano()` do mesmo jeito (só as linhas de `obs`).
3. Em `metodos_sem_pinn.py::shooting_fit`, usar só as colunas `obs` no resíduo e no Jacobiano.
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
