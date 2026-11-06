"""
Otimizadores que operam sobre o vetor plano theta de um ProblemaPINN.

  adam(...)   : Adam com schedule cosseno (opcional: pesos causais, Wang et al. 2022)
  lbfgs(...)  : L-BFGS do PyTorch com busca de linha strong-Wolfe
  ssbfgs(...) : BFGS denso com auto-escalonamento (Oren-Luenberger / Al-Baali), a família
                de quasi-Newton que Urbán, Stefanou & Pons (JCP 2025, arXiv:2405.04230)
                mostraram dar ganhos de ordens de grandeza em PINNs pequenos.
  lm(...)     : Levenberg-Marquardt (Gauss-Newton amortecido) sobre o vetor de resíduos.
                Quando há mais parâmetros que resíduos, resolve no espaço dos resíduos
                (J J^T + mu I) a = r,  delta = -J^T a  (mesma ideia do "dual natural gradient",
                Jnini & Vella 2025). Gauss-Newton/ENGD: Müller & Zeinhofer (ICML 2023).

Todos registram o histórico da loss e respeitam um limite de tempo opcional.
"""
import math
import time
import torch as tc
from torch.optim.lbfgs import _strong_wolfe


def _loss_grad(prob, theta, **kw):
    th = theta.detach().requires_grad_(True)
    L, _ = prob.loss(th, **kw)
    g, = tc.autograd.grad(L, th)
    return L.detach(), g.detach()


def adam(prob, theta, epochs, lr=1e-3, sched="cos", lr_min_frac=1e-2, causal_eps=0.0,
         reamostrar=True, varpro=False, pesos_adapt=False, adapt_cada=100, adapt_alpha=0.1,
         congelar_campo=0.0, log=None, t_max=None):
    """Adam. A cada época:
      reamostrar  : sorteia novos pontos de colocação aleatórios (além dos pontos dos dados)
      varpro      : (campos lineares) coeficientes do campo resolvidos em forma fechada
      pesos_adapt : peso da loss de dados ajustado pela razão das normas dos gradientes,
                    w = (1-a) w + a * max|grad L_edo| / mean|grad L_dados|
                    (Wang, Teng & Perdikaris, SIAM J. Sci. Comput. 2021). O peso volta ao
                    valor do protocolo no fim do estágio, para os estágios seguintes
                    minimizarem a mesma função objetivo em todos os métodos.
      congelar_campo : fração inicial das épocas em que os coeficientes do campo ficam fixos
                    (campos lineares iniciados por mínimos quadrados): a rede do estado aprende
                    os dados primeiro, sem "estragar" a boa estimativa inicial do campo.
    """
    th = theta.detach().clone().requires_grad_(True)
    opt = tc.optim.Adam([th], lr=lr)
    sch = (tc.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=lr * lr_min_frac)
           if sched == "cos" else None)
    w0 = prob.w_data
    varpro = varpro and prob.linear
    n_cong = int(congelar_campo * epochs) if prob.linear else 0
    sl_campo = slice(*(lambda o, k: (o, o + k))(*prob.layout.offset("field.c"))) if prob.linear else None
    t0 = time.time()
    for ep in range(epochs):
        if reamostrar and prob.n_rand > 0:
            prob.reamostrar()
        if varpro:
            prob.varpro(th.data)
        if pesos_adapt and ep % adapt_cada == 0:
            l_ode, l_data, _ = prob.partes(th, causal_eps=causal_eps)
            g_ode, = tc.autograd.grad(l_ode, th, retain_graph=True)
            g_dat, = tc.autograd.grad(l_data, th)
            lam = float(g_ode.abs().max() / (g_dat.abs().mean() + 1e-30))
            prob.w_data = (1 - adapt_alpha) * prob.w_data + adapt_alpha * lam
        L, parts = prob.loss(th, causal_eps=causal_eps)
        opt.zero_grad(set_to_none=True)
        L.backward()
        if ep < n_cong:
            th.grad[sl_campo] = 0.0
        opt.step()
        if sch:
            sch.step()
        if log is not None and (ep % 500 == 0 or ep == epochs - 1):
            log.append(("adam", ep, float(L.detach())))
        if t_max and time.time() - t0 > t_max:
            break
    if varpro:
        prob.varpro(th.data)
    if pesos_adapt and log is not None:
        log.append(("adam_w_data_final", epochs, prob.w_data))
    prob.w_data = w0
    return th.detach()


def lbfgs(prob, theta, iters, history=50, log=None, t_max=None):
    th = theta.detach().clone().requires_grad_(True)
    opt = tc.optim.LBFGS([th], lr=1.0, max_iter=iters, max_eval=int(iters * 1.25),
                         history_size=history, tolerance_grad=1e-15, tolerance_change=1e-18,
                         line_search_fn="strong_wolfe")
    t0 = time.time()
    count = [0]

    def closure():
        opt.zero_grad(set_to_none=True)
        L, _ = prob.loss(th)
        L.backward()
        count[0] += 1
        if log is not None and count[0] % 200 == 0:
            log.append(("lbfgs", count[0], float(L.detach())))
        if t_max and time.time() - t0 > t_max:
            # força a parada: zera o gradiente => critério tolerance_grad
            th.grad.zero_()
        return L
    opt.step(closure)
    return th.detach()


def ssbfgs(prob, theta, iters, log=None, t_max=None, tol=1e-18, auto_escala=True):
    """BFGS denso (inversa da Hessiana H, n x n no device) com auto-escalonamento
    tau_k = min(1, s'y / y'Hy) aplicado a H antes de cada atualização."""
    x = theta.detach().clone()
    n = x.numel()
    f, g = _loss_grad(prob, x)
    f = float(f)
    H = tc.eye(n, dtype=x.dtype, device=x.device)
    first = True
    falhas = 0
    t0 = time.time()

    def obj(xb, t, d):
        Lb, gb = _loss_grad(prob, xb + t * d)
        return float(Lb), gb

    for k in range(iters):
        d = -(H @ g)
        gtd = g @ d
        if gtd >= 0:            # direção não é de descida: reinicia
            H = tc.eye(n, dtype=x.dtype, device=x.device)
            d, gtd, first = -g, -(g @ g), True
        t_init = min(1.0, 1.0 / float(g.abs().sum())) if first else 1.0
        f_new, g_new, t, _ = _strong_wolfe(obj, x, t_init, d, f, g, gtd,
                                           tolerance_change=1e-20, max_ls=30)
        s = t * d
        y = g_new - g
        x = x + s
        ys = float(y @ s)
        if ys > 1e-300:
            Hy = H @ y
            yHy = float(y @ Hy)
            if first:
                H = (ys / float(y @ y)) * tc.eye(n, dtype=x.dtype, device=x.device)
                Hy = H @ y
                yHy = float(y @ Hy)
                tau = 1.0
                first = False
            else:
                tau = min(1.0, ys / yHy) if auto_escala else 1.0
            v = s / ys - Hy / yHy
            H.mul_(tau)
            H.addr_(Hy, Hy, alpha=-tau / yHy)
            H.addr_(v, v, alpha=tau * yHy)
            H.addr_(s, s, alpha=1.0 / ys)
        df = float(f - f_new)
        f, g = f_new, g_new
        if log is not None and k % 200 == 0:
            log.append(("ssbfgs", k, float(f)))
        if df <= tol * max(1.0, abs(f)):
            # sem progresso: reinicia a aproximação da Hessiana; para após 3 falhas seguidas
            falhas += 1
            if falhas >= 3:
                break
            H = tc.eye(n, dtype=x.dtype, device=x.device)
            first = True
        else:
            falhas = 0
        if float(g.abs().max()) < 1e-16:
            break
        if t_max and time.time() - t0 > t_max:
            break
    return x


def lm(prob, theta, iters, mu0=1e-3, log=None, t_max=None, patience=30, reamostrar_cada=0):
    """Levenberg-Marquardt com amortecimento de Levenberg (mu I) e a regra de Nielsen
    para mu, baseada na razão entre a redução real e a prevista pelo modelo quadrático."""
    x = theta.detach().clone()
    r, J = prob.jacobiano(x)
    L = float(r @ r)
    R, P = J.shape
    escala = float((J * J).sum()) / max(R, P)
    mu = mu0 * escala
    mu_min = 1e-20 * escala
    nu = 2.0
    t0 = time.time()
    stall = 0
    dual = P > R
    eye = tc.eye(R if dual else P, dtype=x.dtype, device=x.device)
    for it in range(iters):
        if dual:
            K = J @ J.T
        else:
            A = J.T @ J
            g = J.T @ r
        accepted = False
        for _ in range(15):
            try:
                if dual:
                    Lc = tc.linalg.cholesky(K + mu * eye)
                    a = tc.cholesky_solve(r[:, None], Lc)[:, 0]
                    delta = -(J.T @ a)
                else:
                    Lc = tc.linalg.cholesky(A + mu * eye)
                    delta = -tc.cholesky_solve(g[:, None], Lc)[:, 0]
            except RuntimeError:            # não positiva definida numericamente
                mu *= nu
                nu *= 2.0
                continue
            Jd = J @ delta
            pred = -(2.0 * float(r @ Jd) + float(Jd @ Jd))      # redução prevista (> 0)
            x_new = x + delta
            with tc.no_grad():
                r_new = prob.residuos(x_new)
            L_new = float(r_new @ r_new)
            rho = (L - L_new) / pred if pred > 0 else -1.0
            if math.isfinite(L_new) and L_new < L and rho > 0:
                rel = (L - L_new) / max(L, 1e-300)
                x, L = x_new, L_new
                mu = max(mu * max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3), mu_min)
                nu = 2.0
                accepted = True
                stall = stall + 1 if rel < 1e-10 else 0
                break
            mu *= nu
            nu *= 2.0
        if log is not None and (it % 20 == 0):
            log.append(("lm", it, L))
        if not accepted or stall >= patience:
            break
        if t_max and time.time() - t0 > t_max:
            break
        if reamostrar_cada and prob.n_rand > 0 and (it + 1) % reamostrar_cada == 0:
            prob.reamostrar()                 # novos pontos de colocação: recalcula a loss
            r, J = prob.jacobiano(x)
            L = float(r @ r)
            stall = 0
            continue
        r, J = prob.jacobiano(x)
    return x


def treinar(prob, theta, estagios, log=None, t_max_total=None):
    """estagios: lista de dicts, ex.
       [{"tipo": "adam", "epochs": 5000, "lr": 1e-3, "sched": "cos"},
        {"tipo": "lm", "iters": 300}]"""
    t0 = time.time()
    for st in estagios:
        restante = None if t_max_total is None else max(1.0, t_max_total - (time.time() - t0))
        tipo = st["tipo"]
        if tipo == "adam":
            theta = adam(prob, theta, st["epochs"], lr=st.get("lr", 1e-3), sched=st.get("sched", "cos"),
                         causal_eps=st.get("causal_eps", 0.0), reamostrar=st.get("reamostrar", True),
                         varpro=st.get("varpro", False), pesos_adapt=st.get("pesos_adapt", False),
                         congelar_campo=st.get("congelar_campo", 0.0),
                         log=log, t_max=restante)
        elif tipo == "lbfgs":
            theta = lbfgs(prob, theta, st["iters"], log=log, t_max=restante)
        elif tipo == "ssbfgs":
            theta = ssbfgs(prob, theta, st["iters"], log=log, t_max=restante,
                           auto_escala=st.get("auto_escala", True))
        elif tipo == "lm":
            theta = lm(prob, theta, st["iters"], mu0=st.get("mu0", 1e-3), log=log, t_max=restante,
                       reamostrar_cada=st.get("reamostrar_cada", 0))
        else:
            raise ValueError(tipo)
    return theta
