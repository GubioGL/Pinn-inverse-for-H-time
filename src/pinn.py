"""
PINN unificado para o problema inverso  H(t) = 0.5 sz + 0.5 f(t) sx.

    dX/dt = -Y,   dY/dt = X - f Z,   dZ/dt = f Y

Características
---------------
* Todos os parâmetros (rede do estado + campo) vivem num ÚNICO vetor plano `theta`.
  Assim Adam, L-BFGS, BFGS denso e Levenberg-Marquardt operam sobre o mesmo objeto.
* Derivada temporal analítica em modo forward: cada camada propaga (h, dh/dt).
* Parametrizações do campo:
      "nn"      : 4a saída da rede do estado (modelo_2)
      "fourier" : série de Fourier periódica, período T (modelo_3)
      "cosseno" : série de cossenos em [0, T] (extensão par; sem Gibbs quando f(0) != f(T))
      "bspline" : B-spline cúbica com nós uniformes (modelo_4)
      "mixfunn" : rede MixFunn (modelo_5)
  As bases são avaliadas em torch, então funcionam em qualquer ponto e na GPU.
* Pontos de colocação da EDO = pontos dos dados + `n_rand` pontos aleatórios em [0, T],
  sorteados de novo a cada época do Adam (`reamostrar()`).
* Restrições rígidas na saída da rede:
      estado : r(t) = r0 + (t/T) NN(t)                      -> r(0) = r0 exatamente
      campo  : f(t) = g(t) - (1-tau) g(0) - tau g(T) + (1-tau) a + tau b,  tau = t/T
               -> f(0) = a e f(T) = b exatamente, quando (a, b) é conhecido a priori
               (ex.: pulso que liga e desliga em zero). Linear em g, então as bases
               continuam lineares nos coeficientes.
* Loss = EDO (média nos pontos de colocação) + w_data * dados (média)
         + lam_f * (1/T) * integral f''(t)^2 dt  (diferenças finitas num grid uniforme fixo).
  O campo verdadeiro NUNCA entra na loss. A loss é exatamente ||r(theta)||^2.
* Para campos lineares (fourier, cosseno, bspline):
      init_lsq : coeficientes iniciais por mínimos quadrados a partir dos dados
      varpro   : coeficientes resolvidos em forma fechada a cada passo (variable projection)
"""
import math
import numpy as np
import torch as tc
from torch.func import functional_call, vmap, jacrev

import bloch as cm
import mixfunn as mfmod

CAMPOS_LINEARES = ("fourier", "cosseno", "bspline")


# ---------------------------------------------------------------------------
# Bases em torch
# ---------------------------------------------------------------------------
def nos_bspline(T, n_basis, k):
    return tc.tensor(cm.open_uniform_knots(0.0, T, n_basis, k), dtype=tc.float64)


def bspline_torch(t, knots, k):
    """Matriz de B-splines (Cox-de Boor), t: (n,), knots: (m,) -> (n, m-k-1)."""
    t = t.clamp(knots[0], knots[-1])
    left, right = knots[:-1], knots[1:]
    B = ((t[:, None] >= left) & (t[:, None] < right)).to(t.dtype)
    # t == T: último intervalo não degenerado
    ult = int(tc.nonzero(right > left)[-1])
    B[:, ult] = tc.where(t >= knots[-1], tc.ones_like(t), B[:, ult])
    m = knots.numel()
    for d in range(1, k + 1):
        nd = m - 1 - d
        ki, kid = knots[:nd], knots[d:d + nd]
        ki1, kid1 = knots[1:1 + nd], knots[d + 1:d + 1 + nd]
        den1, den2 = kid - ki, kid1 - ki1
        a = tc.where(den1 > 0, (t[:, None] - ki) / tc.where(den1 > 0, den1, tc.ones_like(den1)), tc.zeros(()).to(t))
        b = tc.where(den2 > 0, (kid1 - t[:, None]) / tc.where(den2 > 0, den2, tc.ones_like(den2)), tc.zeros(()).to(t))
        B = a * B[:, :nd] + b * B[:, 1:nd + 1]
    return B


def base_linear(kind, t, T, size, k=3, knots=None):
    """Matriz da base (n, nb) em torch, igual às de bloch.py (numpy)."""
    if kind == "fourier":
        n = tc.arange(1, size + 1, device=t.device, dtype=t.dtype)
        ang = 2 * math.pi / T * t[:, None] * n
        return tc.cat([tc.ones_like(t)[:, None], tc.cos(ang), tc.sin(ang)], dim=1)
    if kind == "cosseno":
        n = tc.arange(0, size + 1, device=t.device, dtype=t.dtype)
        return tc.cos(math.pi / T * t[:, None] * n)
    if kind == "bspline":
        return bspline_torch(t, knots.to(t), k)
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Layout do vetor de parâmetros
# ---------------------------------------------------------------------------
class Layout:
    def __init__(self):
        self.items = []        # (nome, shape, offset, numel)
        self.n = 0

    def add(self, name, shape):
        numel = int(np.prod(shape)) if len(shape) else 1
        self.items.append((name, tuple(shape), self.n, numel))
        self.n += numel

    def unflatten(self, theta):
        return {name: theta[o:o + k].view(shape) for name, shape, o, k in self.items}

    def offset(self, name):
        for n, shape, o, k in self.items:
            if n == name:
                return o, k
        raise KeyError(name)


# ---------------------------------------------------------------------------
class ProblemaPINN:
    """Monta o problema para UM conjunto de dados.

    cfg (dicionário):
        field        : "nn" | "fourier" | "cosseno" | "bspline" | "mixfunn"
        field_size   : M (fourier/cosseno) ou n_basis (bspline)
        spline_k     : grau da B-spline (3)
        width, depth : rede do estado;  act: "sin" | "tanh"
        norm_time    : entrada s = 2t/T - 1 (senão t cru)
        hard_ic      : r(t) = r0 + (t/T) NN(t)
        n_rand       : nº de pontos de colocação aleatórios além dos pontos dos dados
                       (0 = só os pontos dos dados, como nos notebooks originais)
        n_reg        : nº de pontos do grid uniforme da regularização do campo
        bc_campo     : None ou (a, b): impõe f(0)=a e f(T)=b na saída (restrição rígida)
        lam_f        : peso da regularização de suavidade do campo
        w_data       : peso da loss de dados
        mixfunn      : dict(N_layers, N_neuro, second_order_function)
        mixfunn_input: "t" (cru) ou "s" (normalizado)
        seed         : semente do sorteio dos pontos de colocação
    """

    def __init__(self, t_data, data, r0, T, cfg, device="cpu", dtype=tc.float64):
        self.cfg = cfg
        self.dev = tc.device(device)
        self.dt = dtype
        self.T = float(T)
        kw = dict(device=self.dev, dtype=dtype)
        self.kw = kw

        self.N = len(t_data)
        self.t_data = tc.tensor(t_data, **kw)
        self.data = tc.tensor(data, **kw)
        self.r0 = tc.tensor(r0, **kw)
        self.n_rand = int(cfg.get("n_rand", 0) or 0)
        self.gen = tc.Generator(device="cpu").manual_seed(int(cfg.get("seed", 0)) + 12345)

        n_reg = int(cfg.get("n_reg", 200))
        self.t_reg = tc.linspace(0.0, self.T, n_reg, **kw)
        self.h_reg = self.T / (n_reg - 1)

        f = cfg["field"]
        self.linear = f in CAMPOS_LINEARES
        self.knots = None
        if f == "bspline":
            self.knots = nos_bspline(self.T, cfg["field_size"], cfg.get("spline_k", 3)).to(self.dev)
        bc = cfg.get("bc_campo")
        self.bc = None if bc is None else (float(bc[0]), float(bc[1]))

        # --- layout dos parâmetros
        L = Layout()
        w, d = cfg["width"], cfg["depth"]
        n_out = 4 if f == "nn" else 3
        dims = [1] + [w] * d
        for i in range(d):
            L.add(f"net.W{i}", (dims[i], dims[i + 1]))
            L.add(f"net.b{i}", (dims[i + 1],))
        L.add("net.Wout", (w, n_out))
        L.add("net.bout", (n_out,))
        self.mf = None
        if self.linear:
            self.nb = self._base(self.t_data[:1]).shape[1]
            L.add("field.c", (self.nb,))
        elif f == "mixfunn":
            mcfg = cfg.get("mixfunn", dict(N_layers=2, N_neuro=4, second_order_function=True))
            self.mf = mfmod.MixFunn(**mcfg).to(device=self.dev, dtype=dtype)
            for m in self.mf.modules():                  # índices auxiliares no device certo
                if hasattr(m, "ids") and isinstance(m.ids, tc.Tensor):
                    m.ids = m.ids.to(self.dev)
            self.mf_names = []
            for name, p in self.mf.named_parameters():
                L.add("mf." + name, tuple(p.shape))
                self.mf_names.append(name)
        self.layout = L
        self.n_params = L.n

        # grid da regularização: operador de 2a diferença e base fixa
        n = n_reg
        D = tc.zeros(n - 2, n, **kw)
        ar = tc.arange(n - 2, device=self.dev)
        D[ar, ar], D[ar, ar + 1], D[ar, ar + 2] = 1.0, -2.0, 1.0
        self.D2 = D / self.h_reg ** 2
        self.reg_scale = math.sqrt(cfg.get("lam_f", 0.0) * self.h_reg / self.T)
        self.w_data = float(cfg.get("w_data", 1.0))

        self.M_data, self.m_data = self._afim(self.t_data)
        self.M_reg, self.m_reg = self._afim(self.t_reg)
        self.reamostrar()

    # ------------------------------------------------------------------
    def _base(self, t):
        c = self.cfg
        return base_linear(c["field"], t, self.T, c["field_size"], c.get("spline_k", 3), self.knots)

    def _afim(self, t):
        """Campo linear = M(t) c + m(t), já com a restrição rígida de borda (se houver)."""
        if not self.linear:
            return None, None
        B = self._base(t)
        if self.bc is None:
            return B, tc.zeros_like(t)
        tau = (t / self.T)[:, None]
        ends = self._base(tc.tensor([0.0, self.T], **self.kw))
        M = B - (1 - tau) * ends[0] - tau * ends[1]
        a, b = self.bc
        return M, (1 - tau[:, 0]) * a + tau[:, 0] * b

    def reamostrar(self):
        """Pontos de colocação = dados + n_rand pontos uniformes em [0, T] (novo sorteio)."""
        if self.n_rand > 0:
            tr = tc.rand(self.n_rand, generator=self.gen, dtype=self.dt).to(self.dev) * self.T
            self.t_col = tc.cat([self.t_data, tr])
        else:
            self.t_col = self.t_data
        self.n_col = self.t_col.numel()
        self.M_col, self.m_col = self._afim(self.t_col)

    # ------------------------------------------------------------------
    def init_theta(self, seed, init_lsq=False, suavizar=False):
        g = tc.Generator(device="cpu").manual_seed(int(seed))
        th = tc.zeros(self.n_params, dtype=self.dt)
        prev_fan_in = 1
        for name, shape, o, k in self.layout.items:
            if name.startswith("net."):
                is_w = name.split(".")[1].startswith("W")
                fan_in = shape[0] if is_w else prev_fan_in
                bound = 1.0 / math.sqrt(fan_in)
                th[o:o + k] = (tc.rand(k, generator=g, dtype=self.dt) * 2 - 1) * bound
                if is_w:
                    prev_fan_in = fan_in
        if self.mf is not None:
            tc.manual_seed(int(seed))
            for m in self.mf.modules():
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()
            for name, p in self.mf.named_parameters():
                if name.endswith(".p"):       # pesos de mistura do MixFunn: randn (como no original)
                    p.data.normal_()
            params = dict(self.mf.named_parameters())
            for name, _, o, k in self.layout.items:
                if name.startswith("mf."):
                    th[o:o + k] = params[name[3:]].detach().cpu().reshape(-1)
        th = th.to(self.dev)
        if self.linear and init_lsq:
            # coeficientes iniciais por mínimos quadrados usando só os dados (splines)
            vals, ders = cm.data_derivative(self.t_data.cpu().numpy(), self.data.cpu().numpy(),
                                            smooth=suavizar)
            V = tc.tensor(vals, **self.kw)
            Dv = tc.tensor(ders, **self.kw)
            c = self._resolver_c(V[:, 0], V[:, 1], V[:, 2], Dv[:, 1], Dv[:, 2], self.M_data, self.m_data)
            o, k = self.layout.offset("field.c")
            th[o:o + k] = c
        return th

    # ------------------------------------------------------------------
    def _mlp(self, P, s, ds):
        """Rede do estado com derivada em modo forward: retorna (out, dout/dt)."""
        act = self.cfg.get("act", "sin")
        h = s
        dh = ds * tc.ones_like(s)
        for i in range(self.cfg["depth"]):
            z = h @ P[f"net.W{i}"] + P[f"net.b{i}"]
            dz = dh @ P[f"net.W{i}"]
            if act == "sin":
                h, dh = tc.sin(z), tc.cos(z) * dz
            else:
                h = tc.tanh(z)
                dh = (1 - h * h) * dz
        return h @ P["net.Wout"] + P["net.bout"], dh @ P["net.Wout"]

    def _entrada(self, t):
        if self.cfg.get("norm_time", True):
            return 2.0 * t / self.T - 1.0, 2.0 / self.T
        return t, 1.0

    def _campo_nao_linear(self, P, t, s, out):
        """g(t) para os campos 'nn' e 'mixfunn' (antes da restrição de borda)."""
        if self.cfg["field"] == "nn":
            return out[:, 3:4]
        pm = {n: P["mf." + n] for n in self.mf_names}
        inp = s if self.cfg.get("mixfunn_input", "t") == "s" else t
        return functional_call(self.mf, pm, (inp,))

    def quantidades(self, theta, t, Mrow=None, mrow=None):
        """t: (n,) ou escalar; Mrow/mrow: linhas da base afim (campos lineares).
        Retorna (n, 7) = [X, Y, Z, dX/dt, dY/dt, dZ/dt, f]."""
        scalar = t.dim() == 0
        t = t.reshape(-1, 1)
        P = self.layout.unflatten(theta)
        s, ds = self._entrada(t)
        out, dout = self._mlp(P, s, ds)
        y, dy = out[:, :3], dout[:, :3]
        if self.cfg.get("hard_ic", True):
            tau = t / self.T
            y, dy = self.r0 + tau * y, y / self.T + tau * dy
        if self.linear:
            f = Mrow.reshape(-1, Mrow.shape[-1]) @ P["field.c"].reshape(-1, 1) + mrow.reshape(-1, 1)
        else:
            g = self._campo_nao_linear(P, t, s, out)
            if self.bc is not None:
                # restrição rígida: f(0) = a, f(T) = b
                tb = tc.tensor([[0.0], [self.T]], dtype=t.dtype, device=t.device)
                sb, _ = self._entrada(tb)
                outb, _ = self._mlp(P, sb, 1.0)
                gb = self._campo_nao_linear(P, tb, sb, outb)
                tau = t / self.T
                a, b = self.bc
                g = g - (1 - tau) * gb[0] - tau * gb[1] + (1 - tau) * a + tau * b
            f = g
        q = tc.cat([y, dy, f], dim=1)
        return q[0] if scalar else q

    # ------------------------------------------------------------------
    @staticmethod
    def _ode_res(q):
        X, Y, Z, dX, dY, dZ, f = q.unbind(-1)
        return tc.stack([dX + Y, dY - X + f * Z, dZ - f * Y], dim=-1)

    def partes(self, theta, causal_eps=0.0, causal_chunks=20):
        """(l_ode, l_data, l_reg) com grafo, para a loss e para os pesos adaptativos."""
        qc = self.quantidades(theta, self.t_col, self.M_col, self.m_col)
        e = (self._ode_res(qc) ** 2).sum(1)
        if causal_eps > 0:
            # Wang, Sankaran & Perdikaris (2022): pesos causais por blocos de tempo
            ordem = tc.argsort(self.t_col)
            chunks = tc.stack([c.mean() for c in e[ordem].tensor_split(causal_chunks)])
            with tc.no_grad():
                w = tc.exp(-causal_eps * tc.cat([chunks.new_zeros(1), tc.cumsum(chunks, 0)[:-1]]))
            l_ode = (w * chunks).mean()
        else:
            l_ode = e.mean()
        qd = qc[:self.N]            # os pontos dos dados são os N primeiros pontos de colocação
        l_data = ((qd[:, :3] - self.data) ** 2).sum(1).mean()
        if self.reg_scale > 0:
            fr = self.quantidades(theta, self.t_reg, self.M_reg, self.m_reg)[:, 6]
            l_reg = (self.reg_scale * (self.D2 @ fr)).pow(2).sum()
        else:
            l_reg = tc.zeros((), **self.kw)
        return l_ode, l_data, l_reg

    def loss(self, theta, causal_eps=0.0, causal_chunks=20):
        l_ode, l_data, l_reg = self.partes(theta, causal_eps, causal_chunks)
        return l_ode + self.w_data * l_data + l_reg, (l_ode.detach(), l_data.detach(), l_reg.detach())

    def residuos(self, theta):
        qc = self.quantidades(theta, self.t_col, self.M_col, self.m_col)
        qd = qc[:self.N]
        r = [self._ode_res(qc).reshape(-1) / math.sqrt(self.n_col),
             (qd[:, :3] - self.data).reshape(-1) * math.sqrt(self.w_data / self.N)]
        if self.reg_scale > 0:
            fr = self.quantidades(theta, self.t_reg, self.M_reg, self.m_reg)[:, 6]
            r.append(self.reg_scale * (self.D2 @ fr))
        return tc.cat(r)

    def jacobiano(self, theta):
        """Resíduos r e Jacobiano J = dr/dtheta, a partir de Jacobianos POR PONTO (vmap de
        jacrev). Custo O(n_pontos * n_params)."""
        def qfun(th, t, M, m):
            q = self.quantidades(th, t, M, m)
            return q, q

        lin = 0 if self.linear else None
        jfun = vmap(jacrev(qfun, has_aux=True), in_dims=(None, 0, lin, lin))
        P = theta.numel()
        Jc, qc = jfun(theta, self.t_col, self.M_col, self.m_col)   # (n_col, 7, P), (n_col, 7)
        Jd, qd = Jc[:self.N], qc[:self.N]        # pontos dos dados = N primeiros da colocação
        X, Y, Z, dX, dY, dZ, f = qc.unbind(-1)
        JX, JY, JZ, JdX, JdY, JdZ, Jf = Jc.unbind(1)
        f_, Y_, Z_ = f[:, None], Y[:, None], Z[:, None]
        J_ode = tc.stack([JdX + JY,
                          JdY - JX + f_ * JZ + Z_ * Jf,
                          JdZ - f_ * JY - Y_ * Jf], dim=1).reshape(-1, P)
        r_ode = tc.stack([dX + Y, dY - X + f * Z, dZ - f * Y], dim=1).reshape(-1)
        s = 1.0 / math.sqrt(self.n_col)
        sd = math.sqrt(self.w_data / self.N)
        rs = [r_ode * s, (qd[:, :3] - self.data).reshape(-1) * sd]
        Js = [J_ode * s, Jd[:, :3, :].reshape(-1, P) * sd]
        if self.reg_scale > 0:
            Jr, qr = jfun(theta, self.t_reg, self.M_reg, self.m_reg)
            rs.append(self.reg_scale * (self.D2 @ qr[:, 6]))
            Js.append(self.reg_scale * (self.D2 @ Jr[:, 6, :]))
        return tc.cat(rs), tc.cat(Js)

    # ------------------------------------------------------------------
    def _resolver_c(self, X, Y, Z, dY, dZ, M, m):
        """Minimiza em c (campo afim f = M c + m) a parte da loss que depende de c:
           (1/n) sum[(Z f - (X - dY))^2 + (Y f - dZ)^2] + reg_scale^2 ||D2 f_reg||^2."""
        n = X.numel()
        w = (Y ** 2 + Z ** 2)
        A = M.T @ (w[:, None] * M) / n
        rhs = -(M.T @ (Z * (dY - X + m * Z) - Y * (dZ - m * Y))) / n
        if self.reg_scale > 0:
            DM = self.D2 @ self.M_reg
            A = A + self.reg_scale ** 2 * DM.T @ DM
            rhs = rhs - self.reg_scale ** 2 * DM.T @ (self.D2 @ self.m_reg)
        A = A + 1e-12 * tc.trace(A) / A.shape[0] * tc.eye(A.shape[0], **self.kw)
        return tc.linalg.solve(A, rhs)

    def varpro(self, theta):
        """Variable projection: troca os coeficientes do campo pelo ótimo em forma fechada,
        dado o estado atual da rede (pontos de colocação atuais)."""
        with tc.no_grad():
            q = self.quantidades(theta, self.t_col, self.M_col, self.m_col)
            c = self._resolver_c(q[:, 0], q[:, 1], q[:, 2], q[:, 4], q[:, 5], self.M_col, self.m_col)
            o, k = self.layout.offset("field.c")
            theta[o:o + k] = c
        return theta

    # ------------------------------------------------------------------
    def campo(self, theta, t):
        t = tc.as_tensor(t, **self.kw)
        M, m = self._afim(t)
        with tc.no_grad():
            return self.quantidades(theta, t, M, m)[:, 6].cpu().numpy()

    def campo_dados(self, theta):
        with tc.no_grad():
            return self.quantidades(theta, self.t_data, self.M_data, self.m_data)[:, 6].cpu().numpy()

    def estado_dados(self, theta):
        with tc.no_grad():
            return self.quantidades(theta, self.t_data, self.M_data, self.m_data)[:, :3].cpu().numpy()
