"""
PINN unificado para o problema inverso  H(t) = 0.5 sz + 0.5 f(t) sx.

    dX/dt = -Y,   dY/dt = X - f Z,   dZ/dt = f Y

Características
---------------
* Todos os parâmetros (rede do estado + campo) vivem num ÚNICO vetor plano `theta`.
  Isso permite usar Adam, L-BFGS, BFGS denso e Levenberg-Marquardt sobre o mesmo objeto.
* Derivada temporal calculada em modo forward, de forma analítica, numa única passada
  (h, dh/dt) pelas camadas -> não precisa de autograd.grad(create_graph=True).
* 4 parametrizações do campo (os 4 métodos do repositório):
      "nn"      : 4a saída da mesma rede do estado (modelo_2)
      "fourier" : série de Fourier periódica com período T (modelo_3)
      "bspline" : B-spline (modelo_4; padrão cúbica n=40)
      "mixfunn" : rede MixFunn (src/mixfunn.py = notebooks/Modelo_5(Mixfunn)/mixfunn.py)
* Loss = EDO (média nos pontos de colocação) + dados (média) + regularização de
  suavidade do campo  lam_f * (1/T) * integral f''(t)^2 dt  (diferenças finitas).
  O campo verdadeiro NUNCA entra na loss.
* A loss é exatamente ||r(theta)||^2, onde r é o vetor de resíduos usado pelo LM.
* Funciona em CPU ou GPU (device = "cuda", "cuda:1", "cpu").
"""
import math
import numpy as np
import torch as tc
from torch.func import functional_call, vmap, jacrev

import bloch as cm
import mixfunn as mfmod


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

    def slice_of(self, prefix):
        idx = [(o, o + k) for name, _, o, k in self.items if name.startswith(prefix)]
        return idx


# ---------------------------------------------------------------------------
class ProblemaPINN:
    """Monta o problema para UM conjunto de dados.

    cfg (dicionário):
        field        : "nn" | "fourier" | "bspline" | "mixfunn"
        field_size   : M (fourier) ou n_basis (bspline)
        spline_k     : grau da B-spline (3)
        width, depth : rede do estado
        act          : "sin" | "tanh"
        norm_time    : entrada s = 2t/T - 1 (senão t cru)
        hard_ic      : r(t) = r0 + (t/T) NN(t)
        n_col        : nº de pontos de colocação (None = pontos dos dados)
        lam_f        : peso da regularização de suavidade do campo
        w_data       : peso da loss de dados
        mixfunn      : dict(N_layers, N_neuro, second_order_function)
    """

    def __init__(self, t_data, data, r0, T, cfg, device="cpu", dtype=tc.float64):
        self.cfg = cfg
        self.dev = tc.device(device)
        self.dt = dtype
        self.T = float(T)
        kw = dict(device=self.dev, dtype=dtype)

        self.N = len(t_data)
        self.t_data = tc.tensor(t_data, **kw)
        self.data = tc.tensor(data, **kw)
        self.r0 = tc.tensor(r0, **kw)

        t_col = t_data if cfg.get("n_col") is None else np.linspace(0.0, T, cfg["n_col"])
        self.n_col = len(t_col)
        self.t_col = tc.tensor(t_col, **kw)
        self.h = float(t_col[1] - t_col[0])

        # --- base do campo (fourier / bspline): matrizes nos pontos de colocação e de dados
        f = cfg["field"]
        self.B_col = self.B_data = None
        if f == "fourier":
            self.B_col = tc.tensor(cm.fourier_matrix(t_col, T, cfg["field_size"]), **kw)
            self.B_data = tc.tensor(cm.fourier_matrix(t_data, T, cfg["field_size"]), **kw)
        elif f == "bspline":
            k = cfg.get("spline_k", 3)
            self.B_col = tc.tensor(cm.bspline_matrix(t_col, T, cfg["field_size"], k), **kw)
            self.B_data = tc.tensor(cm.bspline_matrix(t_data, T, cfg["field_size"], k), **kw)

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
        if f in ("fourier", "bspline"):
            L.add("field.c", (self.B_col.shape[1],))
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

        # operador de 2a diferença para a regularização do campo
        n = self.n_col
        D = tc.zeros(n - 2, n, **kw)
        ar = tc.arange(n - 2, device=self.dev)
        D[ar, ar] = 1.0
        D[ar, ar + 1] = -2.0
        D[ar, ar + 2] = 1.0
        self.D2 = D / self.h ** 2
        self.reg_scale = math.sqrt(cfg.get("lam_f", 0.0) * self.h / self.T)

        self.w_data = cfg.get("w_data", 1.0)

    # ------------------------------------------------------------------
    def init_theta(self, seed):
        g = tc.Generator(device="cpu").manual_seed(int(seed))
        th = tc.zeros(self.n_params, dtype=self.dt)
        for name, shape, o, k in self.layout.items:
            if name.startswith("net."):
                fan_in = shape[0] if name.split(".")[1].startswith("W") else None
                if fan_in is None:   # bias: mesmo fan_in da matriz anterior (padrão do nn.Linear)
                    fan_in = prev_fan_in
                bound = 1.0 / math.sqrt(fan_in)
                th[o:o + k] = (tc.rand(k, generator=g, dtype=self.dt) * 2 - 1) * bound
                if name.split(".")[1].startswith("W"):
                    prev_fan_in = fan_in
        if self.mf is not None:
            tc.manual_seed(int(seed))
            for m in self.mf.modules():
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()
            for name, p in self.mf.named_parameters():
                if name.endswith(".p"):       # pesos de mistura do MixFunn: randn (como no original)
                    p.data.normal_()
            for name, _, o, k in self.layout.items:
                if name.startswith("mf."):
                    th[o:o + k] = dict(self.mf.named_parameters())[name[3:]].detach().cpu().reshape(-1)
        # coeficientes do campo começam em zero
        return th.to(self.dev)

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

    def quantidades(self, theta, t, Brow=None):
        """t: (n,) ou escalar; Brow: (n, nb) ou (nb,). Retorna (n, 7) = [X,Y,Z,dX,dY,dZ,f]."""
        scalar = t.dim() == 0
        t = t.reshape(-1, 1)
        P = self.layout.unflatten(theta)
        if self.cfg.get("norm_time", True):
            s, ds = 2.0 * t / self.T - 1.0, 2.0 / self.T
        else:
            s, ds = t, 1.0
        out, dout = self._mlp(P, s, ds)
        y, dy = out[:, :3], dout[:, :3]
        if self.cfg.get("hard_ic", True):
            tau = t / self.T
            y, dy = self.r0 + tau * y, y / self.T + tau * dy
        fld = self.cfg["field"]
        if fld == "nn":
            f = out[:, 3:4]
        elif fld in ("fourier", "bspline"):
            f = Brow.reshape(-1, Brow.shape[-1]) @ P["field.c"].reshape(-1, 1)
        else:  # mixfunn: entrada t crua (como no Modelo_5) ou normalizada (mixfunn_input="s")
            pm = {n: P["mf." + n] for n in self.mf_names}
            inp = s if self.cfg.get("mixfunn_input", "t") == "s" else t
            f = functional_call(self.mf, pm, (inp,))
        q = tc.cat([y, dy, f], dim=1)
        return q[0] if scalar else q

    # ------------------------------------------------------------------
    @staticmethod
    def _ode_res(q):
        X, Y, Z, dX, dY, dZ, f = q.unbind(-1)
        return tc.stack([dX + Y, dY - X + f * Z, dZ - f * Y], dim=-1)

    def loss(self, theta, causal_eps=0.0, causal_chunks=20):
        qc = self.quantidades(theta, self.t_col, self.B_col)
        res = self._ode_res(qc)
        e = (res ** 2).sum(1)
        if causal_eps > 0:
            # Wang, Sankaran & Perdikaris (2022): pesos causais por blocos de tempo
            chunks = e.reshape(causal_chunks, -1).mean(1) if self.n_col % causal_chunks == 0 \
                else tc.stack([c.mean() for c in e.tensor_split(causal_chunks)])
            with tc.no_grad():
                w = tc.exp(-causal_eps * tc.cat([chunks.new_zeros(1), tc.cumsum(chunks, 0)[:-1]]))
            l_ode = (w * chunks).mean()
        else:
            l_ode = e.mean()
        qd = self.quantidades(theta, self.t_data, self.B_data)
        l_data = ((qd[:, :3] - self.data) ** 2).sum(1).mean()
        l_reg = (self.reg_scale * (self.D2 @ qc[:, 6])).pow(2).sum()
        return l_ode + self.w_data * l_data + l_reg, (l_ode.detach(), l_data.detach(), l_reg.detach())

    def residuos(self, theta):
        qc = self.quantidades(theta, self.t_col, self.B_col)
        qd = self.quantidades(theta, self.t_data, self.B_data)
        r_ode = self._ode_res(qc).reshape(-1) / math.sqrt(self.n_col)
        r_dat = (qd[:, :3] - self.data).reshape(-1) * math.sqrt(self.w_data / self.N)
        r_reg = self.reg_scale * (self.D2 @ qc[:, 6])
        return tc.cat([r_ode, r_dat, r_reg])

    def jacobiano(self, theta):
        """Resíduos r e Jacobiano J = dr/dtheta, montados a partir de Jacobianos POR PONTO
        (vmap de jacrev), o que custa O(n_pontos * n_params) em vez de O(n_res * custo_total)."""
        def qfun(th, t, b):
            q = self.quantidades(th, t, b)
            return q, q

        jfun = vmap(jacrev(qfun, has_aux=True), in_dims=(None, 0, 0 if self.B_col is not None else None))
        Bc = self.B_col
        Bd = self.B_data
        Jc, qc = jfun(theta, self.t_col, Bc)       # (n_col, 7, P), (n_col, 7)
        Jd, qd = jfun(theta, self.t_data, Bd)
        X, Y, Z, dX, dY, dZ, f = qc.unbind(-1)
        JX, JY, JZ, JdX, JdY, JdZ, Jf = Jc.unbind(1)
        f_, Y_, Z_ = f[:, None], Y[:, None], Z[:, None]
        J_ode = tc.stack([JdX + JY,
                          JdY - JX + f_ * JZ + Z_ * Jf,
                          JdZ - f_ * JY - Y_ * Jf], dim=1).reshape(-1, theta.numel())
        r_ode = tc.stack([dX + Y, dY - X + f * Z, dZ - f * Y], dim=1).reshape(-1)
        s = 1.0 / math.sqrt(self.n_col)
        sd = math.sqrt(self.w_data / self.N)
        r_dat = (qd[:, :3] - self.data).reshape(-1) * sd
        J_dat = Jd[:, :3, :].reshape(-1, theta.numel()) * sd
        r_reg = self.reg_scale * (self.D2 @ f)
        J_reg = self.reg_scale * (self.D2 @ Jf)
        r = tc.cat([r_ode * s, r_dat, r_reg])
        J = tc.cat([J_ode * s, J_dat, J_reg])
        return r, J

    # ------------------------------------------------------------------
    def campo_dados(self, theta):
        with tc.no_grad():
            return self.quantidades(theta, self.t_data, self.B_data)[:, 6].cpu().numpy()

    def estado_dados(self, theta):
        with tc.no_grad():
            return self.quantidades(theta, self.t_data, self.B_data)[:, :3].cpu().numpy()
