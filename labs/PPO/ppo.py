import shutil
import tempfile

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from manimlib import *
from PIL import Image
from torch.distributions import Categorical

# ── Helpers ──────────────────────────────────────────────────


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def mlp(dims, out_std=0.01):
    layers = []
    for i in range(len(dims) - 1):
        last = i == len(dims) - 2
        layers.append(
            layer_init(
                nn.Linear(dims[i], dims[i + 1]),
                std=out_std if last else np.sqrt(2),
            )
        )
        if not last:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class PPO:
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden=(64, 64),
        α=2.5e-4,
        γ=0.99,
        λ=0.95,
        ε=0.2,
        c1=0.5,
        c2=0.01,
        K=4,
        M=128,
    ):
        self.γ, self.λ, self.ε = γ, λ, ε
        self.c1, self.c2, self.K, self.M = c1, c2, K, M
        self.π_θ = mlp([obs_dim, *hidden, act_dim], 0.01)
        self.V_θ = mlp([obs_dim, *hidden, 1], 1.0)
        self.params = [*self.π_θ.parameters(), *self.V_θ.parameters()]
        self.opt = torch.optim.Adam(self.params, lr=α, eps=1e-5)

    def enc(self, s):
        x = torch.zeros(48)
        x[int(s)] = 1.0
        return x

    @torch.inference_mode()
    def logits(self, s):
        return self.π_θ(self.enc(s))

    def π_and_V(self, s_t, a_t=None):
        """Returns (aₜ, log πθ(aₜ|sₜ), S[πθ](sₜ), Vθ(sₜ))"""
        π = Categorical(logits=self.π_θ(s_t))
        if a_t is None:
            a_t = π.sample()
        return a_t, π.log_prob(a_t), π.entropy(), self.V_θ(s_t).squeeze(-1)

    @torch.no_grad()
    def collect(self, env, T):
        """Algorithm 1: run πθ_old for T steps, compute Ât via GAE (Eq. 11–12)"""  # reused in update()
        s, _ = env.reset()
        S, A, Π, V, R, D = [], [], [], [], [], []
        for _ in range(T):
            s_t = self.enc(s)
            a_t, log_π, _, V_st = self.π_and_V(s_t)
            S.append(s_t)
            A.append(a_t)
            Π.append(log_π)
            V.append(V_st)
            s, r, dn, tr, _ = env.step(a_t.item())
            R.append(r)
            D.append(dn or tr)
            if dn or tr:
                s, _ = env.reset()

        V_t = torch.stack(V)
        r_t = torch.tensor(R, dtype=torch.float32)
        d_t = torch.tensor(D, dtype=torch.float32)
        V_T = self.V_θ(self.enc(s)).squeeze(-1)
        Â, gae = torch.zeros_like(r_t), 0.0
        for t in reversed(range(T)):
            V_next = V_T if t == T - 1 else V_t[t + 1]
            δ = r_t[t] + self.γ * V_next * (1 - d_t[t]) - V_t[t]
            gae = δ + self.γ * self.λ * (1 - d_t[t]) * gae
            Â[t] = gae
        V_targ = Â + V_t
        return torch.stack(S), torch.stack(A), torch.stack(Π), V_t, Â, V_targ

    def update(self, data):
        """Optimize L^{CLIP+VF+S}(θ) (Eq. 9) for K epochs, minibatch M"""
        S, A, log_π_old, V_old, Â, V_targ = data
        Â_n = (Â - Â.mean()) / (Â.std() + 1e-8)
        for _ in range(self.K):
            idx = np.random.permutation(len(S))
            for i in range(0, len(idx), self.M):
                b = idx[i : i + self.M]
                _, log_π, S_π, V_θ = self.π_and_V(S[b], A[b])

                r_t = (log_π - log_π_old[b]).exp()
                L_CLIP = torch.min(
                    r_t * Â_n[b], r_t.clamp(1 - self.ε, 1 + self.ε) * Â_n[b]
                ).mean()

                V_c = V_old[b] + (V_θ - V_old[b]).clamp(-self.ε, self.ε)
                L_VF = (
                    0.5
                    * torch.max((V_θ - V_targ[b]) ** 2, (V_c - V_targ[b]) ** 2).mean()
                )

                loss = -L_CLIP + self.c1 * L_VF - self.c2 * S_π.mean()

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.params, 0.5)
                self.opt.step()


# ── Gym → Manim bridge ──────────────────────────────────────


class GymScene:
    def __init__(self, eid, w=8):
        self.w, self.tmp, self.n = w, tempfile.mkdtemp(), 0
        self.env = gym.make(eid, render_mode="rgb_array")

    def _snap(self):
        self.n += 1
        p = f"{self.tmp}/{self.n}.png"
        img = Image.fromarray(self.env.render())
        img = img.resize(
            (img.width * 8, img.height * 8),
            resample=Image.NEAREST,
        )
        img.save(p)
        return ImageMobject(p).set_width(self.w)

    def reset(self):
        o, _ = self.env.reset()
        return self._snap(), o

    def step(self, a):
        o, r, d, t, _ = self.env.step(a)
        return self._snap(), o, r, d or t

    def close(self):
        self.env.close()
        shutil.rmtree(self.tmp, True)


# ── Compressed network viz ───────────────────────────────────


def get_activations(agent, state):
    with torch.inference_mode():
        x = agent.enc(state)
        acts = [x.clone()]
        for i, m in enumerate(agent.π_θ):
            x = m(x)
            if isinstance(m, nn.Tanh) or i == len(agent.π_θ) - 1:
                acts.append(x.clone())
    return acts


class ActorViz(VGroup):
    def __init__(self, arch, show=6, sx=3.0, sy=0.4, r=0.06):
        super().__init__()
        self.show = show
        self.nodes, self.edges = [], []
        _e, _d, _n = [], [], []

        for i, n in enumerate(arch):
            k, gap = min(n, show), n > show
            slots = k + gap
            x = (i - (len(arch) - 1) / 2) * sx
            col = []
            for j in range(slots):
                y = ((slots - 1) / 2 - j) * sy
                if gap and j == k // 2:
                    _d.append(Text("⋮", font_size=20, color=GREY_C).move_to([x, y, 0]))
                    continue
                nd = Dot(point=[x, y, 0], radius=r, color=GREY_C, fill_opacity=1)
                col.append(nd)
                _n.append(nd)
            self.nodes.append(col)
            _d.append(
                Text(str(n), font_size=18, color=GREY_B).move_to(
                    [x, -(slots - 1) / 2 * sy - 0.5, 0]
                )
            )

        for i in range(len(arch) - 1):
            layer = [
                Line(
                    a.get_center(),
                    b.get_center(),
                    stroke_width=1.5,
                    stroke_opacity=0.25,
                )
                for a in self.nodes[i]
                for b in self.nodes[i + 1]
            ]
            self.edges.append(layer)
            _e += layer

        self.add(*_e, *_d, *_n)

    def _pick(self, n):
        k = min(n, self.show)
        return np.round(np.linspace(0, n - 1, k)).astype(int)

    def forward_anim(self, acts, pal=(BLUE_D, TEAL, TEAL, GREEN_D)):
        anims = []
        for i, (a, col) in enumerate(zip(acts, self.nodes)):
            c = pal[min(i, len(pal) - 1)]
            v = a.abs()
            v = (v - v.min()) / (v.max() - v.min() + 1e-8)
            vals = v[self._pick(len(a))].tolist()  # ← one lookup
            pulse = [
                nd.animate.set_color(interpolate_color(GREY_C, c, t)).set_opacity(
                    0.4 + 0.6 * t
                )
                for nd, t in zip(col, vals)  # ← clean zip
            ]
            if i:
                pulse += [
                    e.animate.set_stroke(c, 2, opacity=0.5) for e in self.edges[i - 1]
                ]
            anims.append(AnimationGroup(*pulse, run_time=0.4))
            if i:
                anims.append(
                    AnimationGroup(
                        *[
                            e.animate.set_stroke(GREY_C, 1.5, opacity=0.25)
                            for e in self.edges[i - 1]
                        ],
                        run_time=0.15,
                    )
                )
        return Succession(*anims)

    def reset_anim(self, rt=0.3):
        return AnimationGroup(
            *[
                n.animate.set_color(GREY_C).set_opacity(1)
                for col in self.nodes
                for n in col
            ],
            run_time=rt,
        )
