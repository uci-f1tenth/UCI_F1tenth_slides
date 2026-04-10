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
    l = []
    for i in range(len(dims) - 1):
        last = i == len(dims) - 2
        l.append(
            layer_init(
                nn.Linear(dims[i], dims[i + 1]), std=out_std if last else np.sqrt(2)
            )
        )
        if not last:
            l.append(nn.Tanh())
    return nn.Sequential(*l)


# ── PPO-Clip (Schulman et al., 2017 — arXiv:1707.06347) ────


class PPO:
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden=(64, 64),
        α=2.5e-4,
        γ=0.99,
        λ=0.95,
        ε=0.2,  # Table 3
        c1=0.5,
        c2=0.01,
        K=4,
        M=128,
    ):  # Eq. 9, Algo 1
        self.γ, self.λ, self.ε = γ, λ, ε
        self.c1, self.c2, self.K, self.M = c1, c2, K, M
        self.π_θ = mlp([obs_dim, *hidden, act_dim], 0.01)  # policy πθ
        self.V_θ = mlp([obs_dim, *hidden, 1], 1.0)  # value  Vθ
        self.params = [*self.π_θ.parameters(), *self.V_θ.parameters()]
        self.opt = torch.optim.Adam(self.params, lr=α, eps=1e-5)

    def enc(self, s):
        x = torch.zeros(48)
        x[int(s)] = 1.0
        return x

    @torch.no_grad()
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
        """Algorithm 1: run πθ_old for T steps, compute Ât via GAE (Eq. 11–12)"""
        s, _ = env.reset()
        S, A, Π, V, R, D = ([] for _ in range(6))
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

        # Eq. 11–12: Ât = δt + (γλ)δt+1 + ⋯ + (γλ)^{T−t+1} δ_{T−1}
        V_t = torch.stack(V)
        r_t = torch.tensor(R, dtype=torch.float32)
        d_t = torch.tensor(D, dtype=torch.float32)
        V_T = self.V_θ(self.enc(s)).squeeze(-1)  # bootstrap
        Â, gae = torch.zeros_like(r_t), 0.0
        for t in reversed(range(T)):
            V_next = V_T if t == T - 1 else V_t[t + 1]
            δ = r_t[t] + self.γ * V_next * (1 - d_t[t]) - V_t[t]  # Eq. 12
            gae = δ + self.γ * self.λ * (1 - d_t[t]) * gae  # Eq. 11
            Â[t] = gae
        V_targ = Â + V_t  # Vᵗᵃʳᵍ for L^VF
        return torch.stack(S), torch.stack(A), torch.stack(Π), V_t, Â, V_targ

    def update(self, data):
        """Optimize L^{CLIP+VF+S}(θ) (Eq. 9) for K epochs, minibatch M"""
        S, A, log_π_old, V_old, Â, V_targ = data
        for _ in range(self.K):  # K epochs
            idx = np.arange(len(S))
            np.random.shuffle(idx)
            for i in range(0, len(idx), self.M):  # minibatch M
                b = idx[i : i + self.M]
                _, log_π, S_π, V_θ = self.π_and_V(S[b], A[b])

                # Eq. 6–7: L^CLIP = Ê[min(rₜ·Âₜ, clip(rₜ, 1±ε)·Âₜ)]
                r_t = (log_π - log_π_old[b]).exp()  # Eq. 6
                Â_n = (Â[b] - Â[b].mean()) / (Â[b].std() + 1e-8)
                L_CLIP = torch.min(
                    r_t * Â_n, r_t.clamp(1 - self.ε, 1 + self.ε) * Â_n
                ).mean()  # Eq. 7

                # L^VF = (Vθ(s) − Vᵗᵃʳᵍ)² with clipping
                V_c = V_old[b] + (V_θ - V_old[b]).clamp(-self.ε, self.ε)
                L_VF = (
                    0.5
                    * torch.max((V_θ - V_targ[b]) ** 2, (V_c - V_targ[b]) ** 2).mean()
                )

                # Eq. 9: maximize  L^CLIP − c₁·L^VF + c₂·S[πθ]
                #         minimize −L^CLIP + c₁·L^VF − c₂·S[πθ]
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
        Image.fromarray(self.env.render()).save(p)
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


# ── Output action viz ────────────────────────────────────────


class ActionViz(VGroup):
    def __init__(self, labels, r=0.3, gap=1.4):
        super().__init__()
        self.dots = [
            Circle(
                radius=r, stroke_color=WHITE, fill_color=BLUE_E, fill_opacity=0.15
            ).move_to(i * gap * DOWN)
            for i in range(len(labels))
        ]
        self.add(
            *self.dots,
            *[
                Text(l, font_size=24).next_to(self.dots[i], RIGHT)
                for i, l in enumerate(labels)
            ],
        )
        self.center()

    def pulse(self, logits, rt=0.15):
        p = torch.softmax(torch.as_tensor(logits, dtype=torch.float32), 0)
        p = (p - p.min()) / (p.max() - p.min() + 1e-8)
        return AnimationGroup(
            *[
                self.dots[i].animate.set_fill(
                    interpolate_color(BLUE_E, YELLOW, p[i].item()),
                    opacity=0.15 + 0.85 * p[i].item(),
                )
                for i in range(len(self.dots))
            ],
            run_time=rt,
        )

    def dim(self, rt=0.08):
        return AnimationGroup(
            *[d.animate.set_fill(BLUE_E, opacity=0.15) for d in self.dots], run_time=rt
        )
