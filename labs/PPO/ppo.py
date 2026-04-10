import shutil
import tempfile

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from manimlib import *
from PIL import Image
from torch.distributions import Categorical


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


class PPO:
    def __init__(
        self,
        od,
        ad,
        h=(64, 64),
        lr=2.5e-4,
        γ=0.99,
        λ=0.95,
        ε=0.2,
        ent=0.01,
        vf=0.5,
        epochs=4,
        mb=128,
    ):
        self.γ, self.λ, self.ε, self.ent, self.vf = γ, λ, ε, ent, vf
        self.epochs, self.mb = epochs, mb
        self.actor = mlp([od, *h, ad], 0.01)
        self.critic = mlp([od, *h, 1], 1.0)
        self.params = [*self.actor.parameters(), *self.critic.parameters()]
        self.opt = torch.optim.Adam(self.params, lr=lr, eps=1e-5)

    def enc(self, o):
        t = torch.zeros(48)
        t[int(o)] = 1.0
        return t

    @torch.no_grad()
    def logits(self, o):
        return self.actor(self.enc(o))

    def get_action_and_value(self, obs, action=None):
        logits = self.actor(obs)
        d = Categorical(logits=logits)
        if action is None:
            action = d.sample()
        return action, d.log_prob(action), d.entropy(), self.critic(obs).squeeze(-1)

    @torch.no_grad()
    def collect(self, env, n):
        o, _ = env.reset()
        O, A, LP, V, R, D = ([] for _ in range(6))
        for _ in range(n):
            t = self.enc(o)
            a, lp, _, v = self.get_action_and_value(t)
            O.append(t)
            A.append(a)
            LP.append(lp)
            V.append(v)
            o, r, dn, tr, _ = env.step(a.item())
            R.append(r)
            D.append(dn or tr)
            if dn or tr:
                o, _ = env.reset()
        Vt = torch.stack(V)
        Rt = torch.tensor(R, dtype=torch.float32)
        Dt = torch.tensor(D, dtype=torch.float32)
        lv = self.critic(self.enc(o)).squeeze(-1)
        adv, g = torch.zeros_like(Rt), 0.0
        for i in reversed(range(n)):
            nv = lv if i == n - 1 else Vt[i + 1]
            nnt = 1 - Dt[i]
            g = Rt[i] + self.γ * nv * nnt - Vt[i] + self.γ * self.λ * nnt * g
            adv[i] = g
        return torch.stack(O), torch.stack(A), torch.stack(LP), Vt, adv, adv + Vt

    def update(self, data):
        O, A, olp, oldv, adv, ret = data
        for _ in range(self.epochs):
            idx = np.arange(len(O))
            np.random.shuffle(idx)
            for s in range(0, len(idx), self.mb):
                b = idx[s : s + self.mb]
                _, lp, ent, nv = self.get_action_and_value(O[b], A[b])
                ratio = (lp - olp[b]).exp()
                a = adv[b]
                a = (a - a.mean()) / (a.std() + 1e-8)
                pg = torch.max(
                    -a * ratio, -a * ratio.clamp(1 - self.ε, 1 + self.ε)
                ).mean()
                vc = oldv[b] + (nv - oldv[b]).clamp(-self.ε, self.ε)
                vl = 0.5 * torch.max((nv - ret[b]) ** 2, (vc - ret[b]) ** 2).mean()
                loss = pg - self.ent * ent.mean() + self.vf * vl
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


# ── Output-only action viz ───────────────────────────────────


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
        p = (p - p.min()) / (p.max() - p.min() + 1e-8)  # ← brightness fix
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
