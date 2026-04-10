import gymnasium as gym
from manimlib import *
from PIL import Image
from torch.distributions import Categorical

from labs.PPO.ppo import *

# ── Training scene ───────────────────────────────────────────


class Train(Scene):
    def construct(self):
        ag = PPO(48, 4)
        gs = GymScene("CliffWalking-v1")
        te = gym.make("CliffWalking-v1")
        viz = ActionViz(["↑ Up", "→ Right", "↓ Down", "← Left"]).to_edge(
            RIGHT, buff=0.8
        )
        lbl = Text("Untrained", font_size=28).to_edge(UP)
        self.play(FadeIn(viz), FadeIn(lbl))

        for rnd in range(8):
            if rnd:
                for _ in range(5):
                    ag.update(ag.collect(te, 2048))
            f, o = gs.reset()
            f.to_edge(LEFT, buff=0.3)
            nl = Text(
                f"{rnd * 5} updates" if rnd else "Untrained", font_size=28
            ).to_edge(UP)
            self.play(FadeOut(lbl), FadeIn(nl), FadeIn(f), run_time=0.25)
            lbl = nl
            for _ in range(60):
                lg = ag.logits(o)
                self.play(viz.pulse(lg, 0.12))
                a = Categorical(logits=lg).sample().item()
                nf, o, _, done = gs.step(a)
                nf.to_edge(LEFT, buff=0.3)
                self.remove(f)
                self.add(nf)
                f = nf
                self.play(viz.dim(0.06))
                if done:
                    break
            self.play(FadeOut(f), run_time=0.15)

        gs.close()
        te.close()
