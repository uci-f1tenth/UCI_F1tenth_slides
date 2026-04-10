import gymnasium as gym
from manimlib import *
from PIL import Image
from torch.distributions import Categorical

from labs.PPO.ppo import *

# class Train(Scene):
#     def construct(self):
#         ag = PPO(48, 4)
#         gs = GymScene("CliffWalking-v1")
#         te = gym.make("CliffWalking-v1")
#         viz = ActionViz(["↑ Up", "→ Right", "↓ Down", "← Left"]).to_edge(
#             RIGHT, buff=0.8
#         )
#         lbl = Text("Untrained", font_size=28).to_edge(UP)
#         self.play(FadeIn(viz), FadeIn(lbl))

#         for rnd in range(8):
#             if rnd:
#                 for _ in range(5):
#                     ag.update(ag.collect(te, 2048))
#             f, o = gs.reset()
#             f.to_edge(LEFT, buff=0.3)
#             nl = Text(
#                 f"{rnd * 5} updates" if rnd else "Untrained", font_size=28
#             ).to_edge(UP)
#             self.play(FadeOut(lbl), FadeIn(nl), FadeIn(f), run_time=0.25)
#             lbl = nl
#             for _ in range(60):
#                 lg = ag.logits(o)
#                 self.play(viz.pulse(lg, 0.12))
#                 a = Categorical(logits=lg).sample().item()
#                 nf, o, _, done = gs.step(a)
#                 nf.to_edge(LEFT, buff=0.3)
#                 self.remove(f)
#                 self.add(nf)
#                 f = nf
#                 self.play(viz.dim(0.06))
#                 if done:
#                     break
#             self.play(FadeOut(f), run_time=0.15)

#         gs.close()
#         te.close()


class Intro(Scene):
    def construct(self):
        # Title
        # title = TexText("PPO: Explained Simply", font_size=72)
        # self.play(Write(title))
        # self.wait()
        # self.play(FadeOut(title))

        # Part 0: The Environment
        title = TexText("Part 0: The Environment", font_size=72)
        self.play(Write(title))
        self.wait()
        self.play(title.animate.scale(0.5).to_edge(UP))

        scene = GymScene("CliffWalking-v1")
        frame, _ = scene.reset()
        self.play(FadeIn(frame), run_time=1.5)
        self.wait()

        move_arrows = {
            0: ("↑", "Up"),
            1: ("→", "Right"),
            2: ("↓", "Down"),
            3: ("←", "Left"),
        }

        total_reward = 0
        reward_lbl = Text(
            f"Reward: {total_reward}", font_size=28, color=GREY_B
        ).to_edge(DOWN, buff=1.2)
        self.play(FadeIn(reward_lbl), run_time=0.3)

        for move in [0, 3, 1, 1, 1, 2]:
            sym, name = move_arrows[move]
            arrow_lbl = (
                VGroup(
                    Text(sym, font_size=48),
                    Text(name, font_size=28),
                )
                .arrange(buff=0.2)
                .to_edge(DOWN, buff=0.3)
            )

            self.play(FadeIn(arrow_lbl, shift=UP * 0.2), run_time=0.3)

            new_frame, _, reward, _ = scene.step(move)
            total_reward += reward

            self.remove(frame)
            frame = new_frame
            self.add(frame)

            # update reward display
            color = RED if reward < 0 else (GREEN if reward > 0 else GREY_B)
            new_reward_lbl = Text(
                f"Reward: {int(total_reward)}", font_size=28, color=color
            ).move_to(reward_lbl)
            self.play(
                FadeTransform(reward_lbl, new_reward_lbl),
                run_time=0.3,
            )
            reward_lbl = new_reward_lbl
            self.wait(0.3)

            self.play(FadeOut(arrow_lbl), run_time=0.2)

        self.wait()
        self.play(FadeOut(frame), FadeOut(title), FadeOut(reward_lbl), run_time=1.5)
        scene.close()

        # Part 1: PPO Inference
        # inference = TexText("PPO Inference", font_size=48).to_edge(UP)
        # self.play(Write(inference))
        # self.wait()

        # net = ActorViz([48, 64, 64, 4])
        # self.play(FadeIn(net), run_time=1.5)
        # self.wait()

        # agent = PPO(48, 4)
        # acts = get_activations(agent, 0)
        # self.play(net.forward_anim(acts))
        # self.wait()
        # self.play(FadeOut(net))
