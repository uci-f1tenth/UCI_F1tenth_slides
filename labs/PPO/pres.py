import gymnasium as gym
from manimlib import *
from PIL import Image
from torch.distributions import Categorical

from labs.PPO.ppo import *

gym.register(
    id="TinyCliffWalking-v0",
    entry_point="labs.PPO.tiny_cliff:TinyCliffWalkingEnv",
    max_episode_steps=50,
)


class EnvPlayer:
    """Reusable Manim helper: animate an agent (or manual moves) in a Gym env."""

    ARROWS = {0: ("↑", "Up"), 1: ("→", "Right"), 2: ("↓", "Down"), 3: ("←", "Left")}

    def __init__(self, scene, eid="CliffWalking-v1", w=8):
        self.sc = scene
        self.gs = GymScene(eid, w)
        self.frame = None
        self.obs = None
        self.total_reward = 0
        self.reward_lbl = None

    def reset(self, rt=0.5, show_reward=True):
        """Reset env, fade in frame. Returns initial obs."""
        f, o = self.gs.reset()
        if self.frame is not None:
            f.match_width(self.frame).move_to(self.frame)
        self.frame, self.obs, self.total_reward = f, o, 0

        anims = [FadeIn(f)]
        if show_reward:
            self.reward_lbl = Text("Reward: 0", font_size=28).to_edge(DOWN, buff=1.2)
            anims.append(FadeIn(self.reward_lbl))
        self.sc.play(*anims, run_time=rt)
        return o

    def step(self, action, rt=0.3, show_arrow=True):
        """One env step. Returns (old_obs, action, reward, new_obs, done)."""
        old_obs = self.obs
        sym, name = self.ARROWS[action]

        arrow = None
        if show_arrow:
            arrow = (
                VGroup(Text(sym, font_size=48), Text(name, font_size=28))
                .arrange(buff=0.2)
                .to_edge(DOWN, buff=0.3)
            )
            self.sc.play(FadeIn(arrow, shift=UP * 0.2), run_time=rt)

        nf, o, r, done = self.gs.step(action)
        nf.match_width(self.frame).move_to(self.frame)
        self.sc.remove(self.frame)
        self.sc.add(nf)
        self.frame, self.obs = nf, o
        self.total_reward += r

        if self.reward_lbl:
            new_rl = Text(f"Reward: {int(self.total_reward)}", font_size=28).move_to(
                self.reward_lbl
            )
            self.sc.play(FadeTransform(self.reward_lbl, new_rl), run_time=rt)
            self.reward_lbl = new_rl

        if arrow:
            self.sc.wait(0.3)
            self.sc.play(FadeOut(arrow), run_time=rt * 0.6)

        return old_obs, action, r, o, done

    def play_moves(self, moves, **kw):
        """Animate a list of manual actions."""
        results = []
        for m in moves:
            results.append(self.step(m, **kw))
        return results

    def play_agent(self, agent, max_steps=60, **kw):
        """Let the agent's policy choose actions."""
        results = []
        for _ in range(max_steps):
            logits = agent.logits(self.obs)
            a = Categorical(logits=logits).sample().item()
            tup = self.step(a, **kw)
            results.append(tup)
            if tup[-1]:  # done
                self.reset(rt=0.2, show_reward=bool(self.reward_lbl))
        return results

    def fadeout(self, rt=0.3):
        objs = [o for o in [self.frame, self.reward_lbl] if o is not None]
        if objs:
            self.sc.play(*[FadeOut(o) for o in objs], run_time=rt)
        self.frame = self.reward_lbl = None

    def close(self):
        self.gs.close()


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
        # # Title
        # title = TexText("PPO: Explained Simply", font_size=72)
        # self.play(Write(title))
        # self.wait()
        # self.play(FadeOut(title))

        # # Add a why PPO section (LLMs, Robotics, etc.)

        # # Part 0: The Environment
        # title = TexText("Part 0: The Environment", font_size=72)
        # self.play(Write(title))
        # self.wait()
        # self.play(title.animate.scale(0.5).to_edge(UP))

        # ep = EnvPlayer(self)
        # ep.reset(show_reward=True)
        # self.wait()

        # ep.play_moves([0, 3, 1, 1, 1, 2])

        # ep.fadeout()
        # self.play(FadeOut(title))
        # ep.close()

        # # Part 1: PPO Inference
        # title = TexText("Part 1: PPO Inference", font_size=72)
        # self.play(Write(title))
        # self.wait()

        # self.play(title.animate.scale(0.5).to_edge(UP))
        # scene = GymScene("CliffWalking-v1")
        # frame, _ = scene.reset()
        # self.play(FadeIn(frame), run_time=1.5)
        # rows, cols = 4, 12

        # cell_w = frame.get_width() / cols
        # cell_h = frame.get_height() / rows
        # origin = np.array([frame.get_left()[0], frame.get_top()[1], 0])

        # labels = VGroup()
        # for r in range(rows):
        #     for c in range(cols):
        #         s = r * cols + c
        #         pos = origin + RIGHT * (c + 0.5) * cell_w + DOWN * (r + 0.5) * cell_h
        #         labels.add(Text(str(s), font_size=14, font="Monospace").move_to(pos))

        # self.play(LaggedStartMap(FadeIn, labels, lag_ratio=0.03), run_time=1.5)
        # self.wait()

        # scene_group = Group(frame, labels)
        # self.play(scene_group.animate.scale(0.25).to_edge(LEFT, buff=0.1))
        # net = ActorViz([48, 64, 64, 4])

        # self.play(FadeIn(net), run_time=1.5)
        # self.wait()

        # agent = PPO(48, 4)
        # acts = get_activations(agent, 0)
        # self.play(net.forward_anim(acts))
        # self.wait()

        # action_labels = VGroup(
        #     *[
        #         Text(lbl, font_size=20).next_to(nd, RIGHT, buff=0.15)
        #         for nd, lbl in zip(
        #             net.nodes[-1], ["↑ Up", "→ Right", "↓ Down", "← Left"]
        #         )
        #     ]
        # )
        # self.play(FadeIn(action_labels), run_time=1.5)
        # self.wait()

        # self.play(
        #     FadeOut(net), FadeOut(action_labels), FadeOut(scene_group), FadeOut(title)
        # )

        # # Part 2: Critic Inference
        # title = TexText("Part 2: Critic Inference", font_size=72)
        # self.play(Write(title))
        # self.wait()

        # self.play(title.animate.scale(0.5).to_edge(UP))
        # scene = GymScene("CliffWalking-v1")
        # frame, _ = scene.reset()
        # self.play(FadeIn(frame), run_time=1.5)
        # rows, cols = 4, 12

        # cell_w = frame.get_width() / cols
        # cell_h = frame.get_height() / rows
        # origin = np.array([frame.get_left()[0], frame.get_top()[1], 0])

        # labels = VGroup()
        # for r in range(rows):
        #     for c in range(cols):
        #         s = r * cols + c
        #         pos = origin + RIGHT * (c + 0.5) * cell_w + DOWN * (r + 0.5) * cell_h
        #         labels.add(Text(str(s), font_size=14, font="Monospace").move_to(pos))

        # self.play(LaggedStartMap(FadeIn, labels, lag_ratio=0.03), run_time=1.5)
        # self.wait()

        # scene_group = Group(frame, labels)
        # self.play(scene_group.animate.scale(0.25).to_edge(LEFT, buff=0.1))
        # net = ActorViz([48, 64, 64, 1])
        # self.play(FadeIn(net), run_time=1.5)
        # self.wait()

        # agent = PPO(48, 4)
        # acts = get_activations(agent, 0)
        # self.play(net.forward_anim(acts))
        # self.wait()

        # critic_reward = Tex("V(s)").next_to(net.nodes[-1][0], RIGHT)
        # self.play(Write(critic_reward))
        # self.wait()

        # self.play(
        #     FadeOut(net), FadeOut(scene_group), FadeOut(critic_reward), FadeOut(title)
        # )

        # Part 3: Time to train
        title = TexText("Part 3: Time to train", font_size=72)
        self.play(Write(title))
        self.wait()
        self.play(title.animate.scale(0.5).to_edge(UP))

        # # Step 1: Initialize the agent
        # subtitle = TexText("Step 1: Initialize the agent", font_size=48)
        # self.play(Write(subtitle))
        # self.wait()

        # self.play(subtitle.animate.scale(0.5).next_to(title, DOWN))
        # # show a 48, 64, 64, 4 net initlized to all zero
        # net = ActorViz([48, 64, 64, 4])
        # self.play(FadeIn(net), run_time=1.5)
        # self.wait()

        # agent = PPO(48, 4)
        # for p in agent.params:
        #     p.data.zero_()

        # acts = get_activations(agent, 0)
        # self.play(net.forward_anim(acts))
        # self.wait()

        # pct_labels = VGroup(
        #     *[
        #         VGroup(
        #             Text("25%", font_size=18, color=GREEN_D),
        #             Text(lbl, font_size=18),
        #         )
        #         .arrange(RIGHT, buff=0.1)
        #         .next_to(nd, RIGHT, buff=0.15)
        #         for nd, lbl in zip(
        #             net.nodes[-1], ["↑ Up", "→ Right", "↓ Down", "← Left"]
        #         )
        #     ]
        # )
        # self.play(FadeIn(pct_labels), run_time=1.0)
        # self.wait()

        # self.play(
        #     FadeOut(net),
        #     FadeOut(pct_labels),
        #     FadeOut(subtitle),
        #     run_time=0.8,
        # )

        # # Step 2: Initialize the critic
        # subtitle = TexText("Step 2: Initialize the critic", font_size=48)
        # self.play(Write(subtitle))
        # self.wait()
        # self.play(subtitle.animate.scale(0.5).next_to(title, DOWN))

        # critic_net = ActorViz([48, 64, 64, 1])
        # self.play(FadeIn(critic_net), run_time=1.5)
        # self.wait()

        # agent = PPO(48, 4)
        # for p in agent.params:
        #     p.data.zero_()

        # critic_acts = get_critic_activations(agent, 0)
        # self.play(critic_net.forward_anim(critic_acts))
        # self.wait()

        # v_label = Tex(r"V(s) = -50").next_to(critic_net.nodes[-1][0], RIGHT, buff=0.2)
        # group = VGroup(v_label, critic_net)
        # self.play(Write(v_label), group.animate.shift(LEFT))
        # self.wait()

        # self.play(
        #     FadeOut(subtitle),
        #     FadeOut(group),
        #     run_time=0.8,
        # )

        # # Step 3: collect a batch of experiences
        # subtitle = TexText("Step 3: Collect a batch of trajectories", font_size=48)
        # self.play(Write(subtitle))
        # self.wait()
        # self.play(subtitle.animate.scale(0.5).next_to(title, DOWN))

        # ARROWS = {0: "↑", 1: "→", 2: "↓", 3: "←"}
        # trajectories = [
        #     [1],  # a) →           (into cliff)
        #     [0, 1, 1, 2],  # b) ↑ → → ↓     (to goal)
        #     [3, 0, 1, 1, 2],  # c) ← ↑ → → ↓   (wall then goal)
        # ]
        # xs = [LEFT * 4.5, ORIGIN, RIGHT * 4.5]

        # envs = [GymScene("TinyCliffWalking-v0") for _ in range(3)]
        # frames, hdrs, rows, lbls = [], [], [], []

        # for i, gs in enumerate(envs):
        #     f, _ = gs.reset()
        #     f.scale(0.35).move_to(xs[i])
        #     frames.append(f)

        #     l = Text(f"Trajectory {chr(ord('a') + i)}", font_size=20, color=YELLOW)
        #     l.next_to(f, UP, buff=0.15)
        #     lbls.append(l)

        #     h = Text(" a     r", font_size=18, font="Monospace", color=GREEN)
        #     h.next_to(f, DOWN, buff=0.25)
        #     hdrs.append(h)
        #     rows.append(VGroup())

        # self.play(
        #     *[FadeIn(f) for f in frames],
        #     *[FadeIn(l) for l in lbls],
        #     *[FadeIn(h) for h in hdrs],
        #     run_time=0.6,
        # )
        # self.wait(0.5)

        # max_steps = max(len(t) for t in trajectories)
        # for step in range(max_steps):
        #     step_anims = []
        #     for i, traj in enumerate(trajectories):
        #         if step >= len(traj):
        #             continue
        #         a = traj[step]
        #         nf, _, r, done = envs[i].step(a)
        #         nf.match_width(frames[i]).move_to(frames[i])
        #         self.remove(frames[i])
        #         self.add(nf)
        #         frames[i] = nf

        #         row = Text(
        #             f" {ARROWS[a]}   {int(r):>4}",
        #             font_size=18,
        #             font="Monospace",
        #         )
        #         anchor = hdrs[i] if len(rows[i]) == 0 else rows[i][-1]
        #         row.next_to(anchor, DOWN, aligned_edge=LEFT, buff=0.06)
        #         rows[i].add(row)
        #         step_anims.append(FadeIn(row, shift=UP * 0.1))

        #     self.play(*step_anims, run_time=0.4)
        #     self.wait(0.2)

        # self.wait()

        # all_objs = [*frames, *lbls, *hdrs, *rows, subtitle]
        # self.play(*[FadeOut(o) for o in all_objs], run_time=0.8)
        # for gs in envs:
        #     gs.close()

        # Step 4: Calculate Rewards-to-Go
        subtitle = TexText("Step 4: Calculate Rewards-to-Go", font_size=48)
        self.play(Write(subtitle))
        self.wait()
        self.play(subtitle.animate.scale(0.5).next_to(title, DOWN))

        # ── Formula (expanded + recursive) ──
        formula = Tex(
            r"\hat R_t = r_t + \gamma\, r_{t+1} + \gamma^2 r_{t+2} + \cdots",
            font_size=36,
        )
        recursive = Tex(
            r"= r_t + \gamma\,\hat R_{t+1}",
            font_size=36,
            color=BLUE,
        )
        gamma_lbl = Tex(r"\gamma = 0.99", font_size=30, color=YELLOW)
        form_grp = VGroup(formula, recursive, gamma_lbl)
        form_grp.arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        form_grp.next_to(subtitle, DOWN, buff=0.4)

        self.play(Write(formula))
        self.wait(0.5)
        self.play(Write(recursive))
        self.play(FadeIn(gamma_lbl))
        self.wait()

        GAM = 0.9
        rews = [-1, -1, -1, -1]
        syms = ["↑", "→", "→", "↓"]
        n = len(rews)

        rtg_vals = [0.0] * n
        rtg_vals[-1] = float(rews[-1])
        for i in range(n - 2, -1, -1):
            rtg_vals[i] = rews[i] + GAM * rtg_vals[i + 1]

        cols = VGroup()
        for i in range(n):
            t_lbl = Text(f"t={i}", font_size=16, color=GREY)
            a_txt = Text(syms[i], font_size=32)
            r_txt = Text(f"r = {rews[i]}", font_size=20)
            col = VGroup(t_lbl, a_txt, r_txt).arrange(DOWN, buff=0.1)
            cols.add(col)
        cols.arrange(RIGHT, buff=1.2)
        cols.next_to(form_grp, DOWN, buff=0.6)

        conn = VGroup()
        for i in range(n - 1):
            conn.add(
                Arrow(
                    cols[i].get_right(),
                    cols[i + 1].get_left(),
                    buff=0.1,
                    stroke_width=2,
                    color=GREY,
                )
            )

        self.play(
            LaggedStartMap(FadeIn, cols, lag_ratio=0.15),
            *[GrowArrow(a) for a in conn],
            run_time=0.8,
        )
        self.wait(0.5)

        rtg_txts = []
        for i in range(n):
            t = Text("?", font_size=22, color=GREEN)
            t.next_to(cols[i], DOWN, buff=0.3)
            rtg_txts.append(t)

        rtg_hdr = Tex(r"\hat R_t \;:", font_size=22, color=GREEN)
        rtg_hdr.next_to(rtg_txts[0], LEFT, buff=0.5)

        self.play(FadeIn(rtg_hdr), *[FadeIn(t) for t in rtg_txts], run_time=0.4)
        self.wait(0.3)

        for i in range(n - 1, -1, -1):
            hl = SurroundingRectangle(
                VGroup(cols[i], rtg_txts[i]), color=YELLOW, buff=0.12
            )

            if i == n - 1:
                calc_str = f"\\hat R_{i} = {rtg_vals[i]:.2f}"
            else:
                calc_str = (
                    f"\\hat R_{i} = {rews[i]:.0f}"
                    f" + {GAM} \\times ({rtg_vals[i + 1]:.2f})"
                    f" = {rtg_vals[i]:.2f}"
                )

            calc = Tex(calc_str, font_size=26)
            calc.to_edge(DOWN, buff=0.5)

            new_rtg = Text(f"{rtg_vals[i]:.2f}", font_size=22, color=GREEN)
            new_rtg.move_to(rtg_txts[i])

            self.play(ShowCreation(hl), FadeIn(calc), run_time=0.3)
            self.play(FadeTransform(rtg_txts[i], new_rtg), run_time=0.4)
            rtg_txts[i] = new_rtg
            self.wait(0.4)
            self.play(FadeOut(hl), FadeOut(calc), run_time=0.3)

        self.wait()

        self.play(
            *[
                FadeOut(m)
                for m in [
                    cols,
                    conn,
                    rtg_hdr,
                    *rtg_txts,
                    formula,
                    recursive,
                    gamma_lbl,
                    subtitle,
                ]
            ],
            run_time=0.8,
        )
