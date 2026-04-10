from turtle import right

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
        # Title
        title = TexText("PPO: Explained Simply", font_size=72)
        self.play(Write(title))
        self.wait()
        self.play(FadeOut(title))

        # Add a why PPO section (LLMs, Robotics, etc.)

        # Part 0: The Environment
        title = TexText("Part 0: The Environment", font_size=72)
        self.play(Write(title))
        self.wait()
        self.play(title.animate.scale(0.5).to_edge(UP))

        ep = EnvPlayer(self)
        ep.reset(show_reward=True)
        self.wait()

        ep.play_moves([0, 3, 1, 1, 1, 2])

        ep.fadeout()
        self.play(FadeOut(title))
        ep.close()

        # Part 1: PPO Inference
        title = TexText("Part 1: PPO Inference", font_size=72)
        self.play(Write(title))
        self.wait()

        self.play(title.animate.scale(0.5).to_edge(UP))
        scene = GymScene("CliffWalking-v1")
        frame, _ = scene.reset()
        self.play(FadeIn(frame), run_time=1.5)
        rows, cols = 4, 12

        cell_w = frame.get_width() / cols
        cell_h = frame.get_height() / rows
        origin = np.array([frame.get_left()[0], frame.get_top()[1], 0])

        labels = VGroup()
        for r in range(rows):
            for c in range(cols):
                s = r * cols + c
                pos = origin + RIGHT * (c + 0.5) * cell_w + DOWN * (r + 0.5) * cell_h
                labels.add(Text(str(s), font_size=14, font="Monospace").move_to(pos))

        self.play(LaggedStartMap(FadeIn, labels, lag_ratio=0.03), run_time=1.5)
        self.wait()

        scene_group = Group(frame, labels)
        self.play(scene_group.animate.scale(0.25).to_edge(LEFT, buff=0.1))
        net = ActorViz([48, 64, 64, 4])

        self.play(FadeIn(net), run_time=1.5)
        self.wait()

        agent = PPO(48, 4)
        acts = get_activations(agent, 0)
        self.play(net.forward_anim(acts))
        self.wait()

        action_labels = VGroup(
            *[
                Text(lbl, font_size=20).next_to(nd, RIGHT, buff=0.15)
                for nd, lbl in zip(
                    net.nodes[-1], ["↑ Up", "→ Right", "↓ Down", "← Left"]
                )
            ]
        )
        self.play(FadeIn(action_labels), run_time=1.5)
        self.wait()

        self.play(
            FadeOut(net), FadeOut(action_labels), FadeOut(scene_group), FadeOut(title)
        )

        # Part 2: Critic Inference
        title = TexText("Part 2: Critic Inference", font_size=72)
        self.play(Write(title))
        self.wait()

        self.play(title.animate.scale(0.5).to_edge(UP))
        scene = GymScene("CliffWalking-v1")
        frame, _ = scene.reset()
        self.play(FadeIn(frame), run_time=1.5)
        rows, cols = 4, 12

        cell_w = frame.get_width() / cols
        cell_h = frame.get_height() / rows
        origin = np.array([frame.get_left()[0], frame.get_top()[1], 0])

        labels = VGroup()
        for r in range(rows):
            for c in range(cols):
                s = r * cols + c
                pos = origin + RIGHT * (c + 0.5) * cell_w + DOWN * (r + 0.5) * cell_h
                labels.add(Text(str(s), font_size=14, font="Monospace").move_to(pos))

        self.play(LaggedStartMap(FadeIn, labels, lag_ratio=0.03), run_time=1.5)
        self.wait()

        scene_group = Group(frame, labels)
        self.play(scene_group.animate.scale(0.25).to_edge(LEFT, buff=0.1))
        net = ActorViz([48, 64, 64, 1])
        self.play(FadeIn(net), run_time=1.5)
        self.wait()

        agent = PPO(48, 4)
        acts = get_activations(agent, 0)
        self.play(net.forward_anim(acts))
        self.wait()

        critic_reward = Tex("V(s)").next_to(net.nodes[-1][0], RIGHT)
        self.play(Write(critic_reward))
        self.wait()

        self.play(
            FadeOut(net), FadeOut(scene_group), FadeOut(critic_reward), FadeOut(title)
        )

        # Part 3: Time to train
        title = TexText("Part 3: Time to train", font_size=72)
        self.play(Write(title))
        self.wait()
        self.play(title.animate.scale(0.5).to_edge(UP))

        # Step 1: Initialize the agent
        subtitle = TexText("Step 1: Initialize the agent", font_size=48)
        self.play(Write(subtitle))
        self.wait()

        self.play(subtitle.animate.scale(0.5).next_to(title, DOWN))
        # show a 48, 64, 64, 4 net initlized to all zero
        net = ActorViz([48, 64, 64, 4])
        self.play(FadeIn(net), run_time=1.5)
        self.wait()

        agent = PPO(48, 4)
        for p in agent.params:
            p.data.zero_()

        acts = get_activations(agent, 0)
        self.play(net.forward_anim(acts))
        self.wait()

        pct_labels = VGroup(
            *[
                VGroup(
                    Text("25%", font_size=18, color=GREEN_D),
                    Text(lbl, font_size=18),
                )
                .arrange(RIGHT, buff=0.1)
                .next_to(nd, RIGHT, buff=0.15)
                for nd, lbl in zip(
                    net.nodes[-1], ["↑ Up", "→ Right", "↓ Down", "← Left"]
                )
            ]
        )
        self.play(FadeIn(pct_labels), run_time=1.0)
        self.wait()

        self.play(
            FadeOut(net),
            FadeOut(pct_labels),
            FadeOut(subtitle),
            run_time=0.8,
        )

        # Step 2: Initialize the critic
        subtitle = TexText("Step 2: Initialize the critic", font_size=48)
        self.play(Write(subtitle))
        self.wait()
        self.play(subtitle.animate.scale(0.5).next_to(title, DOWN))

        critic_net = ActorViz([48, 64, 64, 1])
        self.play(FadeIn(critic_net), run_time=1.5)
        self.wait()

        agent = PPO(48, 4)
        for p in agent.params:
            p.data.zero_()

        critic_acts = get_critic_activations(agent, 0)
        self.play(critic_net.forward_anim(critic_acts))
        self.wait()

        v_label = Tex(r"V(s) = -50").next_to(critic_net.nodes[-1][0], RIGHT, buff=0.2)
        group = VGroup(v_label, critic_net)
        self.play(Write(v_label), group.animate.shift(LEFT))
        self.wait()

        self.play(
            FadeOut(subtitle),
            FadeOut(group),
            run_time=0.8,
        )

        # Step 3: collect a batch of experiences
        subtitle = TexText("Step 3: Collect a batch of trajectories", font_size=48)
        self.play(Write(subtitle))
        self.wait()
        self.play(subtitle.animate.scale(0.5).next_to(title, DOWN))

        ARROWS = {0: "↑", 1: "→", 2: "↓", 3: "←"}
        trajectories = [
            [1],  # a) →           (into cliff)
            [0, 1, 1, 2],  # b) ↑ → → ↓     (to goal)
            [3, 0, 1, 1, 2],  # c) ← ↑ → → ↓   (wall then goal)
        ]
        xs = [LEFT * 4.5, ORIGIN, RIGHT * 4.5]

        envs = [GymScene("TinyCliffWalking-v0") for _ in range(3)]
        frames, hdrs, rows, lbls = [], [], [], []

        for i, gs in enumerate(envs):
            f, _ = gs.reset()
            f.scale(0.35).move_to(xs[i])
            frames.append(f)

            l = Text(f"Trajectory {chr(ord('a') + i)}", font_size=20, color=YELLOW)
            l.next_to(f, UP, buff=0.15)
            lbls.append(l)

            h = Text(" a     r", font_size=18, font="Monospace", color=GREEN)
            h.next_to(f, DOWN, buff=0.25)
            hdrs.append(h)
            rows.append(VGroup())

        self.play(
            *[FadeIn(f) for f in frames],
            *[FadeIn(l) for l in lbls],
            *[FadeIn(h) for h in hdrs],
            run_time=0.6,
        )
        self.wait(0.5)

        max_steps = max(len(t) for t in trajectories)
        for step in range(max_steps):
            step_anims = []
            for i, traj in enumerate(trajectories):
                if step >= len(traj):
                    continue
                a = traj[step]
                nf, _, r, done = envs[i].step(a)
                nf.match_width(frames[i]).move_to(frames[i])
                self.remove(frames[i])
                self.add(nf)
                frames[i] = nf

                row = Text(
                    f" {ARROWS[a]}   {int(r):>4}",
                    font_size=18,
                    font="Monospace",
                )
                anchor = hdrs[i] if len(rows[i]) == 0 else rows[i][-1]
                row.next_to(anchor, DOWN, aligned_edge=LEFT, buff=0.06)
                rows[i].add(row)
                step_anims.append(FadeIn(row, shift=UP * 0.1))

            self.play(*step_anims, run_time=0.4)
            self.wait(0.2)

        self.wait()

        all_objs = [*frames, *lbls, *hdrs, *rows, subtitle]
        self.play(*[FadeOut(o) for o in all_objs], run_time=0.8)
        for gs in envs:
            gs.close()

        # Step 4: Calculate Rewards-to-Go
        subtitle = TexText("Step 4: Calculate Rewards-to-Go", font_size=48)
        self.play(Write(subtitle))
        self.wait()
        self.play(subtitle.animate.scale(0.5).next_to(title, DOWN))

        formula = Tex(
            r"\hat R_t = r_t + \gamma\, r_{t+1} + \gamma^2 r_{t+2} + \cdots",
            font_size=32,
        )
        recursive = Tex(
            r"= r_t + \gamma\,\hat R_{t+1}",
            font_size=32,
            color=BLUE,
        )
        gamma_lbl = Tex(r"\gamma = 0.9", font_size=28, color=YELLOW)
        form_grp = VGroup(formula, recursive, gamma_lbl)
        form_grp.arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        form_grp.next_to(subtitle, DOWN, buff=0.3)

        self.play(Write(formula))
        self.wait(0.5)
        self.play(Write(recursive))
        self.play(FadeIn(gamma_lbl))
        self.wait()

        GAM = 0.9
        traj_data = [
            ("a", ["→"], [-100]),
            ("b", ["↑", "→", "→", "↓"], [-1, -1, -1, -1]),
            ("c", ["←", "↑", "→", "→", "↓"], [-1, -1, -1, -1, -1]),
        ]

        all_rtgs = []
        for _, _, rews in traj_data:
            n = len(rews)
            rtg = [0.0] * n
            rtg[-1] = float(rews[-1])
            for i in range(n - 2, -1, -1):
                rtg[i] = rews[i] + GAM * rtg[i + 1]
            all_rtgs.append(rtg)

        all_rows = VGroup()
        all_rtg_mobs = []

        for ti, (name, syms, rews) in enumerate(traj_data):
            lbl = Text(f"{name})", font_size=22, color=YELLOW)
            step_cols = VGroup()
            rtg_mobs = []
            for si in range(len(syms)):
                arrow = Text(syms[si], font_size=26)
                rew = Text(f"r={rews[si]}", font_size=14, color=GREY_B)
                rtg_q = Text("?", font_size=16, color=GREEN)
                col = VGroup(arrow, rew, rtg_q).arrange(DOWN, buff=0.05)
                step_cols.add(col)
                rtg_mobs.append(rtg_q)
            step_cols.arrange(RIGHT, buff=0.5)
            row = VGroup(lbl, step_cols).arrange(RIGHT, buff=0.3)
            all_rows.add(row)
            all_rtg_mobs.append(rtg_mobs)

        all_rows.arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        all_rows.next_to(form_grp, DOWN, buff=0.35)

        self.play(LaggedStartMap(FadeIn, all_rows, lag_ratio=0.2), run_time=0.8)
        self.wait(0.5)

        for ti, (name, syms, rews) in enumerate(traj_data):
            n = len(rews)
            rtg_vals = all_rtgs[ti]
            rtg_mobs = all_rtg_mobs[ti]

            for i in range(n - 1, -1, -1):
                col_mob = all_rows[ti][1][i]
                hl = SurroundingRectangle(col_mob, color=YELLOW, buff=0.08)

                if i == n - 1:
                    calc_str = f"\\hat R = {rtg_vals[i]:.2f}"
                else:
                    calc_str = (
                        f"\\hat R = {rews[i]:.0f}"
                        f" + 0.9 \\times ({rtg_vals[i + 1]:.2f})"
                        f" = {rtg_vals[i]:.2f}"
                    )
                calc = Tex(calc_str, font_size=24)
                calc.to_edge(DOWN, buff=0.4)

                new_rtg = Text(f"{rtg_vals[i]:.2f}", font_size=16)
                new_rtg.move_to(rtg_mobs[i])

                self.play(ShowCreation(hl), FadeIn(calc), run_time=0.25)
                self.play(FadeTransform(rtg_mobs[i], new_rtg), run_time=0.3)
                rtg_mobs[i] = new_rtg
                self.wait(0.3)
                self.play(FadeOut(hl), FadeOut(calc), run_time=0.2)

        self.wait()

        all_on_screen = [
            all_rows,
            formula,
            recursive,
            gamma_lbl,
            subtitle,
            *[mob for traj_mobs in all_rtg_mobs for mob in traj_mobs],
        ]
        self.play(*[FadeOut(m) for m in all_on_screen], run_time=0.8)

        # Step 5: Calculate Advantage
        subtitle = TexText("Step 5: Calculate Advantage", font_size=48)
        self.play(Write(subtitle))
        self.wait()
        self.play(subtitle.animate.scale(0.5).next_to(title, DOWN))

        formula = Tex(
            r"A(s, a) = Q(s, a) - V(s) \approx \hat{R}_t - V(s)",
            font_size=32,
        )
        v_note = Tex(
            r"V(s) = -50 \text{ (untrained critic)}", font_size=28, color=YELLOW
        )
        form_grp = VGroup(formula, v_note)
        form_grp.arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        form_grp.next_to(subtitle, DOWN, buff=0.3)

        self.play(Write(formula))
        self.wait(0.5)
        self.play(FadeIn(v_note))
        self.wait()

        GAM = 0.9
        V_S = -50.0
        traj_data = [
            ("a", ["→"], [-100]),
            ("b", ["↑", "→", "→", "↓"], [-1, -1, -1, -1]),
            ("c", ["←", "↑", "→", "→", "↓"], [-1, -1, -1, -1, -1]),
        ]

        all_rtgs = []
        for _, _, rews in traj_data:
            n = len(rews)
            rtg = [0.0] * n
            rtg[-1] = float(rews[-1])
            for i in range(n - 2, -1, -1):
                rtg[i] = rews[i] + GAM * rtg[i + 1]
            all_rtgs.append(rtg)

        all_rows = VGroup()
        all_adv_mobs = []

        for ti, (name, syms, rews) in enumerate(traj_data):
            lbl = Text(f"{name})", font_size=22, color=YELLOW)
            step_cols = VGroup()
            adv_mobs = []
            rtg_vals = all_rtgs[ti]
            for si in range(len(syms)):
                arrow = Text(syms[si], font_size=26)
                rtg_txt = Text(f"R̂={rtg_vals[si]:.2f}", font_size=14, color=GREY_B)
                adv_q = Text("?", font_size=16, color=GREEN)
                col = VGroup(arrow, rtg_txt, adv_q).arrange(DOWN, buff=0.05)
                step_cols.add(col)
                adv_mobs.append(adv_q)
            step_cols.arrange(RIGHT, buff=0.5)
            row = VGroup(lbl, step_cols).arrange(RIGHT, buff=0.3)
            all_rows.add(row)
            all_adv_mobs.append(adv_mobs)

        all_rows.arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        all_rows.next_to(form_grp, DOWN, buff=0.35)

        self.play(LaggedStartMap(FadeIn, all_rows, lag_ratio=0.2), run_time=0.8)
        self.wait(0.5)

        for ti, (name, syms, rews) in enumerate(traj_data):
            rtg_vals = all_rtgs[ti]
            adv_mobs = all_adv_mobs[ti]

            for i in range(len(syms)):
                adv = rtg_vals[i] - V_S
                col_mob = all_rows[ti][1][i]
                hl = SurroundingRectangle(col_mob, color=YELLOW, buff=0.08)

                calc_str = f"A = {rtg_vals[i]:.2f} - ({int(V_S)}) = {adv:.2f}"
                calc = Tex(calc_str, font_size=24)
                calc.to_edge(DOWN, buff=0.4)

                adv_color = GREEN if adv > 0 else RED
                new_adv = Text(f"{adv:.2f}", font_size=16, color=adv_color)
                new_adv.move_to(adv_mobs[i])

                self.play(ShowCreation(hl), FadeIn(calc), run_time=0.25)
                self.play(FadeTransform(adv_mobs[i], new_adv), run_time=0.3)
                adv_mobs[i] = new_adv
                self.wait(0.3)
                self.play(FadeOut(hl), FadeOut(calc), run_time=0.2)

        self.wait()

        all_on_screen = [
            all_rows,
            formula,
            v_note,
            subtitle,
            *[mob for traj_mobs in all_adv_mobs for mob in traj_mobs],
        ]
        self.play(*[FadeOut(m) for m in all_on_screen], run_time=0.8)

        # Step 6: Optimize Critic Network
        subtitle = TexText("Step 6: Optimize Critic Network", font_size=48)
        self.play(Write(subtitle))
        self.wait()
        self.play(subtitle.animate.scale(0.5).next_to(title, DOWN))

        formula = Tex(
            r"\text{loss} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2",
            font_size=32,
        )
        legend = VGroup(
            Tex(r"y_i = \hat{R}_t \text{ (Rewards-to-Go)}", font_size=26, color=GREY_B),
            Tex(
                r"\hat{y}_i = V(s) \text{ (Critic output)}", font_size=26, color=GREY_B
            ),
        ).arrange(DOWN, buff=0.08, aligned_edge=LEFT)
        form_grp = VGroup(formula, legend)
        form_grp.arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        form_grp.next_to(subtitle, DOWN, buff=0.3)

        self.play(Write(formula))
        self.wait(0.5)
        self.play(FadeIn(legend))
        self.wait()

        GAM = 0.9
        V_S = -50.0
        traj_data = [
            ("a", ["→"], [-100]),
            ("b", ["↑", "→", "→", "↓"], [-1, -1, -1, -1]),
            ("c", ["←", "↑", "→", "→", "↓"], [-1, -1, -1, -1, -1]),
        ]

        first_rtgs = []
        for _, _, rews in traj_data:
            n = len(rews)
            rtg = [0.0] * n
            rtg[-1] = float(rews[-1])
            for i in range(n - 2, -1, -1):
                rtg[i] = rews[i] + GAM * rtg[i + 1]
            first_rtgs.append(rtg[0])

        COL_X = [-4.0, -2.2, -0.4, 1.8]  # traj | R̂ | V(s) | squared error

        headers = VGroup(
            Text("traj", font_size=22, color=YELLOW),
            Tex(r"\hat{R}_t", font_size=22, color=YELLOW),
            Tex(r"V(s)", font_size=22, color=YELLOW),
            Tex(r"(\hat{R}_t - V(s))^2", font_size=22, color=YELLOW),
        )
        for hdr, x in zip(headers, COL_X):
            hdr.set_x(x)
        headers.set_y(0)

        rows_group = VGroup()
        sq_err_vals = []
        row_mobs = []

        for ti, (name, _, _) in enumerate(traj_data):
            rtg = first_rtgs[ti]
            sq = (rtg - V_S) ** 2
            sq_err_vals.append(sq)

            cells = VGroup(
                Text(f"{name})", font_size=20),
                Text(f"{rtg:.2f}", font_size=20),
                Text(f"{int(V_S)}", font_size=20),
                Text("?", font_size=20, color=GREEN),
            )
            for cell, x in zip(cells, COL_X):
                cell.set_x(x)
            rows_group.add(cells)
            row_mobs.append(cells[3])

        rows_group.arrange(DOWN, buff=0.3)
        for row in rows_group:
            for cell, x in zip(row, COL_X):
                cell.set_x(x)

        table = VGroup(headers, rows_group).arrange(DOWN, buff=0.25)

        for hdr, x in zip(table[0], COL_X):
            hdr.set_x(x)
        for row in table[1]:
            for cell, x in zip(row, COL_X):
                cell.set_x(x)

        table.next_to(form_grp, DOWN, buff=0.35)

        self.play(FadeIn(headers), run_time=0.5)
        self.play(LaggedStartMap(FadeIn, rows_group, lag_ratio=0.2), run_time=0.6)
        self.wait(0.5)

        for ti in range(len(traj_data)):
            rtg = first_rtgs[ti]
            sq = sq_err_vals[ti]
            row = rows_group[ti]

            calc_str = f"({rtg:.2f} - ({int(V_S)}))^2 = ({rtg - V_S:.2f})^2 = {sq:.2f}"
            calc = Tex(calc_str, font_size=24)
            calc.to_edge(DOWN, buff=0.4)

            new_sq = Text(f"{sq:.2f}", font_size=20, color=GREEN)
            new_sq.move_to(row_mobs[ti])

            self.play(FadeIn(calc), run_time=0.25)
            self.play(FadeTransform(row_mobs[ti], new_sq), run_time=0.3)
            row_mobs[ti] = new_sq

            # Build a fresh VGroup with the ACTUAL live mobs so the box fits correctly
            live_row = VGroup(row[0], row[1], row[2], new_sq)
            hl = SurroundingRectangle(live_row, color=YELLOW, buff=0.1)
            self.play(ShowCreation(hl), run_time=0.2)
            self.wait(0.3)
            self.play(FadeOut(hl), FadeOut(calc), run_time=0.2)

        self.wait(0.4)

        mean_loss = sum(sq_err_vals) / len(sq_err_vals)
        loss_str = (
            r"\text{loss} = \frac{1}{3}("
            + " + ".join(f"{v:.2f}" for v in sq_err_vals)
            + rf") = {mean_loss:.2f}"
        )
        loss_final = Tex(loss_str, font_size=28, color=BLUE)
        loss_final.to_edge(DOWN, buff=0.5)

        live_rows = VGroup(
            *[
                VGroup(row[0], row[1], row[2], row_mobs[ti])
                for ti, row in enumerate(rows_group)
            ]
        )
        hl_all = SurroundingRectangle(live_rows, color=BLUE, buff=0.1)
        self.play(ShowCreation(hl_all), run_time=0.3)
        self.play(FadeIn(loss_final), run_time=0.5)
        self.wait()
        self.play(FadeOut(hl_all), run_time=0.2)
        self.wait()

        all_on_screen = [table, formula, legend, subtitle, loss_final, *row_mobs]
        self.play(*[FadeOut(m) for m in all_on_screen], run_time=0.8)

        # Step 7: Optimize Actor Network
        subtitle = TexText("Step 7: Optimize Actor Network", font_size=48)
        self.play(Write(subtitle))
        self.wait()
        self.play(subtitle.animate.scale(0.5).next_to(title, DOWN))

        LEFT_COLOR = GREEN
        RIGHT_COLOR = BLUE

        clip = Tex(
            "L^{CLIP}(\\theta)=\\mathbb{E}_t \\bigl[\\min\\bigl(r_t(\\theta)\\hat{A_t}, \\text{clip}(r_t(\\theta), 1-\\varepsilon, 1+\\varepsilon)\\hat{A_t}\\bigr)\\bigr]",
            tex_to_color_map={
                "r_t(\\theta)\\hat{A_t}": LEFT_COLOR,
                "\\text{clip}(r_t(\\theta), 1-\\varepsilon, 1+\\varepsilon)\\hat{A_t}": RIGHT_COLOR,
            },
        )
        self.play(Write(clip), run_time=2)
        self.wait()
        self.play(clip.animate.shift(UP * 2))

        clip_min_left = Tex("\\text{Left: }r_t(\\theta)\\hat{A_t}").next_to(clip, DOWN)
        clip_min_left.set_color(LEFT_COLOR)
        clip_min_left.set_color_by_tex("\\text{Left}", WHITE)
        self.play(Write(clip_min_left), run_time=0.8)
        self.wait()

        clip_min_left2 = Tex(
            "\\text{Left: } \\frac{\\text{action prob}_{\\text{current}}}{\\text{action prob}_{\\text{old}}}\\hat{A_t}"
        ).next_to(clip, DOWN * 1.5)
        clip_min_left2.set_color(LEFT_COLOR)
        clip_min_left2.set_color_by_tex("\\text{Left}", WHITE)
        self.play(TransformMatchingTex(clip_min_left, clip_min_left2), run_time=0.8)
        self.wait()

        clip_min_left3 = Tex(
            "\\text{Left: } \\frac{\\text{action prob}_{\\text{current}}}{\\text{action prob}_{\\text{old}}}\\cdot\\text{advantage}"
        ).next_to(clip, DOWN * 1.5)
        clip_min_left3.set_color(LEFT_COLOR)
        clip_min_left3.set_color_by_tex("\\text{Left}", WHITE)
        self.play(TransformMatchingTex(clip_min_left2, clip_min_left3), run_time=0.8)
        self.wait()

        clip_min_right = Tex(
            "\\text{Right: } \\text{clip}(r_t(\\theta), 1-\\varepsilon, 1+\\varepsilon)\\hat{A_t}"
        ).next_to(clip_min_left3, DOWN * 1.5)
        clip_min_right.set_color(RIGHT_COLOR)
        clip_min_right.set_color_by_tex("\\text{Right}", WHITE)
        self.play(Write(clip_min_right), run_time=0.8)
        self.wait()

        clip_min_right2 = Tex(
            "\\text{Right: } \\text{clip}(r_t(\\theta), 0.8, 1.2)\\hat{A_t}"
        ).next_to(clip_min_left3, DOWN * 1.5)
        clip_min_right2.set_color(RIGHT_COLOR)
        clip_min_right2.set_color_by_tex("\\text{Right}", WHITE)
        self.play(TransformMatchingTex(clip_min_right, clip_min_right2), run_time=0.8)
        self.wait()

        clip_min_right3 = Tex(
            "\\text{Right: } \\text{clip}(\\frac{\\text{action prob}_{\\text{current}}}{\\text{action prob}_{\\text{old}}}, 0.8, 1.2)\\cdot\\text{advantage}"
        ).next_to(clip_min_left3, DOWN * 1.5)
        clip_min_right3.set_color(RIGHT_COLOR)
        clip_min_right3.set_color_by_tex("\\text{Right}", WHITE)
        self.play(TransformMatchingTex(clip_min_right2, clip_min_right3), run_time=0.8)
        self.wait()

        self.play(FadeOut(clip_min_left3), FadeOut(clip_min_right3))

        eps = 0.3

        left_axis = Axes(
            x_range=(0, 2, 1),
            y_range=(0, 2, 1),
            axis_config=dict(
                stroke_width=3,
            ),
            height=4,
            width=4,
            num_sampled_graph_points_per_tick=100,
        ).shift(LEFT * 3.5 + DOWN * 1.5)
        left_axis_x_label, left_axis_y_label = left_axis.get_axis_labels(
            "r", "L^{CLIP}"
        )
        left_axis_graph = left_axis.get_graph(
            lambda x: min(x, 1 + eps),
            use_smoothing=False,
        )
        left_axis_line1 = left_axis.get_v_line(left_axis.c2p(1, 1))
        left_axis_line2 = left_axis.get_v_line(left_axis.c2p(1 + eps, 1 + eps))
        left_axis_label1 = Tex("1", font_size=24).next_to(left_axis.c2p(1, 0), DOWN)
        left_axis_label2 = Tex("1+\\varepsilon", font_size=24).next_to(
            left_axis.c2p(1 + eps, 0), DOWN
        )
        left_axis_dot = Dot(left_axis.c2p(1, 1))
        left_axis_title = Tex("A>0", font_size=24).next_to(left_axis, UP)
        self.play(
            ShowCreation(left_axis),
            ShowCreation(left_axis_graph),
            ShowCreation(left_axis_line1),
            ShowCreation(left_axis_line2),
            ShowCreation(left_axis_dot),
            ShowCreation(left_axis_label1),
            ShowCreation(left_axis_label2),
            ShowCreation(left_axis_x_label),
            ShowCreation(left_axis_y_label),
            ShowCreation(left_axis_title),
            run_time=0.8,
        )
        self.wait()

        right_axis = Axes(
            x_range=(0, 2, 1),
            y_range=(0, -2, 1),
            axis_config=dict(
                stroke_width=3,
            ),
            height=4,
            width=4,
            num_sampled_graph_points_per_tick=100,
        ).shift(RIGHT * 3.5 + DOWN * 1.5)
        right_axis_x_label = right_axis.get_x_axis_label("r")
        right_axis_y_label = right_axis.get_y_axis_label(
            "L^{CLIP}", edge=DOWN, direction=UR
        )
        right_axis_graph = right_axis.get_graph(
            lambda x: min(-x, -1 + eps),
            use_smoothing=False,
        )
        right_axis_line1 = right_axis.get_v_line(right_axis.c2p(1, -1))
        right_axis_line2 = right_axis.get_v_line(right_axis.c2p(1 - eps, -1 + eps))
        right_axis_label1 = Tex("1", font_size=24).next_to(right_axis.c2p(1, 0), UP)
        right_axis_label2 = Tex("1-\\varepsilon", font_size=24).next_to(
            right_axis.c2p(1 - eps, 0), UP
        )
        right_axis_dot = Dot(right_axis.c2p(1, -1))
        right_axis_title = Tex("A<0", font_size=24).next_to(right_axis, DOWN)
        self.play(
            ShowCreation(right_axis),
            ShowCreation(right_axis_graph),
            ShowCreation(right_axis_line1),
            ShowCreation(right_axis_line2),
            ShowCreation(right_axis_dot),
            ShowCreation(right_axis_label1),
            ShowCreation(right_axis_label2),
            ShowCreation(right_axis_x_label),
            ShowCreation(right_axis_y_label),
            ShowCreation(right_axis_title),
            run_time=0.8,
        )

        self.wait()

        self.play(
            FadeOut(clip),
            FadeOut(right_axis),
            FadeOut(right_axis_graph),
            FadeOut(right_axis_line1),
            FadeOut(right_axis_line2),
            FadeOut(right_axis_dot),
            FadeOut(right_axis_label1),
            FadeOut(right_axis_label2),
            FadeOut(right_axis_x_label),
            FadeOut(right_axis_y_label),
            FadeOut(right_axis_title),
            FadeOut(left_axis),
            FadeOut(left_axis_graph),
            FadeOut(left_axis_line1),
            FadeOut(left_axis_line2),
            FadeOut(left_axis_dot),
            FadeOut(left_axis_label1),
            FadeOut(left_axis_label2),
            FadeOut(left_axis_x_label),
            FadeOut(left_axis_y_label),
            FadeOut(left_axis_title),
            run_time=0.8,
        )

        # Step 8: Repeat steps 6 & 7
        subtitle = TexText("Step 8: Repeat steps 6 and 7 n times", font_size=48)
        self.play(Write(subtitle))
        self.wait()
        self.play(FadeOut(subtitle))

        # Step 9: Repeat steps 3-8
        subtitle = TexText("Step 9: Repeat steps 3-8 T times", font_size=48)
        self.play(Write(subtitle))
        self.wait()
        self.play(FadeOut(subtitle))
