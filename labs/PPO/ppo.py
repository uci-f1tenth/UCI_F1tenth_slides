import os
import shutil
import tempfile

import gymnasium as gym
from manimlib import *
from PIL import Image


class GymScene:
    def __init__(self, env_id, width=12, **kwargs):
        self.width = width
        self.tmp_dir = tempfile.mkdtemp(prefix="manim_gym_")
        self.frame_count = 0
        self.env = gym.make(env_id, render_mode="rgb_array", **kwargs)
        self.env.reset()

    def render(self):
        self.frame_count += 1
        path = os.path.join(self.tmp_dir, f"{self.frame_count}.png")
        Image.fromarray(self.env.render()).save(path)
        return ImageMobject(path).set_width(self.width)

    def step(self, action):
        """Take a step and return (frame, state, reward, done)."""
        state, reward, done, trunc, info = self.env.step(action)
        return self.render(), state, reward, done or trunc

    def close(self):
        self.env.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)


class Intro(Scene):
    def construct(self):
        gs = GymScene("CliffWalking-v1")

        frame = gs.render()
        self.play(FadeIn(frame))
        self.wait(0.5)

        for action in [0] + [1] * 11 + [2]:
            new_frame, _, _, done = gs.step(action)
            self.remove(frame)
            self.add(new_frame)
            frame = new_frame
            self.wait(0.2)
            if done:
                break

        self.play(Flash(frame.get_center(), flash_radius=0.5, color=GREEN))
        self.wait()
        gs.close()
