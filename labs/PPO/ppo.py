from manim_imports_ext import *


class Intro(InteractiveScene):
    def construct(self):
        title = Text("PPO").scale(2)
        self.play(Write(title))
        self.wait()
        self.play(FadeOut(title))
