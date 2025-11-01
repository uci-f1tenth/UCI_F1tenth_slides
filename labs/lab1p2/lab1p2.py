from manimlib import *


class Lab1p2(Scene):
    def construct(self):
        # Title
        title = TexText("F1tenth Lab 1 Part 2:").shift(1 * UP)
        title2 = TexText("Wall Following Code")

        self.play(Write(title))
        self.play(Write(title2))
        self.wait()
        self.play(FadeOut(title), FadeOut(title2))

        # Outline
        outline_title = TexText("Outline:", font_size=100).shift(UP * 2)
        outline = (
            VGroup(
                TexText("1. Implementing PID"),
                TexText("2. Implementing Wall Following"),
                TexText("3. Competition!"),
            )
            .arrange(DOWN, aligned_edge=LEFT)
            .shift(DOWN)
        )
        self.play(Write(outline_title))
        self.play(Write(outline))
        self.wait()
        self.play(FadeOut(outline_title), FadeOut(outline))

        # Implementing PID
        title = TexText("Implementing PID")

        self.play(Write(title))
        self.wait()
        self.play(FadeOut(title))

        # Calculating Integral
        axes = Axes((-1, 12), (-1, 6))
        axes.add_coordinate_labels()
        x_label = axes.get_x_axis_label("Time").shift(DOWN * 0.25)
        y_label = axes.get_y_axis_label("Error").shift(RIGHT * 0.25)
        self.play(Write(axes), Write(x_label), Write(y_label))

        graph = axes.get_graph(
            lambda x: 0.03 * (x - 2) * (x - 7) * (x - 9) + 3.2,
            color=BLUE,
        )
        self.play(
            ShowCreation(graph),
        )

        x_tracker = ValueTracker(0)
        dot1 = Dot(color=RED)
        rectangles = always_redraw(
            lambda: axes.get_riemann_rectangles(
                graph,
                x_range=[0, x_tracker.get_value()],
                dx=0.5,
                fill_opacity=0.6,
                stroke_width=1,
            )
        )
        f_always(dot1.move_to, lambda: axes.i2gp(x_tracker.get_value(), graph))
        self.play(FadeIn(dot1), ShowCreation(rectangles))
        self.play(x_tracker.animate.set_value(5), run_time=5)
        self.wait()
        self.play(
            FadeOut(axes),
            FadeOut(x_label),
            FadeOut(y_label),
            FadeOut(graph),
            FadeOut(dot1),
            FadeOut(rectangles),
        )

        # Integral code
        code = Code(
            "def update(self, measurement: float, dt: Optional[float]) -> float:\n"
            "   self.integral += error * dt"
        )
        self.play(Write(code))
        self.wait()
        self.play(FadeOut(code))

        # Calculating Derivative
        axes = Axes((-1, 12), (-1, 6))
        axes.add_coordinate_labels()
        x_label = axes.get_x_axis_label("Time").shift(DOWN * 0.25)
        y_label = axes.get_y_axis_label("Error").shift(RIGHT * 0.25)
        self.play(Write(axes), Write(x_label), Write(y_label))
        graph = axes.get_graph(
            lambda x: 0.03 * (x - 2) * (x - 7) * (x - 9) + 3.2,
            color=BLUE,
        )
        self.play(
            ShowCreation(graph),
        )

        x_tracker = ValueTracker(0)
        dt = 1
        dot1 = Dot()
        dot2 = Dot()
        f_always(dot1.move_to, lambda: axes.i2gp(x_tracker.get_value(), graph))
        f_always(dot2.move_to, lambda: axes.i2gp(x_tracker.get_value() + dt, graph))
        line = always_redraw(
            lambda: Line(
                axes.i2gp(x_tracker.get_value(), graph),
                axes.i2gp(x_tracker.get_value() + dt, graph),
                color=RED,
            ).set_length(4)
        )

        self.play(FadeIn(dot1), FadeIn(dot2), ShowCreation(line))
        self.play(x_tracker.animate.set_value(9), run_time=5)
        self.wait()
        self.play(
            FadeOut(axes),
            FadeOut(x_label),
            FadeOut(y_label),
            FadeOut(graph),
            FadeOut(dot1),
            FadeOut(dot2),
            FadeOut(line),
        )

        # Derivative code
        code = Code(
            "def update(self, measurement: float, dt: Optional[float]) -> float:\n"
            "   derivative = (error - self.previous_error) / dt\n"
            "   self.previous_error = error"
        )
        self.play(Write(code))
        self.wait()
        self.play(FadeOut(code))

        # Full PID code
        code = Code(
            """def update(self, measurement: float, dt: Optional[float]) -> float:
            error: float = self.setpoint - measurement
            if dt and dt > 0.0:
                self.integral += error * dt
            derivative: float = (
                (error - self.previous_error) / dt
                if dt and dt > 0.0 and self.previous_error is not None
                else 0.0
            )
            u: float = self.kp * error + self.ki * self.integral + self.kd * derivative
            low, high = self.out_limits
            if low is not None and u < low:
                u = low
            if high is not None and u > high:
                u = high
            self.previous_error = error
            return u
        """
        )
        self.play(Write(code))
        self.wait()
        self.play(FadeOut(code))

        # Implementing Wall Following
        title = TexText("Implementing Wall Following")
        self.play(Write(title))
        self.wait()
        self.play(FadeOut(title))

        # Wall Following code
        alpha_equation = Tex(
            r"\alpha",
            " = ",
            r"\arctan\left(\frac{a\cos(\theta) - b}{a\sin(\theta)}\right)",
        )
        D_equation = Tex("D", " = ", "b", r"\cos(\alpha)").next_to(alpha_equation, DOWN)
        alpha_code = Code(
            "alpha = np.arctan2(a * np.cos(theta) - b, a * np.sin(theta))"
        )
        D_code = Code("D = b * np.cos(alpha)").next_to(alpha_code, DOWN)
        self.play(Write(alpha_equation))
        self.play(Write(D_equation))
        self.wait()
        self.play(Transform(alpha_equation, alpha_code))
        self.wait()
        self.play(Transform(D_equation, D_code))
        self.wait()
        full_wall_following_code = Code(
            """def get_distance_from_wall(lidar_range_array: np.ndarray[Any] | None):
            theta = np.radians(45.0)
            b = get_range(lidar_range_array, np.radians(90))
            a = get_range(lidar_range_array, np.radians(45))
            alpha = np.arctan2(a * np.cos(theta) - b, a * np.sin(theta))
            D = b * np.cos(alpha)
            error = D + 1.5 * np.sin(alpha)
            
            return error"""
        )
        self.play(
            FadeOut(D_equation),
            FadeOut(alpha_equation),
            FadeIn(full_wall_following_code),
        )
        self.wait()
        self.play(FadeOut(full_wall_following_code))

        # Competition
        title = TexText("Competition!")
        self.play(Write(title))
        self.wait()
        self.play(FadeOut(title))

        # End
        end_text = TexText("Thank you!")
        self.play(Write(end_text))
        self.wait()
        self.play(FadeOut(end_text))
