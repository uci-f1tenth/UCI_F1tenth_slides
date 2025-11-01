from manimlib import *


def get_is_outside_track(track_inner: Ellipse, track_outer: Ellipse):
    def is_outside(sample_point):
        x_inner = (sample_point[0] - track_inner.get_center()[0]) / (
            track_inner.get_width() / 2
        )
        y_inner = (sample_point[1] - track_inner.get_center()[1]) / (
            track_inner.get_height() / 2
        )

        x_outer = (sample_point[0] - track_outer.get_center()[0]) / (
            track_outer.get_width() / 2
        )
        y_outer = (sample_point[1] - track_outer.get_center()[1]) / (
            track_outer.get_height() / 2
        )

        return x_inner**2 + y_inner**2 <= 1 or x_outer**2 + y_outer**2 >= 1

    return is_outside


def ray_updater(
    car: Mobject,
    car_angle: ValueTracker,
    rays: list[Line],
    is_outside,
    max_ray_length=10,
    dx=0.01,
):
    def update_rays(mob: Mobject, dt: float):
        for ray, angle in zip(
            rays,
            np.linspace(
                car_angle.get_value() - np.pi / 2,
                car_angle.get_value() + np.pi / 2,
                len(rays),
            ),
        ):
            unit_vector = np.cos(angle) * RIGHT + np.sin(angle) * UP
            for t in np.arange(0, max_ray_length, dx):
                sample_point = car.get_center() + t * unit_vector

                if is_outside(sample_point):
                    ray.put_start_and_end_on(car.get_center(), sample_point)
                    break

    return update_rays


def car_updater(
    car_velocity: ValueTracker,
    car_angle: ValueTracker,
    rays: list[Line],
):
    def update_car(car: Mobject, dt: float):
        if car_velocity.get_value() < 1:
            car_velocity.set_value(car_velocity.get_value() + dt)

        max_ray = max(rays, key=lambda ray: ray.get_length())
        target_angle = max_ray.get_angle()
        print(
            f"Current car angle: {car_angle.get_value():.2f}, Max ray angle: {target_angle:.2f}, length: {max_ray.get_length():.2f}"
        )
        rotation = np.clip(
            0.1 * (target_angle - car_angle.get_value()), -2 * dt, 2 * dt
        )
        car.rotate(rotation)
        car_angle.increment_value(rotation)

        forward_direction = np.array(
            [np.cos(car_angle.get_value()), np.sin(car_angle.get_value()), 0]
        )
        car.shift(car_velocity.get_value() * dt * forward_direction)

    return update_car


class Lab2(Scene):
    def construct(self):
        # Title
        title = TexText("F1tenth Lab 2:").shift(1 * UP)
        title2 = TexText("Follow The Gap")

        self.play(Write(title))
        self.play(Write(title2))
        self.wait()
        self.play(FadeOut(title), FadeOut(title2))

        # Naive Approach
        title = TexText("Naive Approach")
        self.play(Write(title))
        self.wait()
        self.play(FadeOut(title))

        car = (
            ImageMobject("labs/lab1/car_topview.png")
            .scale(0.1)
            .rotate(PI / 2)
            .shift(LEFT * 2)
        )
        car_velocity = ValueTracker(0)
        car_angle = ValueTracker(np.pi / 2)

        track_outer = Ellipse(width=6, height=8, color=WHITE, stroke_width=4)
        track_inner = Ellipse(width=3, height=5, color=WHITE, stroke_width=4)
        track = VGroup(track_outer, track_inner)

        rays = [
            Line(
                car.get_center(),
                car.get_center(),
                stroke_width=2,
                color=RED,
                stroke_opacity=0.5,
            )
            for _ in range(20)
        ]
        rays_group = VGroup(*rays)
        rays_updater_instance = ray_updater(
            car, car_angle, rays, get_is_outside_track(track_inner, track_outer)
        )
        rays_group.add_updater(rays_updater_instance)

        self.play(
            FadeIn(car),
            FadeIn(rays_group),
            Write(track),
        )
        car_updater_instance = car_updater(car_velocity, car_angle, rays)

        car.add_updater(car_updater_instance)
        self.wait_until(
            lambda: sum(
                get_is_outside_track(track_inner, track_outer)(corner)
                for corner in car.get_all_corners()
            )
            >= 4,
            max_time=10,
        )
        car.remove_updater(car_updater_instance)
        rays_group.remove_updater(rays_updater_instance)
        self.wait()
        self.play(FadeOut(car), FadeOut(rays_group), FadeOut(track))
