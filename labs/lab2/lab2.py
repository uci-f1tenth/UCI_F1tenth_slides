from enum import Enum
from manimlib import *
from manim_slides import Slide


class ObstacleType(Enum):
    POSITIVE_SPACE = 1
    NEGATIVE_SPACE = 2


def get_is_in(*obstacles: tuple[Mobject, ObstacleType]):
    """
    Check if a point is inside any of the obstacles.

    Args:
        obstacles: tuples of (obstacle, is_negative_space) where is_negative_space
                  indicates if the obstacle should be treated as negative space (inverted)
    """

    def is_in(sample_point):
        for obstacle, obstacle_type in obstacles:
            inside = False
            if type(obstacle) is Circle:
                if (
                    np.linalg.norm(sample_point - obstacle.get_center())
                    <= obstacle.get_radius()
                ):
                    inside = True
            elif type(obstacle) is Ellipse:
                x = (sample_point[0] - obstacle.get_center()[0]) / (
                    obstacle.get_width() / 2
                )
                y = (sample_point[1] - obstacle.get_center()[1]) / (
                    obstacle.get_height() / 2
                )
                if x**2 + y**2 <= 1:
                    inside = True
            elif type(obstacle) is Rectangle:
                upper_left = obstacle.get_corner(UL)
                lower_right = obstacle.get_corner(DR)
                if (
                    upper_left[0] <= sample_point[0] <= lower_right[0]
                    and lower_right[1] <= sample_point[1] <= upper_left[1]
                ):
                    inside = True
            if obstacle_type == ObstacleType.NEGATIVE_SPACE:
                inside = not inside

            if inside:
                return True
        return False

    return is_in


def ray_updater(
    car: Mobject,
    car_angle: ValueTracker,
    rays: list[Line],
    is_outside,
    max_ray_length=20,
    dx=0.1,
    binary_search_iterations=10,
    use_disparity_extender=False,
    threshold: float = 2.0,
    bubble_size: float = 0.3,
):
    def update_rays(mob: Mobject, dt: float):
        angles = np.linspace(
            car_angle.get_value() - np.pi / 2,
            car_angle.get_value() + np.pi / 2,
            len(rays),
        )
        for ray, angle in zip(
            rays,
            angles,
        ):
            unit_vector = np.cos(angle) * RIGHT + np.sin(angle) * UP
            for t in np.arange(0, max_ray_length, dx):
                sample_point = car.get_center() + t * unit_vector

                if is_outside(sample_point):
                    low, high = t - dx, t
                    for _ in range(binary_search_iterations):
                        mid = (low + high) / 2
                        mid_point = car.get_center() + mid * unit_vector
                        if is_outside(mid_point):
                            high = mid
                        else:
                            low = mid
                    sample_point = car.get_center() + high * unit_vector
                    ray.put_start_and_end_on(car.get_center(), sample_point)
                    break

        if use_disparity_extender:
            lidar_range_array = np.array([ray.get_length() for ray in rays])
            disparities = np.where(abs(np.diff(lidar_range_array)) > threshold)[0]
            for d in disparities:
                if lidar_range_array[d] < lidar_range_array[d + 1]:
                    bubble_indices = int(
                        bubble_size
                        * len(lidar_range_array)
                        / (lidar_range_array[d] * np.pi)
                    )
                    lidar_range_array[d + 1 : d + bubble_indices + 2] = (
                        lidar_range_array[d]
                    )
                else:
                    bubble_indices = int(
                        bubble_size
                        * len(lidar_range_array)
                        / (lidar_range_array[d + 1] * np.pi)
                    )
                    lidar_range_array[d - bubble_indices : d + 1] = lidar_range_array[
                        d + 1
                    ]
            for i, ray in enumerate(rays):
                angle = angles[i]
                unit_vector = np.cos(angle) * RIGHT + np.sin(angle) * UP
                ray.put_start_and_end_on(
                    car.get_center(),
                    car.get_center() + lidar_range_array[i] * unit_vector,
                )

    return update_rays


def car_updater(
    car_velocity: ValueTracker,
    car_angle: ValueTracker,
    rays: list[Line],
):
    previous_max_ray = None

    def update_car(car: Mobject, dt: float):
        nonlocal previous_max_ray
        if car_velocity.get_value() < 1:
            car_velocity.set_value(car_velocity.get_value() + dt)

        max_ray = max(rays, key=lambda ray: ray.get_length())
        if previous_max_ray is not None:
            previous_max_ray.set_color(RED)
        previous_max_ray = max_ray
        max_ray.set_color(YELLOW)
        target_angle = max_ray.get_angle()
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


class Lab2(Slide):
    def construct(self):
        # Title
        title = TexText("F1tenth Lab 2:").shift(1 * UP)
        title2 = TexText("Follow The Gap")

        self.play(Write(title))
        self.play(Write(title2))
        self.next_slide()
        self.play(FadeOut(title), FadeOut(title2))

        # Naive Approach
        title = TexText("Naive Approach")
        self.play(Write(title))
        self.next_slide()
        title2 = TexText(
            "Naive Approach:\\\\",
            "1. Find the farthest ray\\\\",
            "2. Drive towards the farthest ray",
        )
        self.play(TransformMatchingTex(title, title2))
        self.next_slide()
        self.play(FadeOut(title2))

        # Visualize Naive Approach With Obstacles
        car = (
            ImageMobject("labs/lab1/car_topview.png").scale(0.1).shift(LEFT * 4 + DOWN)
        )
        self.add(car)
        car_velocity = ValueTracker(0)
        car_angle = ValueTracker(0)

        bounding_rectangle = Rectangle(width=12, height=6)
        obstacle_1 = Circle(radius=1, stroke_color=WHITE, stroke_width=4).shift(
            RIGHT * 2 + UP * 2
        )
        obstacle_2 = Circle(radius=1.5, stroke_color=WHITE, stroke_width=4).shift(
            DOWN * 1.5
        )

        obstacle_3 = Circle(radius=1, stroke_color=WHITE, stroke_width=4).shift(
            LEFT * 2 + UP * 0.5
        )
        obstacles = VGroup(bounding_rectangle, obstacle_1, obstacle_2, obstacle_3)
        is_outside_track = get_is_in(
            (obstacle_1, ObstacleType.POSITIVE_SPACE),
            (obstacle_2, ObstacleType.POSITIVE_SPACE),
            (obstacle_3, ObstacleType.POSITIVE_SPACE),
            (bounding_rectangle, ObstacleType.NEGATIVE_SPACE),
        )

        rays = [
            Line(
                car.get_center(),
                car.get_center(),
                stroke_width=2,
                color=RED,
            )
            for _ in range(15)
        ]
        rays_group = VGroup(*rays)
        rays_updater_instance = ray_updater(car, car_angle, rays, is_outside_track)

        self.play(
            FadeIn(car),
            FadeIn(rays_group),
            Write(obstacles),
        )
        rays_group.add_updater(rays_updater_instance)
        self.next_slide()
        car_updater_instance = car_updater(car_velocity, car_angle, rays)

        car.add_updater(car_updater_instance)
        self.wait_until(
            lambda: sum(is_outside_track(corner) for corner in car.get_points()) >= 1,
            max_time=10,
        )
        car.remove_updater(car_updater_instance)
        rays_group.remove_updater(rays_updater_instance)
        self.next_slide()
        self.play(FadeOut(car), FadeOut(rays_group), FadeOut(obstacles))

        # Visualize Naive Approach On Track
        car = (
            ImageMobject("labs/lab1/car_topview.png")
            .scale(0.1)
            .rotate(PI / 2)
            .shift(LEFT * 2.5 + DOWN)
        )
        car_velocity = ValueTracker(0)
        car_angle = ValueTracker(np.pi / 2)

        track_outer = Ellipse(width=6, height=8, stroke_color=WHITE, stroke_width=4)
        track_inner = Ellipse(width=3, height=5, stroke_color=WHITE, stroke_width=4)
        obstacles = VGroup(track_outer, track_inner)
        is_outside_track = get_is_in(
            (track_inner, ObstacleType.POSITIVE_SPACE),
            (track_outer, ObstacleType.NEGATIVE_SPACE),
        )

        rays = [
            Line(
                car.get_center(),
                car.get_center(),
                stroke_width=2,
                color=RED,
            )
            for _ in range(15)
        ]
        rays_group = VGroup(*rays)
        rays_updater_instance = ray_updater(car, car_angle, rays, is_outside_track)
        rays_group.add_updater(rays_updater_instance)

        self.play(
            FadeIn(car),
            FadeIn(rays_group),
            Write(obstacles),
        )
        self.next_slide()
        car_updater_instance = car_updater(car_velocity, car_angle, rays)

        car.add_updater(car_updater_instance)
        self.wait_until(
            lambda: sum(is_outside_track(corner) for corner in car.get_points()) >= 2,
            max_time=10,
        )
        car.remove_updater(car_updater_instance)
        rays_group.remove_updater(rays_updater_instance)
        self.next_slide()
        self.play(FadeOut(car), FadeOut(rays_group), FadeOut(obstacles))

        # Disparity Extender
        title = TexText("Disparity Extender")
        self.play(Write(title))
        self.next_slide()
        title2 = TexText(
            "Naive Approach:\\\\",
            "1. Find the farthest ray\\\\",
            "2. Drive towards the farthest ray",
        )
        self.play(TransformMatchingTex(title, title2))
        self.next_slide()
        title3 = TexText(
            "Disparity Extender:\\\\",
            "1. Extend large disparities\\\\",
            "2. Find the farthest ray\\\\",
            "3. Drive towards the farthest ray",
        )
        self.play(TransformMatchingTex(title2, title3))
        self.next_slide()
        title4 = TexText(
            "$\\text{Disparities:}$\\\\",
            "$\\left|\\frac{d}{d \\theta}\\text{ray\\_length}\\right|>\\text{threshold}$",
        )
        self.play(TransformMatchingTex(title3, title4))
        self.next_slide()
        title5 = TexText(
            "$\\text{Disparities:}$\\\\",
            "$abs(ray\\_lengths[i+1]-ray\\_lengths[i])->\\text{threshold}$",
        )
        self.play(TransformMatchingTex(title4, title5))
        self.next_slide()
        self.play(FadeOut(title5))

        # Visualize Disparity Extender With Obstacles
        car = (
            ImageMobject("labs/lab1/car_topview.png").scale(0.1).shift(LEFT * 4 + DOWN)
        )
        self.add(car)
        car_velocity = ValueTracker(0)
        car_angle = ValueTracker(0)

        bounding_rectangle = Rectangle(width=12, height=6)
        obstacle_1 = Circle(radius=1, stroke_color=WHITE, stroke_width=4).shift(
            RIGHT * 2 + UP * 2
        )
        obstacle_2 = Circle(radius=1.5, stroke_color=WHITE, stroke_width=4).shift(
            DOWN * 1.5
        )

        obstacle_3 = Circle(radius=1, stroke_color=WHITE, stroke_width=4).shift(
            LEFT * 2 + UP * 0.5
        )
        obstacles = VGroup(bounding_rectangle, obstacle_1, obstacle_2, obstacle_3)
        is_outside_track = get_is_in(
            (obstacle_1, ObstacleType.POSITIVE_SPACE),
            (obstacle_2, ObstacleType.POSITIVE_SPACE),
            (obstacle_3, ObstacleType.POSITIVE_SPACE),
            (bounding_rectangle, ObstacleType.NEGATIVE_SPACE),
        )

        rays = [
            Line(
                car.get_center(),
                car.get_center(),
                stroke_width=0.5,
                color=RED,
            )
            for _ in range(60)
        ]
        rays_group = VGroup(*rays)
        rays_updater_instance = ray_updater(
            car, car_angle, rays, is_outside_track, use_disparity_extender=True
        )

        self.play(
            FadeIn(car),
            FadeIn(rays_group),
            Write(obstacles),
        )
        rays_group.add_updater(rays_updater_instance)
        self.next_slide()
        car_updater_instance = car_updater(car_velocity, car_angle, rays)

        car.add_updater(car_updater_instance)
        self.wait_until(
            lambda: sum(is_outside_track(corner) for corner in car.get_points()) >= 2,
            max_time=10,
        )
        car.remove_updater(car_updater_instance)
        rays_group.remove_updater(rays_updater_instance)
        self.next_slide()
        self.play(FadeOut(car), FadeOut(rays_group), FadeOut(obstacles))

        # Visualize Naive Approach On Track
        car = (
            ImageMobject("labs/lab1/car_topview.png")
            .scale(0.1)
            .rotate(PI / 2)
            .shift(LEFT * 2.5 + DOWN)
        )
        car_velocity = ValueTracker(0)
        car_angle = ValueTracker(np.pi / 2)

        track_outer = Ellipse(width=6, height=8, stroke_color=WHITE, stroke_width=4)
        track_inner = Ellipse(width=3, height=5, stroke_color=WHITE, stroke_width=4)
        obstacles = VGroup(track_outer, track_inner)
        is_outside_track = get_is_in(
            (track_inner, ObstacleType.POSITIVE_SPACE),
            (track_outer, ObstacleType.NEGATIVE_SPACE),
        )

        rays = [
            Line(
                car.get_center(),
                car.get_center(),
                stroke_width=0.5,
                color=RED,
            )
            for _ in range(60)
        ]
        rays_group = VGroup(*rays)
        rays_updater_instance = ray_updater(
            car,
            car_angle,
            rays,
            is_outside_track,
            use_disparity_extender=True,
        )
        rays_group.add_updater(rays_updater_instance)

        self.play(
            FadeIn(car),
            FadeIn(rays_group),
            Write(obstacles),
        )
        self.next_slide()
        car_updater_instance = car_updater(car_velocity, car_angle, rays)

        car.add_updater(car_updater_instance)
        self.wait_until(
            lambda: sum(is_outside_track(corner) for corner in car.get_points()) >= 2,
            max_time=10,
        )
        car.remove_updater(car_updater_instance)
        rays_group.remove_updater(rays_updater_instance)
        self.next_slide()
        self.play(FadeOut(car), FadeOut(rays_group), FadeOut(obstacles))
