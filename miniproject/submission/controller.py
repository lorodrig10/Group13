import numpy as np
from miniproject.simulation import MiniprojectSimulation
from enum import Enum, auto
import matplotlib.pyplot as plt

SHOW_PRINTS = True
print_frequency = 1000
GO_STRAIGHT_THRESHOLD = 2 / 100


class State(Enum):
    FOLLOW_SCENT = auto()
    AVOID_OBSTACLE = auto()
    DODGE_DRAGON = auto()


class Controller:
    def __init__(self, sim: MiniprojectSimulation):
        from flygym.examples.locomotion import TurningController

        self.turning_controller = TurningController(sim.timestep)
        self.state = State.FOLLOW_SCENT
        self.show_prints = False

        self.odor_smooth = None
        self.alpha = 0.0005

        self.obstacle_threshold = 0.015
        self.sky_region_ratio = 0.55        #allow to modify the % of height seen from the sky (the smaller the more high we see)
        self.min_green_height_ratio = 0.10

        #UNCOMMENT TO ACTIVATE TIME FOR DETECTION
        #self.avoid_duration = 300
        #self.avoid_timer = 0

    def step(self, sim: MiniprojectSimulation, step):
        self.show_prints = SHOW_PRINTS and step % print_frequency == 0

        raw_olfaction = sim.get_olfaction(sim.fly.name)

        if self.odor_smooth is None:
            self.odor_smooth = raw_olfaction.copy()
        else:
            self.odor_smooth = (
                (1 - self.alpha) * self.odor_smooth
                + self.alpha * raw_olfaction
            )

        left_score, right_score = self.detect_green_obstacle(sim)

        if self.show_prints:
            print(f"Vision obstacle scores: L={left_score:.4f}, R={right_score:.4f}")

        if left_score > self.obstacle_threshold or right_score > self.obstacle_threshold:
            self.state = State.AVOID_OBSTACLE
        else:
            self.state = State.FOLLOW_SCENT

        if self.state == State.FOLLOW_SCENT:
            drives = self.follow_scent()
        elif self.state == State.AVOID_OBSTACLE:
            drives = self.avoid_obstacle(left_score, right_score)
        else:
            drives = np.array([0.0, 0.0])

        joint_angles, adhesion = self.turning_controller.step(drives)

        #if step > 0 and self.show_prints:
        #    fly_vision = np.concatenate(sim.get_raw_vision(sim.fly.name), axis=-2)
        #    plt.figure(figsize=(10, 4))
        #    plt.imshow(fly_vision)
        #   plt.title(f"Vision step {step} — {self.state.name}")
        #    plt.axis("off")
        #    plt.show()

        return joint_angles, adhesion

    def detect_green_obstacle(self, sim):
        """
        Utilise directement les images RGB de sim.get_raw_vision().
        """

        eye_imgs = sim.get_raw_vision(sim.fly.name)
        scores = []

        for img in eye_imgs:
            img = np.asarray(img)

            if img.max() <= 1.0:
                img = img * 255.0

            h, w, c = img.shape

            sky_region = img[: int(h * self.sky_region_ratio), :, :]

            r = sky_region[:, :, 0].astype(float)
            g = sky_region[:, :, 1].astype(float)
            b = sky_region[:, :, 2].astype(float)

            green_mask = (
                (g > 90)
                & (g > r * 1.25)
                & (g > b * 1.05)
            )

            column_green_ratio = green_mask.mean(axis=0)
            tall_green_columns = column_green_ratio > self.min_green_height_ratio

            score = tall_green_columns.mean()
            scores.append(score)

        return scores[0], scores[1]

    def avoid_obstacle(self, left_score, right_score):
        if self.show_prints:
            print(f"AVOID_OBSTACLE: L={left_score:.4f}, R={right_score:.4f}")

        if left_score > right_score:
            if self.show_prints:
                print("Obstacle à gauche → tourner à droite")
            return np.array([1.2, 0.2])
        else:
            if self.show_prints:
                print("Obstacle à droite → tourner à gauche")
            return np.array([0.2, 1.2])

    def follow_scent(self):
        left_odor_a = self.odor_smooth[0, 0]
        left_odor_b = self.odor_smooth[2, 0]
        right_odor_a = self.odor_smooth[1, 0]
        right_odor_b = self.odor_smooth[3, 0]

        left_signal = left_odor_a + left_odor_b
        right_signal = right_odor_a + right_odor_b

        eps = 1e-8
        ratio = abs(left_signal) / (abs(right_signal) + eps)

        if self.show_prints:
            print(f"Olfaction: {self.odor_smooth}")
            print(f"Left signal: {left_signal}, Right signal: {right_signal}")
            print(f"ratio: {ratio}")

        if abs(ratio - 1) < GO_STRAIGHT_THRESHOLD:
            if self.show_prints:
                print("FOLLOW_SCENT: going straight")
            return np.array([2.0, 2.0])

        elif left_signal > right_signal:
            if self.show_prints:
                print("FOLLOW_SCENT: turning left")
            return np.array([0.2, 1.0])

        else:
            if self.show_prints:
                print("FOLLOW_SCENT: turning right")
            return np.array([1.0, 0.2])