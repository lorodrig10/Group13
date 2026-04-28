import numpy as np
from miniproject.simulation import MiniprojectSimulation
from enum import Enum, auto
import matplotlib.pyplot as plt
# import pygame

SHOW_PRINTS = True #enable prints everywhere
print_frequency = 1000 #every x timesteps
GO_STRAIGHT_THRESHOLD = 2/100 #if the ratio of left_signal to right_signal is within this threshold, go straight

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
        # --- EMA Variables ---
        self.odor_smooth = None
        self.alpha = 0.0005

        
        # self.display_size = (1024, 512)
        # self.screen = pygame.display.set_mode(self.display_size)
        # pygame.display.set_caption("COBAR 2026 Miniproject")

    # def print_fly_vision(self, vision_data, step):
    #     # Extract left and right eyes
    #     left_eye = vision_data[0]
    #     right_eye = vision_data[1]
        
    #     # The data is usually a flat array (e.g., length 729). 
    #     # We find the square root to reshape it into a 2D image (e.g., 27x27).
    #     n_pixels = left_eye.shape[0]
    #     grid_size = int(np.sqrt(n_pixels))
        
    #     fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        
    #     # Check if it can form a perfect square grid
    #     if grid_size * grid_size == n_pixels:
    #         left_img = left_eye.reshape((grid_size, grid_size, -1))
    #         right_img = right_eye.reshape((grid_size, grid_size, -1))
            
    #         # Squeeze color channel if it's grayscale
    #         if left_img.shape[-1] == 1:
    #             left_img = left_img.squeeze(-1)
    #             right_img = right_img.squeeze(-1)
    #             axes[0].imshow(left_img, cmap='gray', vmin=0, vmax=1)
    #             axes[1].imshow(right_img, cmap='gray', vmin=0, vmax=1)
    #         else:
    #             axes[0].imshow(left_img)
    #             axes[1].imshow(right_img)
    #     else:
    #         # Fallback: Plot as a flat strip if not a perfect square
    #         axes[0].imshow(left_eye[np.newaxis, :], aspect='auto', cmap='gray')
    #         axes[1].imshow(right_eye[np.newaxis, :], aspect='auto', cmap='gray')

    #     axes[0].set_title(f"Left Eye View (Step {step})")
    #     axes[1].set_title(f"Right Eye View (Step {step})")
    #     axes[0].axis('off')
    #     axes[1].axis('off')
        
    #     # This prevents Jupyter from making 40,000 separate plots
    #     # clear_output(wait=True) 
    #     plt.show()

    def step(self, sim: MiniprojectSimulation, step):
        if  SHOW_PRINTS and step % print_frequency == 0:
            self.show_prints = True
        else:
            self.show_prints = False

        raw_olfaction = sim.get_olfaction(sim.fly.name)
        vision_data   = sim.get_ommatidia_readouts(sim.fly.name)
        if self.odor_smooth is None: # Initialized once
            self.odor_smooth = raw_olfaction.copy()
        else:
            self.odor_smooth = (1 - self.alpha) * self.odor_smooth + (self.alpha * raw_olfaction)
        
        if self.state == State.FOLLOW_SCENT:
            drives = self.follow_scent()
        else:
            drives = np.array([0.0, 0.0])

        joint_angles, adhesion = self.turning_controller.step(drives)
        if step > 0 and self.show_prints:
            # self.print_fly_vision(vision_data, step)
            frame = np.concatenate(
                [frames[-1] for frames in sim.renderer.frames.values()], axis=-2
            )
            fly_vision = np.concatenate(sim.get_raw_vision(sim.fly.name), axis=-2)
            fly_vision = np.pad(
                fly_vision,
                (
                    [0] * 2,
                    [(frame.shape[1] - fly_vision.shape[1]) // 2] * 2,
                    [0] * 2,
                ),
            )
            frame = np.vstack((fly_vision, frame))
            # self.render(frame)
            # plt.figure(figsize=(10, 8)) # You can adjust this size
            # plt.imshow(frame)
            # plt.title(f"Simulation Dashboard (Step {step})")
            # plt.axis('off') # Hides the axes/numbers
            # plt.show()
            plt.figure(figsize=(10, 4)) # You can adjust this size
            plt.imshow(fly_vision)
            plt.title(f"Vision (Step {step})")
            # plt.axis('off') # Hides the axes/numbers
            plt.show()
            print(f"Fly vision shape: {fly_vision.shape}")
        return joint_angles, adhesion


    def follow_scent(self):
        
        left_odor_a = self.odor_smooth[0, 0]
        left_odor_b = self.odor_smooth[2, 0]
        right_odor_a = self.odor_smooth[1, 0]
        right_odor_b = self.odor_smooth[3, 0]

        left_signal = left_odor_a + left_odor_b
        right_signal = right_odor_a + right_odor_b
        if self.show_prints:
            print (f"Olfaction: {self.odor_smooth}")
            print( f"Left signal: {left_signal}", f"Right signal: {right_signal}")
            print( f"ratio: {left_signal/right_signal}")
        
        if abs(left_signal)/abs(right_signal)-1<GO_STRAIGHT_THRESHOLD:
            if self.show_prints: print("Going straight")
            drives = np.array([2.0, 2.0])
        elif left_signal > right_signal:
            if self.show_prints: print("Turning left")
            drives = np.array([0.2, 1.0])
        elif right_signal > left_signal:
            if self.show_prints: print("Turning right")
            drives = np.array([1.0, 0.2])

        return drives
    
    # def render(self, frame: np.ndarray):
            # frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            # if frame_surface.get_size() != self.display_size:
            #     frame_surface = pygame.transform.smoothscale(
            #         frame_surface, self.display_size
            #     )
            # self.screen.blit(frame_surface, (0, 0))
            # pygame.display.flip()

    