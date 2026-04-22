from cv2 import data
import numpy as np
from miniproject.simulation import MiniprojectSimulation
from enum import Enum, auto
from matplotlib import pyplot as plt

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

    def step(self, sim: MiniprojectSimulation, step):
        if  SHOW_PRINTS and step % print_frequency == 0:
            self.show_prints = True
        else:
            self.show_prints = False

        raw_olfaction = sim.get_olfaction(sim.fly.name)
        if self.odor_smooth is None: # Initialized once
            self.odor_smooth = raw_olfaction.copy()
        else:
            self.odor_smooth = (1 - self.alpha) * self.odor_smooth + (self.alpha * raw_olfaction)
        
        if self.state == State.FOLLOW_SCENT:
            drives = self.follow_scent()
        else:
            drives = np.array([0.0, 0.0])

        #######
        import matplotlib.pyplot as plt
        ommatidia_readouts = sim.get_ommatidia_readouts(sim.fly.name)
        print(f"Ommatidia readouts shape: {ommatidia_readouts.shape}")
        ommatidia_readouts = ommatidia_readouts[0]
        print(f"Ommatidia readouts shape after indexing: {ommatidia_readouts.shape} ")
        retina = sim.world.fly_lookup[sim.fly.name].retina
        img = retina.hex_pxls_to_human_readable(ommatidia_readouts, color_8bit=True)
        plt.figure()
        plt.imshow(img[:, :, 0], cmap="gray", vmin=0, vmax=255)
        plt.title("channel 0")

        plt.figure()
        plt.imshow(img[:, :, 1], cmap="gray", vmin=0, vmax=255)
        plt.title("channel 1")
    
        plt.show()
        ######
        joint_angles, adhesion = self.turning_controller.step(drives)
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