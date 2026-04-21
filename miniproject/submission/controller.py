import numpy as np
from miniproject.simulation import MiniprojectSimulation
from enum import Enum, auto

SHOW_prints = True #enable prints everywhere
print_frequency = 1000 #every x timesteps
GO_STRAIGHT_THRESHOLD = 5/100 #if the ratio of left_signal to right_signal is within this threshold, go straight

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

    def step(self, sim: MiniprojectSimulation, step):
        if step % print_frequency == 0 and SHOW_prints:
            self.show_prints = True
        else:
            self.show_prints = False

        olfaction = sim.get_olfaction(sim.fly.name)
        
        if self.state == State.FOLLOW_SCENT:
            drives = self.follow_scent(olfaction)
        else:
            drives = np.array([0.0, 0.0])

        joint_angles, adhesion = self.turning_controller.step(drives)
        return joint_angles, adhesion


    def follow_scent(self, olfaction):
        
        left_odor_a = olfaction[0, 0]
        right_odor_a = olfaction[1, 0]
        left_odor_b = olfaction[2, 0]
        right_odor_b = olfaction[3, 0]

        left_signal = left_odor_a + left_odor_b
        right_signal = right_odor_a + right_odor_b
        if self.show_prints:
            print (f"Olfaction: {olfaction}")
            print( f"Left signal: {left_signal}", f"Right signal: {right_signal}")
            print( f"ratio: {left_signal/right_signal}")
        # get other observations as needed
        
        if abs(left_signal/right_signal-1)<GO_STRAIGHT_THRESHOLD:
            if self.show_prints: print("Going straight")
            drives = np.array([2.0, 2.0])
        elif left_signal > right_signal:
            if self.show_prints: print("Turning left")
            drives = np.array([0.2, 1.0])
        elif right_signal > left_signal:
            if self.show_prints: print("Turning right")
            drives = np.array([1.0, 0.2])

        return drives