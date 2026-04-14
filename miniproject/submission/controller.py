import numpy as np
from miniproject.simulation import MiniprojectSimulation


class Controller:
    def __init__(self, sim: MiniprojectSimulation):
        # you may also implement your own turning controller
        from flygym.examples.locomotion import TurningController

        self.turning_controller = TurningController(sim.timestep)

    def step(self, sim: MiniprojectSimulation):
        # implement your control algorithm here
        olfaction = sim.get_olfaction(sim.fly.name)
        
        left_antenna = olfaction[0, 0]
        right_antenna = olfaction[1, 0]
        left_palp = olfaction[2, 0]
        right_palp = olfaction[3, 0]

        # For a simple steering controller, you might just average the left side 
        # and average the right side to decide which way to turn:
        left_signal = left_antenna + left_palp
        right_signal = right_antenna + right_palp
        # print(f"Olfaction: {olfaction}", f"Left signal: {left_signal}", f"Right signal: {right_signal}")
        # get other observations as needed
        if left_signal > right_signal:
            # turn left
            drives = np.array([0.2, 1.0])
        elif right_signal > left_signal:
            # turn right
            drives = np.array([1.0, 0.2])
        joint_angles, adhesion = self.turning_controller.step(drives)
        return joint_angles, adhesion
