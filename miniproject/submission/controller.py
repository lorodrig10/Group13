import numpy as np
from miniproject.simulation import MiniprojectSimulation

SHOW_prints = False
show_prints = False
print_frequency = 1000

class Controller:
    def __init__(self, sim: MiniprojectSimulation):
        from flygym.examples.locomotion import TurningController

        self.turning_controller = TurningController(sim.timestep)

    def step(self, sim: MiniprojectSimulation, step):
        if step % print_frequency == 0 and SHOW_prints:
            show_prints = True
        else:
            show_prints = False

        olfaction = sim.get_olfaction(sim.fly.name)
        
        left_antenna = olfaction[0, 0]
        right_antenna = olfaction[1, 0]
        left_palp = olfaction[2, 0]
        right_palp = olfaction[3, 0]

        left_signal = left_antenna + left_palp
        right_signal = right_antenna + right_palp
        if show_prints:
            print (f"Olfaction: {olfaction}")
            print( f"Left signal: {left_signal}", f"Right signal: {right_signal}")
        # get other observations as needed
        if left_signal > right_signal:
            if show_prints: print("Turning left")
            drives = np.array([0.2, 1.0])
        elif right_signal > left_signal:
            if show_prints: print("Turning right")
            drives = np.array([1.0, 0.2])
        joint_angles, adhesion = self.turning_controller.step(drives)
        return joint_angles, adhesion
