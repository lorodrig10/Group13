# Miniproject Controller System Guide

Date: 2026-04-22
Repository focus: COBAR 2026 Group13 miniproject controller workflow

## 1. Why this document exists

You said your immediate goal is to design a controller that passes multiple tests, but first you need a complete understanding of:
- the code layout,
- simulation architecture,
- file/function/class relationships,
- which APIs are safe and useful for controller logic.

This guide is a full map of the controller-relevant part of the project, from entry scripts down to sensor generation and actuator writing.

## 2. Top-level architecture in one page

The miniproject runtime has a clean three-layer split:

1) Task layer (your code and runners)
- miniproject/submission/controller.py
- miniproject/run_controller.ipynb
- miniproject/run_interactive.py

2) Miniproject world/simulation layer
- src/miniproject/simulation.py
- src/miniproject/fly.py
- src/miniproject/arena/*.py
- src/miniproject/interactive/*.py

3) FlyGym engine layer
- src/flygym/simulation.py
- src/flygym/compose/fly.py
- src/flygym/examples/locomotion/turning_controller.py

Mental model:
- Your controller computes a 2D descending command (left, right).
- TurningController transforms that into full leg joint targets + adhesion on/off.
- Simulation applies those actuator arrays in MuJoCo each step.
- Sensors (olfaction and vision) are queried from the same simulation object.

## 3. Repository areas that matter most for passing controller tests

Primary files to master:
- miniproject/submission/controller.py
- src/miniproject/simulation.py
- src/miniproject/arena/banana.py
- src/flygym/simulation.py
- src/flygym/examples/locomotion/turning_controller.py
- miniproject/run_controller.ipynb

Secondary but useful:
- miniproject/run_interactive.py
- src/miniproject/fly.py
- src/miniproject/arena/terrain.py
- src/miniproject/arena/grass.py
- src/miniproject/arena/dragonfly.py
- src/miniproject/interactive/controls.py

Environment/dependencies:
- pyproject.toml

## 4. Execution paths and how files connect

### 4.1 Evaluation-like path (run_controller.ipynb)

Typical sequence in the notebook:
1. Construct simulation
   - sim = MiniprojectSimulation(level=..., seed=...)
2. Construct your controller
   - controller = Controller(sim)
3. Each timestep:
   - joint_angles, adhesion = controller.step(sim, i)
   - sim.set_actuator_inputs(sim.fly.name, ActuatorType.POSITION, joint_angles)
   - sim.set_actuator_inputs(sim.fly.name, ActuatorType.ADHESION, adhesion)
   - sim.step()
   - sim.render_as_needed()

So tests can verify:
- Controller constructor behavior,
- Controller.step signature and output shape/type,
- Stability over many steps,
- Compatibility with all levels.

### 4.2 Interactive debugging path (run_interactive.py)

This script is a manual baseline and debugging aid.

Flow:
1. Parse level/seed and keyboard mode.
2. Build MiniprojectSimulation.
3. Use FlyGym TurningController directly with keyboard-generated left/right gains.
4. Apply actuator arrays, step simulation, render frame.

Why this matters:
- It provides a known-good gait driver and control magnitudes.
- You can compare your policy outputs against keyboard behavior.

## 5. Core class and method map

### 5.1 src/miniproject/simulation.py: MiniprojectSimulation

Class role:
- Specializes generic FlyGym Simulation into this course task world.
- Controls level mechanics (terrain/grass/wind/dragonfly).

Constructor flags derived from level:
- enable_terrain: levels 1..4
- enable_grass: levels 2..4
- enable_wind: levels 3..4
- enable_dragonfly: level 4

World build sequence:
1. create_fly() from src/miniproject/fly.py
2. optionally add tracking and top cameras
3. build MiniprojectWorld (mixins + terrain)
4. place banana target at sampled polar location
5. optionally place grass
6. optionally add dragonfly
7. add fly spawn at world height
8. call Simulation base constructor
9. set renderer
10. reset and warm-up 2000 physics steps

Important methods:
- set_wind(magnitude, angle_deg)
  - Sets MuJoCo wind and world flow velocity used by olfaction field.
- step()
  - Applies dynamic wind changes in wind-enabled levels.
  - Updates dragonfly motion in level 4.
  - Calls base Simulation.step().

Dragonfly subsystem methods:
- _init_dragonfly_controller
- _get_fly_state
- _update_dragonfly_velocity_buffer
- _get_mean_fly_velocity
- _should_trigger_dragonfly
- _set_dragonfly_rest_pose
- _start_dragonfly_attack
- _step_dragonfly

Controller impact:
- Levels are behaviorally different worlds, not only visual skins.
- A policy that works in level 0 can fail in level 3/4 due to wind/predator perturbations.

### 5.2 src/flygym/simulation.py: Simulation (base API you call)

Key API surface used by controllers:

State/sensors:
- get_olfaction(fly_name, **kwargs)
  - Returns shape approximately [n_sensors, n_odor_dimensions].
  - In this project odor_dimensions is effectively 1.
- get_ommatidia_readouts(fly_name)
  - Returns [n_cameras, n_ommatidia, 2].
- get_body_positions(fly_name)
- get_joint_angles(fly_name), get_joint_velocities(fly_name)

Actuation:
- set_actuator_inputs(fly_name, actuator_type, inputs)
  - Validates length against actuator count.
  - Writes to MuJoCo control vector.

Stepping/rendering:
- step()
- render_as_needed()

Critical point:
- If your output arrays do not match required lengths, set_actuator_inputs raises ValueError.

### 5.3 src/flygym/examples/locomotion/turning_controller.py: TurningController

Role:
- Converts 2D descending command into full locomotor command.

Input:
- action: np.ndarray shape (2,)
  - [delta_L, delta_R]

Output:
- joint_angles: flattened shape (42,)
- adhesion: shape (6,)

Internal behavior:
- CPG network updates amplitudes/frequencies from action.
- PreprogrammedSteps maps phase/magnitude to per-leg joint targets and adhesion.

Implication for your controller:
- You normally only decide the 2D command.
- You do not need to directly synthesize 42 joint targets from scratch.

### 5.4 src/miniproject/fly.py: create_fly

Defines fly body/actuators/sensors at construction:
- Adds joints and actuators (position control by default).
- Adds odor sensors.
- Adds vision system.
- Adds adhesion actuators on tarsus segments.
- Adds force sensors and antenna joints.

This explains why these APIs are available on MiniprojectSimulation:
- get_olfaction works because odor sensors exist.
- get_ommatidia_readouts works because vision exists.
- POSITION and ADHESION actuator writes both exist.

### 5.5 src/flygym/compose/fly.py: ActuatorType enum

Relevant enum values include:
- POSITION
- ADHESION
(and others such as MOTOR, VELOCITY, MUSCLE, etc.)

In this miniproject control loop, you use:
- POSITION for 42 joint targets,
- ADHESION for 6 leg contact signals.

## 6. World mechanics and sensor generation

### 6.1 Target/odor field: src/miniproject/arena/banana.py

BananaSliceMixin does two jobs:
1) Creates banana visual/collision geoms in world.
2) Implements olfaction model via compute_log_concentration.

Odor depends on:
- source position,
- flow velocity (wind),
- diffusivity,
- decay,
- emission rate,
- sensor positions.

get_olfaction returns exp(log_concentration) by default.

Why this is important:
- Wind changes (level 3/4) alter odor gradient direction and shape.
- Gradient-following policy must be robust to time-varying flow.

### 6.2 Terrain: src/miniproject/arena/terrain.py

RollingHills builds a heightfield terrain from smoothed noise.
Provides:
- get_height(x, y)
- get_normal(x, y)

Used in simulation setup for:
- fly spawn z,
- object placement normals/heights.

### 6.3 Grass: src/miniproject/arena/grass.py

Grass blades are dynamic objects with hinge joints.
Potential effects:
- visual clutter,
- physical interaction perturbations,
- altered traversal dynamics.

### 6.4 Dragonfly predator: src/miniproject/arena/dragonfly.py + simulation logic

DragonFlyMixin provides movable mocap predator body.
MiniprojectSimulation level 4 drives looming attacks probabilistically.

Policy consequence:
- Robust controllers need a fallback behavior when visual looming is detected.

### 6.5 Skybox: src/miniproject/arena/sky.py

Adds environmental texture only (no direct control effect), but changes visual background statistics for vision-based policies.

## 7. Input/output contracts for your controller

Expected constructor and step interface (based on runner usage):
- Controller(sim: MiniprojectSimulation)
- step(sim: MiniprojectSimulation, step: int) -> (joint_angles, adhesion)

Expected return compatibility:
- joint_angles length must match POSITION actuators (42 with current fly setup)
- adhesion length must match ADHESION actuators (6)

Practical safe pattern:
- Use TurningController.step(drives) to generate these arrays.

## 8. Relationship graph (file-level dependencies)

Main chain:
- miniproject/run_controller.ipynb
  -> miniproject/submission/controller.py
  -> src/miniproject/simulation.py (MiniprojectSimulation)
  -> src/flygym/simulation.py (base engine)
  -> MuJoCo physics + renderer

Controller internal chain:
- miniproject/submission/controller.py
  -> src/flygym/examples/locomotion/turning_controller.py
  -> CPG network + preprogrammed steps
  -> returns joint_angles + adhesion

World build chain:
- src/miniproject/simulation.py
  -> src/miniproject/fly.py
  -> src/miniproject/arena/banana.py
  -> src/miniproject/arena/terrain.py
  -> src/miniproject/arena/grass.py
  -> src/miniproject/arena/dragonfly.py

Interactive debug chain:
- miniproject/run_interactive.py
  -> src/miniproject/interactive/controls.py
  -> src/miniproject/interactive/game_state.py
  -> src/flygym/examples/locomotion/turning_controller.py

## 9. What can break hidden tests quickly

Based on current code in miniproject/submission/controller.py:

1) Undefined variable usage in step
- Calls plt.imshow(img, ...) but img is not defined.
- Any execution path hitting this fails immediately.

2) Heavy plotting inside control loop
- Blocking plots in every step are not evaluation-safe.
- Causes major slowdown or crashes in headless environments.

3) Potential divide-by-zero in follow_scent
- ratio uses left/right signals directly without epsilon guard.

4) Overly tiny smoothing factor
- alpha = 0.0005 makes adaptation extremely slow.
- Could underperform in changing wind fields.

5) State machine placeholders not implemented
- AVOID_OBSTACLE and DODGE_DRAGON exist but no behavior logic.

## 10. Design strategy to pass multiple levels/tests

### 10.1 Keep architecture simple and test-safe

Recommended structure:
- Per-step sensor acquisition
- Feature extraction
- State selection (odor-follow / avoid / evade)
- Convert desired steering into drives
- TurningController.step(drives)

Avoid inside step:
- plotting,
- expensive logging each frame,
- random uncontrolled branching.

### 10.2 Build progressively by levels

Level 0-1:
- reliable odor gradient ascent,
- stable forward movement,
- anti-oscillation steering hysteresis.

Level 2:
- add obstacle cues (vision flow asymmetry or collision proxy).

Level 3:
- wind-robust odor integration (EMA with realistic alpha, short memory).

Level 4:
- visual looming trigger for temporary escape behavior.

### 10.3 Keep deterministic behavior where possible

- Respect seed in simulation.
- Avoid random action sampling in controller.
- Use bounded gains and clipping.

## 11. Useful API cheat sheet

From simulation object in your step:
- sim.get_olfaction(sim.fly.name)
- sim.get_ommatidia_readouts(sim.fly.name)
- sim.get_body_positions(sim.fly.name)
- sim.get_joint_angles(sim.fly.name)

For action synthesis:
- self.turning_controller.step(np.array([left_drive, right_drive]))

For runner loop (outside controller):
- sim.set_actuator_inputs(sim.fly.name, ActuatorType.POSITION, joint_angles)
- sim.set_actuator_inputs(sim.fly.name, ActuatorType.ADHESION, adhesion)
- sim.step()

## 12. Fast orientation checklist for new work sessions

1. Read current controller and strip non-test-safe code.
2. Confirm step returns valid shapes every call.
3. Validate level 0 first in run_controller notebook.
4. Check level 3 behavior under wind changes.
5. Add predator/obstacle handling only after core odor-follow is stable.

## 13. Where this guide maps in code

Core docs and runners:
- README.md
- miniproject/README.md
- miniproject/run_controller.ipynb
- miniproject/run_interactive.py

Submission target:
- miniproject/submission/controller.py

Simulation and world:
- src/miniproject/simulation.py
- src/miniproject/fly.py
- src/miniproject/utils.py
- src/miniproject/arena/banana.py
- src/miniproject/arena/terrain.py
- src/miniproject/arena/grass.py
- src/miniproject/arena/dragonfly.py
- src/miniproject/arena/sky.py

Input helpers:
- src/miniproject/interactive/controls.py
- src/miniproject/interactive/game_state.py

FlyGym engine and locomotion bridge:
- src/flygym/simulation.py
- src/flygym/compose/fly.py
- src/flygym/examples/locomotion/turning_controller.py

Dependencies:
- pyproject.toml
