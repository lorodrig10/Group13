"""
Export static figures for the mini-project report (levels 2 and 3).

Usage (from repo root, with the course venv active):
    python miniproject/export_report_figures.py
    python miniproject/export_report_figures.py --grass-seed 1 --olfaction-seed 127

Outputs (by default):
    overleaf_report/Images/grass_detection.png
    overleaf_report/Images/olfaction_ema_wind.png
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flygym.compose import ActuatorType

MINIPROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = MINIPROJECT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from miniproject.simulation import MiniprojectSimulation
from miniproject.utils import close_simulation
from submission.controller import GO_STRAIGHT_THRESHOLD, SENSOR_FREQ, Controller

DEFAULT_IMAGES_DIR = REPO_ROOT / "overleaf_report" / "Images"
MAX_STEPS = 100_000
WIND_FIRST_STEP = 2000
WIND_PERIOD_STEPS = 1000


def olfaction_asymmetry(odor: np.ndarray) -> float:
    """|L|/(|R|+eps) — same aggregation as Controller.follow_scent."""
    left = float(odor[0, 0] + odor[2, 0])
    right = float(odor[1, 0] + odor[3, 0])
    return abs(left) / (abs(right) + 1e-8)


def wind_change_steps(max_step: int) -> list[int]:
    return list(range(WIND_FIRST_STEP, max_step + 1, WIND_PERIOD_STEPS))


def _simulation_step(sim: MiniprojectSimulation, controller: Controller, step: int) -> None:
    joint_angles, adhesion = controller.step(sim, step)
    sim.set_actuator_inputs(sim.fly.name, ActuatorType.POSITION, joint_angles)
    sim.set_actuator_inputs(sim.fly.name, ActuatorType.ADHESION, adhesion)
    sim.step()


def _capture_grass_frame(
    level: int,
    seed: int,
    save_path: Path,
    max_search_steps: int,
    *,
    require_threat: bool,
) -> int | None:
    sim = MiniprojectSimulation(level=level, seed=seed, back_cam=False, top_cam=False)
    controller = Controller(sim)
    captured_step = None
    try:
        for step in range(max_search_steps):
            _simulation_step(sim, controller, step)
            if step % SENSOR_FREQ != 0:
                continue
            (
                _,
                _,
                obstacle_threat,
                obstacle_seen,
                *_,
            ) = controller.detect_obstacle(sim, visualize=False)
            if not obstacle_seen:
                continue
            if require_threat and not obstacle_threat:
                continue
            controller.current_step = step
            controller.detect_obstacle(
                sim,
                visualize=True,
                save_path=str(save_path),
                english=True,
            )
            captured_step = step
            break
    finally:
        del controller
        close_simulation(sim)
        del sim
        gc.collect()
    return captured_step


def export_grass_detection_figure(
    save_path: Path,
    *,
    level: int = 2,
    seed: int = 1,
    max_search_steps: int = 40_000,
    prefer_threat: bool = True,
) -> dict:
    """
    Run level 2 until grass is detected, then save the 2x2 vision panel
    (compound eyes + green masks) via Controller.visualize_detection.
    """
    save_path = Path(save_path)
    captured_step = None
    if prefer_threat:
        captured_step = _capture_grass_frame(
            level, seed, save_path, max_search_steps, require_threat=True
        )
    if captured_step is None:
        captured_step = _capture_grass_frame(
            level, seed, save_path, max_search_steps, require_threat=False
        )

    if captured_step is None:
        raise RuntimeError(
            f"No grass detected before step {max_search_steps} "
            f"(level={level}, seed={seed}). Try another seed."
        )
    return {"level": level, "seed": seed, "step": captured_step, "path": save_path}


def export_olfaction_wind_figure(
    save_path: Path,
    *,
    level: int = 3,
    seed: int = 127,
    max_steps: int = 35_000,
    log_every: int = 20,
) -> dict:
    """
    Log raw vs EMA olfactory asymmetry during a windy run and plot vs time (s).
    Vertical lines mark wind direction updates (every 1000 steps from step 2000).
    """
    save_path = Path(save_path)
    sim = MiniprojectSimulation(level=level, seed=seed, back_cam=False, top_cam=False)
    controller = Controller(sim)
    dt = float(sim.timestep)

    steps_log: list[int] = []
    rho_raw_log: list[float] = []
    rho_ema_log: list[float] = []
    last_step = 0

    try:
        for step in range(max_steps):
            last_step = step
            raw = sim.get_olfaction(sim.fly.name)
            if controller.odor_smooth is None:
                ema = raw.copy()
            else:
                ema = controller.odor_smooth.copy()

            if step % log_every == 0:
                steps_log.append(step)
                rho_raw_log.append(olfaction_asymmetry(raw))
                rho_ema_log.append(olfaction_asymmetry(ema))

            _simulation_step(sim, controller, step)
    finally:
        del controller
        close_simulation(sim)
        del sim
        gc.collect()

    t = np.array(steps_log, dtype=float) * dt
    rho_raw = np.array(rho_raw_log)
    rho_ema = np.array(rho_ema_log)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, rho_raw, color="0.55", linewidth=0.9, alpha=0.85, label="Raw")
    ax.plot(t, rho_ema, color="C0", linewidth=1.4, label="EMA")
    ax.axhline(1.0, color="k", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.axhspan(
        1.0 - GO_STRAIGHT_THRESHOLD,
        1.0 + GO_STRAIGHT_THRESHOLD,
        color="green",
        alpha=0.08,
        label=f"Straight band ($\\pm${GO_STRAIGHT_THRESHOLD:.3f})",
    )

    for ws in wind_change_steps(last_step):
        ax.axvline(ws * dt, color="C3", linestyle="--", linewidth=0.7, alpha=0.65)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Olfactory asymmetry $|L|/(|R|+\varepsilon)$")
    ax.set_title(f"Level {level}, seed {seed} — raw vs EMA under periodic wind")
    handles, labels = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D

    handles.append(
        Line2D([0], [0], color="C3", linestyle="--", label="Wind update")
    )
    labels.append("Wind update")
    ax.legend(handles, labels, loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "level": level,
        "seed": seed,
        "last_step": last_step,
        "dt_s": dt,
        "path": save_path,
    }


def export_all_report_figures(
    output_dir: Path | None = None,
    *,
    grass_seed: int = 1,
    olfaction_seed: int = 127,
) -> dict:
    """Write grass_detection.png and olfaction_ema_wind.png."""
    output_dir = Path(output_dir or DEFAULT_IMAGES_DIR)
    grass_path = output_dir / "grass_detection.png"
    olfaction_path = output_dir / "olfaction_ema_wind.png"
    return {
        "grass": export_grass_detection_figure(grass_path, seed=grass_seed),
        "olfaction": export_olfaction_wind_figure(olfaction_path, seed=olfaction_seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export report figures for levels 2 and 3.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_IMAGES_DIR,
        help="Directory for PNG outputs",
    )
    parser.add_argument("--grass-seed", type=int, default=1)
    parser.add_argument("--olfaction-seed", type=int, default=127)
    parser.add_argument("--grass-only", action="store_true")
    parser.add_argument("--olfaction-only", action="store_true")
    args = parser.parse_args()

    out = args.output_dir
    if not args.olfaction_only:
        info = export_grass_detection_figure(
            out / "grass_detection.png", seed=args.grass_seed
        )
        print(f"Grass figure: step {info['step']} -> {info['path']}")
    if not args.grass_only:
        info = export_olfaction_wind_figure(
            out / "olfaction_ema_wind.png", seed=args.olfaction_seed
        )
        print(f"Olfaction figure: {info['last_step']} steps, dt={info['dt_s']} s -> {info['path']}")


if __name__ == "__main__":
    main()
