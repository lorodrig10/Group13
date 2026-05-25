from importlib.resources import files as _importlib_resources_files
from pathlib import Path as _Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from miniproject.simulation import MiniprojectSimulation

miniproject_assets_dir = _Path(
    str(_importlib_resources_files("miniproject") / "assets")
)


def close_simulation(sim: "MiniprojectSimulation") -> None:
    """Release MuJoCo GPU renderers before creating another simulation.

    Without this, a second Renderer (or re-running a notebook cell) often records
    black frames because the previous OpenGL/EGL context was not freed.
    """
    renderer = getattr(sim, "renderer", None)
    if renderer is not None:
        renderer.close()
        sim.renderer = None

    eye_renderer = sim.__dict__.get("eye_renderer")
    if eye_renderer is not None:
        eye_renderer.close()
        del sim.__dict__["eye_renderer"]
