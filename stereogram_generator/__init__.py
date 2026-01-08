"""Mystereogram package - autostereogram generator with depth estimation."""

from pathlib import Path


def _get_version() -> str:
    """Get version from pyproject.toml file."""
    try:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        # Get the project root directory (parent of stereogram_generator package)
        project_root = Path(__file__).parent.parent
        pyproject_path = project_root / "pyproject.toml"

        if pyproject_path.exists():
            with open(pyproject_path, "rb") as f:
                pyproject = tomllib.load(f)
            return pyproject["project"]["version"]
    except Exception:
        pass

    # Fallback version if we can't read from pyproject.toml
    return "0.1.0"


__version__ = _get_version()
__author__ = "codeprimate"
__description__ = "A Python command-line tool that generates autostereograms from 2D images using AI depth estimation"

__all__ = [
    "__version__",
    "__author__",
    "__description__",
]

