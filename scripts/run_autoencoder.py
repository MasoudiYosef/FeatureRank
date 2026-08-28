"""Backward-compatible command that forwards to :mod:`feature_ranking`."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.feature_ranking import main as run_feature_ranking


def main() -> None:
    """Keep the historical command name without duplicating the workflow."""
    run_feature_ranking()


if __name__ == "__main__":
    main()
