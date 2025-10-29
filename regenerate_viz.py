#!/usr/bin/env python3
"""Regenerate visualizations for a results directory."""

import json
import sys
from pathlib import Path
from typing import Any

from visualization import generate_all_plots

if len(sys.argv) < 2:
    print("Usage: python regenerate_viz.py <results_directory>")
    sys.exit(1)

results_dir = Path(sys.argv[1])
json_path = results_dir / "evolution_history.json"

if not json_path.exists():
    print(f"Error: {json_path} not found")
    sys.exit(1)

with open(json_path) as f:
    data: dict[str, Any] = json.load(f)
    history: list[dict[str, Any]] = data.get("history", data)  # Handle both formats


# Mock cost tracker
class MockTracker:
    """Mock cost tracker for regenerating visualizations without cost data."""

    def __init__(self) -> None:
        self.calls: list[Any] = []


cost_tracker = MockTracker()
generate_all_plots(history, cost_tracker, str(results_dir))
print(f"✓ Visualizations generated in {results_dir}")
