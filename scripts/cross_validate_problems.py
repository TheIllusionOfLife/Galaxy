#!/usr/bin/env python3
"""Cross-problem generalization analysis for evolved surrogate models.

This script tests whether surrogate models trained on one N-body test problem
can generalize to other problems, answering the scientific question: "Do LLM-
discovered strategies use universal physics approximations or problem-specific tricks?"

Usage:
    python scripts/cross_validate_problems.py

Output:
    - results/analysis/cross_validation_YYYYMMDD_HHMMSS/cross_validation_matrix.md
    - results/analysis/cross_validation_YYYYMMDD_HHMMSS/cross_validation_results.json
"""

import json
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from prototype import (
    CosmologyCrucible,
    compile_external_surrogate,
    get_initial_particles,
    make_parametric_surrogate,
)
from scripts.extract_best_model import find_best_model, load_evolution_history


@dataclass
class CrossValidationResult:
    """Single cross-validation test result."""

    trained_on: str
    tested_on: str
    trained_particles: int
    tested_particles: int
    fitness: float
    accuracy: float
    speed: float
    generalization_penalty: float  # % drop from original fitness


def load_best_model(run_dir: Path) -> dict[str, Any]:
    """Load best model from evolution run directory.

    Args:
        run_dir: Path to evolution run directory containing evolution_history.json

    Returns:
        Dictionary containing best model data plus metadata (test_problem, num_particles)

    Raises:
        FileNotFoundError: If evolution_history.json doesn't exist
        KeyError: If metadata is missing from evolution history
    """
    history_path = run_dir / "evolution_history.json"
    history = load_evolution_history(history_path)

    # Extract metadata
    if "metadata" not in history:
        raise KeyError("metadata field missing from evolution history")

    metadata = history["metadata"]
    best_model = find_best_model(history)

    # Add metadata to best model
    best_model["test_problem"] = metadata["test_problem"]
    best_model["num_particles"] = metadata["num_particles"]

    return best_model


def _build_model_callable(model: dict) -> Callable[[list[float], list[list[float]]], list[float]]:
    """Build callable function from model dictionary.

    Args:
        model: Model dictionary with either 'raw_code' or 'theta' fields

    Returns:
        Callable with signature predict(particle, all_particles) -> predicted_particle

    Raises:
        ValueError: If model has neither raw_code nor theta
    """
    # LLM-generated models have raw_code
    if "raw_code" in model:
        # Need validation particles for compile_external_surrogate
        # Use simple 2-particle system as validation
        validation_particles = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
        return compile_external_surrogate(model["raw_code"], validation_particles)

    # Parametric models have theta parameters
    if "theta" in model:
        theta = model["theta"]
        # make_parametric_surrogate needs all_particles argument (unused but required)
        validation_particles = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
        return make_parametric_surrogate(theta, validation_particles)

    raise ValueError("Model must have either 'raw_code' or 'theta' field")


def evaluate_model_on_problem(
    model: dict, test_problem: str, num_particles: int
) -> dict[str, float]:
    """Evaluate a trained model on a different test problem.

    Args:
        model: Model dictionary from load_best_model()
        test_problem: Test problem name (two_body, figure_eight, plummer)
        num_particles: Number of particles for test problem

    Returns:
        Dictionary with keys: fitness, accuracy, speed

    Raises:
        ValueError: If test_problem is invalid
    """
    # Get initial conditions for test problem
    particles = get_initial_particles(test_problem, num_particles)

    # Build callable from model
    model_callable = _build_model_callable(model)

    # Create crucible with test problem particles
    crucible = CosmologyCrucible.with_particles(particles)

    # Evaluate model
    accuracy, speed = crucible.evaluate_surrogate_model(model_callable)

    # Calculate fitness (same formula as evolution)
    if speed > 0:
        fitness = accuracy / speed
    else:
        fitness = float("inf")

    return {"fitness": fitness, "accuracy": accuracy, "speed": speed}


def compute_generalization_penalty(original_fitness: float, new_fitness: float) -> float:
    """Calculate % fitness drop when tested on different problem.

    Args:
        original_fitness: Fitness on training problem
        new_fitness: Fitness on test problem

    Returns:
        Percentage drop in fitness (positive = worse, negative = better)

    Examples:
        >>> compute_generalization_penalty(1000.0, 500.0)
        50.0  # 50% drop
        >>> compute_generalization_penalty(100.0, 150.0)
        -50.0  # 50% improvement (rare)
    """
    if original_fitness == 0.0:
        return float("inf")

    return ((original_fitness - new_fitness) / original_fitness) * 100.0


def create_cross_validation_matrix(run_dirs: dict[str, Path]) -> list[CrossValidationResult]:
    """Create full cross-validation matrix for all problem pairs.

    Args:
        run_dirs: Dictionary mapping test_problem name to run directory path

    Returns:
        List of CrossValidationResult for all (trained, tested) pairs
    """
    results = []

    for trained_problem, trained_run_dir in run_dirs.items():
        # Load best model from this problem
        model = load_best_model(trained_run_dir)
        trained_particles = model["num_particles"]
        original_fitness = model["fitness"]

        # Test on all problems (including itself)
        for tested_problem in run_dirs.keys():
            # Get particle count for tested problem
            tested_model = load_best_model(run_dirs[tested_problem])
            tested_particles = tested_model["num_particles"]

            # Evaluate model on test problem
            eval_result = evaluate_model_on_problem(model, tested_problem, tested_particles)

            # Compute generalization penalty
            penalty = compute_generalization_penalty(original_fitness, eval_result["fitness"])

            # Create result
            result = CrossValidationResult(
                trained_on=trained_problem,
                tested_on=tested_problem,
                trained_particles=trained_particles,
                tested_particles=tested_particles,
                fitness=eval_result["fitness"],
                accuracy=eval_result["accuracy"],
                speed=eval_result["speed"],
                generalization_penalty=penalty,
            )

            results.append(result)

    return results


def export_results_markdown(results: list[CrossValidationResult], output_path: Path) -> None:
    """Export 3x3 cross-validation matrix as markdown table.

    Args:
        results: List of CrossValidationResult
        output_path: Path to save markdown file
    """
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get unique problems (ordered) - need both trained_on and tested_on
    problems = sorted({r.trained_on for r in results} | {r.tested_on for r in results})

    # Build matrix
    lines = ["# Cross-Problem Generalization Analysis", "", "## 3x3 Cross-Validation Matrix", ""]

    # Header row
    header = ["Trained On → Tested On"] + problems
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))

    # Data rows
    for trained_problem in problems:
        row = [f"**{trained_problem}**"]

        for tested_problem in problems:
            # Find result for this pair
            result = next(
                (
                    r
                    for r in results
                    if r.trained_on == trained_problem and r.tested_on == tested_problem
                ),
                None,
            )

            if result:
                # Format: fitness (penalty%)
                fitness_str = f"{int(result.fitness):,}" if result.fitness != float("inf") else "∞"
                penalty_str = (
                    f"{result.generalization_penalty:+.0f}%"
                    if result.generalization_penalty != float("inf")
                    else "N/A"
                )
                cell = f"{fitness_str} ({penalty_str})"
            else:
                cell = "N/A"

            row.append(cell)

        lines.append("| " + " | ".join(row) + " |")

    # Add summary statistics
    lines.extend(["", "*Values: Fitness (Generalization Penalty %)*", "", "## Key Findings", ""])

    # Calculate average penalties per model
    avg_penalties = {}
    for trained_problem in problems:
        penalties = [
            r.generalization_penalty
            for r in results
            if r.trained_on == trained_problem and r.generalization_penalty != float("inf")
        ]
        if penalties:
            avg_penalties[trained_problem] = sum(penalties) / len(penalties)

    # Find best generalizer (lowest average penalty)
    if avg_penalties:
        best_generalizer = min(avg_penalties, key=avg_penalties.get)
        lines.append(
            f"- **Best Generalizer**: {best_generalizer} (avg penalty: {avg_penalties[best_generalizer]:.1f}%)"
        )

        worst_generalizer = max(avg_penalties, key=avg_penalties.get)
        lines.append(
            f"- **Most Specialized**: {worst_generalizer} (avg penalty: {avg_penalties[worst_generalizer]:.1f}%)"
        )

    # Write file
    output_path.write_text("\n".join(lines) + "\n")


def export_results_json(results: list[CrossValidationResult], output_path: Path) -> None:
    """Export detailed cross-validation results as JSON.

    Args:
        results: List of CrossValidationResult
        output_path: Path to save JSON file
    """
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert results to dict, handling inf values
    matrix = []
    for result in results:
        result_dict = asdict(result)

        # Convert inf to null for JSON serialization
        for key, value in result_dict.items():
            if isinstance(value, float) and (value == float("inf") or value == float("-inf")):
                result_dict[key] = None

        matrix.append(result_dict)

    # Calculate summary statistics
    problems = sorted({r.trained_on for r in results})
    avg_penalty_by_model = {}

    for trained_problem in problems:
        penalties = [
            r.generalization_penalty
            for r in results
            if r.trained_on == trained_problem
            and r.generalization_penalty != float("inf")
            and r.generalization_penalty != float("-inf")
        ]
        if penalties:
            avg_penalty_by_model[trained_problem] = sum(penalties) / len(penalties)

    summary = {"avg_penalty_by_model": avg_penalty_by_model}

    if avg_penalty_by_model:
        summary["best_generalizer"] = min(avg_penalty_by_model, key=avg_penalty_by_model.get)
        summary["most_specialized"] = max(avg_penalty_by_model, key=avg_penalty_by_model.get)

    # Write JSON
    output = {"matrix": matrix, "summary": summary}

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


def find_latest_runs() -> dict[str, Path]:
    """Find latest evolution run for each test problem.

    Returns:
        Dictionary mapping test_problem to run directory path

    Raises:
        FileNotFoundError: If no runs found for any required problem
    """
    results_dir = Path("results")
    required_problems = ["two_body", "figure_eight", "plummer"]
    latest_runs = {}

    # Find all run directories
    run_dirs = sorted([d for d in results_dir.glob("run_*") if d.is_dir()], reverse=True)

    for run_dir in run_dirs:
        history_path = run_dir / "evolution_history.json"
        if not history_path.exists():
            continue

        try:
            history = load_evolution_history(history_path)
            if "metadata" in history:
                test_problem = history["metadata"].get("test_problem")
                if test_problem in required_problems and test_problem not in latest_runs:
                    latest_runs[test_problem] = run_dir
        except (json.JSONDecodeError, KeyError):
            continue

        # Early exit if we found all problems
        if len(latest_runs) == len(required_problems):
            break

    # Check if we found all required problems
    missing = set(required_problems) - set(latest_runs.keys())
    if missing:
        raise FileNotFoundError(
            f"No evolution runs found for test problems: {', '.join(sorted(missing))}"
        )

    return latest_runs


def main() -> None:
    """Main entry point for cross-validation analysis."""
    print("🔬 Cross-Problem Generalization Analysis")
    print("=" * 60)

    # Find latest runs for each problem
    print("\n📂 Finding latest evolution runs...")
    try:
        run_dirs = find_latest_runs()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    for problem, run_dir in sorted(run_dirs.items()):
        print(f"  ✓ {problem}: {run_dir.name}")

    # Create cross-validation matrix
    print("\n🧪 Running cross-validation...")
    results = create_cross_validation_matrix(run_dirs)
    print(f"  ✓ Completed {len(results)} evaluations ({len(run_dirs)}×{len(run_dirs)} matrix)")

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results") / "analysis" / f"cross_validation_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export results
    print(f"\n💾 Saving results to: {output_dir}")

    md_path = output_dir / "cross_validation_matrix.md"
    export_results_markdown(results, md_path)
    print(f"  ✓ Markdown table: {md_path}")

    json_path = output_dir / "cross_validation_results.json"
    export_results_json(results, json_path)
    print(f"  ✓ JSON data: {json_path}")

    print("\n✅ Cross-validation analysis complete!")
    print(f"\nView results: {md_path}")


if __name__ == "__main__":
    main()
