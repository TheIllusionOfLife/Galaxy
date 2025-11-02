#!/usr/bin/env python3
"""Physics validation for evolved surrogate models.

This script validates evolved models for physical plausibility by computing:
- Energy drift (energy conservation violation)
- Trajectory RMSE (position accuracy vs ground truth)
- Angular momentum conservation (rotational conservation)

Usage:
    # Validate single run
    python scripts/validate_evolved_model.py --run-dir results/run_YYYYMMDD_HHMMSS

    # Validate all runs
    python scripts/validate_evolved_model.py --all

Output:
    - results/analysis/physics_validation_YYYYMMDD_HHMMSS/validation_results.json
    - results/analysis/physics_validation_YYYYMMDD_HHMMSS/validation_report.md
"""

import argparse
import json
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from prototype import (
    CosmologyCrucible,
    compile_external_surrogate,
    get_initial_particles,
    make_parametric_surrogate,
)
from scripts.extract_best_model import find_best_model, load_evolution_history
from validation_metrics import (
    compute_angular_momentum_conservation,
    compute_energy_drift,
    compute_trajectory_rmse,
)

# Constants
VALIDATION_TIMESTEPS = 100  # Simulation length for validation
ENERGY_DRIFT_THRESHOLD = 0.01  # 1% threshold for "good" conservation

# Validation particles for model compilation (2-particle system)
_VALIDATION_PARTICLES = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
]


def load_model_for_validation(run_dir: Path) -> dict[str, Any]:
    """Load best model from evolution run with validation metadata.

    Args:
        run_dir: Path to evolution run directory

    Returns:
        Dictionary with model data plus metadata (test_problem, num_particles, etc.)

    Raises:
        FileNotFoundError: If evolution_history.json doesn't exist
        KeyError: If metadata is missing
        ValueError: If model has neither raw_code nor theta
    """
    history_path = run_dir / "evolution_history.json"
    history = load_evolution_history(history_path)

    # Extract metadata
    if "metadata" not in history:
        raise KeyError("metadata field missing from evolution history")

    metadata = history["metadata"]
    best_model = find_best_model(history)

    # Add metadata to model
    best_model["test_problem"] = metadata["test_problem"]
    best_model["num_particles"] = metadata["num_particles"]
    best_model["run_dir"] = str(run_dir)

    # Verify model has code
    if "raw_code" not in best_model and "theta" not in best_model:
        raise ValueError(
            "Model must have either 'raw_code' or 'theta' field. "
            "This evolution run does not save model code. "
            "Re-run evolution with code saving enabled (prototype.py lines 864-874)."
        )

    return best_model


def _build_model_callable(
    model: dict,
) -> Callable[[list[float], list[list[float]]], list[float]]:
    """Build callable function from model dictionary.

    Args:
        model: Model dictionary with either 'raw_code' or 'theta' fields

    Returns:
        Callable with signature predict(particle, all_particles) -> predicted_particle

    Raises:
        ValueError: If model has neither raw_code nor theta
    """
    # LLM-generated models have raw_code
    if "raw_code" in model and model["raw_code"]:
        return compile_external_surrogate(model["raw_code"], _VALIDATION_PARTICLES)

    # Parametric models have theta parameters
    if "theta" in model:
        return make_parametric_surrogate(model["theta"], _VALIDATION_PARTICLES)

    raise ValueError("Model must have either 'raw_code' or 'theta' field")


def compute_physics_metrics(
    model_callable: Callable,
    test_problem: str,
    num_particles: int,
    timesteps: int = VALIDATION_TIMESTEPS,
) -> dict[str, float]:
    """Run model and compute energy drift, trajectory RMSE, angular momentum.

    Args:
        model_callable: Surrogate model function
        test_problem: Test problem name (two_body, figure_eight, plummer)
        num_particles: Number of particles
        timesteps: Number of simulation timesteps

    Returns:
        Dictionary with keys: energy_drift, trajectory_rmse, angular_momentum_drift
    """
    # Get initial conditions
    particles = get_initial_particles(test_problem, num_particles)

    # Create crucible for ground truth simulation
    crucible_truth = CosmologyCrucible.with_particles([p[:] for p in particles])

    # Run ground truth simulation
    truth_trajectory = []
    for _ in range(timesteps):
        truth_trajectory.append([p[:] for p in crucible_truth.particles])
        crucible_truth.particles = crucible_truth.brute_force_step(crucible_truth.particles)

    # Capture final state after all timesteps complete
    truth_final = [p[:] for p in crucible_truth.particles]

    # Run surrogate model simulation
    # Start from same initial conditions
    model_particles = [p[:] for p in particles]
    model_trajectory = []

    for _ in range(timesteps):
        model_trajectory.append([p[:] for p in model_particles])
        # Use surrogate model for next step prediction
        model_particles = [
            model_callable(particle, model_particles) for particle in model_particles
        ]

    # Capture final state after all timesteps complete
    model_final = [p[:] for p in model_particles]

    # Compute energy drift on model trajectory to validate physics preservation
    # Use initial state (trajectory[0]) and actual final state (model_final)
    energy_drift = compute_energy_drift(model_trajectory[0], model_final)

    # Compute trajectory RMSE (model accuracy vs ground truth)
    # Compare actual final states
    trajectory_rmse = compute_trajectory_rmse(model_final, truth_final)

    # Compute angular momentum conservation on model trajectory
    # Use initial state and actual final state
    angular_momentum_drift = compute_angular_momentum_conservation(model_trajectory[0], model_final)

    return {
        "energy_drift": energy_drift,
        "trajectory_rmse": trajectory_rmse,
        "angular_momentum_drift": angular_momentum_drift,
    }


def validate_single_run(run_dir: Path) -> dict[str, Any]:
    """Validate one evolution run with physics metrics.

    Args:
        run_dir: Path to evolution run directory

    Returns:
        Validation result dictionary with all metrics
    """
    # Load model
    model = load_model_for_validation(run_dir)

    # Build callable
    model_callable = _build_model_callable(model)

    # Compute physics metrics
    metrics = compute_physics_metrics(
        model_callable=model_callable,
        test_problem=model["test_problem"],
        num_particles=model["num_particles"],
        timesteps=VALIDATION_TIMESTEPS,
    )

    # Combine model info and metrics
    result = {
        "run_dir": model["run_dir"],
        "test_problem": model["test_problem"],
        "num_particles": model["num_particles"],
        "fitness": model["fitness"],
        "accuracy": model["accuracy"],
        "speed": model["speed"],
        **metrics,
    }

    return result


def validate_all_runs(results_dir: Path = Path("results")) -> list[dict]:
    """Validate all evolution runs with valid model code.

    Args:
        results_dir: Directory containing run_* subdirectories

    Returns:
        List of validation results (only successful validations)
    """
    results = []
    skipped = []

    # Find all run directories
    run_dirs = sorted([d for d in results_dir.glob("run_*") if d.is_dir()])

    print(f"Found {len(run_dirs)} evolution runs")

    for run_dir in run_dirs:
        try:
            print(f"Validating {run_dir.name}...", end=" ")
            result = validate_single_run(run_dir)
            results.append(result)
            print(f"✓ (fitness={result['fitness']:.1f})")
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"✗ Skipped ({type(e).__name__})")
            skipped.append({"run_dir": str(run_dir), "error": str(e)})

    print(f"\nValidated: {len(results)} runs")
    print(f"Skipped: {len(skipped)} runs")

    return results


def export_validation_json(results: list[dict], output_path: Path) -> None:
    """Export validation results as JSON.

    Args:
        results: List of validation result dictionaries
        output_path: Path to save JSON file
    """
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Add metadata
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_runs": len(results),
            "validation_timesteps": VALIDATION_TIMESTEPS,
        },
        "results": results,
    }

    # Write JSON
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


def export_validation_markdown(results: list[dict], output_path: Path) -> None:
    """Export validation results as markdown table.

    Args:
        results: List of validation result dictionaries
        output_path: Path to save markdown file
    """
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Group results by test problem
    by_problem: dict[str, list[dict]] = {}
    for result in results:
        problem = result["test_problem"]
        if problem not in by_problem:
            by_problem[problem] = []
        by_problem[problem].append(result)

    # Build markdown
    lines = ["# Physics Validation of Evolved Models", ""]

    # Summary section
    lines.extend(["## Summary", ""])
    lines.append(f"- **Total runs validated**: {len(results)}")

    # Count runs with good energy conservation
    good_energy = sum(1 for r in results if r["energy_drift"] < ENERGY_DRIFT_THRESHOLD)
    lines.append(
        f"- **Runs with good energy conservation (<{ENERGY_DRIFT_THRESHOLD * 100:.0f}%)**:   {good_energy}"
    )

    # Average metrics
    if results:
        avg_rmse = sum(r["trajectory_rmse"] for r in results) / len(results)
        lines.append(f"- **Average trajectory RMSE**: {avg_rmse:.6f}")

        # Find best performer (lowest energy drift)
        best = min(results, key=lambda r: r["energy_drift"])
        lines.append(
            f"- **Best energy conservation**: {Path(best['run_dir']).name} "
            f"(drift: {best['energy_drift'] * 100:.4f}%)"
        )

    lines.append("")

    # Results by test problem
    lines.extend(["## Results by Test Problem", ""])

    for problem in sorted(by_problem.keys()):
        problem_results = by_problem[problem]

        lines.append(f"### {problem.replace('_', ' ').title()}")
        lines.append("")

        # Table header
        headers = [
            "Run",
            "Fitness",
            "Energy Drift (%)",
            "Trajectory RMSE",
            "Angular Mom. Drift",
        ]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")

        # Table rows
        for result in sorted(problem_results, key=lambda r: r["fitness"], reverse=True):
            run_name = Path(result["run_dir"]).name
            row = [
                run_name,
                f"{int(result['fitness']):,}",
                f"{result['energy_drift'] * 100:.4f}",
                f"{result['trajectory_rmse']:.6f}",
                f"{result['angular_momentum_drift']:.6f}",
            ]
            lines.append("| " + " | ".join(row) + " |")

        lines.append("")

    # Physics interpretation
    lines.extend(["## Physics Interpretation", ""])

    lines.append("### Energy Conservation")
    lines.append(
        f"Energy drift < {ENERGY_DRIFT_THRESHOLD * 100:.0f}% indicates good conservation. "
        "Higher drift suggests numerical integration errors or unstable model predictions."
    )
    lines.append("")

    lines.append("### Trajectory Accuracy")
    lines.append(
        "RMSE measures average position error vs ground truth. "
        "Lower values indicate more accurate surrogate model predictions."
    )
    lines.append("")

    lines.append("### Angular Momentum Conservation")
    lines.append(
        "Drift measures violation of rotational conservation. "
        "Good models should preserve angular momentum (drift near zero)."
    )
    lines.append("")

    # Write file
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    """Main entry point for physics validation."""
    parser = argparse.ArgumentParser(
        description="Validate evolved models for physical plausibility"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Single evolution run directory to validate",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Validate all evolution runs in results/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/analysis"),
        help="Output directory for validation results",
    )

    args = parser.parse_args()

    print("🔬 Physics Validation of Evolved Models")
    print("=" * 60)

    # Validate run(s)
    if args.run_dir:
        print(f"\n📂 Validating single run: {args.run_dir}")
        try:
            result = validate_single_run(args.run_dir)
            results = [result]
            print(f"  ✓ Validation complete (fitness={result['fitness']:.1f})")
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    elif args.all:
        print("\n📂 Validating all runs in results/")
        results = validate_all_runs()
        if not results:
            print("❌ No valid runs found to validate")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output / f"physics_validation_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export results
    print(f"\n💾 Saving results to: {output_dir}")

    json_path = output_dir / "validation_results.json"
    export_validation_json(results, json_path)
    print(f"  ✓ JSON data: {json_path}")

    md_path = output_dir / "validation_report.md"
    export_validation_markdown(results, md_path)
    print(f"  ✓ Markdown report: {md_path}")

    print("\n✅ Physics validation complete!")
    print(f"\nView results: {md_path}")


if __name__ == "__main__":
    main()
