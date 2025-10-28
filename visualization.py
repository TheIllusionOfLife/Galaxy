"""
Evolution visualization and data export module.

This module provides functions to visualize evolution results including:
- Fitness progression over generations
- Accuracy vs speed trade-offs
- Cost progression over time
- JSON export of evolution history
"""

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def _create_empty_plot(output_path: str, title: str, xlabel: str, ylabel: str) -> None:
    """
    Create and save an empty plot with a message for when no data is available.

    Args:
        output_path: Path where plot image will be saved
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(
        0.5,
        0.5,
        "No evolution data available",
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_fitness_progression(history: list[dict[str, Any]], output_path: str) -> None:
    """
    Plot fitness progression over generations.

    Shows best, average, and worst fitness for each generation as line plots.

    Args:
        history: List of generation data dicts with fitness statistics
        output_path: Path where plot image will be saved
    """
    if not history:
        _create_empty_plot(
            output_path,
            "Fitness Progression Over Generations",
            "Generation",
            "Fitness",
        )
        return

    generations = [entry["generation"] for entry in history]
    best_fitness = [entry["best_fitness"] for entry in history]
    avg_fitness = [entry["avg_fitness"] for entry in history]
    worst_fitness = [entry["worst_fitness"] for entry in history]

    # Filter out inf/nan values
    def clean_data(gens, values):
        cleaned_gens = []
        cleaned_values = []
        for g, v in zip(gens, values, strict=True):
            if math.isfinite(v):
                cleaned_gens.append(g)
                cleaned_values.append(v)
        return cleaned_gens, cleaned_values

    best_gens, best_vals = clean_data(generations, best_fitness)
    avg_gens, avg_vals = clean_data(generations, avg_fitness)
    worst_gens, worst_vals = clean_data(generations, worst_fitness)

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot lines
    if best_vals:
        ax.plot(best_gens, best_vals, "g-o", label="Best", linewidth=2, markersize=8)
    if avg_vals:
        ax.plot(avg_gens, avg_vals, "b-s", label="Average", linewidth=2, markersize=6)
    if worst_vals:
        ax.plot(worst_gens, worst_vals, "r-^", label="Worst", linewidth=2, markersize=6)

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Fitness (Accuracy / Speed)", fontsize=12)
    ax.set_title("Fitness Progression Over Generations", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_accuracy_vs_speed(history: list[dict[str, Any]], output_path: str) -> None:
    """
    Plot accuracy vs speed scatter plot with fitness as color.

    Shows the trade-off between accuracy and speed for all models across all generations.

    Args:
        history: List of generation data dicts with population details
        output_path: Path where plot image will be saved
    """
    if not history:
        _create_empty_plot(
            output_path,
            "Accuracy vs Speed Trade-off",
            "Speed (seconds)",
            "Accuracy",
        )
        return

    # Collect all models from all generations
    speeds = []
    accuracies = []
    fitnesses = []
    generations = []

    for entry in history:
        for model in entry["population"]:
            speed = model["speed"]
            accuracy = model["accuracy"]
            fitness = model["fitness"]

            # Filter out inf/nan values
            if math.isfinite(speed) and math.isfinite(accuracy) and math.isfinite(fitness):
                if speed > 0 and 0 <= accuracy <= 1:
                    speeds.append(speed)
                    accuracies.append(accuracy)
                    fitnesses.append(fitness)
                    generations.append(entry["generation"])

    if not speeds:
        # No valid data - use the same helper but with a different message
        _create_empty_plot(
            output_path,
            "Accuracy vs Speed Trade-off",
            "Speed (seconds)",
            "Accuracy",
        )
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    # Create scatter plot with fitness as color
    scatter = ax.scatter(
        speeds,
        accuracies,
        c=fitnesses,
        cmap="viridis",
        s=100,
        alpha=0.6,
        edgecolors="black",
        linewidth=0.5,
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Fitness", fontsize=12)

    ax.set_xlabel("Speed (seconds, lower is better)", fontsize=12)
    ax.set_ylabel("Accuracy (higher is better)", fontsize=12)
    ax.set_title("Accuracy vs Speed Trade-off", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Use log scale for speed if range is large
    if speeds and min(speeds) > 0 and max(speeds) / min(speeds) > 10:
        ax.set_xscale("log")

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_cost_progression(cost_tracker: Any, output_path: str) -> None:
    """
    Plot cumulative cost progression over API calls.

    Args:
        cost_tracker: CostTracker object with API call history
        output_path: Path where plot image will be saved
    """
    if not cost_tracker or not cost_tracker.calls:
        _create_empty_plot(
            output_path,
            "Cost Progression",
            "API Call Number",
            "Cumulative Cost (USD)",
        )
        return

    # Calculate cumulative costs
    cumulative_cost = []
    total = 0.0
    for call in cost_tracker.calls:
        total += call.get("cost_usd", 0.0)
        cumulative_cost.append(total)

    call_numbers = list(range(1, len(cumulative_cost) + 1))

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot cumulative cost
    ax.plot(call_numbers, cumulative_cost, "b-o", linewidth=2, markersize=6)

    # Add horizontal line for final cost
    if cumulative_cost:
        final_cost = cumulative_cost[-1]
        ax.axhline(y=final_cost, color="r", linestyle="--", alpha=0.5, linewidth=1)
        ax.text(
            len(call_numbers) * 0.7,
            final_cost * 1.05,
            f"Final: ${final_cost:.4f}",
            fontsize=10,
            color="red",
        )

    ax.set_xlabel("API Call Number", fontsize=12)
    ax.set_ylabel("Cumulative Cost (USD)", fontsize=12)
    ax.set_title("Cost Progression Over API Calls", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Format y-axis to show currency
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"${x:.4f}"))

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_all_plots(history: list[dict[str, Any]], cost_tracker: Any, output_dir: str) -> None:
    """
    Generate all visualization plots.

    Creates fitness progression, accuracy vs speed, and cost progression plots
    in the specified output directory.

    Args:
        history: Evolution history data
        cost_tracker: CostTracker object (optional, can be None)
        output_dir: Directory where plots will be saved
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate fitness progression plot
    plot_fitness_progression(history, str(output_path / "fitness_progression.png"))

    # Generate accuracy vs speed plot
    plot_accuracy_vs_speed(history, str(output_path / "accuracy_vs_speed.png"))

    # Generate cost progression plot if tracker available
    if cost_tracker and hasattr(cost_tracker, "calls") and cost_tracker.calls:
        plot_cost_progression(cost_tracker, str(output_path / "cost_progression.png"))


def _sanitize_for_json(obj: Any) -> Any:
    """
    Recursively sanitize data structure for JSON export.

    Replaces non-finite float values (NaN, Inf, -Inf) with None.

    Args:
        obj: Object to sanitize (dict, list, or primitive)

    Returns:
        Sanitized object safe for JSON serialization
    """
    if isinstance(obj, dict):
        return {key: _sanitize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(item) for item in obj]
    elif isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    else:
        return obj


def export_history_json(history: list[dict[str, Any]], output_path: str) -> None:
    """
    Export evolution history to JSON file.

    Saves the complete evolution history along with summary statistics.
    Non-finite float values (NaN, Inf) are replaced with null for JSON compatibility.

    Args:
        history: Evolution history data
        output_path: Path where JSON file will be saved
    """
    # Calculate summary statistics
    if history:
        all_fitness = []
        for entry in history:
            for model in entry["population"]:
                if math.isfinite(model["fitness"]):
                    all_fitness.append(model["fitness"])

        summary = {
            "total_generations": len(history),
            "best_overall_fitness": max(all_fitness) if all_fitness else 0.0,
            "avg_overall_fitness": sum(all_fitness) / len(all_fitness) if all_fitness else 0.0,
            "total_models_evaluated": sum(len(entry["population"]) for entry in history),
        }
    else:
        summary = {
            "total_generations": 0,
            "best_overall_fitness": 0.0,
            "avg_overall_fitness": 0.0,
            "total_models_evaluated": 0,
        }

    # Sanitize history to remove non-finite values
    sanitized_history = _sanitize_for_json(history)
    data = {"history": sanitized_history, "summary": summary}

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Write JSON with allow_nan=False to catch any remaining non-finite values
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, allow_nan=False)
