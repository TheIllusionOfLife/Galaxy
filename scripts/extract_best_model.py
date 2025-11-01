"""Extract best model from evolution history.

This script finds the model with the highest fitness across all generations
and saves its code to a file for further analysis.

Usage:
    python scripts/extract_best_model.py results/run_YYYYMMDD_HHMMSS/

Output:
    - best_model.py: Code for the best model
    - best_model_info.json: Metadata about the best model
"""

import argparse
import json
import sys
from pathlib import Path


def load_evolution_history(history_path: Path) -> dict:
    """Load evolution history from JSON file.

    Args:
        history_path: Path to evolution_history.json

    Returns:
        Dictionary containing evolution history

    Raises:
        FileNotFoundError: If history file doesn't exist
        json.JSONDecodeError: If JSON is malformed
    """
    if not history_path.exists():
        raise FileNotFoundError(f"History file not found: {history_path}")

    with open(history_path) as f:
        return json.load(f)


def find_best_model(history: dict) -> dict:
    """Find model with highest fitness across all generations.

    Args:
        history: Evolution history dictionary

    Returns:
        Dictionary containing best model data with 'generation' field added

    Raises:
        ValueError: If no valid models found (all infinite fitness)
    """
    best_model = None
    best_fitness = float("-inf")

    # Handle both "history" and "generations" keys for compatibility
    generations = history.get("history", history.get("generations", []))

    for gen_data in generations:
        gen_num = gen_data["generation"]
        for model in gen_data["population"]:
            fitness = model["fitness"]

            # Skip invalid fitness values (inf means validation failed)
            if fitness == float("inf") or fitness == float("-inf"):
                continue

            if fitness > best_fitness:
                best_fitness = fitness
                best_model = model.copy()
                best_model["generation"] = gen_num

    if best_model is None:
        raise ValueError("No valid models found in evolution history")

    return best_model


def extract_model_code(model: dict) -> str | None:
    """Extract code from model dictionary.

    Args:
        model: Model dictionary containing 'raw_code' or 'theta' fields

    Returns:
        Model code string, or None if parametric model or missing
    """
    # LLM-generated models have raw_code
    if "raw_code" in model:
        return model["raw_code"]

    # Parametric models only have theta parameters
    return None


def save_model_to_file(code: str, output_path: Path) -> None:
    """Save model code to Python file.

    Args:
        code: Python code string
        output_path: Path to save file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(code)


def format_model_summary(model: dict) -> str:
    """Format model information for display.

    Args:
        model: Model dictionary

    Returns:
        Formatted summary string
    """
    lines = [
        "=" * 70,
        "Best Model Summary",
        "=" * 70,
        f"Civilization ID: {model['civ_id']}",
        f"Generation: {model['generation']}",
        f"Fitness: {model['fitness']:.2f}",
    ]

    # Optional fields
    if "accuracy" in model:
        lines.append(f"Accuracy: {model['accuracy']:.4f}")
    if "speed" in model:
        lines.append(f"Speed: {model['speed']:.6f}s")
    if "token_count" in model:
        lines.append(f"Token Count: {model['token_count']}")
    if "description" in model:
        lines.append(f"Description: {model['description']}")

    lines.append("=" * 70)

    return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Extract best model from evolution history")
    parser.add_argument(
        "run_dir", type=Path, help="Path to evolution run directory (results/run_XXX/)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (default: <run_dir>/best_model.py)",
    )

    args = parser.parse_args()

    # Load evolution history
    history_path = args.run_dir / "evolution_history.json"
    print(f"Loading evolution history from: {history_path}")

    try:
        history = load_evolution_history(history_path)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return 1
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON: {e}")
        return 1

    # Find best model
    print("Finding best model across all generations...")

    try:
        best_model = find_best_model(history)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1

    # Display summary
    print()
    print(format_model_summary(best_model))
    print()

    # Extract code
    code = extract_model_code(best_model)

    if code is None:
        print("⚠️  This is a parametric model (no code to extract)")
        print("Theta parameters:", best_model.get("theta", []))
        return 0

    # Save code
    output_path = args.output or (args.run_dir / "best_model.py")
    save_model_to_file(code, output_path)
    print(f"✅ Best model code saved to: {output_path}")

    # Save metadata
    metadata_path = output_path.parent / "best_model_info.json"
    metadata = {k: v for k, v in best_model.items() if k != "raw_code"}  # Exclude code
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Model metadata saved to: {metadata_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
