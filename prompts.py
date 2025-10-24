"""Prompt templates for Gemini code generation.

This module contains all prompts used to generate and mutate surrogate models.
Prompts are carefully crafted to maximize code quality and physics accuracy.
"""

from typing import Optional


# System instruction used for all prompts
SYSTEM_INSTRUCTION = """You are an expert in numerical methods and physics simulation.
Generate Python code for a surrogate model that approximates N-body gravitational dynamics.

CRITICAL REQUIREMENTS:
1. Define EXACTLY this function signature:
   def predict(particle, attractor):
       # Your code here
       return [new_x, new_y, new_vx, new_vy]

2. Input format:
   - particle: [x, y, vx, vy] (position x, y and velocity vx, vy)
   - attractor: [ax, ay] (central gravity source position)

3. Output: [new_x, new_y, new_vx, new_vy] (next state after one timestep)

4. Physics: Approximate gravity (force ∝ 1/r² or similar)

5. Constraints:
   - The 'math' module is ALREADY AVAILABLE (do NOT import it)
   - Use math.sqrt(), math.sin(), etc. directly
   - Use basic Python functions: abs, min, max, sum, len
   - FORBIDDEN: import statements, loops over large arrays, recursion, file I/O
   - Fast and simple (will be called 50 times per evaluation)

6. Output ONLY the Python code with the predict function, NO import statements, NO markdown, NO explanations."""


def get_initial_prompt(seed: int) -> str:
    """Generate diverse initial surrogate model prompts.

    Uses different numerical integration approaches to create diversity
    in the initial population.

    Args:
        seed: Seed value for selecting approach (0-indexed)

    Returns:
        Complete prompt for initial model generation
    """
    approaches = [
        "Use classic Euler integration with adaptive timestep based on distance from attractor",
        "Use semi-implicit Euler (symplectic) integration for better energy conservation",
        "Use polynomial approximation for distance calculations to improve speed",
        "Use damping proportional to velocity to stabilize the system",
        "Use a Verlet-inspired approach with position and velocity updates",
        "Use softened gravity (add epsilon to distance) to prevent singularities at close range",
    ]

    approach = approaches[seed % len(approaches)]

    return f"""{SYSTEM_INSTRUCTION}

TASK: Create an initial surrogate model for N-body gravity simulation.

Approach to try: {approach}

Remember:
- Gravity force typically: F = G / r²
- Acceleration = Force × direction / distance
- Update velocity: v_new = v + a × dt
- Update position: x_new = x + v × dt
- Keep code simple, fast, and numerically stable
- Return [new_x, new_y, new_vx, new_vy]

Generate the complete Python code now."""


def get_mutation_prompt(
    parent_code: str,
    fitness: float,
    accuracy: float,
    speed: float,
    generation: int,
    mutation_type: str = "explore"
) -> str:
    """Generate mutation prompt based on parent performance.

    Args:
        parent_code: Python code of parent model
        fitness: Parent's fitness score (accuracy/speed)
        accuracy: Parent's accuracy (0-1)
        speed: Parent's execution time in seconds
        generation: Current generation number
        mutation_type: "explore" for large changes, "exploit" for refinement

    Returns:
        Complete prompt for mutating parent model
    """
    # Determine strategy based on mutation type
    if mutation_type == "explore":
        strategy = """Try a DIFFERENT approach from the parent:
- Change the integration method (e.g., from Euler to semi-implicit, or vice versa)
- Modify how you calculate forces (e.g., different softening, different power law)
- Add or remove stabilization terms (damping, velocity correction)
- Experiment with different timestep scaling or adaptive timesteps
- Try a completely novel numerical trick

Be creative and don't be afraid to make significant changes!"""
    else:  # exploit
        strategy = """REFINE the existing approach:
- Adjust numerical constants for better accuracy (e.g., timestep, epsilon, damping)
- Fine-tune force calculations to match ground truth more closely
- Optimize calculations for speed (simplify expressions, reduce operations)
- Fix any numerical instabilities (NaN, overflow)
- Improve energy conservation or accuracy

Make targeted improvements while keeping the core approach."""

    # Analyze performance
    perf_analysis = []
    if accuracy > 0.8:
        perf_analysis.append("✓ Good accuracy - try to maintain while improving speed")
    else:
        perf_analysis.append("✗ Low accuracy - focus on matching true physics more closely")

    if speed < 0.01:
        perf_analysis.append("✓ Fast execution - try to maintain speed")
    else:
        perf_analysis.append("✗ Slow execution - simplify calculations, reduce operations")

    if fitness < 50:
        perf_analysis.append("⚠ Low fitness - significant improvements needed")
    elif fitness > 100:
        perf_analysis.append("✓ High fitness - make incremental refinements")

    performance_context = f"""
Parent model performance (Generation {generation}):
- Fitness: {fitness:.4f} (higher is better = accuracy/speed)
- Accuracy: {accuracy:.4f} (how close to true physics, 0-1 scale)
- Speed: {speed:.5f} seconds (lower is better)

Performance analysis:
{chr(10).join(f'  {item}' for item in perf_analysis)}
"""

    return f"""{SYSTEM_INSTRUCTION}

TASK: Improve this surrogate model (Generation {generation + 1})

{performance_context}

PARENT CODE:
```python
{parent_code}
```

MUTATION STRATEGY:
{strategy}

Generate improved code that maintains the predict(particle, attractor) signature and returns [new_x, new_y, new_vx, new_vy]."""


def get_crossover_prompt(
    parent1_code: str,
    parent1_fitness: float,
    parent2_code: str,
    parent2_fitness: float
) -> str:
    """Generate crossover prompt combining two high-performing models.

    Note: This is optional and not currently used in the main evolution loop.

    Args:
        parent1_code: Python code of first parent
        parent1_fitness: First parent's fitness
        parent2_code: Python code of second parent
        parent2_fitness: Second parent's fitness

    Returns:
        Complete prompt for creating hybrid model
    """
    return f"""{SYSTEM_INSTRUCTION}

TASK: Combine insights from two high-performing surrogate models.

PARENT 1 (Fitness: {parent1_fitness:.4f}):
```python
{parent1_code}
```

PARENT 2 (Fitness: {parent2_fitness:.4f}):
```python
{parent2_code}
```

Create a HYBRID model that:
- Combines the best ideas from both parents
- Uses the more accurate force calculation approach
- Uses the faster computational method
- May introduce novel elements that neither parent has

Generate the complete hybrid code with predict(particle, attractor) signature."""
