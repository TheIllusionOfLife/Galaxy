"""Prompt templates for Gemini code generation.

This module contains all prompts used to generate and mutate surrogate models.
Prompts are carefully crafted to maximize code quality and physics accuracy.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prototype import SurrogateGenome

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

6. CODE COMPLETENESS (CRITICAL - READ CAREFULLY):
   - ALWAYS generate COMPLETE, syntactically VALID Python code with NO syntax errors
   - BEFORE finishing, count ALL opening and closing brackets to ensure they match: ( ) [ ] { }
   - VERIFY the function has a proper return statement at the very end
   - DO NOT truncate code, leave brackets unclosed, or create incomplete expressions
   - If generating long/complex code (>2000 characters), PAUSE and verify completeness
   - FORBIDDEN: Incomplete code, unmatched brackets, missing return statements
   - If you're unsure, generate SIMPLER code that you KNOW is syntactically complete

7. Output format (CRITICAL):
   - Output ONLY raw Python code - NO other text
   - NO import statements (math module is already available)
   - NO markdown formatting (no ```python, no ```, no backticks)
   - NO explanatory text, NO comments outside the function
   - NO incomplete or placeholder code
   - Start directly with "def predict" - nothing before it
   - End with the return statement - nothing after it
   - The output must be directly executable Python code with ZERO syntax errors"""


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
    mutation_type: str = "explore",
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
{chr(10).join(f"  {item}" for item in perf_analysis)}
"""

    # Extra reminder for code completeness (especially important in exploit phase)
    completion_reminder = """
CRITICAL - Final Verification Before Submitting:
You MUST verify these items BEFORE finishing (this prevents syntax errors):

✓ Count brackets: ALL opening ( [ { MUST have matching closing ) ] }
✓ Return statement: The function MUST end with "return [new_x, new_y, new_vx, new_vy]"
✓ All blocks closed: Every if/for/while MUST have proper closing and indentation
✓ No truncation: The code MUST be complete - no "..." or incomplete lines
✓ Zero syntax errors: The code must parse as valid Python

SPECIAL WARNING for long/complex code:
- If your code is >2000 characters, PAUSE and verify completeness twice
- When in doubt, generate SIMPLER code that you KNOW is complete
- One syntax error = complete failure; completeness is MORE important than cleverness
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

{completion_reminder}

Generate improved code that maintains the predict(particle, attractor) signature and returns [new_x, new_y, new_vx, new_vy]."""


def get_crossover_prompt(
    parent1: "SurrogateGenome", parent2: "SurrogateGenome", generation: int
) -> str:
    """Generate crossover prompt combining two high-performing models.

    Args:
        parent1: First parent SurrogateGenome (must have raw_code)
        parent2: Second parent SurrogateGenome (must have raw_code)
        generation: Current generation number

    Returns:
        Complete prompt for creating hybrid model

    Raises:
        ValueError: If either parent lacks raw_code (parametric genome)
    """
    # Validate both parents have LLM-generated code
    if parent1.raw_code is None or parent2.raw_code is None:
        raise ValueError("Crossover requires LLM-generated parents with raw_code")
    # Use completion reminder for later generations when code gets more complex
    completion_reminder = (
        """

CRITICAL - Verify code completeness:
- Count ALL opening ( [ {{ MUST have matching closing ) ] }}
- Verify the predict function is complete with all calculations
- If your code exceeds 2000 characters, STOP and simplify - avoid truncation
- Completeness > Cleverness (a simple complete model beats a truncated complex one)"""
        if generation > 1
        else ""
    )

    return f"""{SYSTEM_INSTRUCTION}

OBJECTIVE: Create a HYBRID surrogate model by combining the strengths of TWO parent models.

PARENT 1 (Fitness: {parent1.fitness or 0.0:.2f}, Accuracy: {parent1.accuracy or 0.0:.4f}, Speed: {parent1.speed or 0.01:.6f}s):
```python
{parent1.raw_code}
```

PARENT 2 (Fitness: {parent2.fitness or 0.0:.2f}, Accuracy: {parent2.accuracy or 0.0:.4f}, Speed: {parent2.speed or 0.01:.6f}s):
```python
{parent2.raw_code}
```

TASK:
1. Analyze what makes each parent successful
2. Identify complementary strengths (e.g., Parent 1 has better accuracy, Parent 2 is faster)
3. Design a NEW approach that combines these strengths
4. You may introduce novel elements beyond just merging (creative synthesis)

GENERATION: {generation} (Explore phase: try bold combinations)

{completion_reminder}

Generate the hybrid predict function:"""
