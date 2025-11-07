"""Prompt templates for Gemini code generation.

This module contains all prompts used to generate and mutate surrogate models.
Prompts are carefully crafted to maximize code quality and physics accuracy.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prototype import SurrogateGenome

from config import settings

# System instruction used for all prompts
SYSTEM_INSTRUCTION = """You are an expert in numerical methods and physics simulation.
Generate Python code for a surrogate model that approximates 3D N-body gravitational dynamics.

CRITICAL REQUIREMENTS:
1. Define EXACTLY this function signature:
   def predict(particle, all_particles):
       # Your code here
       return [new_x, new_y, new_z, new_vx, new_vy, new_vz, mass]

2. Input format:
   - particle: [x, y, z, vx, vy, vz, mass] (3D position, velocity, and mass of THIS particle)
   - all_particles: list of ALL particles [[x,y,z,vx,vy,vz,mass], ...] (entire particle system)

3. Output: [new_x, new_y, new_z, new_vx, new_vy, new_vz, mass] (next state after timestep dt=0.1)

4. Physics: True 3D N-body gravity between ALL particles (not single attractor!)
   - Each particle exerts gravitational force on 'particle': F_ij = G * m_i * m_j / r²
   - Direction: unit vector from 'particle' to other particle
   - Acceleration on 'particle': a = Σ(F_ij / m_particle) for all other particles
   - Integration timestep: dt = 0.1
   - Gravitational constant: G = 1.0
   - Softening (optional): epsilon = 0.01 to prevent singularities

5. Physics Conservation (CRITICAL FOR SCIENTIFIC VALIDITY):
   - Your model MUST preserve energy conservation (drift <1% over 10 timesteps)
   - Your model MUST preserve angular momentum conservation (drift <1%)
   - These are MEASURED and heavily PENALIZED in fitness function (up to 90% penalty)
   - Consider using symplectic integrators (e.g., Verlet, leapfrog, semi-implicit Euler)
   - Energy: E = 0.5*m*v² - Σ(G*m_i*m_j/r_ij) should stay constant
   - Angular momentum: L = Σ(m * r × v) should stay constant
   - Bad conservation = severe fitness penalty even if trajectory looks accurate

6. Constraints:
   - The 'math' module is ALREADY AVAILABLE (do NOT import it)
   - Use math.sqrt(), math.sin(), etc. directly
   - Use basic Python functions: abs, min, max, sum, len, range, enumerate, zip
   - FORBIDDEN: import statements, recursion, file I/O
   - PERFORMANCE: Ground truth is O(N²). Your approximation MUST be faster (e.g., cutoff radius, K-NN, grid)
   - Fast and simple (will be called 50 times per evaluation with 50 particles)

7. CODE COMPLETENESS (CRITICAL - READ CAREFULLY):
   - ALWAYS generate COMPLETE, syntactically VALID Python code with NO syntax errors
   - BEFORE finishing, count ALL opening and closing brackets to ensure they match: ( ) [ ] { }
   - VERIFY the function has a proper return statement at the very end
   - DO NOT truncate code, leave brackets unclosed, or create incomplete expressions
   - If generating long/complex code (>2000 characters), PAUSE and verify completeness
   - FORBIDDEN: Incomplete code, unmatched brackets, missing return statements
   - If you're unsure, generate SIMPLER code that you KNOW is syntactically complete

8. Output format (CRITICAL):
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
        "Use CUTOFF RADIUS approximation: only compute forces from particles within distance R_cutoff=20.0 (ignore distant particles)",
        "Use K-NEAREST NEIGHBORS: find K=5 closest particles, only compute forces from those (O(N) with simple linear search)",
        "Use GRID-BASED APPROXIMATION: divide space into cells, only compute forces from particles in nearby cells",
        "Use SOFTENED GRAVITY with large epsilon=1.0 to approximate long-range forces as weaker (reduces sensitivity to distant particles)",
        "Use HIERARCHICAL APPROXIMATION: compute center-of-mass for distant particle clusters, treat as single particle",
        "Use VELOCITY-WEIGHTED SELECTION: prioritize forces from fast-moving particles, ignore slow/static ones",
        "Use SEMI-IMPLICIT EULER (symplectic): Update velocity using NEW position's forces to preserve energy better than standard Euler",
        "Use LEAPFROG INTEGRATION (symplectic): Interleave position/velocity updates (kick-drift-kick) to conserve energy and angular momentum",
    ]

    approach = approaches[seed % len(approaches)]

    return f"""{SYSTEM_INSTRUCTION}

TASK: Create an initial surrogate model for 3D N-body gravity simulation.

Approach to try: {approach}

Remember:
- Ground truth is O(N²) all-pairs. Your model MUST be faster (O(N) or O(N log N))
- Timestep: dt = 0.1
- Gravity force from particle j to particle i: F = G * m_i * m_j / r²  (G=1.0)
- Acceleration on particle i: a_i = Σ(F_ij / m_i) for selected particles j
- Update velocity: v_new = v + a × dt
- Update position: x_new = x + v × dt
- Preserve mass: mass stays constant
- Return [new_x, new_y, new_z, new_vx, new_vy, new_vz, mass]

PERFORMANCE EXAMPLES:
- Cutoff radius: Loop over all_particles, skip if distance > R_cutoff
- K-NN: Find K closest, compute only those forces
- Grid: Only check particles in same/adjacent cells

PHYSICS CONSERVATION REQUIREMENTS (CRITICAL):
- Energy drift <1%: Your integration must preserve E = 0.5*m*v² - Σ(G*m_i*m_j/r_ij)
- Angular momentum drift <1%: Verify L = Σ(m * r × v) stays constant
- Symplectic integrators strongly recommended (Verlet, leapfrog, semi-implicit Euler)
- Standard Euler integration will cause >10% energy drift (heavy fitness penalty)

Generate the complete Python code now."""


def get_mutation_prompt(
    parent_code: str,
    fitness: float,
    accuracy: float,
    speed: float,
    generation: int,
    mutation_type: str = "explore",
    energy_drift: float = 0.0,
    momentum_drift: float = 0.0,
    test_problem: str = "plummer",
) -> str:
    """Generate mutation prompt based on parent performance.

    Args:
        parent_code: Python code of parent model
        fitness: Parent's fitness score (accuracy/speed)
        accuracy: Parent's accuracy (0-1)
        speed: Parent's execution time in seconds
        generation: Current generation number
        mutation_type: "explore" for large changes, "exploit" for refinement
        energy_drift: Parent's energy conservation drift (0-1+, lower is better)
        momentum_drift: Parent's angular momentum drift (0-1+, lower is better)
        test_problem: Test problem name for threshold lookup (two_body, figure_eight, plummer)

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
- Try symplectic integrators (leapfrog, semi-implicit Euler) if not already using
- Add conservation-preserving terms (energy correction, momentum adjustment)
- Try a completely novel numerical trick

Be creative and don't be afraid to make significant changes!"""
    else:  # exploit
        strategy = """REFINE the existing approach:
- Adjust numerical constants for better accuracy (e.g., timestep, epsilon, damping)
- Fine-tune force calculations to match ground truth more closely
- Optimize calculations for speed (simplify expressions, reduce operations)
- Fix any numerical instabilities (NaN, overflow)
- Improve energy conservation (reduce energy drift below 1%)
- Preserve angular momentum (ensure L = m*r×v stays constant)
- Fine-tune integration scheme for better physics preservation

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

    # Add conservation analysis with dynamic per-problem thresholds
    energy_threshold = settings.get_physics_threshold(test_problem, "energy")
    momentum_threshold = settings.get_physics_threshold(test_problem, "momentum")

    # Categorize energy drift relative to actual threshold
    if energy_drift > energy_threshold:
        # ELIMINATED by hard constraint
        perf_analysis.append(
            f"✗ Energy drift {energy_drift * 100:.2f}% - ELIMINATED (exceeds {energy_threshold * 100:.1f}% threshold)"
        )
    elif energy_drift > 0.01:
        # Acceptable but not ideal
        perf_analysis.append(
            f"⚠ Energy drift {energy_drift * 100:.2f}% - acceptable but not ideal (threshold: {energy_threshold * 100:.1f}%, ideal: <1%)"
        )
    else:
        # Excellent conservation
        perf_analysis.append(f"✓ Energy drift {energy_drift * 100:.2f}% - excellent conservation")

    # Categorize momentum drift relative to actual threshold
    if momentum_drift > momentum_threshold:
        # ELIMINATED by hard constraint
        perf_analysis.append(
            f"✗ Angular momentum drift {momentum_drift * 100:.2f}% - ELIMINATED (exceeds {momentum_threshold * 100:.1f}% threshold)"
        )
    elif momentum_drift > 0.01:
        # Acceptable but not ideal
        perf_analysis.append(
            f"⚠ Angular momentum drift {momentum_drift * 100:.2f}% - acceptable but not ideal (threshold: {momentum_threshold * 100:.1f}%, ideal: <1%)"
        )
    else:
        # Excellent conservation
        perf_analysis.append(
            f"✓ Angular momentum drift {momentum_drift * 100:.2f}% - excellent conservation"
        )

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
✓ Return statement: The function MUST end with "return [new_x, new_y, new_z, new_vx, new_vy, new_vz, mass]"
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

Generate improved code that maintains the predict(particle, all_particles) signature and returns [new_x, new_y, new_z, new_vx, new_vy, new_vz, mass]."""


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

    # Format conservation metrics
    # Handle None explicitly to avoid misclassifying 0.0 as poor conservation
    p1_energy_status = (
        "✓ Good"
        if parent1.energy_drift is not None and parent1.energy_drift < 0.01
        else f"✗ Poor ({(parent1.energy_drift or 1.0) * 100:.1f}%)"
    )
    p2_energy_status = (
        "✓ Good"
        if parent2.energy_drift is not None and parent2.energy_drift < 0.01
        else f"✗ Poor ({(parent2.energy_drift or 1.0) * 100:.1f}%)"
    )

    return f"""{SYSTEM_INSTRUCTION}

OBJECTIVE: Create a HYBRID surrogate model by combining the strengths of TWO parent models.

PARENT 1 (Fitness: {parent1.fitness or 0.0:.2f}, Accuracy: {parent1.accuracy or 0.0:.4f}, Speed: {parent1.speed or 0.01:.6f}s):
Energy Drift: {p1_energy_status}
```python
{parent1.raw_code}
```

PARENT 2 (Fitness: {parent2.fitness or 0.0:.2f}, Accuracy: {parent2.accuracy or 0.0:.4f}, Speed: {parent2.speed or 0.01:.6f}s):
Energy Drift: {p2_energy_status}
```python
{parent2.raw_code}
```

TASK:
1. Analyze what makes each parent successful (accuracy, speed, AND conservation)
2. Identify complementary strengths (e.g., Parent 1 conserves energy, Parent 2 is faster)
3. Design a NEW approach that combines these strengths WHILE preserving physics
4. Prioritize conservation: Energy and momentum drift MUST be <1% (heavily penalized)
5. You may introduce novel elements beyond just merging (creative synthesis)

GENERATION: {generation} (Explore phase: try bold combinations)

{completion_reminder}

Generate the hybrid predict function:"""
