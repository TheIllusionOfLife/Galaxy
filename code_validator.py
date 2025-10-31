"""Code validation and safe execution for LLM-generated surrogate models.

This module provides multi-layer security validation to ensure generated code
is safe to execute and meets required specifications.
"""

import ast
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of code validation.

    Attributes:
        valid: Whether code passed all validation checks
        errors: List of validation errors (blocking)
        warnings: List of warnings (non-blocking)
    """

    valid: bool
    errors: list[str]
    warnings: list[str]


class CodeValidator:
    """Multi-layer validation for LLM-generated code.

    Provides AST-based static analysis and runtime safety checks to prevent
    malicious or buggy code from executing.
    """

    # Allowed built-in functions in sandbox
    SAFE_BUILTINS: ClassVar[set[str]] = {
        "abs",
        "min",
        "max",
        "sum",
        "len",
        "range",
        "enumerate",
        "zip",
        "float",
        "int",
        "list",  # Type constructors needed by generated code
    }

    # Disallowed AST node types
    FORBIDDEN_NODES: ClassVar[set[type[ast.AST]]] = {
        ast.Import,  # No imports
        ast.ImportFrom,  # No from X import Y
        ast.AsyncFunctionDef,  # No async
        ast.Global,  # No global variables
        ast.Nonlocal,  # No nonlocal
    }

    @classmethod
    def validate(cls, code: str) -> ValidationResult:
        """Run all validation checks on code.

        Args:
            code: Python code string to validate

        Returns:
            ValidationResult with errors and warnings
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Check 1: Parse AST
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return ValidationResult(False, [f"Syntax error: {e}"], [])

        # Check 2: Find predict function
        predict_func = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "predict":
                predict_func = node
                break

        if not predict_func:
            errors.append("No 'predict' function found")
            return ValidationResult(False, errors, warnings)

        # Check 3: Validate signature (must take exactly 2 arguments)
        num_args = len(predict_func.args.args)
        if num_args != 2:
            errors.append(
                f"predict() must take exactly 2 arguments (particle, all_particles), found {num_args}"
            )

        # Check 4: Scan for forbidden operations
        for node in ast.walk(tree):
            # Check forbidden node types
            if type(node) in cls.FORBIDDEN_NODES:
                errors.append(f"Forbidden operation: {type(node).__name__}")

            # Check for dangerous builtin calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in ("eval", "exec", "compile", "__import__"):
                        errors.append(f"Dangerous function call: {func_name}")
                    elif func_name == "open":
                        errors.append("File I/O not allowed: open()")

            # Check for infinite loops (while True/1 without break)
            if isinstance(node, ast.While):
                # Check for always-true conditions: True, 1, or other truthy constants
                is_always_true = False
                test_value = None
                if isinstance(node.test, ast.Constant):
                    # Catches: while True, while 1, while 2, etc.
                    if node.test.value is True or (
                        isinstance(node.test.value, int) and node.test.value != 0
                    ):
                        is_always_true = True
                        test_value = node.test.value

                if is_always_true and test_value is not None:
                    # Check if there's a break statement
                    has_break = any(isinstance(n, ast.Break) for n in ast.walk(node))
                    if not has_break:
                        errors.append(f"Infinite loop detected: while {test_value!r} without break")

        # Check 5: Warn about potentially expensive comprehensions
        for node in ast.walk(tree):
            if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp)):
                # Check for large range() calls in comprehensions
                for gen in node.generators:
                    if isinstance(gen.iter, ast.Call):
                        if isinstance(gen.iter.func, ast.Name) and gen.iter.func.id == "range":
                            # Check if range has a large constant argument
                            if gen.iter.args and isinstance(gen.iter.args[0], ast.Constant):
                                range_size = gen.iter.args[0].value
                                if isinstance(range_size, int) and range_size > 10000:
                                    warnings.append(
                                        f"List comprehension with large range ({range_size}) may be slow"
                                    )

        # Check 6: Warn about complexity
        num_loops = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While)))
        if num_loops > 2:
            warnings.append(f"Code has {num_loops} loops, may be slow (expected 0-2)")

        num_functions = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
        if num_functions > 1:
            warnings.append(f"Code defines {num_functions} functions, only 'predict' is needed")

        return ValidationResult(len(errors) == 0, errors, warnings)

    @classmethod
    def compile_safely(cls, code: str, all_particles: list[list[float]]) -> Callable | None:
        """Compile code in restricted namespace and return predict function.

        Args:
            code: Python code string containing predict function
            all_particles: List of all particles for testing (each: [x,y,z,vx,vy,vz,mass])

        Returns:
            Wrapped predict function, or None if compilation fails
        """
        # Create sandbox with limited builtins
        import builtins

        allowed_builtins = {
            name: getattr(builtins, name) for name in cls.SAFE_BUILTINS if hasattr(builtins, name)
        }

        sandbox_globals = {
            "__builtins__": allowed_builtins,
            "math": math,
        }
        local_namespace: dict[str, Any] = {}

        # Execute code in sandbox
        try:
            exec(code, sandbox_globals, local_namespace)
        except Exception:
            logger.exception("Execution error during compilation")
            return None

        if "predict" not in local_namespace:
            logger.error("predict function not found after exec")
            return None

        predict_func = local_namespace["predict"]

        # Test call to verify it works with 3D N-body particles
        # Use a realistic test scenario: 3 particles in 3D space
        # Test particles: [x, y, z, vx, vy, vz, mass]
        try:
            test_all_particles = [
                [10.0, 20.0, 30.0, 0.1, 0.2, 0.3, 1.0],  # Particle 1
                [40.0, 50.0, 60.0, -0.1, -0.2, -0.3, 1.5],  # Particle 2
                [70.0, 80.0, 90.0, 0.0, 0.0, 0.0, 2.0],  # Particle 3
            ]
            test_particle = test_all_particles[0]

            result = predict_func(test_particle, test_all_particles)

            # Validate output format
            if not isinstance(result, (list, tuple)):
                logger.error(f"predict() must return list/tuple, got {type(result)}")
                return None

            if len(result) != 7:
                logger.error(
                    f"predict() must return 7 values [x,y,z,vx,vy,vz,mass], got {len(result)}"
                )
                return None

            # Check all values are numeric
            try:
                result_floats = [float(x) for x in result]
            except (ValueError, TypeError) as e:
                logger.error(f"predict() must return numeric values: {e}")
                return None

            # Check for NaN or Inf
            if any(math.isnan(x) or math.isinf(x) for x in result_floats):
                logger.error("predict() returned NaN or Inf on test call")
                return None

        except Exception as e:
            logger.error(f"Test call failed: {e}")
            return None

        # Return wrapped function with all_particles bound
        def wrapped(particle: list[float], particles_list: list[list[float]]) -> list[float]:
            """Wrapped predict function that accepts (particle, all_particles)."""
            return list(predict_func(particle, particles_list))

        return wrapped


def validate_and_compile(
    code: str, all_particles: list[list[float]]
) -> tuple[Callable | None, ValidationResult]:
    """Convenience function to validate and compile in one step.

    Args:
        code: Python code string
        all_particles: List of all particles for testing

    Returns:
        Tuple of (compiled_function or None, validation_result)
    """
    validation = CodeValidator.validate(code)

    if not validation.valid:
        logger.warning(f"Code validation failed: {validation.errors}")
        return None, validation

    if validation.warnings:
        for warning in validation.warnings:
            logger.info(f"Code warning: {warning}")

    compiled_func = CodeValidator.compile_safely(code, all_particles)

    if compiled_func is None:
        # Add error to validation result
        validation.errors.append("Failed to compile or test function")
        validation.valid = False

    return compiled_func, validation
