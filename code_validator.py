"""Code validation and safe execution for LLM-generated surrogate models.

This module provides multi-layer security validation to ensure generated code
is safe to execute and meets required specifications.
"""

import ast
import math
import logging
from typing import Callable, Any
from dataclasses import dataclass

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
    SAFE_BUILTINS = {"abs", "min", "max", "sum", "len", "range", "enumerate", "zip"}

    # Disallowed AST node types
    FORBIDDEN_NODES = {
        ast.Import,              # No imports
        ast.ImportFrom,          # No from X import Y
        ast.AsyncFunctionDef,    # No async
        ast.Global,              # No global variables
        ast.Nonlocal,            # No nonlocal
    }

    @classmethod
    def validate(cls, code: str) -> ValidationResult:
        """Run all validation checks on code.

        Args:
            code: Python code string to validate

        Returns:
            ValidationResult with errors and warnings
        """
        errors = []
        warnings = []

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
                f"predict() must take exactly 2 arguments (particle, attractor), "
                f"found {num_args}"
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

            # Check for infinite loops (while True without break)
            if isinstance(node, ast.While):
                if isinstance(node.test, ast.Constant) and node.test.value is True:
                    # Check if there's a break statement
                    has_break = any(
                        isinstance(n, ast.Break) for n in ast.walk(node)
                    )
                    if not has_break:
                        errors.append("Infinite loop detected: while True without break")

        # Check 5: Warn about complexity
        num_loops = sum(
            1 for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While))
        )
        if num_loops > 2:
            warnings.append(
                f"Code has {num_loops} loops, may be slow (expected 0-2)"
            )

        num_functions = sum(
            1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        )
        if num_functions > 1:
            warnings.append(
                f"Code defines {num_functions} functions, only 'predict' is needed"
            )

        return ValidationResult(len(errors) == 0, errors, warnings)

    @classmethod
    def compile_safely(
        cls,
        code: str,
        attractor: list[float]
    ) -> Callable | None:
        """Compile code in restricted namespace and return predict function.

        Args:
            code: Python code string containing predict function
            attractor: Attractor position to bind to function

        Returns:
            Wrapped predict function, or None if compilation fails
        """
        # Create sandbox with limited builtins
        import builtins
        allowed_builtins = {
            name: getattr(builtins, name)
            for name in cls.SAFE_BUILTINS
            if hasattr(builtins, name)
        }

        sandbox_globals = {
            "__builtins__": allowed_builtins,
            "math": math,
        }
        local_namespace = {}

        # Execute code in sandbox
        try:
            exec(code, sandbox_globals, local_namespace)
        except Exception as e:
            logger.error(f"Execution error during compilation: {e}")
            return None

        if "predict" not in local_namespace:
            logger.error("predict function not found after exec")
            return None

        predict_func = local_namespace["predict"]

        # Test call to verify it works
        # Use a particle away from attractor to avoid division by zero
        try:
            test_particle = [45.0, 45.0, 0.5, 0.5]
            result = predict_func(test_particle, attractor)

            # Validate output format
            if not isinstance(result, (list, tuple)):
                logger.error(f"predict() must return list/tuple, got {type(result)}")
                return None

            if len(result) != 4:
                logger.error(f"predict() must return 4 values, got {len(result)}")
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

        # Return wrapped function with attractor bound
        def wrapped(particle: list[float]) -> list[float]:
            """Wrapped predict function with attractor pre-bound."""
            return list(predict_func(particle, attractor))

        return wrapped


def validate_and_compile(
    code: str,
    attractor: list[float]
) -> tuple[Callable | None, ValidationResult]:
    """Convenience function to validate and compile in one step.

    Args:
        code: Python code string
        attractor: Attractor position

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

    compiled_func = CodeValidator.compile_safely(code, attractor)

    if compiled_func is None:
        # Add error to validation result
        validation.errors.append("Failed to compile or test function")
        validation.valid = False

    return compiled_func, validation
