"""Benchmark suite for Galaxy Prometheus N-body simulation.

This package provides systematic benchmarking of baseline surrogate models
across multiple test problems and particle counts.

Modules:
    benchmark_runner: Core benchmark orchestration
    scaling_analysis: Particle count scaling analysis
    visualization: Benchmark-specific plotting utilities
"""

from benchmarks.benchmark_runner import BenchmarkResult, BenchmarkRunner

__all__ = ["BenchmarkResult", "BenchmarkRunner"]
