# Codex Improvement Plan

## Objective
Lay out the concrete next steps required to evolve the prototype into a trustworthy small-scale proving ground for surrogate-model discovery driven by LLM strategies.

## Roadmap

### 1. LLM Integration & Interface Hardening
- Define the precise prompt/response contract for surrogate proposals, including expected `predict(particle, attractor)` signature and guardrails on state access.
- Implement the call-out layer around the target LLM (mock locally if offline) and wire the response into `SurrogateGenome.raw_code`.
- Extend `compile_external_surrogate` to validate generated code (AST inspection or static checks) before execution and surface structured error feedback to the evolution loop.

### 2. Fitness Signal Deepening
- Augment the crucible with multi-step rollouts (e.g., 5–10 steps) to capture compounding error; weight accuracy vs. stability explicitly.
- Incorporate velocity and acceleration error components alongside positional error for a richer score.
- Track historical fitness and implement early rejection for unstable surrogates (NaNs, exploding velocities) to conserve compute.

### 3. Experimentation Harness & Observability
- Add run configuration (population size, mutation rate, rollout horizon) via CLI or config file to enable repeatable experiments.
- Instrument the engine with structured logs/metrics (CSV or JSONL) capturing generation stats, best genomes, and failure reasons.
- Provide quick visualization notebooks or scripts to graph fitness trajectories and compare surrogate behaviours.

### 4. Safety & Sandboxing
- Containerize surrogate execution or use a restricted Python interpreter to further limit capabilities of LLM-generated code.
- Enforce execution timeouts and memory guards per surrogate evaluation; abort or penalize violators.
- Document the security posture and review process for approving new primitives accessible to surrogate code.

### 5. Validation & Benchmarking
- Establish regression scenarios: fixed random seeds plus baseline genomes to detect behavioural drift.
- Compare evolved surrogates against classical integrators (e.g., Verlet) as a baseline.
- Define success metrics for the small-scale trial (target accuracy thresholds, speedups) and set up automated reports after each experiment batch.

### 6. Collaboration & PR Workflow
- Break down upcoming changes into PR-sized milestones (LLM integration, fitness deepening, observability, etc.).
- Maintain codex_improvement_plan.md as a living roadmap, updating progress and adjusting priorities after each iteration.
- Share experiment artefacts and insights in the PR description or dedicated docs to accelerate review cycles.
