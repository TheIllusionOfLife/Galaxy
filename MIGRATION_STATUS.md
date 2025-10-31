# 3D N-Body Migration Status

## Goal
Transform the project from a toy 2D single-attractor problem to true 3D particle-particle N-body gravitational simulation.

## Completed ✅

### 1. Core Physics (DONE)
- **File**: `prototype.py` - CosmologyCrucible class
- **Changes**:
  - Particles: `[x, y, vx, vy]` → `[x, y, z, vx, vy, vz, mass]` (4→7 elements)
  - Physics: Single attractor O(N) → All-pairs O(N²) gravitational interactions
  - Integration: Euler → Leapfrog (Verlet) for better energy conservation
  - Removed: Artificial `time.sleep(0.05)` delay
  - Added: Real computational cost from nested loops
- **Tests**: `tests/test_nbody_physics.py` - 15/15 passing
  - ✅ 7-element particle format
  - ✅ Particle-particle interactions (not attractor)
  - ✅ O(N²) complexity scaling
  - ✅ Inverse square law verification
  - ✅ 3D force calculations
  - ✅ Mass-dependent forces
  - ✅ Two-body energy conservation (<10% drift)
  - ✅ Angular momentum conservation
  - ✅ No self-interaction
  - ✅ No artificial delays

### 2. Surrogate Interface (DONE)
- **File**: `code_validator.py`
- **Changes**:
  - Signature: `predict(particle, attractor)` → `predict(particle, all_particles)`
  - Test particles: 4-element → 7-element
  - Validation: Returns 4 values → Returns 7 values
  - Error messages: Updated to reference `all_particles` not `attractor`
- **Tests**: `tests/test_surrogate_interface.py` - 13/13 passing
  - ✅ 2-parameter signature validation
  - ✅ 7-element output format
  - ✅ 3D particle handling
  - ✅ Mass preservation
  - ✅ all_particles iteration
  - ✅ Access to other particle masses
  - ✅ Cutoff radius approximation example
  - ✅ K-nearest neighbors example
  - ✅ Edge cases (empty list, single particle)

### 3. Test Updates (PARTIAL)
- **File**: `tests/test_crossover.py`
- **Updated**: `test_crossover_preserves_function_signature()` for 3D N-body
- **Status**: 1/1 updated test passing

## In Progress 🚧

### 4. Call Site Updates (BLOCKED)
Breaking changes affect multiple files. Need to update all call sites from:
```python
validate_and_compile(code, attractor=[x, y])
```
To:
```python
validate_and_compile(code, all_particles=[[x,y,z,vx,vy,vz,mass], ...])
```

**Affected files** (from grep):
- `prototype.py` - 3 call sites (lines 58, 330)
  - `build_callable()` method
  - `LLM_propose_surrogate_model()` function
- `tests/test_integration.py` - Unknown count
- `test_gemini_connection.py` - Unknown count
- `ARCHITECTURE.md` - Documentation only

## Not Started ❌

### 5. Evaluation Function
- **File**: `prototype.py` - `evaluate_surrogate_model()`
- **Current**: Expects `model(particle)` with attractor pre-bound
- **Needed**: Update to `model(particle, all_particles)`
- **Impact**: Changes how surrogate models are called during evaluation

### 6. Fitness Calculation
- **File**: `prototype.py` - `EvolutionaryEngine.run_evolutionary_cycle()`
- **Current**: `fitness = accuracy / speed`
- **Needed**: `fitness = accuracy * speedup` where speedup = brute_force_time / surrogate_time
- **Reason**: Reward surrogates that are faster than O(N²) baseline

### 7. Prompts
- **File**: `prompts.py`
- **Current**: Describes 2D single-attractor problem
- **Needed**:
  - Update to 3D N-body context
  - Provide approximation strategy examples (cutoff, K-NN, grid)
  - Show concrete code examples for 3D particles
  - Emphasize O(N) or O(N log N) complexity targets

### 8. Mock Functions
- **File**: `prototype.py` - `make_mock_surrogate()`, `LLM_mutate_surrogate_mock()`
- **Current**: Generate 4-element particle code
- **Needed**: Generate 7-element particle code for mock mode

### 9. Configuration
- **File**: `config.yaml`
- **Needed**: Add `num_particles` configuration (default: 50, range: 50-200)
- **Needed**: Add dimensionality flag (though 3D is now hardcoded)

### 10. DEFAULT_ATTRACTOR
- **File**: `prototype.py`
- **Current**: `DEFAULT_ATTRACTOR = [50.0, 50.0]`
- **Needed**: Remove entirely (no longer used)

### 11. Integration Tests
- **File**: `tests/test_integration.py`
- **Status**: Not examined yet
- **Expected**: Multiple test failures due to signature changes

### 12. Other Test Files
Need to examine and update:
- `tests/test_evolution.py`
- `tests/test_history_tracking.py`
- `tests/test_visualization_integration.py`
- `tests/test_code_length_penalty.py`
- `tests/test_config.py`
- `tests/test_visualization.py`
- `tests/test_crossover_integration.py`

### 13. Documentation
- `README.md` - Update problem description, examples
- `ARCHITECTURE.md` - Update technical details
- CLI help text - Update descriptions

### 14. Phase 2 Tasks (Not Started)
- scipy.spatial KDTree baseline (use library instead of implementing Barnes-Hut)
- Standard test problems (Plummer sphere, two-body, etc.)
- Validation metrics (energy, momentum, trajectory error)
- Benchmark suite

## Test Status

### Passing ✅
- `tests/test_nbody_physics.py` - 15/15
- `tests/test_surrogate_interface.py` - 13/13
- `tests/test_crossover.py::test_crossover_preserves_function_signature` - 1/1

### Failing ❌
- All tests using `validate_and_compile()` with old signature
- All tests expecting 4-element particles
- All tests using `DEFAULT_ATTRACTOR`
- All integration tests (not yet examined)

### Not Run
- Full test suite (will have many failures)
- Real API integration test
- User acceptance testing

## Estimated Remaining Work

### Phase 1 Completion (Core Migration)
- Call site updates: 4-6 hours
- Evaluation/fitness updates: 2-3 hours
- Prompts update: 2-3 hours
- Test fixes: 6-8 hours
- Documentation: 2-3 hours
- **Total: 16-23 hours** (~2-3 days)

### Phase 2 (Baselines & Benchmarks)
- As per original plan: 2-3 days

## Next Steps for Intermediate PR

1. Document current state (this file) ✅
2. Commit what we have with clear message
3. Create PR explaining partial progress
4. Get feedback on approach before continuing
5. Resume with systematic call site updates

## Technical Decisions Made

1. **Leapfrog Integration**: Chose symplectic integrator for better energy conservation
   - Tested with two-body orbits: <10% energy drift over 50 timesteps
   - Acceptable for prototype; can optimize later

2. **7-Element Particles**: `[x, y, z, vx, vy, vz, mass]`
   - More realistic than unit masses
   - Allows testing mass-dependent forces
   - Consistent with astrophysics N-body conventions

3. **O(N²) Baseline**: True all-pairs calculation
   - No shortcuts in ground truth
   - Real computational cost (no artificial delays)
   - Surrogates must beat this for meaningful speedup

4. **Wrapped Function Signature**: `wrapped(particle, particles_list)`
   - Maintains flexibility for different evaluation contexts
   - Avoids pre-binding particles at compile time
   - Allows surrogates to access full particle list

## Open Questions

1. **Timestep**: Current dt=0.1 may be too large for some orbits. Configurable?
2. **Softening**: epsilon=0.01 prevents singularities. Is this appropriate value?
3. **Integration Method**: Leapfrog good enough or need RK4/adaptive?
4. **Backward Compatibility**: Support both 2D and 3D, or clean break?
   - **Current decision**: Clean break to 3D (simpler, clearer intent)

## Lessons Learned

1. **Scope Creep**: "Add 3D" cascades to 50+ file changes
2. **TDD Value**: Physics tests caught integration issues early
3. **Interface Changes**: Ripple through entire codebase
4. **Mock Updates**: Must stay in sync with real implementation
5. **Documentation**: Essential for handoff/pause points
