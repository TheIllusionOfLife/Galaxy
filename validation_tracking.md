# Full-Scale Conservation Prompts Validation - Tracking

## Baseline Metrics (PR #45 - WITHOUT Conservation Prompts)

**Run**: results/run_20251102_215639
**Config**: 10 pop × 5 gen, plummer (N=50), physics_penalty enabled
**Prompts**: NO conservation emphasis

**Key Metrics**:
- Best energy drift: 4.21% (civ_1_7)
- Mean energy drift: 391.67%
- Models <1% drift: 0/50 (0%)
- Models <10% drift: Unknown (estimated <5%)
- Best fitness: 1,518.11

## PR #47 Small Run (WITH Conservation Prompts)

**Run**: results/run_20251103_095527
**Config**: 3 pop × 2 gen, two_body (N=2), physics_penalty enabled
**Prompts**: WITH conservation emphasis

**Key Metrics**:
- Best energy drift: 0.10% (civ_1_2) - **162x better than PR #45**
- Mean energy drift (Gen 0): 1.12%
- Models <1% drift: 1/6 (17%)
- Models <10% drift: 3/6 (50%)
- Best fitness: 187,907

## This Validation (Full Scale + Complex Problems)

### Run 1: two_body (10 pop × 5 gen)
- **Run directory**: results/run_20251103_111004
- **Start time**: 2025-11-03 11:04:54
- **End time**: 2025-11-03 11:10:04 (5m 10s)
- **API calls**: 60
- **Cost**: $0.0528
- **Best fitness**: 320,157 (civ_0_3, Gen 0)
- **Best energy drift**: 0.16% (degraded from 0.10% in PR #47)
- **Mean energy drift**: 161,061% (catastrophic degradation)
- **Conservation rate**: 2% (<1% drift)
- **LLM Success Rate**: 100% (60/60 successful)
- **Validation directory**: results/analysis/physics_validation_20251103_111538/

### Run 2: plummer (10 pop × 5 gen)
- **Run directory**: results/run_20251103_111514
- **Start time**: 2025-11-03 11:10:33
- **End time**: 2025-11-03 11:15:14 (4m 41s)
- **API calls**: 60
- **Cost**: $0.0350
- **Best fitness**: 1,345 (civ_4_2, Gen 4)
- **Best energy drift**: 16.25% (WORSE than 4.21% PR #45 baseline)
- **Mean energy drift**: 1,658% (WORSE than 391.67% baseline)
- **Conservation rate**: 0% (<1% drift)
- **LLM Success Rate**: 98.3% (59/60 successful, 1 invalid)
- **Validation directory**: N/A (validation failed due to invalid model)

### CRITICAL FINDING
⚠️ **Conservation prompts FAILED to generalize to full scale**
- Small run (PR #47): 0.10% drift, 17% conservation rate
- Full run (this): 0.16% drift (two_body), 16.25% (plummer), 0-2% conservation rate
- **Conclusion**: Prompt-based approach insufficient, architectural changes required

## Notes
- Conservation prompts confirmed active in prompts.py (line 35-42)
- Current config.yaml test_problem: plummer (will change to two_body first)
- Physics penalty enabled: energy_weight=0.3, momentum_weight=0.1
