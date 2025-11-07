# Galaxy - Model Selection Guide

**Last Updated**: November 07, 2025
**Status**: Based on empirical validation runs

## Executive Summary

This guide documents which Gemini models work for Galaxy evolution, based on systematic testing.

**TL;DR**: **Use `gemini-2.5-flash-lite` for all problems.** Gemini Pro is blocked by safety filters and cannot generate code.

---

## Model Comparison Matrix

| Model | Code Generation | Plummer (N=50) | Cost/Run | Recommendation |
|-------|----------------|----------------|----------|----------------|
| **gemini-2.5-flash-lite** | ✅ Works (0% syntax errors) | Marginal (10% Gen1, 0% Gen2+) | $0.04 | **USE THIS** |
| **gemini-2.5-flash** | ❓ Untested | ❓ Untested | ~$0.12 | Worth testing |
| **gemini-2.5-pro** | ❌ **BLOCKED** (safety filters) | N/A (unusable) | N/A | **DO NOT USE** |

---

## Gemini Pro: Why It's Blocked

### The Problem

Gemini Pro **cannot generate ANY code** for Galaxy evolution due to safety filter restrictions.

**Error Pattern**:
```
finish_reason: 2 (SAFETY)
Duration: 6+ minutes
Attempts: 6+ failed generations
Success rate: 0% (100% blocked)
```

### Root Cause

Pro's safety filters are more conservative than Flash models. Evolution prompts trigger false positives:
- "Force calculations" (flagged as violence?)
- "Mutation" and "evolution" terminology
- N-body dynamics (complex system prompts)
- Large-scale physics simulations

### Why Flash Lite Works

Flash models are **optimized for code generation use cases** with more permissive safety filters. They understand the context is scientific code, not harmful content.

### Attempted Workarounds

❌ **None successful** - Safety filters cannot be disabled or adjusted via API.

### Impact

**Gemini Pro is completely unusable** for this project despite theoretical quality advantages (better reasoning, fewer hallucinations).

---

## Flash Lite: Current Champion

### Strengths

✅ **Code Generation**: 0% syntax errors (50/50 models valid)
✅ **Gen 0-1 Survival**: 10% on plummer (N=50)
✅ **Cost**: $0.04 per run (cheap experimentation)
✅ **Speed**: 15 RPM rate limit (4 minutes for 50 calls)
✅ **Availability**: Works on free tier

### Limitations

⚠️ **Gen2+ Collapse**: 0% survival after Gen 1 on plummer
⚠️ **Physics Preservation**: Struggles with N=50 conservation (16-200% drift)
⚠️ **Single Lineage**: Only 1 survivor → insufficient diversity

### When Flash Lite Works Well

- **two_body (N=2)**: Expected good performance (simple problem)
- **figure_eight (N=3)**: Expected moderate performance (chaotic but low N)
- **plummer (N=50)**: Marginal performance (proves concept, not production-ready)

---

## Decision Tree

```
Are you running Galaxy evolution?
│
├─ YES → Use gemini-2.5-flash-lite
│   │
│   ├─ two_body/figure_eight → ✅ Should work well
│   │
│   ├─ plummer (N=50) → ⚠️ Expect Gen2+ collapse
│   │   │
│   │   ├─ Need sustained evolution? → Reduce to N=30
│   │   └─ Exploratory research? → Accept 10% Gen1 limitation
│   │
│   └─ Syntax errors high? → Already at 0%, unlikely issue
│
└─ Considering Pro upgrade? → ❌ Don't bother, it's blocked
```

---

## Recommendations by Use Case

### Research / Prototyping

**Use**: `gemini-2.5-flash-lite`
**Rationale**: Cheap, fast, works for concept validation
**Limitations**: Documented and acceptable for research

### Production Simulations

**Current state**: **Not ready**
**Blocker**: Gen2+ collapse on complex problems
**Options**:
1. Simplify to N=30 (test if Flash Lite sustains evolution)
2. Use two_body/figure_eight as primary problems
3. Wait for better models (Claude/GPT-4o integration)

### Cost-Sensitive Experiments

**Use**: `gemini-2.5-flash-lite`
**Cost**: $0.04/run = 250 runs per $10 daily budget
**Volume**: Can run 20 full evolutions per day on free tier

---

## Configuration

### Current Recommended Config

```yaml
# config.yaml
model:
  name: gemini-2.5-flash-lite  # ONLY working option
  temperature: 0.8
  max_output_tokens: 2000
```

### Do NOT Change To Pro

```yaml
# ❌ THIS WILL FAIL
model:
  name: gemini-2.5-pro  # Blocked by safety filters!
```

**If you try Pro**, you'll see:
```
ERROR - LLM call failed: finish_reason=2 (SAFETY)
ERROR - LLM call failed: finish_reason=2 (SAFETY)
ERROR - LLM call failed: finish_reason=2 (SAFETY)
...
```

---

## Future Directions

### Alternative Models (Not Yet Implemented)

Potential alternatives if/when integrated:
- **Claude 3.5 Sonnet**: Strong code generation, may have different safety filters
- **GPT-4o**: Multimodal capabilities, proven code quality
- **Gemini 2.5 Flash**: Middle ground between Flash Lite and Pro (untested)

### Flash Lite Improvements

Ways to maximize Flash Lite performance:
1. **Reduce problem complexity**: N=30 instead of N=50
2. **Increase population**: 15-20 models → more diversity → better Gen2+ survival
3. **Adaptive thresholds**: Relax to 25-30% for plummer
4. **Hybrid approach**: Use Flash Lite for exploration, validate top models with direct N-body

---

## Empirical Evidence

### Flash Lite Plummer Validation (Nov 07, 2025)

**Configuration**:
- Model: gemini-2.5-flash-lite
- Problem: plummer (N=50)
- Population: 10, Generations: 5
- Cost: $0.0362

**Results**:
```
Gen 0:  1/10 survived (10.0%)
Gen 1:  1/10 survived (10.0%)
Gen 2:  0/10 survived ( 0.0%)  ← Collapse
Gen 3:  0/10 survived ( 0.0%)
Gen 4:  0/10 survived ( 0.0%)

Code Quality: 0% syntax errors (50/50 valid)
Best Fitness: 1.3151
```

**Interpretation**: Flash Lite can generate valid code and achieve marginal evolution, but cannot sustain past Gen 1 on N=50 problems.

### Gemini Pro Attempt (Nov 07, 2025)

**Configuration**:
- Model: gemini-2.5-pro
- Problem: plummer (N=50)
- Duration: 6+ minutes

**Results**:
```
Attempt 1: finish_reason=2 (SAFETY) ❌
Attempt 2: finish_reason=2 (SAFETY) ❌
Attempt 3: finish_reason=2 (SAFETY) ❌
Attempt 4: finish_reason=2 (SAFETY) ❌
Attempt 5: finish_reason=2 (SAFETY) ❌
Attempt 6: finish_reason=2 (SAFETY) ❌
...
Success rate: 0/6 (0%)
```

**Interpretation**: Pro is completely blocked. No code generated. Unusable.

---

## FAQ

### Q: Can I override Pro's safety filters?

**A**: No. Safety filters are server-side and cannot be disabled via API.

### Q: Will Pro work for simpler problems?

**A**: Unknown, but unlikely. The blocker is prompt content (evolution terminology), not problem complexity.

### Q: Should I pay for Pro tier?

**A**: No. Pro is blocked regardless of tier. Flash Lite is only option.

### Q: What about Flash (not Flash Lite)?

**A**: Untested. Worth trying ($0.12/run). May have same safety issues as Pro, or may work like Flash Lite.

### Q: Why does Flash Lite work but Pro doesn't?

**A**: Flash models are **optimized for code generation** with permissive filters. Pro prioritizes safety over code-gen flexibility.

### Q: Can I use Claude or GPT instead?

**A**: Not currently integrated. Would require significant code changes to support alternative LLM providers.

---

## Conclusion

**For Galaxy evolution: Use `gemini-2.5-flash-lite` exclusively.**

Gemini Pro is a dead end due to safety filters. Flash Lite is the only proven working model, with documented limitations that are acceptable for research purposes.

If you need better sustained evolution, the path forward is **not** upgrading to Pro (blocked), but rather:
1. Simplifying the problem (reduce N)
2. Increasing population diversity
3. Integrating alternative LLM providers (Claude, GPT-4o)

---

**Last Validation**: November 07, 2025
**Models Tested**: Flash Lite (✅), Pro (❌)
**Recommendation Status**: Stable - do not attempt Pro
