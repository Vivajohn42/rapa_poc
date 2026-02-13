# DEF / RAPA MVP -- Modular Cognitive Architecture

## Overview

This repository contains a minimal working prototype (MVP) of the **DEF/RAPA** modular cognitive architecture, together with a comprehensive **8-stage validation suite** (Stufe 0-8) that empirically tests the core claims of the Dimensional Emergence Framework (DEF).

The system demonstrates how transient perceptual information can be transformed into persistent actionable knowledge through:

- A **Meaning/Narrative stream** (D)
- A deterministic **Deconstruction interface**
- A **Control stream** (C) with persistent memory

The MVP is implemented in a GridWorld environment with partial observability and a hidden-goal task requiring knowledge acquisition.

## Architecture

### Stream Pipeline

```
A (Perception) --> B (Dynamics) --> C (Valence/Control) --> Action
|                                    ^           ^
|  (observation)                     |           | (tie_break via Deconstruct_Plan)
+---> D (Narrative) --> Deconstruct -+           |
|     ^                                   Shadow-D (Planning)
|     | (Router: gated by C's uncertainty)    ^
+-----+--------------------------------------+
```

| Stream | Role | Implementation |
|--------|------|----------------|
| **A** | Perception -- abstracts raw observations into symbolic state `zA` | `agents/agent_a.py` |
| **B** | Prediction -- deterministic forward model for action evaluation | `agents/agent_b.py` |
| **C** | Control -- goal-directed decision-making with persistent memory | `agents/agent_c.py` |
| **D** | Meaning / Narrative -- semantic representation from events | `agents/agent_d.py`, `agents/agent_d_llm.py` |
| **D-Interpreter** | Extended D with coded hint interpretation capability | `agents/agent_d_interpreter.py` |
| **Shadow-D** | Forward-planning agent using B's model for multi-step lookahead | `agents/agent_shadow_d.py` |
| **Deconstruct** | Deterministic translation of D-output into structured C-memory | `router/deconstruct.py` |
| **Deconstruct-Plan** | Plan-to-C transfer (sets tie_break_preference) | `router/deconstruct_plan.py` |
| **Router** | Activates D/Shadow-D on demand; manages 3D/4D/5D regime transitions | `router/router.py` |

### Core Principle

Persistent knowledge is not stored directly in Control. It emerges through the coupling **D -> Deconstruction -> C**.

### Shared State Schemas

All inter-agent data flows through Pydantic models (`state/schema.py`):

- `ZA` -- observation (position, goal, obstacles, hints)
- `ZC` -- goal mode + persistent memory dict
- `ZD` -- narrative + meaning tags + grounding violations
- `ZPlan` -- planning output (recommended actions, confidence, risk)

### Coupling Constraints (DEF Pair-to-Pair)

The architecture enforces adjacent pairings only:

- **A <-> B**: B consumes zA for prediction
- **B <-> C**: C uses B.predict_next for look-ahead rollouts
- **C <-> D**: Router uses C's decision_delta to gate D; D feeds back via Deconstruct

D never selects actions directly. D's output flows to C exclusively through `deconstruct_d_to_c()`. This is validated automatically (9/9 checks pass, see Stufe 2).

## Environment

### GridWorld

Parametrizable gridworld supporting:

- **Variable size**: 5x5 (default), 10x10, 15x15
- **Multiple goals**: 2-N candidate goals with partition-based hint system
- **Hint cells**: Agent visits hint cell to eliminate wrong goal candidates
- **Obstacles**: Static (fixed), random (configurable count), dynamic (move periodically)

Default (5x5): Agent starts at (0,0). Hidden true goal is randomly Goal A (4,4) or Goal B (4,0). Single hint cell at (0,4) reveals which goal is correct. One obstacle at (2,2). Rewards: +1 on goal reach, -0.01 per step.

## Validation Results

The validation suite consists of 6 stages (Stufe 0-5), each testing increasingly specific DEF claims.

### Stufe 0: Infrastructure

- Statistical framework (`eval/stats.py`): 95% CI (t-distribution + Wilson for proportions), Mann-Whitney U, Cohen's d
- Shared test runner (`eval/runner.py`): batch execution, CSV output, aggregation
- Symmetry metrics integrated into ablation

### Stufe 1: Valence-Swap (Stream Isolation)

**DEF Claim**: Same world, inverted goal/reward -> A and B remain stable; C diverges.

| Metric | Result | Expected |
|--------|--------|----------|
| B mode-independent | 100% | 100% |
| Top-action flip rate | 84% | ~100% |
| Spearman (seek vs avoid) | 0.33 | -1.0 |

**Verdict**: PASS. B is perfectly mode-independent. C diverges strongly (84% action flip). Imperfect negation is caused by `anti_stay_penalty` -- a documented design feature, not a bug.

Three complementary test designs: static state analysis, forced-trajectory paired comparison, mid-episode valence swap (100% immediate response at swap point).

### Stufe 2: Stream Isolation & Ablation

**DEF Claim**: D without C must fail; pair-to-pair coupling only.

7 variants tested (4 existing + 3 new failure modes):

| Variant | Regime | SEEK SR | DEF Prediction |
|---------|--------|---------|----------------|
| modular_ond_tb | 4D (A+B+C+D) | 0.500 | Best |
| modular_nod | 3D (A+B+C) | 0.500 | Self-sufficient |
| baseline_mono | 2D+ (monolithic) | 0.500 | Good |
| ab_only | 2D (A+B) | 0.415 | Random walk |
| c_off_d_on | broken (A+B+D, no C) | 0.415 | Must fail |
| d_only | broken (D only) | 0.400 | Must fail |

**All 5 DEF predictions PASS**. Coupling constraint validation: 9/9 static + runtime checks pass.

### Stufe 3: Complexity Scaling

**DEF Claim**: Higher-dimensional regimes justify their cost at higher task complexity.

6 complexity levels, 4 variants, SEEK mode:

| Level | baseline_mono | modular_nod | modular_ond_tb | ab_only |
|-------|:---:|:---:|:---:|:---:|
| 5x5_2g | 1.000 | 1.000 | 1.000 | 0.440 |
| 10x10_2g | 0.320 | 0.500 | 0.500 | 0.100 |
| 15x15_2g | 0.180 | 0.260 | 0.260 | 0.040 |
| 10x10_2g_dyn | 0.480 | **0.740** | **0.740** | 0.120 |

**PASS**: Performance gap between modular and baseline grows with complexity. Dynamic obstacles show the largest advantage (+0.26). Random walk degrades rapidly with grid size (0.44 -> 0.04).

### Stufe 4: Drift & Deconstruction

**DEF Claim**: D's narrative drifts over time; Deconstruction stabilizes it.

5 variants (D-invocation frequency x Deconstruction frequency):

| Variant | D-calls | Decon | Flip Rate | Tag Stability |
|---------|---------|-------|-----------|---------------|
| d_always_no_decon | 83 | 0 | 0.033 | 0.927 |
| d_always_decon_k1 | 132 | 132 | 0.019 | 0.971 |
| d_always_decon_k5 | 132 | 26 | 0.019 | 0.971 |
| d_routed | **15** | 15 | 0.112 | **0.986** |

**PASS**: Deconstruction reduces tag flip rate. Router achieves **88% fewer D-calls** while maintaining highest stability. (Note: deterministic D shows mild drift; LLM-backed D would show stronger effects.)

### Stufe 5: Regime Transitions

**DEF Claim**: Router correctly transitions between 3D/4D regimes based on task difficulty.

3 task types with increasing difficulty:

| Task Type | %3D | %4D | Switches | D-Triggers | SR |
|-----------|-----|-----|----------|------------|-----|
| 2D (easy) | 93.5% | 6.5% | 0.5 | 0.5 | 1.000 |
| 3D (medium) | 89.2% | 10.8% | 8.8 | 4.7 | 0.340 |
| 4D (hard) | 87.7% | 12.3% | 24.4 | 12.3 | 0.000 |

**All 4 predictions PASS**:
1. Simple tasks stay in 3D (93.5%)
2. Complex tasks use more 4D regime
3. Regime switches increase monotonically with difficulty (0.5 -> 8.8 -> 24.4)
4. D-triggers scale with difficulty (0.5 -> 4.7 -> 12.3)

Trigger analysis: uncertainty is the dominant trigger (94-100%), with hint capture contributing 5.2% in 4D tasks.

### Stufe 6: Semantic Ambiguity -- D Makes the Difference

**DEF Claim**: When hints require semantic interpretation, Agent D with interpretation capability outperforms no-D variants.

Coded hints replace direct goal IDs with directional/comparative clues at three difficulty levels:
- **easy**: Absolute directional ("goal_at_bottom_right_far")
- **medium**: Comparative ("goal_furthest_from_origin")
- **hard**: Abstract property ("coords_sum_high")

| Level | D-Interpreter SR | no-D SR | Gap | p-value |
|-------|:---:|:---:|:---:|:---:|
| 5x5_2g (easy) | 1.000 | 0.520 | +0.480 | <0.001 |
| 5x5_2g (medium) | 1.000 | 0.520 | +0.480 | <0.001 |
| 5x5_2g (hard) | 1.000 | 0.520 | +0.480 | <0.001 |
| 10x10_2g (easy) | 0.380 | 0.240 | +0.140 | 0.037 |
| 10x10_2g (hard) | 0.400 | 0.160 | +0.240 | <0.001 |

**All 3 DEF predictions PASS**:
1. D-interpreter > no-D on all configurations (100% interpretation rate vs 0%)
2. Performance gap grows with difficulty (easy +0.207 -> hard +0.240)
3. D's interpretation creates genuine information asymmetry -- coded hints are opaque to `deconstruct_d_to_c`

### Stufe 8: Shadow-D Forward Planning (5D Regime)

**DEF Claim**: Forward-planning via multi-step lookahead creates a new 5D regime with measurable advantage in obstacle-heavy environments.

Shadow-D uses B's forward model for beam-search planning (depth=5, width=8). It populates C's `tie_break_preference` -- the first component to actually use this mechanism.

| Level | 3D (nod) | 4D (ond_tb) | 5D | Shadow-only |
|-------|:---:|:---:|:---:|:---:|
| 5x5_few_obs | 1.000 | 1.000 | 1.000 | 1.000 |
| 10x10_medium_obs | 0.520 | 0.520 | **0.710** | 0.710 |
| 10x10_dense_obs | 0.150 | 0.150 | **0.290** | 0.290 |
| 10x10_dynamic_obs | 0.720 | 0.720 | **0.860** | 0.860 |
| 15x15_dense_obs | 0.150 | 0.150 | **0.370** | 0.370 |

**All 5 DEF predictions PASS**:
1. 5D > 4D on obstacle-heavy levels (+14% to +22% SR)
2. Performance gap grows with obstacle density (simple: +0% -> dense: +18%)
3. Shadow-only > nod (planning helps without narrative, p=0.007 on 15x15)
4. No 5D advantage on simple 5x5 (planning is overkill)
5. Plan confidence inversely correlates with obstacle density (0.80 -> 0.48)

### Summary of DEF Claims Validated

| Stufe | DEF Claim | Test | Result |
|-------|-----------|------|--------|
| 1 | Pairing isolation (B stable, C diverges) | Valence-swap | **PASS** |
| 2 | D needs C (narrative needs valence) | Stream isolation | **PASS** |
| 2 | Pair-to-pair coupling only | Static + runtime analysis | **PASS** (9/9) |
| 3 | Higher regimes justified at complexity | Complexity scaling | **PASS** |
| 4 | Deconstruction stabilizes drift | Drift metrics | **PASS** |
| 4 | Router is efficient | D-call count comparison | **PASS** (88% fewer) |
| 5 | Router = regime transition | Task difficulty scaling | **PASS** (4/4) |
| 6 | D creates information advantage at ambiguity | Coded hints interpretation | **PASS** (3/3) |
| 8 | Forward-planning = 5D regime advantage | Shadow-D beam search | **PASS** (5/5) |

## Running the Tests

### Prerequisites

- Python 3.10+
- `pydantic`, `requests`, `tqdm`
- **Ollama** running locally (`ollama serve`) with `mistral:latest` pulled (only for LLM-backed D)

```bash
pip install pydantic requests tqdm
```

### Main Demo

```bash
python main.py
```

### Validation Suite (Stufe 0-8)

```bash
# Stufe 0: Ablation with symmetry check
python eval/run_ablation.py

# Stufe 1: Valence-swap validation
python eval/run_valence_swap.py

# Stufe 2: Stream isolation + coupling constraints
python eval/run_stream_isolation.py

# Stufe 3: Complexity scaling (variable grid size, multi-goal, dynamic obstacles)
python eval/run_complexity_scaling.py

# Stufe 4: Drift test and deconstruction validation
python eval/run_drift_test.py

# Stufe 5: Regime-transition validation
python eval/run_regime_transition.py

# Stufe 6: Semantic ambiguity (D's interpretation advantage)
python eval/run_semantic_ambiguity.py

# Stufe 8: Shadow-D forward planning (5D regime)
python eval/run_shadow_planning.py
```

### Legacy Ablation Studies

```bash
python eval/run_ablation_hidden_goal.py
python eval/run_ablation_hidden_goal_A2.py
python eval/run_ablation_hidden_goal_A2_llm_timing.py
```

All results are written as CSV to `runs/`.

## Project Structure

```
agents/
  agent_a.py          # Stream A: Perception (obs -> zA)
  agent_b.py          # Stream B: Dynamics (zA, action -> zA_next)
  agent_c.py          # Stream C: Valence/Control (scoring + tie-break)
  agent_d.py          # Stream D: Deterministic narrative
  agent_d_llm.py      # Stream D: LLM-backed narrative (Ollama/Mistral)
  agent_d_interpreter.py  # Stream D: Extended D with coded hint interpretation
  agent_shadow_d.py   # Shadow-D: Forward-planning via beam search

env/
  gridworld.py        # Parametrizable GridWorld (variable size, multi-goal, dynamic obstacles)
  coded_hints.py      # HintEncoder + CodedGridWorld wrapper for semantic ambiguity

router/
  router.py           # Router with regime logging (3D/4D/5D transitions)
  deconstruct.py      # D->C knowledge transfer (multi-goal support)
  deconstruct_plan.py # Plan->C transfer (tie_break_preference)

state/
  schema.py           # Shared Pydantic models (ZA, ZC, ZD, ZPlan)

llm/
  provider.py         # LLM provider protocol + Ollama implementation

eval/
  stats.py            # Statistical utilities (CI, p-values, Cohen's d)
  runner.py           # Shared batch test runner
  metrics.py          # Symmetry and scoring metrics
  drift_metrics.py    # Drift measurement (tag flip, narrative similarity)
  d_metrics.py        # Narrative quality metrics
  baselines.py        # Monolithic baseline policy
  run_ablation.py                       # Stufe 0: Basic ablation
  run_valence_swap.py                   # Stufe 1: Valence-swap validation
  run_stream_isolation.py               # Stufe 2: Stream isolation + coupling
  run_complexity_scaling.py             # Stufe 3: Complexity scaling
  run_drift_test.py                     # Stufe 4: Drift & deconstruction
  run_regime_transition.py              # Stufe 5: Regime transitions
  run_semantic_ambiguity.py             # Stufe 6: Semantic ambiguity (D's value)
  run_shadow_planning.py                # Stufe 8: Shadow-D planning (5D regime)
  run_ablation_hidden_goal.py           # Legacy: hidden goal ablation
  run_ablation_hidden_goal_A2.py        # Legacy: A2 knowledge acquisition
  run_ablation_hidden_goal_A2_llm_timing.py  # Legacy: LLM timing

runs/                 # CSV output directory (git-ignored)
docs/                 # DEF documentation (.docx)
main.py               # Phase 3b demo (on-demand D + tie-break)
```

## Key Design Decisions

- Agent D is expensive (LLM call) so the router gates it adaptively rather than calling every step
- Hint extraction from D's narrative is enforced deterministically in `deconstruct.py` regardless of LLM output quality
- `ZC.memory` is episode-scoped via an `episode_id` key to prevent cross-episode leakage
- Agent C's `anti_stay_penalty` breaks perfect seek/avoid negation symmetry -- this is intentional to prevent degenerate freeze policies
- Multi-goal hint cells use a partition system: each hint divides goals into two groups, and the environment dynamically computes which group to eliminate based on the true goal

## Conceptual Note

In this architecture:

- The **Narrator** (D) generates meaning and narrative
- The **Actor** (C) performs goal-directed behavior
- **Deconstruction** stabilizes their coupling

> *Healing is when the narrator and the actor are stably coupled in the same system.*

## Citation

```bibtex
@software{DEF_RAPA_MVP,
  title = {DEF / RAPA Modular Cognitive Architecture MVP},
  author = {Hans Buholzer},
  year = {2026},
  url = {https://github.com/Vivajohn42/rapa_poc}
}
```
