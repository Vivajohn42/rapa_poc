# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAPA MVP (Recursive Agent Planning Architecture) is a research codebase implementing a hierarchical agent architecture across three environments (GridWorld, TextWorld, Riddle Rooms). The system decomposes reasoning into specialized agents (A/B/C/D) that inherit from abstract interfaces (`kernel/interfaces.py`), with a governance kernel (`MvpKernel`) that enforces ABI constraints, coupling schedules, and closure invariants — ported from rapa_os but running in-process without ZMQ. Includes a B→C planning extension for multi-step lookahead. A 10-stage validation suite (Stufe 0-9) plus a 3-environment stability matrix tests core claims of the Dimensional Emergence Framework (DEF).

## Commands

```bash
# Activate virtual environment (Windows)
.venv\Scripts\activate

# Run main demo (Phase 3b: on-demand D + tie-break, SEEK and AVOID episodes)
python main.py

# Run validation suite (Stufe 0-9)
python eval/run_ablation.py               # Stufe 0: Basic ablation + symmetry
python eval/run_valence_swap.py            # Stufe 1: Valence-swap validation
python eval/run_stream_isolation.py        # Stufe 2: Stream isolation + coupling constraints
python eval/run_complexity_scaling.py      # Stufe 3: Complexity scaling
python eval/run_drift_test.py              # Stufe 4: Drift & deconstruction
python eval/run_regime_transition.py       # Stufe 5: Regime transitions
python eval/run_semantic_ambiguity.py      # Stufe 6: Semantic ambiguity (D's value)
python eval/run_planning_horizon.py         # Stufe 8: B→C planning horizon extension
python eval/run_task_change.py              # Stufe 9: Task-change stabilization (deconstruct)

# Stufe 6-LLM: Architecture robustness with stochastic narrative (requires Ollama)
python -m eval.run_llm_semantic_ambiguity                          # all available models
python -m eval.run_llm_semantic_ambiguity --model mistral:latest   # single model
python -m eval.run_llm_semantic_ambiguity --n 5 --max-steps 50     # quick test

# Stufe 7: LLM-D multi-model validation (requires Ollama)
python -m eval.run_llm_drift                          # 7a: all available models
python -m eval.run_llm_drift --model mistral:latest   # 7a: single model
python -m eval.run_llm_regime                         # 7b: all available models
python -m eval.run_llm_regime --model phi3:mini       # 7b: single model

# Kernel governance tests
python eval/run_kernel_smoke.py              # Kernel smoke test (50 episodes)
python eval/run_kernel_loop_gain.py          # Loop gain G/F convergence
python eval/run_kernel_jung.py               # Jung personality profiles
python eval/run_kernel_llm_loop_gain.py      # LLM-D loop gain (requires Ollama)
python eval/run_residuum_analysis.py         # Closure Residuum (Delta_8) analysis

# TextWorld: D-essentiality validation
python eval/run_textworld_ablation.py                  # D-ablation (with_d vs no_d vs random)
python eval/run_textworld_ablation.py --llm             # Include LLM-D variant (requires Ollama)
python eval/run_textworld_ablation.py --scenario 0      # Single scenario
python eval/run_textworld_loop_gain.py                  # Persistence Theorem on TextWorld

# Cross-environment stability matrix (GridWorld + TextWorld + Riddle Rooms)
python eval/run_stability_matrix.py          # 3-env matrix, 8 assertions
python eval/run_stability_matrix.py --n 30   # Custom episode count

# Legacy ablation studies
python eval/run_ablation_hidden_goal.py
python eval/run_ablation_hidden_goal_A2.py
python eval/run_ablation_hidden_goal_A2_llm_timing.py
```

No formal test suite, linter, or build system is configured. Validation is empirical via CSV results written to `runs/`.

## Prerequisites

- Python 3.10+ with `pydantic`, `requests`, `tqdm`
- **Ollama** running locally (`ollama serve`) for LLM-backed Agent D
- Supported models: `phi3:mini` (3.8B), `mistral:latest` (7B), `qwen2.5:3b` (3B), `gemma2:2b` (2B)

## Architecture

### Stream Interfaces (`kernel/interfaces.py`)

All domain-specific agents inherit from abstract base classes. The kernel depends only on these interfaces:

- **StreamA** -- `infer_zA(obs) -> ZA`
- **StreamB** -- `predict_next(zA, action) -> ZA`
- **StreamC** -- `choose_action(zA, predict_next_fn, memory, tie_break_delta) -> (action, scored)` + `goal` property (GoalTarget Protocol)
- **StreamD** -- `observe_step(t, zA, action, reward, done)`, `build(goal_mode) -> ZD`, `build_micro(goal_mode) -> ZD`
- **EnvironmentAdapter** -- eval-script convenience class (kernel does NOT depend on it)

### Agent Hierarchy

The agents form a perception->prediction->valuation->narrative+planning pipeline:

- **Agent A** (`agents/agent_a.py`, extends StreamA) -- Perception. Wraps raw gridworld observations into a typed `ZA` state.
- **Agent B** (`agents/agent_b.py`, extends StreamB) -- Dynamics. Deterministic forward model predicting next state given action.
- **Agent C** (`agents/agent_c.py`, extends StreamC) -- Valence/Goals. Scores actions via Manhattan distance in `seek` or `avoid` mode. Supports tie-breaking via persistent memory from D and PlannerBC.
- **Agent D** (`agents/agent_d.py`, `agents/agent_d_llm.py`, extends StreamD) -- Narrative. Meaning extraction from recent events. Has both deterministic and LLM-backed (Ollama/Mistral) implementations.
- **Agent D-Interpreter** (`agents/agent_d_interpreter.py`) -- Extended D with coded hint interpretation capability.
- **Agent D-LLM-Interpreter** (`agents/agent_d_llm_interpreter.py`) -- LLM narrative + deterministic HintEncoder for coded hint interpretation.
- **PlannerBC** (`agents/planner_bc.py`) -- B→C planning extension. Uses B's forward model for multi-step beam-search lookahead, feeding results into C's tie_break_preference. Not a new D-agent — deepens the existing B↔C coupling.

### Shared State Schemas (`state/schema.py`)

All inter-agent data flows through Pydantic models: `ZA` (observation), `ZC` (goal mode + persistent memory dict), `ZD` (narrative + meaning tags + grounding violations), `ZPlan` (planning output with recommended actions + confidence).

### Router (`router/router.py`)

Controls when Agent D and PlannerBC are invoked. Triggers: uncertainty (top-2 action scores too close), stuck detection (repeated positions), periodic interval. Cooldown prevents thrashing. Logs regime (3D/3D+/4D) per step for analysis.

### Knowledge Transfer (`router/deconstruct.py`, `router/deconstruct_plan.py`)

`deconstruct_d_to_c()` parses D's meaning tags into concrete target coordinates stored in C's memory. Supports multi-goal elimination via `not_x_y` tags. Accepts optional `goal_map` parameter for variable grid sizes.

`deconstruct_plan_to_c()` translates PlannerBC's planning output into C's `tie_break_preference` for action selection during ties.

### MvpKernel (`kernel/`)

In-process governance layer ported from rapa_os. Enforces the same ABI constraints, coupling schedules, and closure invariants without ZMQ — agents are called directly via Python method calls.

- **`interfaces.py`** -- Abstract base classes: StreamA, StreamB, StreamC, StreamD, EnvironmentAdapter, GoalTarget Protocol. All domain agents inherit from these.
- **`kernel.py`** -- `MvpKernel` orchestrator. Type-hinted against StreamA/B/C/D. Tick lifecycle: A→Signals→Route→Schedule→ABI→ClosureCore→C→D(out-of-band)→Deconstruct→LoopGain→Result. Accepts optional `jung_profile` and `fallback_actions`.
- **`types.py`** -- `MvpTickSignals`, `MvpKernelDecision`, `MvpLoopGain`, `MvpTickResult`, `ResidualSnapshot`.
- **`abi.py`** -- `enforce_constraints()`: AB always active, gC/gD gates remove couplings, max 1 extra coupling beyond AB.
- **`closure_core.py`** -- `ClosureCore.validate_decision()`: 4 assertion checks per tick. D cannot write to L3 directly.
- **`closure_residuum.py`** -- `ClosureResiduum`: Delta_8 = lambda_1 * c_term + lambda_2 * d_term. Dynamic thresholds derived from Jung profile I/E and S/N weights and schedule topology. EMA smoothing with adaptive alpha.
- **`scheduler.py`** -- `schedule_for()` with deterministic template rotation: 4FoM=[AB], 6FoM=[AB+BC, AB+AC], 8FoM=[AB+CD, AB+BC, AB+AD]. Supports M4 priority_coupling override.
- **`memory_manager.py`** -- `MvpMemoryManager`: L3 persistent memory (c_long, b_priors, semantic_index). D→C→B deconstruction pipeline. Capped at 100 entries.
- **`loop_gain.py`** -- `MvpLoopGainTracker`: G = g_BA * g_CB * g_DC * g_AD. g_AD = 1.0 for deterministic D; for LLM D, validates grounding (hint consistency weight=1.0, position weight=0.5, goal_mode weight=0.5, hallucinated tags weight=0.25). DETERMINISTIC_NARRATIVE_PREFIXES for LLM detection across all 3 environments.
- **`jung_profiles.py`** -- `JungProfile` with I/E, S/N, T/F axes modulating cooldown, stuck_window, tie_break_delta, deconstruct_cooldown. Pre-defined: SENSOR, INTUITIVE, ANALYST, DEFAULT.
- **`state_bridge.py`** -- Pure functions converting Pydantic models (ZA/ZC/ZD/ZPlan) to rapa_os-compatible z-dicts.

### Environments

Three environments with EnvironmentAdapter wrappers for eval scripts:

**GridWorld** (`env/gridworld.py`, `env/gridworld_adapter.py`): Parametrizable gridworld supporting variable size (5x5 to 15x15+), multiple candidate goals (2-N) with partition-based hint system, configurable static/random/dynamic obstacles. Default: 5x5, 2 goals, 1 hint cell, 1 obstacle. `TaskChangeGridWorld` wraps `GridWorld` with a mid-episode goal switch for Stufe 9 testing.

**TextWorld** (`env/textworld.py`, `env/textworld_adapter.py`): Text-based "Clue Rooms" environment where D is architecturally essential. A network of named rooms connected by exits. Clue fragments are scattered across rooms; no single clue identifies the target — only multi-clue synthesis (constraint propagation) reveals it. Agent must explicitly `"claim"` in the target room to succeed. 5 hand-crafted scenarios (4-8 rooms, 2-3 clues each). Results: with_d=100% SR, no_d=0% SR, LLM-D=52% SR.

**Riddle Rooms** (`env/riddle_rooms.py`, `env/riddle_adapter.py`): Non-spatial propositional logic puzzles. No navigation graph — state transitions are logical (test/submit). 5 hand-crafted puzzles (Liar Boxes, Alibi Check, Sequence Rule, Schedule Puzzle, Inference Chain). D is essential for constraint synthesis. Results: with_d=100% SR, no_d=0% SR. Proves the architecture is not bound to spatial domains.

### TextWorld Agents

Domain-specific agents for TextWorld, all inheriting from Stream interfaces:

- **TextAgentA** (`agents/text_agent_a.py`, extends StreamA) -- Parses room observations into ZA using pseudo-positions `(room_index, 0)`.
- **TextAgentB** (`agents/text_agent_b.py`, extends StreamB) -- Graph-based forward model using room exit lookup tables.
- **TextAgentC** (`agents/text_agent_c.py`, extends StreamC) -- BFS graph-distance scoring. Claims when at target room. Explores toward nearest unvisited room when no target set. Has `_TextGoalProxy` for kernel compatibility.
- **TextAgentD** (`agents/text_agent_d.py`, extends StreamD) -- Deterministic clue synthesizer via constraint propagation. Extracts required/negated properties from natural language clues, eliminates candidate rooms until one remains.
- **TextAgentDLLM** (`agents/text_agent_d_llm.py`, extends StreamD) -- LLM-backed clue synthesis. Prompts LLM with room properties + collected clues to identify target. Validates grounding (hallucinated room detection).
- **`router/deconstruct_text.py`** -- Text-specific D→C pipeline: maps `"target:room_id"` tag to pseudo-position in C's memory.

### Riddle Rooms Agents

Non-spatial domain agents, all inheriting from Stream interfaces:

- **RiddleAgentA** (`agents/riddle_agent_a.py`, extends StreamA) -- Maps evidence hash to pseudo-position `(hash % n_answers, 0)`.
- **RiddleAgentB** (`agents/riddle_agent_b.py`, extends StreamB) -- Forward model: test actions shift position +1 mod n_answers; submit actions stay.
- **RiddleAgentC** (`agents/riddle_agent_c.py`, extends StreamC) -- Scores tests by information value (0.5 if unrevealed), scores submits by target match (2.0 if correct answer known). Has `_RiddleGoalProxy`.
- **RiddleAgentD** (`agents/riddle_agent_d.py`, extends StreamD) -- Constraint propagation via `clue_eliminates` mapping. Tags: `answer:`, `target:`, `evidence:`, `candidates:`, `eliminated:`.
- **`router/deconstruct_riddle.py`** -- Riddle-specific D→C pipeline: maps answer/target/eliminated/evidence/candidates tags to C's memory.

### Evaluation (`eval/`)

- `stats.py` -- Statistical utilities (95% CI, Mann-Whitney U, Cohen's d)
- `runner.py` -- Shared batch test runner with CSV output and aggregation
- `metrics.py` -- Scoring and symmetry metrics
- `drift_metrics.py` -- Tag flip rate, narrative similarity, windowed stability
- `llm_utils.py` -- Ollama availability checks, model discovery, timed LLM calls
- Validation scripts: `run_valence_swap.py`, `run_stream_isolation.py`, `run_complexity_scaling.py`, `run_drift_test.py`, `run_regime_transition.py`, `run_semantic_ambiguity.py`, `run_llm_semantic_ambiguity.py`, `run_planning_horizon.py`, `run_llm_drift.py`, `run_llm_regime.py`, `run_task_change.py`
- Kernel governance tests: `run_kernel_smoke.py` (MvpKernel tick lifecycle), `run_kernel_loop_gain.py` (G/F convergence, weakest coupling), `run_kernel_jung.py` (Jung profile behavioral diff), `run_kernel_llm_loop_gain.py` (LLM-D loop gain validation), `run_residuum_analysis.py` (Closure Residuum Delta_8 analysis)
- TextWorld D-essentiality: `run_textworld_ablation.py` (D-ablation with_d/no_d/random/llm), `run_textworld_loop_gain.py` (Persistence Theorem: g_DC progression, G/F collapse)
- Cross-environment: `run_stability_matrix.py` (3-environment stability matrix: GridWorld + TextWorld + Riddle Rooms, 8 assertions)

## Key Design Decisions

- Agent D is expensive (LLM call) so the router gates it adaptively rather than calling every step
- Hint extraction from D's narrative is enforced deterministically in `deconstruct.py` regardless of LLM output quality
- `ZC.memory` is episode-scoped via an `episode_id` key to prevent cross-episode leakage
- Agent C's `anti_stay_penalty` intentionally breaks perfect seek/avoid negation symmetry to prevent degenerate freeze policies
- Multi-goal hint cells use partition-based elimination: environment dynamically computes which goals to eliminate based on the true goal
- DEF pair-to-pair coupling is enforced: D never selects actions directly, output flows through Deconstruct only
- MvpKernel governance rules are transport-independent — same ABI as rapa_os but via in-process Python calls instead of ZMQ
- D runs out-of-band (6FoM+D overlay): D is never in the coupling schedule, communicates only via deconstruction
- Loop gain g_AD distinguishes deterministic D (always 1.0) from LLM D (grounding checks with weighted violations)
- Jung profiles modulate kernel parameters (cooldown, stuck_window, tie_break_delta) without changing agent implementations
- L3 memory persists across episodes; per-episode state is reset via `kernel.reset_episode()`
- All agents inherit from abstract interfaces in `kernel/interfaces.py`. Kernel depends only on StreamA/B/C/D, never on domain-specific implementations
- GoalTarget uses `typing.Protocol` (structural subtyping): GoalSpec and _TextGoalProxy satisfy it without inheritance
- EnvironmentAdapter standardizes eval scripts but kernel does NOT depend on it
- ZA is universal: width/height = state-space dimensions, agent_pos = current state, obstacles = blocked states, hint = external info channel
- TextWorld uses pseudo-positions `(room_index, 0)` to satisfy ZA interface without kernel changes
- Riddle Rooms maps evidence states to `(hash % n_answers, 0)` — no spatial navigation at all
- TextWorld requires explicit `"claim"` action to succeed — prevents accidental success without D's target identification
- Closure Residuum (Delta_8) = lambda_1 * c_term + lambda_2 * d_term with dynamic thresholds from Jung profiles
- `MvpMemoryManager` accepts optional `deconstruct_fn` for domain-specific D→C pipelines (default: GridWorld's `deconstruct_d_to_c`)
- `loop_gain.py` uses dynamic actions from `scored` list (not hardcoded), and `has_agent_d` flag to correctly decay g_DC/g_AD when D is absent
- g_DC detects target-in-memory via hint-capture path (not only via gD=1 route), ensuring accurate coupling measurement
- Kernel accepts optional `fallback_actions` for domain-agnostic AB-only action selection
