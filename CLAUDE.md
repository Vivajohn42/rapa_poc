# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAPA MVP (Recursive Agent Planning Architecture) is a research codebase implementing a hierarchical agent architecture for navigation in a parametrizable gridworld. The system decomposes reasoning into four specialized agents (A/B/C/D) with a router that gates expensive LLM calls based on decision uncertainty and behavioral stagnation. A 6-stage validation suite tests core claims of the Dimensional Emergence Framework (DEF).

## Commands

```bash
# Activate virtual environment (Windows)
.venv\Scripts\activate

# Run main demo (Phase 3b: on-demand D + tie-break, SEEK and AVOID episodes)
python main.py

# Run validation suite (Stufe 0-5)
python eval/run_ablation.py               # Stufe 0: Basic ablation + symmetry
python eval/run_valence_swap.py            # Stufe 1: Valence-swap validation
python eval/run_stream_isolation.py        # Stufe 2: Stream isolation + coupling constraints
python eval/run_complexity_scaling.py      # Stufe 3: Complexity scaling
python eval/run_drift_test.py              # Stufe 4: Drift & deconstruction
python eval/run_regime_transition.py       # Stufe 5: Regime transitions

# Legacy ablation studies
python eval/run_ablation_hidden_goal.py
python eval/run_ablation_hidden_goal_A2.py
python eval/run_ablation_hidden_goal_A2_llm_timing.py
```

No formal test suite, linter, or build system is configured. Validation is empirical via CSV results written to `runs/`.

## Prerequisites

- Python 3.10+ with `pydantic`, `requests`, `tqdm`
- **Ollama** running locally (`ollama serve`) with `mistral:latest` pulled, for LLM-backed Agent D

## Architecture

### Agent Hierarchy

The four agents form a perception->prediction->valuation->narrative pipeline:

- **Agent A** (`agents/agent_a.py`) -- Perception. Wraps raw gridworld observations into a typed `ZA` state.
- **Agent B** (`agents/agent_b.py`) -- Dynamics. Deterministic forward model predicting next state given action.
- **Agent C** (`agents/agent_c.py`) -- Valence/Goals. Scores actions via Manhattan distance in `seek` or `avoid` mode. Supports tie-breaking via persistent memory from D.
- **Agent D** (`agents/agent_d.py`, `agents/agent_d_llm.py`) -- Narrative. Meaning extraction from recent events. Has both deterministic and LLM-backed (Ollama/Mistral) implementations.

### Shared State Schemas (`state/schema.py`)

All inter-agent data flows through Pydantic models: `ZA` (observation), `ZC` (goal mode + persistent memory dict), `ZD` (narrative + meaning tags + grounding violations).

### Router (`router/router.py`)

Controls when Agent D is invoked. Triggers: uncertainty (top-2 action scores too close), stuck detection (repeated positions), periodic interval. Cooldown prevents thrashing. Logs regime (3D/4D) per step for analysis.

### Knowledge Transfer (`router/deconstruct.py`)

`deconstruct_d_to_c()` parses D's meaning tags into concrete target coordinates stored in C's memory. Supports multi-goal elimination via `not_x_y` tags. Accepts optional `goal_map` parameter for variable grid sizes.

### Environment (`env/gridworld.py`)

Parametrizable gridworld supporting variable size (5x5 to 15x15+), multiple candidate goals (2-N) with partition-based hint system, configurable static/random/dynamic obstacles. Default: 5x5, 2 goals, 1 hint cell, 1 obstacle.

### Evaluation (`eval/`)

- `stats.py` -- Statistical utilities (95% CI, Mann-Whitney U, Cohen's d)
- `runner.py` -- Shared batch test runner with CSV output and aggregation
- `metrics.py` -- Scoring and symmetry metrics
- `drift_metrics.py` -- Tag flip rate, narrative similarity, windowed stability
- Validation scripts: `run_valence_swap.py`, `run_stream_isolation.py`, `run_complexity_scaling.py`, `run_drift_test.py`, `run_regime_transition.py`

## Key Design Decisions

- Agent D is expensive (LLM call) so the router gates it adaptively rather than calling every step
- Hint extraction from D's narrative is enforced deterministically in `deconstruct.py` regardless of LLM output quality
- `ZC.memory` is episode-scoped via an `episode_id` key to prevent cross-episode leakage
- Agent C's `anti_stay_penalty` intentionally breaks perfect seek/avoid negation symmetry to prevent degenerate freeze policies
- Multi-goal hint cells use partition-based elimination: environment dynamically computes which goals to eliminate based on the true goal
- DEF pair-to-pair coupling is enforced: D never selects actions directly, output flows through Deconstruct only
