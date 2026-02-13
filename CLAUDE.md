# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAPA MVP (Recursive Agent Planning Architecture) is a research codebase implementing a hierarchical agent architecture for navigation in a 5×5 gridworld. The system decomposes reasoning into four specialized agents (A/B/C/D) with a router that gates expensive LLM calls based on decision uncertainty and behavioral stagnation.

## Commands

```bash
# Activate virtual environment (Windows)
.venv\Scripts\activate

# Run main demo (Phase 3b: on-demand D + tie-break, SEEK and AVOID episodes)
python main.py

# Run ablation studies
python eval/run_ablation.py
python eval/run_ablation_hidden_goal.py
python eval/run_ablation_hidden_goal_A2.py
python eval/run_ablation_hidden_goal_A2_llm_timing.py
```

No formal test suite, linter, or build system is configured. Validation is empirical via episode logs written to `runs/` as JSONL.

## Prerequisites

- Python 3.8+ with `pydantic`, `requests`, `openai`, `tqdm`
- **Ollama** running locally (`ollama serve`) with `mistral:latest` pulled, for LLM-backed Agent D

## Architecture

### Agent Hierarchy

The four agents form a perception→prediction→valuation→narrative pipeline:

- **Agent A** (`agents/agent_a.py`) — Perception. Wraps raw gridworld observations into a typed `ZA` state (position, goal, obstacles, hints).
- **Agent B** (`agents/agent_b.py`) — Dynamics. Deterministic forward model predicting next state given action. Used by C for look-ahead rollouts.
- **Agent C** (`agents/agent_c.py`) — Valence/Goals. Scores actions via Manhattan distance in `seek` or `avoid` mode. Supports tie-breaking via persistent memory from D.
- **Agent D** (`agents/agent_d_llm.py`) — Narrative. LLM-backed (Ollama/Mistral) meaning extraction from recent events. Has a deterministic fallback (`agents/agent_d.py`).

### Shared State Schemas (`state/schema.py`)

All inter-agent data flows through Pydantic models: `ZA` (observation), `ZC` (goal mode + persistent memory dict), `ZD` (narrative + meaning tags + grounding violations).

### Router (`router/router.py`)

Controls when Agent D is invoked. Triggers: uncertainty (top-2 action scores too close), stuck detection (repeated positions), periodic interval. Cooldown prevents thrashing.

### Knowledge Transfer (`router/deconstruct.py`)

`deconstruct_d_to_c()` parses D's meaning tags (e.g., `hint:a`, `hint:b`) into concrete target coordinates stored in C's memory. This is deterministic — LLM output is not fully trusted for hint extraction.

### Environment (`env/gridworld.py`)

5×5 grid. Agent starts at (0,0). Hidden true goal is randomly Goal A (4,4) or Goal B (4,0). Single hint cell at (0,4) reveals which goal is correct. One obstacle at (2,2). Rewards: +1 on goal reach, -0.01 per step.

### LLM Provider (`llm/provider.py`)

`LLMProvider` protocol with `OllamaProvider` implementation. Connects to `localhost:11434/api/chat`. Temperature 0.2 for near-deterministic output.

### Evaluation (`eval/`)

Ablation studies compare four variants: `baseline_mono` (monolithic), `modular_nod` (A+B+C only), `modular_ond` (with router-gated D), `modular_ond_tb` (with tie-break memory). Each runs across SEEK/AVOID modes. Results go to CSV in `runs/`.

## Key Design Decisions

- Agent D is expensive (LLM call) so the router gates it adaptively rather than calling every step
- Hint extraction from D's narrative is enforced deterministically in `deconstruct.py` regardless of LLM output quality
- `ZC.memory` is episode-scoped via an `episode_id` key to prevent cross-episode leakage
- Agent C's tie-break mechanism consults memory populated by D, demonstrating how higher-level reasoning influences lower-level control