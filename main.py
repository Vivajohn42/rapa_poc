import sys
from datetime import datetime, timezone
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).resolve().parent))

from env.gridworld import GridWorld

from agents.agent_a import AgentA
from agents.agent_b import AgentB
from agents.agent_c import AgentC, GoalSpec

from llm.provider import OllamaProvider
from agents.agent_d_llm import AgentDLLM

from router.deconstruct import deconstruct_d_to_c
from router.router import Router, RouterConfig

from state.schema import ZC

from eval.logger import JSONLLogger
from eval.d_metrics import narrative_metrics


def run_episode_with_on_demand_d(goal_mode: str, run_id: str, log_path: str, max_steps: int = 50):
    env = GridWorld()
    obs = env.reset()

    A = AgentA()
    B = AgentB()

    zA0 = A.infer_zA(obs)

    # --- C stream state (per-episode, hard reset) ---
    zC = ZC(goal_mode=goal_mode, memory={})
    zC.memory.clear()

    # Episode-scoped safety marker to avoid memory leaks across episodes
    episode_id = f"{run_id}_{goal_mode}"
    zC.memory["episode_id"] = episode_id

    hint_seen = False
    hint_learned = False

    # Metrics for clarity: router triggers vs hint captures
    d_triggers = 0
    d_hint_captures = 0

    # Default target (used only if no hint learned and no hint-cell acquisition)
    default_target = (zA0.width - 1, zA0.height - 1)

    C = AgentC(goal=GoalSpec(mode=goal_mode, target=default_target), anti_stay_penalty=1.1)

    # LLM-backed D (Ollama)
    D = AgentDLLM(OllamaProvider(model="mistral:latest"))

    # Router: uncertainty-driven + stuck fallback + cooldown
    router = Router(RouterConfig(
        d_every_k_steps=0,
        stuck_window=4,
        enable_stuck_trigger=True,
        enable_uncertainty_trigger=True,
        uncertainty_threshold=0.25,
        d_cooldown_steps=8
    ))

    logger = JSONLLogger(log_path)

    total_reward = 0.0
    done = False
    stay_count = 0

    last_positions = deque(maxlen=20)

    t = -1
    for t in range(max_steps):
        zA = A.infer_zA(obs)

        # ----- A2 Knowledge acquisition: before goal known, go to hint cell -----
        if "target" not in zC.memory and hasattr(env, "hint_cell"):
            C.goal.target = env.hint_cell
        else:
            # If goal learned, use it; otherwise default
            if "target" in zC.memory and zC.memory.get("episode_id") == episode_id:
                C.goal.target = tuple(zC.memory["target"])
            else:
                C.goal.target = default_target

        # ----- If hint appears, capture it immediately via D -> Deconstruct -----
        if zA.hint in ("A", "B"):
            hint_seen = True
            d_hint_captures += 1

            D.observe_step(t=t, zA=zA, action="hint", reward=0.0, done=False)
            zD_hint = D.build_micro(goal_mode=goal_mode, goal_pos=(-1, -1), last_n=1)
            zC = deconstruct_d_to_c(zC, zD_hint)

            # lock memory to this episode
            zC.memory["episode_id"] = episode_id

            hint_learned = ("target" in zC.memory) and (zC.memory.get("episode_id") == episode_id)

            logger.log({
                "run_id": run_id,
                "goal_mode": goal_mode,
                "t": t,
                "hint_event": True,
                "hint_value": zA.hint,
                "zD_hint": zD_hint,
                "zC_memory": zC.memory
            })

        # ----- Choose action (tie-break can use zC.memory) -----
        action, scored = C.choose_action(
            zA,
            B.predict_next,
            memory=zC.memory,
            tie_break_delta=0.25
        )
        decision_delta = scored[0][1] - scored[1][1]

        obs_next, reward, done = env.step(action)
        zA_next = A.infer_zA(obs_next)

        if zA_next.agent_pos == zA.agent_pos:
            stay_count += 1

        last_positions.append(zA_next.agent_pos)

        # Always let D observe (cheap)
        D.observe_step(t=t, zA=zA_next, action=action, reward=reward, done=done)

        total_reward += reward
        obs = obs_next

        # ----- Router gating for on-demand D -----
        activate_d, reason = router.should_activate_d(
            t=t,
            last_positions=tuple(last_positions),
            decision_delta=decision_delta
        )

        if activate_d:
            d_triggers += 1

            zD_micro = D.build_micro(goal_mode=goal_mode, goal_pos=(-1, -1), last_n=5)
            d_m = narrative_metrics(zD_micro)

            zC = deconstruct_d_to_c(zC, zD_micro)
            zC.memory["episode_id"] = episode_id  # keep episode binding

            hint_learned = ("target" in zC.memory) and (zC.memory.get("episode_id") == episode_id)

            logger.log({
                "run_id": run_id,
                "goal_mode": goal_mode,
                "t": t,
                "d_event": True,
                "d_reason": reason,
                "decision_delta": decision_delta,
                "zD": zD_micro,
                "d_metrics": d_m,
                "zC_memory": zC.memory
            })

        if done:
            break

    steps = (t + 1) if t >= 0 else 0

    # ===== End-of-episode D (full narrative) =====
    zD_final = D.build(goal_mode=goal_mode, goal_pos=(-1, -1))
    d_final_m = narrative_metrics(zD_final)

    zC = deconstruct_d_to_c(zC, zD_final)
    zC.memory["episode_id"] = episode_id

    hint_learned = ("target" in zC.memory) and (zC.memory.get("episode_id") == episode_id)

    logger.log({
        "run_id": run_id,
        "goal_mode": goal_mode,
        "episode_summary": True,
        "success": bool(done),
        "steps": steps,
        "total_reward": total_reward,
        "d_triggers": d_triggers,
        "d_hint_captures": d_hint_captures,
        "stay_count": stay_count,
        "stay_rate": (stay_count / steps) if steps > 0 else 0.0,
        "router_cfg": router.cfg.__dict__,
        "zD_final": zD_final,
        "d_final_metrics": d_final_m,
        "zC_memory_final": zC.memory,
        "hint_seen": hint_seen,
        "hint_learned": hint_learned,
        "learned_hint_goal": zC.memory.get("hint_goal"),
        "memory_episode_id": zC.memory.get("episode_id"),
    })

    return {
        "goal_mode": goal_mode,
        "success": bool(done),
        "steps": steps,
        "total_reward": total_reward,
        "d_triggers": d_triggers,
        "d_hint_captures": d_hint_captures,
        "stay_count": stay_count,
        "stay_rate": (stay_count / steps) if steps > 0 else 0.0,
        "d_final_metrics": d_final_m,
        "hint_seen": hint_seen,
        "hint_learned": hint_learned,
        "learned_hint_goal": zC.memory.get("hint_goal"),
    }


def run_phase3b_demo():
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    Path("runs").mkdir(exist_ok=True)

    log_seek = f"runs/{run_id}_seek_onD_tiebreak.jsonl"
    log_avoid = f"runs/{run_id}_avoid_onD_tiebreak.jsonl"

    print("\n=== Running SEEK episode with on-demand D + tie-break ===")
    r_seek = run_episode_with_on_demand_d("seek", run_id, log_seek)

    print("\n=== Running AVOID episode with on-demand D + tie-break ===")
    r_avoid = run_episode_with_on_demand_d("avoid", run_id, log_avoid)

    print("\n=== Phase-3b on-demand D + tie-break Results ===")
    print("SEEK :", r_seek)
    print("AVOID:", r_avoid)

    print("\nLogs written to:")
    print(" ", log_seek)
    print(" ", log_avoid)


if __name__ == "__main__":
    run_phase3b_demo()
