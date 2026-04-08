"""DoorKey Reasoning Demo — Reasoning Chip in a real MiniGrid environment.

Proves the Reasoning Chip works on DoorKey (3-phase subgoals: Key → Door → Goal).
The kernel injects DoorKey state into Canvas, the 200M model reasons about it,
and Agent C navigates based on D's ANSWER.

Usage:
    cd rapa_mvp
    python demo_doorkey_reasoning.py \
        --checkpoint ../runs/reasoning_v4/best_model_sft.pt \
        --config D:/Downloads/E.2KVCanvas/config.yaml
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import GPT2TokenizerFast

from env.doorkey import DoorKeyEnv, DOOR_OPEN
from agents.doorkey_agent_a import DoorKeyAgentA
from agents.doorkey_agent_b import DoorKeyAgentB
from agents.doorkey_agent_c import DoorKeyAgentC
from agents.agent_d_llm import AgentDLLM
from kernel.kernel import MvpKernel
from kernel.kernel_canvas_manager import KernelCanvasManager
from llm.def_provider import DEFProvider
from llm.output_parser import parse_reasoning_output
from router.deconstruct_doorkey import deconstruct_doorkey_d_to_c
from def_transformer.model.config import DEFTransformerConfig
from def_transformer.model.utils import build_model


BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"
RESET = "\033[0m"


def update_doorkey_canvas(canvas: KernelCanvasManager, obs) -> None:
    """Write DoorKey-specific state into Canvas."""
    if hasattr(obs, "agent_pos"):
        canvas.write("agent_pos", str(obs.agent_pos))
    if hasattr(obs, "agent_dir"):
        dirs = {0: "right", 1: "down", 2: "left", 3: "up"}
        canvas.write("agent_dir", dirs.get(obs.agent_dir, str(obs.agent_dir)))

    # Phase
    phase = getattr(obs, "phase", None)
    if phase:
        canvas.write("phase", phase)

    # Key
    key_pos = getattr(obs, "key_pos", None)
    if key_pos:
        canvas.write("key_at", str(key_pos))
    carrying = getattr(obs, "carrying_key", False)
    canvas.write("carrying_key", "yes" if carrying else "no")

    # Door
    door_pos = getattr(obs, "door_pos", None)
    if door_pos:
        canvas.write("door_at", str(door_pos))
    door_state = getattr(obs, "door_state", None)
    if door_state is not None:
        canvas.write("door_state", "open" if door_state == DOOR_OPEN else "locked")

    # Goal (always at size-2, size-2)
    goal_pos = getattr(obs, "goal_pos", None)
    if goal_pos:
        canvas.write("goal_at", str(goal_pos))


def main():
    parser = argparse.ArgumentParser(description="DoorKey Reasoning Demo")
    parser.add_argument("--checkpoint", type=str,
                        default="../runs/reasoning_v4/best_model_sft.pt")
    parser.add_argument("--config", type=str,
                        default="D:/Downloads/E.2KVCanvas/config.yaml")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"  {BOLD}{CYAN}DoorKey Reasoning Demo{RESET}")
    print(f"  {DIM}Reasoning Chip + RAPA-OS Kernel on MiniGrid DoorKey-{args.size}x{args.size}{RESET}")
    print(f"  {DIM}Device: {device}{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    # Load model
    print(f"\n{BLUE}Loading DEF model...{RESET}")
    config = DEFTransformerConfig.from_yaml(args.config)
    try:
        import flash_attn
    except ImportError:
        config.use_flash_attn = False
    model = build_model(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model.eval()
    tokenizer = GPT2TokenizerFast.from_pretrained(config.tokenizer_name)
    print(f"  {GREEN}Loaded{RESET} ({sum(p.numel() for p in model.parameters()):,} params)")

    results = []
    for ep in range(args.episodes):
        canvas = KernelCanvasManager(n_slots=10)
        llm = DEFProvider(model=model, tokenizer=tokenizer, device=device,
                          canvas_manager=canvas, forced_prefix=None)

        env = DoorKeyEnv(size=args.size, seed=args.seed + ep, max_steps=args.max_steps)
        obs = env.reset()

        A = DoorKeyAgentA()
        B = DoorKeyAgentB(door_pos=obs.door_pos, door_open=False)
        C = DoorKeyAgentC(goal_mode="seek")
        D = AgentDLLM(llm)

        kernel = MvpKernel(
            agent_a=A, agent_b=B, agent_c=C, agent_d=D,
            deconstruct_fn=deconstruct_doorkey_d_to_c,
            canvas_manager=canvas,
            fallback_actions=env.available_actions,
        )
        kernel.reset_episode(goal_mode="seek", episode_id=f"dk_ep{ep:03d}")

        print(f"\n{BOLD}{'='*60}{RESET}")
        print(f"  {BOLD}Episode {ep+1}/{args.episodes}{RESET} (seed={args.seed + ep})")
        print(f"  Key: {obs.key_pos}, Door: {obs.door_pos}, Goal: {obs.goal_pos}")
        print(f"{'='*60}")

        solved = False
        total_reward = 0.0

        for t in range(args.max_steps):
            # Update Canvas + C/B metadata
            update_doorkey_canvas(canvas, obs)
            phase = getattr(obs, "phase", "?")

            # Sync C and B
            C.phase = getattr(obs, "phase", "FIND_KEY")
            C.key_pos = obs.key_pos
            C.door_pos = obs.door_pos
            C.carrying_key = getattr(obs, "carrying_key", False)
            C.door_open = (obs.door_state == DOOR_OPEN) if obs.door_state is not None else False
            B.update_door_state(obs.door_pos, C.door_open)

            t0 = time.perf_counter()
            result = kernel.tick(t, obs, done=False)
            dt = time.perf_counter() - t0
            action = result.action

            obs_next, reward, done = env.step(action)
            kernel.observe_reward(reward)
            total_reward += reward

            # Render
            pos = obs.agent_pos if hasattr(obs, "agent_pos") else "?"
            phase_color = CYAN if phase == "FIND_KEY" else (YELLOW if phase == "OPEN_DOOR" else GREEN)
            d_str = ""
            if result.d_activated and hasattr(llm, '_last_generated') and llm._last_generated:
                raw = tokenizer.decode(llm._last_generated, skip_special_tokens=True)
                reasoning, answer, _memo, _, fmt = parse_reasoning_output(raw)
                if reasoning:
                    d_str = f"\n     {BOLD}> REASONING:{RESET} {reasoning[:120]}"
                if answer:
                    d_str += f"\n     {GREEN}{BOLD}> ANSWER:{RESET} {answer[:100]}"

            regime = f"{CYAN}4D{RESET}" if result.d_activated else f"{DIM}3D{RESET}"
            print(f"  t={t:>2} {phase_color}{phase:<11}{RESET} pos={pos} "
                  f"act={action:<12} r={reward:+.1f} {regime} "
                  f"{DIM}({dt*1000:.0f}ms){RESET}{d_str}")

            obs = obs_next
            if done:
                solved = (reward > 0)
                kernel.tick(t + 1, obs, done=True)
                color = GREEN if solved else RED
                print(f"\n  {color}{BOLD}>>> {'SOLVED' if solved else 'TIMEOUT'} "
                      f"in {t+1} steps <<<{RESET}")
                break

        if not done:
            print(f"\n  {RED}{BOLD}>>> TIMEOUT nach {args.max_steps} steps <<<{RESET}")

        results.append({"solved": solved, "steps": t + 1, "reward": total_reward})

    # Summary
    sr = sum(1 for r in results if r["solved"]) / len(results) * 100
    avg_steps = sum(r["steps"] for r in results) / len(results)
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"  {BOLD}{CYAN}ZUSAMMENFASSUNG{RESET}")
    print(f"  SR: {sr:.0f}%  Avg Steps: {avg_steps:.1f}")
    print(f"{BOLD}{'='*60}{RESET}")


if __name__ == "__main__":
    main()
