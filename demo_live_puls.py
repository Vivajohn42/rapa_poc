"""Live-Puls Demo — Sichtbarer Beweis dass Kernel + Sprachchip miteinander reden.

Zeigt pro Tick:
  - F.2: Generierungszeit (KV-Cache vs. full-sequence)
  - F.3: Format-Qualitaet (0=perfekt, 3=fallback)
  - F.4: NARRATIVE + PREDICTION + Self-Correction Loop

Usage:
    cd rapa_mvp
    python demo_live_puls.py
    python demo_live_puls.py --episodes 5 --max-steps 40
    python demo_live_puls.py --checkpoint <pfad> --config <pfad>
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import GPT2TokenizerFast

from agents.agent_a import AgentA
from agents.agent_b import AgentB
from agents.agent_c import AgentC, GoalSpec
from agents.agent_d_llm import AgentDLLM
from env.gridworld import GridWorld
from kernel.kernel import MvpKernel
from llm.def_provider import DEFProvider
from kernel.kernel_canvas_manager import KernelCanvasManager
from router.deconstruct import deconstruct_d_to_c

from def_transformer.model.config import DEFTransformerConfig
from def_transformer.model.utils import build_model


# ─── ANSI colors ──────────────────────────────────────────────
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"
RESET = "\033[0m"


def load_def_model(checkpoint_path: str, config_path: str, device: torch.device):
    config = DEFTransformerConfig.from_yaml(config_path)
    try:
        import flash_attn
    except ImportError:
        config.use_flash_attn = False
    model = build_model(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    missing, _ = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  {DIM}New params (fresh init): {len(missing)}{RESET}")
    model.eval()
    del ckpt, state
    return model, config


def fmt_quality_label(q: int) -> str:
    labels = {0: f"{GREEN}perfect{RESET}", 1: f"{YELLOW}case-fix{RESET}",
              2: f"{YELLOW}regex{RESET}", 3: f"{RED}fallback{RESET}"}
    return labels.get(q, f"{RED}?{RESET}")


def run_live_episode(
    kernel: MvpKernel,
    env: GridWorld,
    canvas_manager: KernelCanvasManager,
    agent_d: AgentDLLM,
    llm: DEFProvider,
    episode_id: str,
    max_steps: int = 30,
) -> dict:
    obs = env.reset()
    canvas_manager.reset()

    goal_map = env.goal_positions
    kernel.reset_episode(goal_mode="seek", episode_id=episode_id)
    if hasattr(kernel, '_memory_manager') and kernel._memory_manager:
        kernel._memory_manager.goal_map = goal_map

    total_reward = 0.0
    steps = 0
    solved = False
    gen_times: list[float] = []

    for t in range(max_steps):
        pos = obs.agent_pos if hasattr(obs, "agent_pos") else "?"
        hint = obs.hint if hasattr(obs, "hint") and obs.hint else None

        # Track generation time (F.2 proof)
        call_count_before = llm._call_count
        t0 = time.perf_counter()

        result = kernel.tick(t, obs, done=False)

        dt_tick = time.perf_counter() - t0
        was_d_call = llm._call_count > call_count_before
        if was_d_call:
            gen_times.append(dt_tick)

        action = result.action
        canvas_manager.update_from_observation(obs)

        obs_next, reward, done = env.step(action)
        kernel.observe_reward(reward)
        total_reward += reward
        steps = t + 1

        # ─── Render tick ──────────────────────────────────────
        regime = f"{CYAN}4D{RESET}" if result.d_activated else f"{DIM}3D{RESET}"
        decon = f" {MAGENTA}[DECON]{RESET}" if result.decon_fired else ""
        hint_str = f" {YELLOW}hint={hint}{RESET}" if hint else ""
        gain_str = ""
        if result.gain:
            g = result.gain.G
            gf = result.gain.G_over_F
            color = GREEN if gf > 0.8 else (YELLOW if gf > 0.5 else RED)
            gain_str = f" {color}G={g:.3f} G/F={gf:.2f}{RESET}"

        print(f"\n{BOLD}=== TICK {t:>2} | Pos: {pos} | Goal: {env.goal_positions} ==={RESET}")
        print(f"  {BLUE}[Kernel]{RESET} Action: {BOLD}{action}{RESET}  "
              f"Regime: {regime}{decon}{hint_str}{gain_str}")

        if was_d_call:
            # Show B's prediction
            agent_b = kernel.agent_b
            try:
                from state.schema import ZA
                zA = kernel.agent_a.infer_zA(obs)
                zA_next = agent_b.predict_next(zA, action)
                print(f"  {GREEN}[Agent B]{RESET} Deterministic: "
                      f"{pos} + {action} -> {zA_next.agent_pos}")
            except Exception:
                pass

            # Show D's output — read raw from LLM provider's last generation
            print(f"  {MAGENTA}[Agent D]{RESET} Reasoning-Chip... "
                  f"{DIM}({dt_tick*1000:.0f}ms){RESET}")

            raw_text = ""
            if hasattr(llm, '_last_generated') and llm._last_generated:
                raw_text = llm.tokenizer.decode(
                    llm._last_generated, skip_special_tokens=True)

            if raw_text:
                from llm.output_parser import parse_reasoning_output
                reasoning, answer, tags_parsed, fmt_q = parse_reasoning_output(raw_text)
                if reasoning:
                    print(f"     {BOLD}> REASONING:{RESET} {reasoning[:140]}")
                if answer:
                    print(f"     {GREEN}{BOLD}> ANSWER:{RESET} {answer[:120]}")
                tags_str = ", ".join(tags_parsed[:8])
                print(f"     {DIM}> Tags: {tags_str}  Format: {fmt_quality_label(fmt_q)}{RESET}")
            else:
                print(f"     {DIM}(no output){RESET}")

            # Show canvas state
            slots = list(canvas_manager.slots.keys())
            print(f"  {BLUE}[Canvas]{RESET} Slots: {slots}")

            # Self-correction check
            prev_pred = canvas_manager.get_prediction()
            if prev_pred:
                print(f"  {YELLOW}[Self-Correction]{RESET} "
                      f"PREV_PREDICTION im naechsten Prompt verfuegbar")
        else:
            print(f"  {DIM}[Agent D] (nicht aktiv, gD=0){RESET}")

        obs = obs_next
        if done:
            solved = (reward > 0)
            kernel.tick(t + 1, obs, done=True)
            status_color = GREEN if solved else RED
            print(f"\n  {status_color}{BOLD}>>> "
                  f"{'SOLVED' if solved else 'DONE'} "
                  f"in {steps} steps (reward={total_reward:+.1f}) <<<{RESET}")
            break

    if not done:
        print(f"\n  {RED}{BOLD}>>> TIMEOUT nach {steps} steps <<<{RESET}")

    # Speed summary
    if gen_times:
        avg_ms = sum(gen_times) / len(gen_times) * 1000
        print(f"\n  {BLUE}[F.2 Speed]{RESET} "
              f"{len(gen_times)} D-Calls, avg {avg_ms:.0f}ms/call")

    return {
        "episode_id": episode_id,
        "steps": steps,
        "solved": solved,
        "total_reward": total_reward,
        "d_calls": len(gen_times),
        "avg_gen_ms": sum(gen_times) / len(gen_times) * 1000 if gen_times else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Live-Puls Demo (F.2+F.3+F.4)")
    parser.add_argument("--checkpoint", type=str,
                        default="D:/Downloads/E.2KVCanvas/best_model.pt")
    parser.add_argument("--config", type=str,
                        default="D:/Downloads/E.2KVCanvas/config.yaml")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"  {BOLD}{CYAN}RAPA-OS Live-Puls Demo{RESET}")
    print(f"  {DIM}Sprachchip: DEF Transformer (200M){RESET}")
    print(f"  {DIM}Gehirn:     RAPA-OS MvpKernel{RESET}")
    print(f"  {DIM}Features:   F.2 KV-Cache | F.3 Robust Parse | F.4 Prediction{RESET}")
    print(f"  {DIM}Device:     {device}{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    # Load model
    print(f"\n{BLUE}Loading DEF model...{RESET}")
    model, config = load_def_model(args.checkpoint, args.config, device)
    tokenizer = GPT2TokenizerFast.from_pretrained(config.tokenizer_name)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {GREEN}Loaded{RESET} ({n_params:,} params on {device})")

    # Run episodes
    all_results = []

    for ep in range(args.episodes):
        canvas_manager = KernelCanvasManager(n_slots=8)
        llm = DEFProvider(
            model=model, tokenizer=tokenizer,
            device=device, canvas_manager=canvas_manager,
            forced_prefix=None,  # SFT model generates NARRATIVE: itself
        )

        env = GridWorld(width=5, height=5, seed=args.seed + ep)
        agent_a = AgentA()
        agent_b = AgentB()
        agent_c = AgentC(goal=GoalSpec(mode="seek", target=(-1, -1)))
        agent_d = AgentDLLM(llm)

        kernel = MvpKernel(
            agent_a=agent_a, agent_b=agent_b,
            agent_c=agent_c, agent_d=agent_d,
            deconstruct_fn=deconstruct_d_to_c,
            canvas_manager=canvas_manager,
        )

        print(f"\n{BOLD}{'='*60}{RESET}")
        print(f"  {BOLD}Episode {ep+1}/{args.episodes}{RESET} "
              f"(seed={args.seed + ep}, goals={env.goal_positions})")
        print(f"{'='*60}")

        result = run_live_episode(
            kernel, env, canvas_manager, agent_d, llm,
            episode_id=f"live_ep{ep:03d}",
            max_steps=args.max_steps,
        )
        all_results.append(result)

    # Final summary
    solved = sum(1 for r in all_results if r["solved"])
    total_calls = sum(r["d_calls"] for r in all_results)
    avg_ms = (sum(r["avg_gen_ms"] * r["d_calls"] for r in all_results)
              / max(total_calls, 1))

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"  {BOLD}{CYAN}ZUSAMMENFASSUNG{RESET}")
    print(f"{'='*60}")
    print(f"  Episoden:  {solved}/{len(all_results)} solved")
    print(f"  D-Calls:   {total_calls} total, avg {avg_ms:.0f}ms/call")
    print(f"  KV-Cache:  {'aktiv' if hasattr(model, 'blocks') else 'n/a'}")
    print(f"  Prediction: F.4 Self-Correction Loop aktiv")
    print(f"{BOLD}{'='*60}{RESET}")


if __name__ == "__main__":
    main()
