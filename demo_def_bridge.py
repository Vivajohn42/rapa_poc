"""DEF-RAPA Bridge Demo — DEF Transformer as Sprachchip in RAPA-OS.

Proves the concept: RAPA-OS kernel is the brain, DEF model is the language chip.
The kernel controls memory (KernelCanvasManager), routing (gC/gD), and governance.
The model just processes language — tokenize, generate, decode.

Uses DEFProvider (drop-in LLMProvider) with existing AgentDLLM.
Zero changes to kernel, agents, or governance logic.

Usage:
    cd rapa_mvp
    python demo_def_bridge.py \
        --checkpoint D:/Downloads/E.2KVCanvas/best_model.pt \
        --config ../def_transformer/configs/def_200m_e2_kv.yaml \
        --episodes 3 --max-steps 30
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add repo root to path for def_transformer imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import GPT2TokenizerFast

from agents.agent_a import AgentA
from agents.agent_b import AgentB
from agents.agent_c import AgentC
from agents.agent_d_llm import AgentDLLM
from agents.agent_d import AgentD
from env.gridworld import GridWorld
from kernel.kernel import MvpKernel
from llm.def_provider import DEFProvider
from kernel.kernel_canvas_manager import KernelCanvasManager
from router.deconstruct import deconstruct_d_to_c
from agents.agent_c import GoalSpec

from def_transformer.model.config import DEFTransformerConfig
from def_transformer.model.utils import build_model


def load_def_model(checkpoint_path: str, config_path: str, device: torch.device):
    """Load DEF transformer for inference."""
    config = DEFTransformerConfig.from_yaml(config_path)
    # Disable flash_attn if not available
    try:
        import flash_attn
    except ImportError:
        config.use_flash_attn = False

    model = build_model(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  New params (fresh init): {len(missing)}")
    model.eval()
    del ckpt, state
    return model, config


def run_episode(
    kernel: MvpKernel,
    env: GridWorld,
    canvas_manager: KernelCanvasManager,
    episode_id: str,
    max_steps: int = 50,
    verbose: bool = True,
) -> dict:
    """Run one GridWorld episode with DEF-backed Agent D."""
    obs = env.reset()
    canvas_manager.reset()

    goal_map = env.goal_positions  # Dict[str, Tuple[int, int]]

    kernel.reset_episode(goal_mode="seek", episode_id=episode_id)
    # Set goal_map for deconstruction (D->C hint mapping)
    if hasattr(kernel, '_memory_manager') and kernel._memory_manager:
        kernel._memory_manager.goal_map = goal_map

    total_reward = 0.0
    steps = 0
    solved = False

    for t in range(max_steps):
        result = kernel.tick(t, obs, done=False)
        action = result.action

        # Update canvas from observation (use raw env obs)
        canvas_manager.update_from_observation(obs)

        # Step environment
        obs_next, reward, done = env.step(action)
        kernel.observe_reward(reward)
        total_reward += reward
        steps = t + 1

        if verbose:
            pos = obs.agent_pos if hasattr(obs, "agent_pos") else "?"
            regime = "4D" if result.d_activated else "3D"
            gain_str = ""
            if result.gain:
                gain_str = f" G={result.gain.G:.3f} G/F={result.gain.G_over_F:.2f}"
            canvas_str = f" canvas={canvas_manager}" if canvas_manager.slots else ""
            decon = " [DECON]" if result.decon_fired else ""
            print(f"  t={t:>3} pos={pos} act={action:<6} r={reward:+.2f} "
                  f"regime={regime}{decon}{gain_str}{canvas_str}")

        obs = obs_next
        if done:
            solved = (reward > 0)
            kernel.tick(t + 1, obs, done=True)
            break

    return {
        "episode_id": episode_id,
        "steps": steps,
        "solved": solved,
        "total_reward": total_reward,
        "canvas_slots": dict(canvas_manager.slots),
    }


def main():
    parser = argparse.ArgumentParser(description="DEF-RAPA Bridge Demo")
    parser.add_argument("--checkpoint", required=True, help="Path to E.2 model checkpoint")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--compare-det", action="store_true",
                        help="Also run with deterministic D for comparison")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 60)
    print("  DEF-RAPA Bridge Demo")
    print(f"  Sprachchip: DEF Transformer (200M)")
    print(f"  Gehirn: RAPA-OS MvpKernel")
    print(f"  Device: {device}")
    print("=" * 60)

    # Load model
    print(f"\nLoading DEF model from {args.checkpoint}...")
    model, config = load_def_model(args.checkpoint, args.config, device)
    tokenizer = GPT2TokenizerFast.from_pretrained(config.tokenizer_name)
    print(f"  Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    # Create Canvas Manager
    canvas_manager = KernelCanvasManager(n_slots=8)

    # Create DEF Provider (drop-in LLMProvider)
    llm = DEFProvider(
        model=model,
        tokenizer=tokenizer,
        device=device,
        canvas_manager=canvas_manager,
    )

    # Run episodes with DEF-backed D
    print(f"\n{'='*60}")
    print(f"  Phase 1: DEF-D (Sprachchip)")
    print(f"{'='*60}")

    def_results = []
    for ep in range(args.episodes):
        env = GridWorld(
            width=5, height=5, seed=args.seed + ep,
        )

        # Create agents
        agent_a = AgentA()
        agent_b = AgentB()
        agent_c = AgentC(goal=GoalSpec(mode="seek", target=(-1, -1)))
        agent_d = AgentDLLM(llm)

        kernel = MvpKernel(
            agent_a=agent_a,
            agent_b=agent_b,
            agent_c=agent_c,
            agent_d=agent_d,
            deconstruct_fn=deconstruct_d_to_c,
        )

        print(f"\n--- Episode {ep + 1}/{args.episodes} (DEF-D) ---")
        t0 = time.time()
        result = run_episode(
            kernel, env, canvas_manager,
            episode_id=f"def_ep{ep:03d}",
            max_steps=args.max_steps,
        )
        dt = time.time() - t0
        status = "SOLVED" if result["solved"] else "TIMEOUT"
        print(f"  -> {status} in {result['steps']} steps ({dt:.1f}s)")
        print(f"  -> Canvas: {result['canvas_slots']}")
        print(f"  -> DEFProvider calls: {llm._call_count}")
        def_results.append(result)

    # Optional: compare with deterministic D
    if args.compare_det:
        print(f"\n{'='*60}")
        print(f"  Phase 2: Deterministic D (baseline)")
        print(f"{'='*60}")

        det_results = []
        for ep in range(args.episodes):
            env = GridWorld(GridConfig(
                width=5, height=5, seed=args.seed + ep,
                n_goals=2, n_obstacles=1,
            ))

            agent_a = AgentA()
            agent_b = AgentB()
            agent_c = AgentC(goal=GoalSpec(mode="seek", target=(-1, -1)))
            agent_d = AgentD()

            kernel = MvpKernel(
                agent_a=agent_a,
                agent_b=agent_b,
                agent_c=agent_c,
                agent_d=agent_d,
                deconstruct_fn=deconstruct_d_to_c,
            )

            canvas_det = KernelCanvasManager(n_slots=8)
            print(f"\n--- Episode {ep + 1}/{args.episodes} (Det-D) ---")
            result = run_episode(
                kernel, env, canvas_det,
                episode_id=f"det_ep{ep:03d}",
                max_steps=args.max_steps,
            )
            status = "SOLVED" if result["solved"] else "TIMEOUT"
            print(f"  -> {status} in {result['steps']} steps")
            det_results.append(result)

        # Summary
        print(f"\n{'='*60}")
        print(f"  COMPARISON")
        print(f"{'='*60}")
        def_sr = sum(1 for r in def_results if r["solved"]) / len(def_results) * 100
        det_sr = sum(1 for r in det_results if r["solved"]) / len(det_results) * 100
        print(f"  DEF-D: {def_sr:.0f}% SR")
        print(f"  Det-D: {det_sr:.0f}% SR")

    # Final summary
    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"{'='*60}")
    def_sr = sum(1 for r in def_results if r["solved"]) / len(def_results) * 100
    print(f"  DEF-D success rate: {def_sr:.0f}%")
    print(f"  Total DEFProvider calls: {llm._call_count}")


if __name__ == "__main__":
    main()
