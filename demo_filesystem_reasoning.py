"""Filesystem Reasoning Demo — the REAL test for the Reasoning Chip.

No BFS possible. No Manhattan distance. No predetermined path.
The agent must UNDERSTAND file names and contents to find the secret.

Usage:
    cd rapa_mvp
    .venv/Scripts/python demo_filesystem_reasoning.py \
        --checkpoint ../runs/reasoning_v5/best_model_sft.pt \
        --config D:/Downloads/E.2KVCanvas/config.yaml \
        --scenario 0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import GPT2TokenizerFast

from env.filesystem import FileSystemEnv
from kernel.kernel_canvas_manager import KernelCanvasManager
from llm.def_provider import DEFProvider
from llm.output_parser import parse_reasoning_output
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


def update_fs_canvas(canvas: KernelCanvasManager, obs, goal: str) -> None:
    """Write filesystem observation into Canvas."""
    canvas.write("current_dir", obs.current_dir)
    canvas.write("directories", ", ".join(obs.dirs) if obs.dirs else "none")
    canvas.write("files", ", ".join(obs.files) if obs.files else "none")
    canvas.write("goal", goal)
    if obs.last_read and obs.last_read_name:
        canvas.write("file_content", f"{obs.last_read_name}: {obs.last_read[:200]}")


def extract_action(answer: str, obs) -> str:
    """Parse D's ANSWER into a filesystem action."""
    lower = answer.lower()

    # "cd logs" / "explore logs" / "go to logs/"
    for d in obs.dirs:
        dname = d.rstrip("/")
        if dname.lower() in lower:
            return f"cd {dname}"

    # "read error.log" / "open nginx.conf"
    for f in obs.files:
        if f.lower() in lower:
            return f"read {f}"

    # "search timeout" / "grep for password"
    if "search" in lower or "grep" in lower:
        # Extract the search term
        for word in ["search", "grep", "find"]:
            if word in lower:
                parts = lower.split(word, 1)
                if len(parts) > 1:
                    term = parts[1].strip().strip("'\"").split()[0]
                    if term and len(term) > 2:
                        return f"search {term}"

    # "answer: upstream_timeout"
    if "answer" in lower or "cause" in lower or "secret" in lower:
        # Try to extract the answer value
        for delim in [":", "is", "="]:
            if delim in answer:
                val = answer.split(delim, 1)[-1].strip().strip(".")
                if val:
                    return f"answer {val}"

    # "go back" / "cd .."
    if ".." in lower or "back" in lower or "parent" in lower:
        return "cd .."

    # Default: ls (just observe)
    return "ls"


def main():
    parser = argparse.ArgumentParser(description="Filesystem Reasoning Demo")
    parser.add_argument("--checkpoint", type=str,
                        default="../runs/reasoning_v5/best_model_sft.pt")
    parser.add_argument("--config", type=str,
                        default="D:/Downloads/E.2KVCanvas/config.yaml")
    parser.add_argument("--scenario", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"\n{BOLD}{'='*65}{RESET}")
    print(f"  {BOLD}{CYAN}Filesystem Reasoning Demo{RESET}")
    print(f"  {DIM}No BFS. No Manhattan. Pure semantic reasoning.{RESET}")
    print(f"  {DIM}Device: {device}{RESET}")
    print(f"{BOLD}{'='*65}{RESET}")

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

    # Create environment
    env = FileSystemEnv(scenario=args.scenario)
    obs = env.reset()

    print(f"\n{BOLD}Scenario: {env.scenario['name']}{RESET}")
    print(f"  {YELLOW}Goal: {env.goal}{RESET}")
    print(f"  {DIM}Secret: [hidden]{RESET}")
    print(f"{'='*65}")

    # Create canvas + reasoning chip
    canvas = KernelCanvasManager(n_slots=8)
    llm = DEFProvider(model=model, tokenizer=tokenizer, device=device,
                      canvas_manager=canvas, forced_prefix=None)

    solved = False
    for step in range(args.max_steps):
        # Update canvas with current observation
        update_fs_canvas(canvas, obs, env.goal)

        # Build reasoning prompt from canvas — adapt question to context
        memory_block = canvas.to_prefix()
        if obs.files and not obs.dirs:
            question = f"QUESTION: We are in {obs.current_dir} with files: {', '.join(obs.files)}. Which file should the agent read to achieve the goal?"
        elif obs.last_read:
            question = f"QUESTION: We read {obs.last_read_name}. Based on the content, what is the answer to the goal?"
        else:
            question = "QUESTION: Based on the current directory and goal, what should the agent do next?"

        prompt = memory_block + "\n" + question + "\n"

        t0 = time.perf_counter()
        txt = llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=150,
        )
        dt = time.perf_counter() - t0

        if isinstance(txt, tuple):
            txt = txt[0]
        txt = txt.strip() if isinstance(txt, str) else ""

        reasoning, answer, tags, fmt = parse_reasoning_output(txt)

        # Convert ANSWER to filesystem action
        action = extract_action(answer or reasoning, obs)

        # Execute
        obs_next, reward, done = env.step(action)

        # Render
        dir_color = CYAN if obs.current_dir != "/" else DIM
        print(f"\n  {BOLD}Step {step}{RESET} {dir_color}{obs.current_dir}{RESET} "
              f"{DIM}({dt*1000:.0f}ms){RESET}")
        print(f"    {DIM}Files: {', '.join(obs.files[:5])}{RESET}")
        print(f"    {DIM}Dirs:  {', '.join(obs.dirs[:5])}{RESET}")
        if reasoning:
            print(f"    {MAGENTA}REASONING:{RESET} {reasoning[:120]}")
        if answer:
            print(f"    {GREEN}ANSWER:{RESET} {answer[:100]}")
        print(f"    {BLUE}ACTION:{RESET} {action}")

        if obs.last_read and obs.last_read_name:
            snippet = obs.last_read[:80].replace("\n", " | ")
            print(f"    {DIM}[Content: {snippet}]{RESET}")

        if reward > 0:
            solved = True
            print(f"\n  {GREEN}{BOLD}>>> SOLVED! Secret found in {step+1} steps <<<{RESET}")
            break
        elif reward < 0:
            print(f"    {RED}Wrong answer!{RESET}")

        obs = obs_next
        if done:
            break

    if not solved:
        print(f"\n  {RED}{BOLD}>>> TIMEOUT nach {args.max_steps} steps <<<{RESET}")
        print(f"  {DIM}(Secret was: {env.secret}){RESET}")

    env.cleanup()


if __name__ == "__main__":
    main()
