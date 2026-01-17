import json
from pathlib import Path
from typing import List, Dict, Tuple
from agents.agent_b import AgentB
from agents.agent_c import AgentC, GoalSpec
from state.schema import ZA

from eval.metrics import score_negation_error


def load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def extract_step_records(records: List[Dict]) -> List[Dict]:
    return [r for r in records if "episode_summary" not in r]


def extract_episode_summary(records: List[Dict]) -> Dict:
    for r in records:
        if r.get("episode_summary"):
            return r
    return {}


def aggregate_pair(seek_log: str, avoid_log: str, out_csv: str):
    seek_records = load_jsonl(seek_log)
    avoid_records = load_jsonl(avoid_log)

    seek_steps = extract_step_records(seek_records)
    avoid_steps = extract_step_records(avoid_records)

    # --- Basic sanity check ---
    paired_steps = min(len(seek_steps), len(avoid_steps))
    seek_steps = seek_steps[:paired_steps]
    avoid_steps = avoid_steps[:paired_steps]
    total_steps = paired_steps

    total_steps = len(seek_steps)

    # --- AB mismatch rate ---
    ab_mismatch = 0
    for s in seek_steps:
        if s["pred_next_pos"] != s["true_next_pos"]:
            ab_mismatch += 1
    ab_mismatch_rate = ab_mismatch / total_steps

    # --- Score negation error per step ---
    neg_errors = []
    top_flip_count = 0

    B = AgentB()

    neg_errors = []
    top_flip_count = 0

    # We use SEEK states as reference trajectory
    for s in seek_steps:
        zA_dict = s["zA"]
        zA = ZA(**zA_dict)

        C_seek = AgentC(goal=GoalSpec(mode="seek", target=zA.goal_pos))
        C_avoid = AgentC(goal=GoalSpec(mode="avoid", target=zA.goal_pos))

        _, scored_seek = C_seek.choose_action(zA, B.predict_next)
        _, scored_avoid = C_avoid.choose_action(zA, B.predict_next)

        neg_errors.append(score_negation_error(scored_seek, scored_avoid))

        top_seek = [a for a, _ in scored_seek][0]
        top_avoid = [a for a, _ in scored_avoid][0]
        if top_seek != top_avoid:
            top_flip_count += 1

    mean_neg_error = sum(neg_errors) / len(neg_errors)
    top_flip_rate = top_flip_count / total_steps

    # --- Episode summaries ---
    seek_ep = extract_episode_summary(seek_records)
    avoid_ep = extract_episode_summary(avoid_records)

    # --- Write CSV ---
    Path(out_csv).parent.mkdir(exist_ok=True)

    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("metric,value\n")
        f.write(f"total_steps,{total_steps}\n")
        f.write(f"ab_mismatch_rate,{ab_mismatch_rate}\n")
        f.write(f"mean_score_negation_error,{mean_neg_error}\n")
        f.write(f"top_action_flip_rate,{top_flip_rate}\n")
        f.write(f"seek_success,{seek_ep.get('success')}\n")
        f.write(f"seek_steps,{seek_ep.get('steps')}\n")
        f.write(f"seek_total_reward,{seek_ep.get('total_reward')}\n")
        f.write(f"avoid_success,{avoid_ep.get('success')}\n")
        f.write(f"avoid_steps,{avoid_ep.get('steps')}\n")
        f.write(f"avoid_total_reward,{avoid_ep.get('total_reward')}\n")

    print("Aggregation complete.")
    print(f"CSV written to: {out_csv}")
    print("\n--- Summary ---")
    print(f"AB mismatch rate: {ab_mismatch_rate:.3f} (expect 0.0)")
    print(f"Mean score negation error: {mean_neg_error:.3f} (expect ~0.0)")
    print(f"Top-action flip rate: {top_flip_rate:.3f} (expect ~1.0)")
