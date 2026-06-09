"""
Batch evaluate PPO checkpoints in Simple-vs-Simple dual-port mode.

Place this file under:
    Python/evaluate_checkpoints.py

Run from the Python directory:
    D:/Anaconda/envs/uav_rl/python.exe evaluate_checkpoints.py --run-dir ./output/run_XX --episodes 10

It will write:
    ./output/eval_<timestamp>/checkpoint_episode_results.csv
    ./output/eval_<timestamp>/checkpoint_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from utils import adaptor, action, initialize, observation, reward


CONFIG_PATH = "./config/envs.yaml"


def pack_initial(my_init: np.ndarray, enemy_init: np.ndarray, unit_id: int) -> np.ndarray:
    """Pack InitData payload: [room, unit_id, my_12, enemy_12]."""
    room = 114514
    packet = np.array([room, unit_id], dtype=np.int32)
    packet = np.append(packet, my_init.astype(np.int32))
    packet = np.append(packet, enemy_init.astype(np.int32))
    return packet


def pack_ctrl(ctrl4: np.ndarray, done: float = 0.0) -> np.ndarray:
    """Pack CtrlData payload: [throttle, pitch, roll, yaw, done]."""
    return np.append(ctrl4.astype(np.float64), float(done)).astype(np.float64)


def split_battle(raw: np.ndarray):
    my_state = raw[0:13].astype(np.float64).copy()
    enemy_state = raw[13:26].astype(np.float64).copy()
    is_done = bool(raw[26] == 1.0)
    return my_state, enemy_state, is_done


def checkpoint_sort_key(path: Path):
    """Sort model_5000_steps.zip by step number; final ppo_single_uav.zip goes last."""
    m = re.search(r"model_(\d+)_steps\.zip$", path.name)
    if m:
        return int(m.group(1))
    if path.name == "ppo_single_uav.zip":
        return 10**18
    return 10**17


def find_checkpoints(run_dir: str | None, explicit_models: list[str] | None) -> list[Path]:
    if explicit_models:
        paths = [Path(p) for p in explicit_models]
    else:
        if not run_dir:
            raise ValueError("Either --run-dir or --models must be provided.")
        model_dir = Path(run_dir) / "model"
        paths = sorted(model_dir.glob("*.zip"), key=checkpoint_sort_key)

    paths = [p for p in paths if p.exists()]
    if not paths:
        raise FileNotFoundError("No checkpoint .zip files found.")

    return paths


def connect_pair(config_path: str):
    """Create two fresh TCP connections: my port and enemy port+1."""
    net_my = adaptor.NetworkAdaptor(config_path)
    net_my.connect()

    net_enemy = adaptor.NetworkAdaptor(config_path)
    net_enemy.port += 1
    net_enemy.connect()

    return net_my, net_enemy


def close_pair(net_my, net_enemy):
    for net in (net_my, net_enemy):
        try:
            net.socket.close()
        except Exception:
            pass


def start_episode(net_my, net_enemy):
    """Send InitData for both sides, then send one CtrlData to trigger first BattleData."""
    init_state = initialize.generate_initial_state()
    my_init = init_state[0:12]
    enemy_init = init_state[12:24]

    packet_my = pack_initial(my_init, enemy_init, 1919810)
    packet_enemy = pack_initial(enemy_init, my_init, 1919811)

    net_my.send_initial_packet(packet_my)
    net_enemy.send_initial_packet(packet_enemy)

    # Simple-vs-Simple needs both clients to send CtrlData before BattleData is produced.
    zero_ctrl = np.zeros(5, dtype=np.float64)
    net_my.send_action_packet(zero_ctrl)
    net_enemy.send_action_packet(zero_ctrl)

    raw_my = net_my.get_observation_packet()
    _ = net_enemy.get_observation_packet()

    my_state, enemy_state, is_done = split_battle(raw_my)
    obs = observation.marshal_observation(my_state, enemy_state)

    return obs, my_state, enemy_state, is_done


def evaluate_one_episode(
    model: PPO,
    model_path: Path,
    episode_idx: int,
    config_path: str,
    max_steps: int,
    sleep_after_connect: float = 0.05,
):
    net_my, net_enemy = None, None

    try:
        net_my, net_enemy = connect_pair(config_path)
        if sleep_after_connect > 0:
            time.sleep(sleep_after_connect)

        obs, my_state, enemy_state, is_done = start_episode(net_my, net_enemy)

        total_damage = 0.0
        total_damage_taken = 0.0
        total_reward = 0.0
        max_damage_step = 0.0
        min_dist_m = float("inf")
        max_dist_m = 0.0
        heading_sum = 0.0
        speed_sum = 0.0
        desertion_steps = 0
        proximity_steps = 0
        damage_steps = 0
        last_comps = {}

        killed = False
        self_dead = False
        terminated_by_done = False

        for step in range(1, max_steps + 1):
            prev_my_state = my_state.copy()
            prev_enemy_state = enemy_state.copy()

            agent_action, _ = model.predict(obs, deterministic=True)
            real_action = action.marshal_action(agent_action)
            net_my.send_action_packet(pack_ctrl(real_action, 0.0))

            # Match current training setup: enemy is a zero-action moving target.
            enemy_ctrl = np.zeros(4, dtype=np.float64)
            net_enemy.send_action_packet(pack_ctrl(enemy_ctrl, 0.0))

            raw_my = net_my.get_observation_packet()
            _ = net_enemy.get_observation_packet()

            my_state, enemy_state, is_done = split_battle(raw_my)
            obs = observation.marshal_observation(my_state, enemy_state)

            damage = max(0.0, (prev_enemy_state[12] - enemy_state[12]) * 1000.0)
            damage_taken = max(0.0, (prev_my_state[12] - my_state[12]) * 1000.0)
            total_damage += damage
            total_damage_taken += damage_taken
            max_damage_step = max(max_damage_step, damage)
            if damage > 0:
                damage_steps += 1

            comps = reward.reward_components(prev_my_state, prev_enemy_state, my_state, enemy_state)
            last_comps = comps
            total_reward += float(comps.get("total", 0.0))

            rel_pos = enemy_state[0:3] - my_state[0:3]
            dist_units = float(np.linalg.norm(rel_pos))
            dist_m = dist_units * 10.0
            min_dist_m = min(min_dist_m, dist_m)
            max_dist_m = max(max_dist_m, dist_m)

            speed = float(np.linalg.norm(my_state[6:9]))
            speed_sum += speed

            if dist_units > 1e-6:
                roll, pitch, yaw = my_state[3], my_state[4], my_state[5]
                forward = np.array([
                    np.cos(yaw) * np.cos(pitch),
                    np.sin(yaw) * np.cos(pitch),
                    np.sin(pitch),
                ])
                heading_dot = float(np.dot(forward, rel_pos / dist_units))
            else:
                heading_dot = 0.0
            heading_sum += heading_dot

            if dist_units > 50.0:
                desertion_steps += 1
            if dist_units < 7.0:
                proximity_steps += 1

            killed = enemy_state[12] <= 0.01
            self_dead = my_state[12] <= 0.01
            terminated_by_done = is_done

            if is_done or killed or self_dead:
                break

        result = {
            "model": str(model_path).replace("\\", "/"),
            "model_name": model_path.name,
            "episode": episode_idx,
            "ok": 1,
            "error": "",
            "steps": step,
            "killed": int(killed),
            "self_dead": int(self_dead),
            "platform_done": int(terminated_by_done),
            "timeout_or_max_steps": int(step >= max_steps and not (killed or self_dead or terminated_by_done)),
            "total_reward": total_reward,
            "total_damage": total_damage,
            "total_damage_taken": total_damage_taken,
            "max_damage_step": max_damage_step,
            "damage_steps": damage_steps,
            "final_enemy_hp": float(enemy_state[12]),
            "final_my_hp": float(my_state[12]),
            "min_dist_m": min_dist_m,
            "max_dist_m": max_dist_m,
            "avg_heading_dot": heading_sum / max(step, 1),
            "avg_speed_units": speed_sum / max(step, 1),
            "desertion_steps": desertion_steps,
            "proximity_steps": proximity_steps,
            "last_reward_total": float(last_comps.get("total", 0.0)) if last_comps else 0.0,
        }

        # Add useful reward components from the final step for debugging.
        for k, v in last_comps.items():
            result[f"last_{k}"] = float(v)

        return result

    except Exception as e:
        return {
            "model": str(model_path).replace("\\", "/"),
            "model_name": model_path.name,
            "episode": episode_idx,
            "ok": 0,
            "error": repr(e),
            "steps": 0,
            "killed": 0,
            "self_dead": 0,
            "platform_done": 0,
            "timeout_or_max_steps": 0,
            "total_reward": 0.0,
            "total_damage": 0.0,
            "total_damage_taken": 0.0,
            "max_damage_step": 0.0,
            "damage_steps": 0,
            "final_enemy_hp": np.nan,
            "final_my_hp": np.nan,
            "min_dist_m": np.nan,
            "max_dist_m": np.nan,
            "avg_heading_dot": np.nan,
            "avg_speed_units": np.nan,
            "desertion_steps": 0,
            "proximity_steps": 0,
            "last_reward_total": 0.0,
        }

    finally:
        if net_my is not None and net_enemy is not None:
            close_pair(net_my, net_enemy)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return

    keys = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    summaries = []
    by_model = {}

    for row in rows:
        by_model.setdefault(row["model_name"], []).append(row)

    for model_name, items in by_model.items():
        ok_items = [r for r in items if int(r["ok"]) == 1]
        n = len(items)
        n_ok = len(ok_items)

        def mean(key, default=np.nan):
            vals = [float(r[key]) for r in ok_items if key in r and r[key] == r[key]]
            return float(np.mean(vals)) if vals else default

        def rate(key):
            vals = [int(r[key]) for r in ok_items if key in r]
            return float(np.mean(vals)) if vals else 0.0

        summaries.append({
            "model_name": model_name,
            "episodes_requested": n,
            "episodes_ok": n_ok,
            "error_count": n - n_ok,
            "kill_rate": rate("killed"),
            "self_dead_rate": rate("self_dead"),
            "max_step_rate": rate("timeout_or_max_steps"),
            "avg_steps": mean("steps"),
            "avg_total_reward": mean("total_reward"),
            "avg_total_damage": mean("total_damage"),
            "avg_damage_taken": mean("total_damage_taken"),
            "avg_final_enemy_hp": mean("final_enemy_hp"),
            "avg_final_my_hp": mean("final_my_hp"),
            "avg_min_dist_m": mean("min_dist_m"),
            "avg_max_dist_m": mean("max_dist_m"),
            "avg_heading_dot": mean("avg_heading_dot"),
            "avg_speed_units": mean("avg_speed_units"),
            "avg_desertion_steps": mean("desertion_steps"),
            "avg_proximity_steps": mean("proximity_steps"),
            # A rough ranking score. Prefer kill rate, lower self-death, shorter kill time, higher damage.
            "score": (
                rate("killed") * 10000.0
                - rate("self_dead") * 5000.0
                - mean("steps", 0.0) * 2.0
                + mean("total_damage", 0.0) * 2.0
                + mean("avg_heading_dot", 0.0) * 500.0
            ),
        })

    summaries.sort(key=lambda r: r["score"], reverse=True)
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=None, help="Example: ./output/run_12")
    parser.add_argument("--models", nargs="*", default=None, help="Explicit checkpoint paths.")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per checkpoint.")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max evaluation steps per episode.")
    parser.add_argument("--config", type=str, default=CONFIG_PATH)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--include-final", action="store_true", help="Include ppo_single_uav.zip if present.")
    parser.add_argument(
        "--only-steps",
        nargs="*",
        type=int,
        default=None,
        help="Only evaluate these checkpoint step numbers, e.g. --only-steps 30000 35000 40000 45000 50000",
    )
    args = parser.parse_args()

    checkpoints = find_checkpoints(args.run_dir, args.models)

    if not args.include_final:
        checkpoints = [p for p in checkpoints if p.name != "ppo_single_uav.zip"]

    if args.only_steps:
        wanted = {f"model_{s}_steps.zip" for s in args.only_steps}
        checkpoints = [p for p in checkpoints if p.name in wanted]

    if not checkpoints:
        raise FileNotFoundError("No checkpoints left after filtering.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path("./output") / f"eval_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Evaluating {len(checkpoints)} checkpoints")
    print(f"[INFO] Episodes per checkpoint: {args.episodes}")
    print(f"[INFO] Max steps per episode: {args.max_steps}")
    print(f"[INFO] Output directory: {out_dir}")

    all_rows = []

    for ckpt_idx, ckpt in enumerate(checkpoints, start=1):
        print(f"\n[{ckpt_idx}/{len(checkpoints)}] Loading {ckpt}", flush=True)
        model = PPO.load(str(ckpt))

        for ep in range(1, args.episodes + 1):
            print(f"  episode {ep}/{args.episodes} ...", flush=True)
            row = evaluate_one_episode(
                model=model,
                model_path=ckpt,
                episode_idx=ep,
                config_path=args.config,
                max_steps=args.max_steps,
            )
            all_rows.append(row)

            status = "OK" if row["ok"] else "ERR"
            print(
                f"    {status}: killed={row['killed']} self_dead={row['self_dead']} "
                f"steps={row['steps']} dmg={row['total_damage']:.1f} "
                f"enemy_hp={row['final_enemy_hp']}",
                flush=True,
            )

            # Save incrementally, so partial results are preserved if the room crashes.
            write_csv(out_dir / "checkpoint_episode_results.csv", all_rows)
            write_csv(out_dir / "checkpoint_summary.csv", summarize(all_rows))

    print("\n[DONE]")
    print(f"Episode results: {out_dir / 'checkpoint_episode_results.csv'}")
    print(f"Summary:         {out_dir / 'checkpoint_summary.csv'}")


if __name__ == "__main__":
    main()
