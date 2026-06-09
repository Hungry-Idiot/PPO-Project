"""
evaluate_junior.py

Deterministic evaluation for Junior-stage PPO model.

Run from the Python directory.

Example:
    D:/Anaconda/envs/uav_rl/python.exe evaluate_junior.py ^
        --model ./output/junior_smoke/model/ppo_single_uav.zip ^
        --config ./config/envs_junior.yaml ^
        --episodes 5 ^
        --max-steps 300
"""

from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from envs.train_env import TrainEnv


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def close_env(env):
    """
    TrainEnv currently does not define close().
    Close sockets manually to avoid leaving TCP connections open.
    """
    try:
        if hasattr(env, "adaptor") and env.adaptor is not None:
            env.adaptor.socket.close()
    except Exception:
        pass

    try:
        if hasattr(env, "adaptor_enemy") and env.adaptor_enemy is not None:
            env.adaptor_enemy.socket.close()
    except Exception:
        pass


def evaluate_one_episode(model, env, episode_idx, max_steps, deterministic=True, verbose=True):
    obs, reset_info = env.reset()

    initial_my_state = env.my_state.copy()
    initial_enemy_state = env.enemy_state.copy()

    total_reward = 0.0
    total_damage_stepwise = 0.0
    total_damage_taken_stepwise = 0.0

    min_dist_m = float("inf")
    max_dist_m = 0.0

    min_z = float(initial_my_state[2])
    max_z = float(initial_my_state[2])
    min_speed = float("inf")
    max_speed = 0.0
    speed_sum = 0.0
    altitude_sum = 0.0
    heading_sum = 0.0

    damage_steps = 0
    proximity_steps = 0
    desertion_steps = 0
    low_altitude_steps = 0
    vertical_fall_steps = 0

    killed = False
    self_dead = False
    terminated_by_platform = False
    truncated_by_env = False

    final_my_state = initial_my_state.copy()
    final_enemy_state = initial_enemy_state.copy()

    last_info = {}

    if verbose:
        header = (
            f"{'step':>5s} {'reward':>10s} {'dist(m)':>9s} "
            f"{'z':>9s} {'speed':>8s} {'enemy_hp':>9s} "
            f"{'my_hp':>8s} {'thr':>6s} {'pitch':>7s} "
            f"{'roll':>7s} {'yaw':>7s} {'term':>6s} {'trunc':>6s}"
        )
        print(f"\n[EPISODE {episode_idx}] reset_info={reset_info}")
        print(header)
        print("-" * len(header))

    for step in range(1, max_steps + 1):
        prev_my_state = env.my_state.copy()
        prev_enemy_state = env.enemy_state.copy()

        agent_action, _ = model.predict(obs, deterministic=deterministic)

        obs, reward, terminated, truncated, info = env.step(agent_action)

        my_state = env.my_state.copy()
        enemy_state = env.enemy_state.copy()

        final_my_state = my_state
        final_enemy_state = enemy_state
        last_info = info

        total_reward += safe_float(reward)

        damage = max(0.0, (prev_enemy_state[12] - enemy_state[12]) * 1000.0)
        damage_taken = max(0.0, (prev_my_state[12] - my_state[12]) * 1000.0)

        total_damage_stepwise += damage
        total_damage_taken_stepwise += damage_taken

        if damage > 0:
            damage_steps += 1

        rel_pos = enemy_state[0:3] - my_state[0:3]
        dist_units = float(np.linalg.norm(rel_pos))
        dist_m = dist_units * 10.0

        min_dist_m = min(min_dist_m, dist_m)
        max_dist_m = max(max_dist_m, dist_m)

        z = float(my_state[2])
        min_z = min(min_z, z)
        max_z = max(max_z, z)
        altitude_sum += z

        speed = float(np.linalg.norm(my_state[6:9]))
        min_speed = min(min_speed, speed)
        max_speed = max(max_speed, speed)
        speed_sum += speed

        vertical_speed = float(my_state[8])
        if vertical_speed < -5.0:
            vertical_fall_steps += 1

        if z < 900.0:
            low_altitude_steps += 1

        if dist_units < 10.0:
            proximity_steps += 1

        if dist_units > 60.0:
            desertion_steps += 1

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

        killed = bool(enemy_state[12] <= 0.01)
        self_dead = bool(my_state[12] <= 0.01)
        terminated_by_platform = bool(terminated)
        truncated_by_env = bool(truncated)

        real_action = None
        try:
            from utils import action as action_utils
            real_action = action_utils.marshal_action(agent_action)
        except Exception:
            real_action = np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float64)

        if verbose:
            print(
                f"{step:5d} {reward:10.2f} {dist_m:9.1f} "
                f"{z:9.2f} {speed:8.2f} {enemy_state[12]:9.4f} "
                f"{my_state[12]:8.4f} {real_action[0]:6.3f} "
                f"{real_action[1]:7.3f} {real_action[2]:7.3f} "
                f"{real_action[3]:7.3f} {str(terminated):>6s} {str(truncated):>6s}"
            )

        if terminated or truncated or killed or self_dead:
            break

    steps = step

    # 两种伤害统计：
    # 1. stepwise：从 reset 后第一步开始累计，不包含 warmup 阶段可能造成的伤害
    # 2. from_full_hp：按最终血量从 1.0 计算，包含 reset/warmup 阶段已经造成的伤害
    total_damage_from_full_hp = max(0.0, (1.0 - final_enemy_state[12]) * 1000.0)
    total_damage_taken_from_full_hp = max(0.0, (1.0 - final_my_state[12]) * 1000.0)

    result = {
        "episode": episode_idx,
        "steps": steps,
        "killed": int(killed),
        "self_dead": int(self_dead),
        "terminated_by_platform": int(terminated_by_platform),
        "truncated_by_env": int(truncated_by_env),
        "timeout_or_max_steps": int(
            steps >= max_steps and not killed and not self_dead and not terminated_by_platform and not truncated_by_env
        ),

        "total_reward": total_reward,

        "total_damage_stepwise_hp": total_damage_stepwise,
        "total_damage_from_full_hp": total_damage_from_full_hp,
        "damage_steps": damage_steps,

        "total_damage_taken_stepwise_hp": total_damage_taken_stepwise,
        "total_damage_taken_from_full_hp": total_damage_taken_from_full_hp,

        "initial_enemy_hp": float(initial_enemy_state[12]),
        "final_enemy_hp": float(final_enemy_state[12]),
        "initial_my_hp": float(initial_my_state[12]),
        "final_my_hp": float(final_my_state[12]),

        "initial_z": float(initial_my_state[2]),
        "final_z": float(final_my_state[2]),
        "min_z": min_z,
        "max_z": max_z,
        "avg_z": altitude_sum / max(steps, 1),

        "min_dist_m": min_dist_m,
        "max_dist_m": max_dist_m,

        "min_speed": min_speed,
        "max_speed": max_speed,
        "avg_speed": speed_sum / max(steps, 1),

        "avg_heading_dot": heading_sum / max(steps, 1),

        "proximity_steps": proximity_steps,
        "desertion_steps": desertion_steps,
        "low_altitude_steps": low_altitude_steps,
        "vertical_fall_steps": vertical_fall_steps,
    }

    reward_comps = last_info.get("reward_comps", {})
    for key, value in reward_comps.items():
        try:
            result[f"last_{key}"] = float(value)
        except Exception:
            pass

    if verbose:
        print("\n[EPISODE SUMMARY]")
        print(f"  steps:              {result['steps']}")
        print(f"  killed:             {bool(result['killed'])}")
        print(f"  self_dead:          {bool(result['self_dead'])}")
        print(f"  terminated:         {bool(result['terminated_by_platform'])}")
        print(f"  truncated:          {bool(result['truncated_by_env'])}")
        print(f"  total_reward:       {result['total_reward']:.2f}")
        print(f"  damage_stepwise:    {result['total_damage_stepwise_hp']:.1f} HP")
        print(f"  damage_from_full:   {result['total_damage_from_full_hp']:.1f} HP")
        print(f"  final enemy hp:     {result['final_enemy_hp']:.4f}")
        print(f"  final my hp:        {result['final_my_hp']:.4f}")
        print(f"  min dist:           {result['min_dist_m']:.1f} m")
        print(f"  min z:              {result['min_z']:.2f}")
        print(f"  max z:              {result['max_z']:.2f}")
        print(f"  avg speed:          {result['avg_speed']:.2f}")
        print(f"  proximity steps:    {result['proximity_steps']}")
        print(f"  low altitude steps: {result['low_altitude_steps']}")

    return result


def write_csv(path, rows):
    if not rows:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    keys = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    if not rows:
        return {}

    def mean(key):
        vals = [float(r[key]) for r in rows if key in r and r[key] == r[key]]
        return float(np.mean(vals)) if vals else np.nan

    def rate(key):
        vals = [int(r[key]) for r in rows if key in r]
        return float(np.mean(vals)) if vals else 0.0

    return {
        "episodes": len(rows),
        "kill_rate": rate("killed"),
        "self_dead_rate": rate("self_dead"),
        "terminated_rate": rate("terminated_by_platform"),
        "truncated_rate": rate("truncated_by_env"),
        "timeout_or_max_steps_rate": rate("timeout_or_max_steps"),

        "avg_steps": mean("steps"),
        "avg_total_reward": mean("total_reward"),

        "avg_total_damage_stepwise_hp": mean("total_damage_stepwise_hp"),
        "avg_total_damage_from_full_hp": mean("total_damage_from_full_hp"),
        "avg_damage_taken_from_full_hp": mean("total_damage_taken_from_full_hp"),

        "avg_final_enemy_hp": mean("final_enemy_hp"),
        "avg_final_my_hp": mean("final_my_hp"),

        "avg_min_dist_m": mean("min_dist_m"),
        "avg_max_dist_m": mean("max_dist_m"),

        "avg_min_z": mean("min_z"),
        "avg_max_z": mean("max_z"),
        "avg_final_z": mean("final_z"),

        "avg_min_speed": mean("min_speed"),
        "avg_max_speed": mean("max_speed"),
        "avg_speed": mean("avg_speed"),

        "avg_heading_dot": mean("avg_heading_dot"),

        "avg_proximity_steps": mean("proximity_steps"),
        "avg_desertion_steps": mean("desertion_steps"),
        "avg_low_altitude_steps": mean("low_altitude_steps"),
        "avg_vertical_fall_steps": mean("vertical_fall_steps"),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="./output/junior_smoke/model/ppo_single_uav.zip",
        help="Path to PPO model zip.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./config/envs_junior.yaml",
        help="Path to env config.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=300,
        help="Max steps per episode.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to save evaluation CSV files.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy instead of deterministic policy.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print per-step logs.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.out_dir is None:
        out_dir = Path("./output") / f"eval_junior_{timestamp}"
    else:
        out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    detail_csv = out_dir / "junior_eval_episode_results.csv"
    summary_csv = out_dir / "junior_eval_summary.csv"

    print("[INFO] Junior evaluation")
    print(f"[INFO] Model:       {args.model}")
    print(f"[INFO] Config:      {args.config}")
    print(f"[INFO] Episodes:    {args.episodes}")
    print(f"[INFO] Max steps:   {args.max_steps}")
    print(f"[INFO] Deterministic:{not args.stochastic}")
    print(f"[INFO] Out dir:     {out_dir}")

    model = PPO.load(args.model, device="cpu")

    env = TrainEnv(config_path=args.config)

    rows = []

    try:
        for ep in range(1, args.episodes + 1):
            result = evaluate_one_episode(
                model=model,
                env=env,
                episode_idx=ep,
                max_steps=args.max_steps,
                deterministic=not args.stochastic,
                verbose=not args.quiet,
            )
            rows.append(result)

            write_csv(detail_csv, rows)
            write_csv(summary_csv, [summarize(rows)])

    finally:
        close_env(env)

    summary = summarize(rows)

    print("\n[FINAL SUMMARY]")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print("\n[DONE]")
    print(f"Episode results: {detail_csv}")
    print(f"Summary:         {summary_csv}")


if __name__ == "__main__":
    main()