import argparse
import csv
import os
import re
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from utils import adaptor
from utils import action
from utils import initialize
from utils import observation
from utils import truncate


DEFAULT_CONFIG_PATH = "./config/envs.yaml"
DEFAULT_MODEL_DIR = "./output/simple_fixed_1000m/run_0/model"
DEFAULT_OUTPUT_CSV = "./output/simple_fixed_1000m/checkpoint_eval_results.csv"


def parse_step_from_name(path: Path) -> int:
    """
    从 model_5000_steps.zip 这种文件名里提取 5000。
    如果不是 checkpoint 命名，则返回一个很大的数，排在后面。
    """
    m = re.search(r"(\d+)_steps", path.name)
    if m:
        return int(m.group(1))
    return 10**18


def find_checkpoints(model_dir: str, selected_steps=None):
    model_dir = Path(model_dir)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    all_models = sorted(
        model_dir.glob("*.zip"),
        key=lambda p: parse_step_from_name(p)
    )

    if selected_steps:
        selected_steps = set(int(x) for x in selected_steps)
        return [
            p for p in all_models
            if parse_step_from_name(p) in selected_steps
        ]

    return all_models


def split_observation(original_observation):
    """
    BattleData:
    m_unit_1[13], m_unit_2[13], m_is_done
    """
    my_state = np.array(original_observation[0:13], dtype=np.float64)
    enemy_state = np.array(original_observation[13:26], dtype=np.float64)
    terminated = bool(original_observation[26] > 0.5)
    return my_state, enemy_state, terminated


def build_initial_packet():
    """
    InitData:
    [room_id, unit_id] + my_init[12] + enemy_init[12]
    共 26 个 int。
    """
    init_state = initialize.generate_initial_state()

    my_init = init_state[0:12].astype(np.int32)
    enemy_init = init_state[12:24].astype(np.int32)

    initial_distance_units = float(np.linalg.norm(enemy_init[0:3] - my_init[0:3]))
    initial_distance_m = initial_distance_units * 10.0

    if initial_distance_m < 1000.0:
        raise ValueError(
            f"Initial distance is {initial_distance_m:.1f}m, "
            f"but it must be >= 1000m."
        )

    packet = np.array([114514, 1919810], dtype=np.int32)
    packet = np.append(packet, my_init)
    packet = np.append(packet, enemy_init)

    return packet, initial_distance_m


def safe_close(net):
    try:
        if getattr(net, "socket", None) is not None:
            net.socket.close()
    except Exception:
        pass


def evaluate_one_episode(
    model,
    model_path,
    repeat_id,
    config_path,
    max_steps,
    deterministic=True,
    verbose_every=0,
    drain_steps=80,
):
    net = adaptor.NetworkAdaptor(config_path)

    result = {
        "model": os.path.basename(model_path),
        "model_path": str(model_path),
        "checkpoint_step": parse_step_from_name(Path(model_path)),
        "repeat": repeat_id,

        "initial_distance_m": None,
        "reset_distance_m": None,
        "reset_enemy_hp": None,
        "reset_self_hp": None,
        "reset_ok": False,

        "steps": 0,
        "drain_steps": 0,
        "killed": False,
        "terminated": False,
        "truncated": False,
        "self_dead": False,

        "enemy_hp": None,
        "self_hp": None,
        "total_damage_hp": 0.0,
        "min_distance_m": None,
        "final_distance_m": None,
        "final_speed": None,

        "error": "",
    }

    try:
        net.connect()

        init_packet, initial_distance_m = build_initial_packet()
        result["initial_distance_m"] = initial_distance_m

        net.send_initial_packet(init_packet)
        raw_obs = net.get_observation_packet()

        my_state, enemy_state, terminated = split_observation(raw_obs)
        obs = observation.marshal_observation(my_state, enemy_state)

        reset_dist_units = float(np.linalg.norm(enemy_state[0:3] - my_state[0:3]))
        reset_distance_m = reset_dist_units * 10.0
        reset_enemy_hp = float(enemy_state[12])
        reset_self_hp = float(my_state[12])

        result["reset_distance_m"] = reset_distance_m
        result["reset_enemy_hp"] = reset_enemy_hp
        result["reset_self_hp"] = reset_self_hp

        reset_ok = (900.0 <= reset_distance_m <= 1100.0) and (reset_enemy_hp >= 0.99)
        result["reset_ok"] = reset_ok

        if not reset_ok:
            print(
                f"[WARN] First BattleData is not a clean reset: "
                f"distance={reset_distance_m:.1f}m, "
                f"enemy_hp={reset_enemy_hp:.3f}, "
                f"self_hp={reset_self_hp:.3f}. "
                f"This episode will continue and be marked reset_ok=False."
            )

        init_enemy_hp = reset_enemy_hp
        min_dist_units = reset_dist_units

        if verbose_every > 0:
            print(
                f"[INIT] {os.path.basename(model_path)} "
                f"repeat={repeat_id} "
                f"target_init_dist={initial_distance_m:.1f}m "
                f"reset_dist={reset_distance_m:.1f}m "
                f"my_hp={reset_self_hp:.3f} "
                f"enemy_hp={reset_enemy_hp:.3f} "
                f"reset_ok={reset_ok}"
            )

        for step in range(1, max_steps + 1):
            agent_action, _ = model.predict(obs, deterministic=deterministic)
            real_action = action.marshal_action(agent_action)

            send_pack = np.append(real_action, 0.0).astype(np.float64)
            net.send_action_packet(send_pack)

            raw_obs = net.get_observation_packet()
            my_state, enemy_state, terminated = split_observation(raw_obs)
            obs = observation.marshal_observation(my_state, enemy_state)

            dist_units = float(np.linalg.norm(enemy_state[0:3] - my_state[0:3]))
            min_dist_units = min(min_dist_units, dist_units)

            enemy_hp = float(enemy_state[12])
            self_hp = float(my_state[12])
            total_damage_hp = max(0.0, (init_enemy_hp - enemy_hp) * 1000.0)
            speed = float(np.linalg.norm(my_state[6:9]))

            local_truncated = truncate.check_truncation(my_state, enemy_state)
            killed = enemy_hp <= 0.01
            self_dead = self_hp <= 0.01

            result["steps"] = step
            result["enemy_hp"] = enemy_hp
            result["self_hp"] = self_hp
            result["total_damage_hp"] = total_damage_hp
            result["min_distance_m"] = min_dist_units * 10.0
            result["final_distance_m"] = dist_units * 10.0
            result["final_speed"] = speed
            result["killed"] = killed
            result["terminated"] = terminated
            result["truncated"] = bool(result["truncated"] or local_truncated)
            result["self_dead"] = self_dead

            if verbose_every > 0 and (
                step % verbose_every == 0
                or killed
                or terminated
                or local_truncated
                or total_damage_hp > 0.0
            ):
                print(
                    f"[STEP {step:04d}] "
                    f"dist={dist_units * 10.0:8.1f}m "
                    f"enemy_hp={enemy_hp:.3f} "
                    f"self_hp={self_hp:.3f} "
                    f"dmg={total_damage_hp:7.1f} "
                    f"thr={real_action[0]:+.3f} "
                    f"pitch={real_action[1]:+.3f} "
                    f"roll={real_action[2]:+.3f} "
                    f"yaw={real_action[3]:+.3f} "
                    f"speed={speed:.2f} "
                    f"term={terminated} "
                    f"local_trunc={local_truncated}"
                )

            # 只相信服务器返回的 m_is_done。
            # 不因为 local_truncated / killed / self_dead 主动 break，
            # 避免服务器 episode 没结束、下一次评测读到残留状态。
            if terminated:
                break

        # 如果 max_steps 跑完后服务器还没返回 terminated，
        # 继续发送零动作做短暂 drain，尽量等平台自然结束当前 episode。
        if not result["terminated"] and drain_steps > 0:
            zero_pack = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

            for drain_i in range(1, drain_steps + 1):
                net.send_action_packet(zero_pack)

                raw_obs = net.get_observation_packet()
                my_state, enemy_state, terminated = split_observation(raw_obs)

                dist_units = float(np.linalg.norm(enemy_state[0:3] - my_state[0:3]))
                enemy_hp = float(enemy_state[12])
                self_hp = float(my_state[12])
                total_damage_hp = max(0.0, (init_enemy_hp - enemy_hp) * 1000.0)
                speed = float(np.linalg.norm(my_state[6:9]))

                local_truncated = truncate.check_truncation(my_state, enemy_state)
                killed = enemy_hp <= 0.01
                self_dead = self_hp <= 0.01

                result["drain_steps"] = drain_i
                result["enemy_hp"] = enemy_hp
                result["self_hp"] = self_hp
                result["total_damage_hp"] = total_damage_hp
                result["min_distance_m"] = min(result["min_distance_m"], dist_units * 10.0)
                result["final_distance_m"] = dist_units * 10.0
                result["final_speed"] = speed
                result["killed"] = killed
                result["terminated"] = terminated
                result["truncated"] = bool(result["truncated"] or local_truncated)
                result["self_dead"] = self_dead

                if verbose_every > 0 and (terminated or drain_i == 1 or drain_i % verbose_every == 0):
                    print(
                        f"[DRAIN {drain_i:04d}] "
                        f"dist={dist_units * 10.0:8.1f}m "
                        f"enemy_hp={enemy_hp:.3f} "
                        f"self_hp={self_hp:.3f} "
                        f"dmg={total_damage_hp:7.1f} "
                        f"term={terminated}"
                    )

                if terminated:
                    break

        return result

    except Exception as e:
        result["error"] = repr(e)
        return result

    finally:
        safe_close(net)


def summarize(results):
    valid = [r for r in results if not r["error"]]

    if not valid:
        print("[SUMMARY] No valid evaluation results.")
        return

    clean_valid = [r for r in valid if r["reset_ok"]]

    print("\n========== CHECKPOINT SUMMARY: CLEAN RESET EPISODES ONLY ==========")

    if not clean_valid:
        print("[WARN] No clean-reset episodes. The room was likely not freshly reset or was left mid-episode.")
        print("       Close the current room, create a new Simple-vs-Fixed room, then rerun.")
        return

    by_model = {}
    for r in clean_valid:
        by_model.setdefault(r["model"], []).append(r)

    summary_rows = []

    for model_name, rows in sorted(
        by_model.items(),
        key=lambda item: parse_step_from_name(Path(item[0]))
    ):
        n = len(rows)
        kill_count = sum(1 for r in rows if r["killed"])
        trunc_count = sum(1 for r in rows if r["truncated"])
        term_count = sum(1 for r in rows if r["terminated"])
        avg_damage = float(np.mean([r["total_damage_hp"] for r in rows]))
        avg_enemy_hp = float(np.mean([r["enemy_hp"] for r in rows]))
        avg_steps = float(np.mean([r["steps"] for r in rows]))
        avg_min_dist = float(np.mean([r["min_distance_m"] for r in rows]))
        avg_final_dist = float(np.mean([r["final_distance_m"] for r in rows]))

        summary_rows.append({
            "model": model_name,
            "checkpoint_step": parse_step_from_name(Path(model_name)),
            "episodes": n,
            "kill_count": kill_count,
            "kill_rate": kill_count / n,
            "trunc_count": trunc_count,
            "trunc_rate": trunc_count / n,
            "term_count": term_count,
            "term_rate": term_count / n,
            "avg_damage_hp": avg_damage,
            "avg_enemy_hp": avg_enemy_hp,
            "avg_steps": avg_steps,
            "avg_min_distance_m": avg_min_dist,
            "avg_final_distance_m": avg_final_dist,
        })

        print(
            f"{model_name:30s} "
            f"kill={kill_count}/{n} "
            f"term={term_count}/{n} "
            f"trunc={trunc_count}/{n} "
            f"avg_dmg={avg_damage:7.1f} "
            f"avg_enemy_hp={avg_enemy_hp:.3f} "
            f"avg_steps={avg_steps:7.1f} "
            f"avg_min_dist={avg_min_dist:8.1f}m "
            f"avg_final_dist={avg_final_dist:8.1f}m"
        )

    best = sorted(
        summary_rows,
        key=lambda x: (
            x["kill_rate"],
            x["avg_damage_hp"],
            -x["trunc_rate"],
            -x["avg_enemy_hp"],
        ),
        reverse=True
    )[0]

    print("\n[BEST CANDIDATE]")
    print(
        f"{best['model']} | "
        f"kill_rate={best['kill_rate']:.2f}, "
        f"avg_damage={best['avg_damage_hp']:.1f}, "
        f"avg_enemy_hp={best['avg_enemy_hp']:.3f}, "
        f"trunc_rate={best['trunc_rate']:.2f}, "
        f"avg_min_dist={best['avg_min_distance_m']:.1f}m"
    )

    unclean_count = len(valid) - len(clean_valid)
    if unclean_count > 0:
        print(
            f"\n[NOTE] {unclean_count} valid episodes were excluded from summary "
            f"because reset_ok=False."
        )


def save_csv(results, output_csv):
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model",
        "model_path",
        "checkpoint_step",
        "repeat",

        "initial_distance_m",
        "reset_distance_m",
        "reset_enemy_hp",
        "reset_self_hp",
        "reset_ok",

        "steps",
        "drain_steps",
        "killed",
        "terminated",
        "truncated",
        "self_dead",

        "enemy_hp",
        "self_hp",
        "total_damage_hp",
        "min_distance_m",
        "final_distance_m",
        "final_speed",

        "error",
    ]

    with output_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\n[CSV] Saved evaluation results to: {output_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--verbose-every", type=int, default=0)
    parser.add_argument("--drain-steps", type=int, default=80)
    parser.add_argument("--sleep", type=float, default=0.5)

    parser.add_argument(
        "--steps",
        nargs="*",
        type=int,
        default=[5000, 10000, 15000, 20000, 25000, 30000, 50000, 100000],
        help="Only evaluate checkpoints with these step numbers. Use --steps with no values to evaluate all.",
    )

    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions instead of deterministic actions.",
    )

    args = parser.parse_args()

    deterministic = not args.stochastic

    checkpoints = find_checkpoints(args.model_dir, args.steps)

    if not checkpoints:
        print("[ERROR] No checkpoint found.")
        print(f"model_dir = {args.model_dir}")
        print(f"steps = {args.steps}")
        return

    print("========== EVALUATION CONFIG ==========")
    print(f"config        : {args.config}")
    print(f"model_dir     : {args.model_dir}")
    print(f"output_csv    : {args.output_csv}")
    print(f"max_steps     : {args.max_steps}")
    print(f"drain_steps   : {args.drain_steps}")
    print(f"repeats       : {args.repeats}")
    print(f"deterministic : {deterministic}")
    print(f"checkpoints   :")
    for p in checkpoints:
        print(f"  - {p}")

    results = []

    for ckpt in checkpoints:
        print(f"\n========== LOAD MODEL: {ckpt.name} ==========")
        model = PPO.load(str(ckpt), device="cpu")

        for repeat_id in range(args.repeats):
            print(f"[EVAL] {ckpt.name} repeat {repeat_id + 1}/{args.repeats}")

            r = evaluate_one_episode(
                model=model,
                model_path=ckpt,
                repeat_id=repeat_id,
                config_path=args.config,
                max_steps=args.max_steps,
                deterministic=deterministic,
                verbose_every=args.verbose_every,
                drain_steps=args.drain_steps,
            )

            results.append(r)

            if r["error"]:
                print(f"  ERROR: {r['error']}")
            else:
                print(
                    f"  reset_ok={r['reset_ok']} "
                    f"reset_dist={r['reset_distance_m']:.1f}m "
                    f"reset_enemy_hp={r['reset_enemy_hp']:.3f} "
                    f"kill={r['killed']} "
                    f"term={r['terminated']} "
                    f"steps={r['steps']} "
                    f"drain={r['drain_steps']} "
                    f"dmg={r['total_damage_hp']:.1f} "
                    f"enemy_hp={r['enemy_hp']:.3f} "
                    f"self_hp={r['self_hp']:.3f} "
                    f"min_dist={r['min_distance_m']:.1f}m "
                    f"final_dist={r['final_distance_m']:.1f}m "
                    f"trunc={r['truncated']}"
                )

            time.sleep(args.sleep)

    save_csv(results, args.output_csv)
    summarize(results)


if __name__ == "__main__":
    main()