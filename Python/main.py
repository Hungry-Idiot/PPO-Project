"""
main.py

PPO training entry.

Examples:

Simple default training:
    D:/Anaconda/envs/uav_rl/python.exe main.py

Junior short smoke training:
    D:/Anaconda/envs/uav_rl/python.exe main.py --config ./config/envs_junior.yaml --timesteps 2000 --run-name junior_smoke
"""

import argparse
import os

from envs.train_env import TrainEnv
from stable_baselines3 import PPO
from torch import nn
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback

from utils.callback import RewardComponentsCallback


def make_run_dir(base_output_dir, run_name=None):
    if run_name:
        run_dir = os.path.join(base_output_dir, run_name)
        if not os.path.exists(run_dir):
            return run_dir

        i = 1
        while True:
            candidate = os.path.join(base_output_dir, f"{run_name}_{i}")
            if not os.path.exists(candidate):
                return candidate
            i += 1

    run_dir = os.path.join(base_output_dir, "run_0")
    i = 0

    while os.path.exists(run_dir):
        i += 1
        run_dir = os.path.join(base_output_dir, f"run_{i}")

    return run_dir


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="./config/envs.yaml",
        help="Environment config path, e.g. ./config/envs_junior.yaml",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50000,
        help="Total PPO training timesteps.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional output run name, e.g. junior_smoke.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Base output directory.",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=5000,
        help="Checkpoint save frequency.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="PPO learning rate.",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.005,
        help="PPO entropy coefficient.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PPO device: cpu or cuda.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    args = parser.parse_args()

    run_dir = make_run_dir(args.output_dir, args.run_name)
    model_dir = os.path.join(run_dir, "model")
    log_dir = os.path.join(run_dir, "logs")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print("[INFO] PPO training")
    print(f"[INFO] Config:       {args.config}")
    print(f"[INFO] Timesteps:    {args.timesteps}")
    print(f"[INFO] Run dir:      {run_dir}")
    print(f"[INFO] Model dir:    {model_dir}")
    print(f"[INFO] Log dir:      {log_dir}")
    print(f"[INFO] Save freq:    {args.save_freq}")
    print(f"[INFO] LR:           {args.lr}")
    print(f"[INFO] Ent coef:     {args.ent_coef}")
    print(f"[INFO] Device:       {args.device}")
    print(f"[INFO] Seed:         {args.seed}")

    logger = configure(log_dir, ["stdout", "csv"])

    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=model_dir,
        name_prefix="model",
        save_replay_buffer=True,
    )

    reward_callback = RewardComponentsCallback(
        csv_path=os.path.join(log_dir, "reward_components.csv")
    )

    base_env = TrainEnv(config_path=args.config)
    env = Monitor(base_env)

    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
        ortho_init=False,
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=args.lr,
        n_steps=512,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.02,
        policy_kwargs=policy_kwargs,
        device=args.device,
        seed=args.seed,
    )

    model.set_logger(logger)

    model.learn(
        total_timesteps=args.timesteps,
        progress_bar=True,
        reset_num_timesteps=False,
        log_interval=1,
        callback=[checkpoint_callback, reward_callback],
    )

    final_path = os.path.join(model_dir, "ppo_single_uav")
    model.save(final_path)

    print(f"[DONE] Model saved to {final_path}.zip")


if __name__ == "__main__":
    main()