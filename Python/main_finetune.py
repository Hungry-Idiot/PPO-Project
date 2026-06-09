"""
Fine-tune a trained PPO checkpoint for Simple-vs-Simple moving-target training.

Place this file under:
    Python/main_finetune.py

Example:
    D:/Anaconda/envs/uav_rl/python.exe main_finetune.py ^
        --model ./output/run_12/model/model_45000_steps.zip ^
        --timesteps 20000 ^
        --lr 5e-5 ^
        --ent-coef 0.001

Notes:
- Run from the Python directory.
- Create a Simple-vs-Simple UE room before running.
- This script creates a new output folder:
    ./output/finetune_YYYYMMDD_HHMMSS/
"""

import argparse
import os
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn

from envs.train_env import TrainEnv
from utils.callback import RewardComponentsCallback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Base checkpoint path, e.g. ./output/run_12/model/model_45000_steps.zip",
    )
    parser.add_argument("--config", type=str, default="./config/envs.yaml")
    parser.add_argument("--timesteps", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--ent-coef", type=float, default=0.001)
    parser.add_argument("--save-freq", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError("Base model not found: " + args.model)

    if args.out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("./output", "finetune_" + timestamp)
    else:
        run_dir = args.out_dir

    model_dir = os.path.join(run_dir, "model")
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print("[INFO] Base model:      " + args.model, flush=True)
    print("[INFO] Output dir:      " + run_dir, flush=True)
    print("[INFO] Timesteps:       " + str(args.timesteps), flush=True)
    print("[INFO] Learning rate:   " + str(args.lr), flush=True)
    print("[INFO] Entropy coef:    " + str(args.ent_coef), flush=True)

    base_env = TrainEnv(config_path=args.config)
    env = Monitor(base_env)

    logger = configure(log_dir, ["stdout", "csv"])

    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=model_dir,
        name_prefix="finetune_model",
        save_replay_buffer=True,
    )
    reward_callback = RewardComponentsCallback(
        csv_path=os.path.join(log_dir, "reward_components.csv")
    )

    # Load old checkpoint and attach the current environment.
    model = PPO.load(
        args.model,
        env=env,
        device="cpu",
        seed=args.seed,
    )

    # Override fine-tuning hyperparameters.
    # SB3 stores the LR as a schedule, so update both fields.
    model.learning_rate = args.lr
    model.lr_schedule = get_schedule_fn(args.lr)
    model.ent_coef = args.ent_coef

    model.set_logger(logger)

    model.learn(
        total_timesteps=args.timesteps,
        progress_bar=True,
        reset_num_timesteps=False,
        log_interval=1,
        callback=[checkpoint_callback, reward_callback],
    )

    final_path = os.path.join(model_dir, "ppo_finetuned")
    model.save(final_path)
    print("[DONE] Final model saved to " + final_path + ".zip", flush=True)


if __name__ == "__main__":
    main()
