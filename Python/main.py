"""
main函数，用于训练模型并保存
"""
import os
from envs.train_env import TrainEnv
from stable_baselines3 import PPO
from torch import nn
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback
from utils.callback import RewardComponentsCallback


def main():
    run_dir = os.path.join("./output", "run_0")
    i = 0
    while os.path.exists(run_dir):
        i += 1
        run_dir = os.path.join("./output", f"run_{i}")
    model_dir = os.path.join(run_dir, "model")
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    logger = configure(log_dir, ["stdout", "csv"])
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=model_dir,
        name_prefix="model",
        save_replay_buffer=True,
    )
    reward_callback = RewardComponentsCallback(
        csv_path=os.path.join(log_dir, "reward_components.csv")
    )

    base_env = TrainEnv(config_path="./config/envs.yaml")
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
        learning_rate=1e-4,
        n_steps=512,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.02,
        policy_kwargs=policy_kwargs,
        device="cpu",
        seed=42,
    )
    model.set_logger(logger)
    model.learn(
        total_timesteps=100000,
        progress_bar=True,
        reset_num_timesteps=False,
        log_interval=1,
        callback=[checkpoint_callback, reward_callback],
    )
    model.save(os.path.join(model_dir, "ppo_single_uav"))
    print(f"Model saved to {model_dir}/ppo_single_uav.zip")


if __name__ == "__main__":
    main()
