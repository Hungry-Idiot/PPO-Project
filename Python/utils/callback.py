"""
修改gym的callback函数，实现训练过程中保存模型和日志
兼容原版 "r/" 前缀读取，并无缝对接 reward.py 中的 comps 字典
"""
from stable_baselines3.common.callbacks import BaseCallback
from collections import defaultdict
import os, csv

class RewardComponentsCallback(BaseCallback):
    def __init__(self, csv_path=None):
        super().__init__()
        self.csv_path = csv_path
        self._csv_file = None
        self._csv_writer = None
        self._reset_sums()

    def _reset_sums(self):
        self.count = 0
        self.sums = defaultdict(float)

    def _on_training_start(self) -> None:
        if self.csv_path:
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            self._csv_file = open(self.csv_path, "w", newline="")
            self._csv_writer = None

    def _on_training_end(self) -> None:
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None

    def _on_rollout_start(self) -> None:
        self._reset_sums()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos:
            return True
        info = infos[0]

        row = {"timesteps": int(self.num_timesteps)}
        wrote = False
        
        # 核心修改：提取我们在 reward.py 中定义的 comps 字典
        comps_dict = info.get("comps", {})
        
        # 为了兼容原版，如果底层加了 "r/" 前缀，我们也把它抓出来合并
        for key, val in info.items():
            if isinstance(key, str) and key.startswith("r/"):
                comps_dict[key.replace("r/", "")] = val

        # 遍历所有收集到的奖励分量并累加
        for key, val in comps_dict.items():
            try:
                v = float(val)
            except Exception:
                continue
            self.sums[key] += v
            row[key] = v
            wrote = True

        # 逐步记录到 CSV 文件（可选）
        if wrote and self._csv_file is not None:
            if self._csv_writer is None:
                headers = ["timesteps"] + sorted(k for k in row.keys() if k != "timesteps")
                self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=headers)
                self._csv_writer.writeheader()
            self._csv_writer.writerow(row)

        if wrote:
            self.count += 1
        return True

    def _on_rollout_end(self) -> None:
        if self.count == 0:
            return
        # 阶段结束时，计算平均值并输出到控制台/TensorBoard
        for k, s in self.sums.items():
            self.logger.record(f"reward_components/{k}_mean", s / self.count)
        self._reset_sums()