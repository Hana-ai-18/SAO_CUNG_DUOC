"""
train.py  –  Entry point để chạy training
------------------------------------------
Cách dùng:

  # Train từ đầu (5000 episodes)
  python train.py

  # Tiếp tục từ checkpoint
  python train.py --resume checkpoints/agent_2000.pt

  # Chỉ định GPU
  python train.py --device cuda

  # Tùy chỉnh tham số
  python train.py --episodes 10000 --lr 1e-4 --n_steps 1024
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import torch
from training.ppo_trainer import PPOTrainer, PPOConfig


# ───────────────────────────────────────────────────────────────────────────
#  Wrapper môi trường  (thay bằng engine thực của BomIT)
# ───────────────────────────────────────────────────────────────────────────

def make_env():
    """
    Factory tạo môi trường.

    ⚠️  Thay đoạn này bằng code khởi tạo engine BomIT thực tế.
    Interface cần:
      env.reset()          → dict[agent_id -> obs]
      env.step(actions)    → (obs_dict, reward_dict, done_dict, info_dict)
      env.close()

    Hiện tại dùng mock env để code chạy được mà không cần engine thật.
    """
    try:
        # Thử import engine BomIT (nếu đã cài)
        from bomit_engine import BomITEnv
        return BomITEnv(grid_size=13, n_agents=4, max_steps=800)
    except ImportError:
        # Mock env để test pipeline
        return MockBomITEnv()


class MockBomITEnv:
    """Môi trường giả cho phép test pipeline training mà không cần engine."""
    import numpy as _np

    def __init__(self):
        self.step_count = 0
        self.max_steps  = 200

    def reset(self):
        self.step_count = 0
        return {i: self._fake_obs(i) for i in range(4)}

    def step(self, actions):
        self.step_count += 1
        done_all = self.step_count >= self.max_steps
        obs = {i: self._fake_obs(i) for i in range(4)}
        rewards = {i: 0.0 for i in range(4)}
        dones   = {i: done_all for i in range(4)}
        dones['__all__'] = done_all
        infos   = {i: {'crate_destroyed': 0, 'item_picked': False,
                       'dead': False, 'win': False} for i in range(4)}
        return obs, rewards, dones, infos

    def close(self):
        pass

    def _fake_obs(self, agent_id: int) -> dict:
        import numpy as np
        board = np.zeros((13, 13), dtype=int)
        positions = [(1,1),(1,11),(11,1),(11,11)]
        return {
            'board'  : board,
            'bombs'  : [],
            'flames' : [],
            'items'  : [],
            'agents' : [
                {'position': positions[i], 'alive': True,
                 'bomb_count': 1, 'blast_strength': 2, 'can_kick': 0}
                for i in range(4)
            ],
        }


# ───────────────────────────────────────────────────────────────────────────
#  Main
# ───────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="BomIT PPO Trainer")
    p.add_argument("--episodes",  type=int,   default=5000)
    p.add_argument("--lr",        type=float, default=3e-4)
    p.add_argument("--n_steps",   type=int,   default=512)
    p.add_argument("--batch_size",type=int,   default=64)
    p.add_argument("--clip_eps",  type=float, default=0.2)
    p.add_argument("--device",    type=str,   default="cpu",
                   help="cpu hoặc cuda")
    p.add_argument("--resume",    type=str,   default=None,
                   help="Đường dẫn checkpoint để tiếp tục train")
    p.add_argument("--agent_id",  type=int,   default=0)
    return p.parse_args()


def main():
    args = parse_args()

    cfg = PPOConfig(
        lr         = args.lr,
        n_steps    = args.n_steps,
        batch_size = args.batch_size,
        clip_eps   = args.clip_eps,
    )

    device  = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA không có sẵn, dùng CPU")
        device = "cpu"

    trainer = PPOTrainer(
        env_factory = make_env,
        cfg         = cfg,
        agent_id    = args.agent_id,
        device      = device,
    )

    if args.resume:
        trainer.load(args.resume)

    trainer.train(total_episodes=args.episodes)


if __name__ == "__main__":
    main()
