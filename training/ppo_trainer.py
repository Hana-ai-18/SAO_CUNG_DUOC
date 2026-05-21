"""
ppo_trainer.py
--------------
Huấn luyện agent bằng PPO (Proximal Policy Optimization) với:
  - Curriculum Learning theo 3 giai đoạn
  - Population-based Self-Play
  - Reward shaping thích nghi (adaptive annealing)
  - GAE (Generalized Advantage Estimation)

Dựa trên paper IJCAI-GAMMAL 2024:
"Multi-Agent Training for Pommerman: Curriculum Learning
 and Population-based Self-Play Approach"
"""

import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agent.network import BomberNet
from utils.state_encoder import encode_state


# ── Hyperparameters ──────────────────────────────────────────────────────────

@dataclass
class PPOConfig:
    # Môi trường
    grid_size     : int   = 13
    n_agents      : int   = 4
    max_steps     : int   = 800     # bước tối đa mỗi episode

    # PPO cốt lõi
    lr            : float = 3e-4
    gamma         : float = 0.99
    gae_lambda    : float = 0.95
    clip_eps      : float = 0.2
    entropy_coef  : float = 0.01
    value_coef    : float = 0.5
    max_grad_norm : float = 0.5

    # Rollout & update
    n_steps       : int   = 512     # steps per rollout
    n_epochs      : int   = 4       # PPO update epochs
    batch_size    : int   = 64

    # Curriculum
    curriculum_phase : int = 0      # 0=explore, 1=survive, 2=fight
    phase_thresholds : tuple = (0.3, 0.55)  # win-rate để tăng phase

    # Reward shaping (adaptive annealing)
    explore_reward_coef  : float = 0.5   # giảm dần theo phase
    survival_reward_coef : float = 0.3
    win_reward           : float = 1.0
    death_penalty        : float = -0.5
    crate_reward         : float = 0.05
    item_reward          : float = 0.1
    step_penalty         : float = -0.001  # khuyến khích chơi nhanh

    # Self-play pool
    pool_size       : int   = 10
    pool_update_freq: int   = 200   # episodes

    # Lưu / log
    save_dir        : str   = "checkpoints"
    log_freq        : int   = 50


# ── Rollout Buffer ────────────────────────────────────────────────────────────

class RolloutBuffer:
    def __init__(self, n_steps: int, device: str = "cpu"):
        self.n_steps = n_steps
        self.device  = device
        self.reset()

    def reset(self):
        self.spatials    = []
        self.scalars     = []
        self.actions     = []
        self.log_probs   = []
        self.rewards     = []
        self.values      = []
        self.dones       = []

    def add(self, spatial, scalar, action, log_prob, reward, value, done):
        self.spatials.append(spatial)
        self.scalars.append(scalar)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns(self, last_value: float, gamma: float, gae_lambda: float):
        """Tính GAE advantages."""
        advantages = []
        gae        = 0.0
        next_value = last_value
        for i in reversed(range(len(self.rewards))):
            delta = self.rewards[i] + gamma * next_value * (1 - self.dones[i]) - self.values[i]
            gae   = delta + gamma * gae_lambda * (1 - self.dones[i]) * gae
            advantages.insert(0, gae)
            next_value = self.values[i]
        returns = [a + v for a, v in zip(advantages, self.values)]
        return advantages, returns

    def get_tensors(self, advantages, returns):
        s  = torch.stack(self.spatials)
        sc = torch.stack(self.scalars)
        a  = torch.tensor(self.actions,   dtype=torch.long)
        lp = torch.stack(self.log_probs)
        adv= torch.tensor(advantages,     dtype=torch.float32)
        ret= torch.tensor(returns,        dtype=torch.float32)
        # Normalize advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return s, sc, a, lp, adv, ret


# ── Reward Shaper ─────────────────────────────────────────────────────────────

class RewardShaper:
    """
    Reward shaping thích nghi dựa theo curriculum phase.
    Dùng adaptive annealing: giảm dense reward khi agent đã thành thạo.
    """
    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg

    def shape(self, raw_reward: float, info: dict, phase: int) -> float:
        r = raw_reward

        # Phạt mỗi bước (khuyến khích hành động nhanh)
        r += self.cfg.step_penalty

        # Thưởng phá thùng
        if info.get('crate_destroyed', 0) > 0:
            r += self.cfg.crate_reward * info['crate_destroyed']

        # Thưởng nhặt vật phẩm
        if info.get('item_picked', False):
            r += self.cfg.item_reward

        # Phạt chết
        if info.get('dead', False):
            r += self.cfg.death_penalty

        # Phase 0 (Explore): thưởng khám phá
        if phase == 0:
            r += info.get('new_tiles_explored', 0) * self.cfg.explore_reward_coef * 0.01

        # Phase 1 (Survive): thưởng tồn tại lâu
        if phase <= 1:
            r += self.cfg.survival_reward_coef * 0.001  # per step

        return r


# ── PPO Trainer ───────────────────────────────────────────────────────────────

class PPOTrainer:
    """
    Training loop chính.

    Cách dùng:
        trainer = PPOTrainer(env_factory, cfg)
        trainer.train(total_episodes=5000)
    """

    def __init__(self, env_factory, cfg: PPOConfig = PPOConfig(),
                 agent_id: int = 0, device: str = "cpu"):
        self.cfg        = cfg
        self.agent_id   = agent_id
        self.device     = torch.device(device)
        self.env_factory = env_factory

        # Mạng chính (đang học)
        self.net        = BomberNet(cfg.grid_size).to(self.device)
        self.optimizer  = optim.Adam(self.net.parameters(), lr=cfg.lr, eps=1e-5)
        self.scheduler  = optim.lr_scheduler.LinearLR(
                              self.optimizer, start_factor=1.0,
                              end_factor=0.1, total_iters=5000)

        # Pool self-play
        self.opponent_pool: List[BomberNet] = []
        self._add_to_pool()

        self.shaper      = RewardShaper(cfg)
        self.buffer      = RolloutBuffer(cfg.n_steps, device)
        self.win_history = deque(maxlen=100)
        self.phase       = 0

        Path(cfg.save_dir).mkdir(exist_ok=True)
        self.global_episode = 0

    # ── Pool management ──────────────────────────────────────────────────────

    def _add_to_pool(self):
        snapshot = BomberNet(self.cfg.grid_size).to(self.device)
        snapshot.load_state_dict(self.net.state_dict())
        snapshot.eval()
        self.opponent_pool.append(snapshot)
        if len(self.opponent_pool) > self.cfg.pool_size:
            self.opponent_pool.pop(0)

    def _sample_opponent(self) -> BomberNet:
        return random.choice(self.opponent_pool)

    # ── Curriculum phase ─────────────────────────────────────────────────────

    def _update_phase(self, win_rate: float):
        if self.phase < 2:
            threshold = self.cfg.phase_thresholds[self.phase]
            if win_rate > threshold:
                self.phase += 1
                print(f"  🎓 Curriculum lên Phase {self.phase}!")
                # Giảm dense reward khi phase tăng
                self.cfg.explore_reward_coef  *= 0.5
                self.cfg.survival_reward_coef *= 0.5

    # ── Main training loop ───────────────────────────────────────────────────

    def train(self, total_episodes: int = 5000):
        print(f"🚀 Bắt đầu training | Device: {self.device} | Episodes: {total_episodes}")
        start = time.time()

        for ep in range(1, total_episodes + 1):
            self.global_episode = ep
            win = self._run_episode()
            self.win_history.append(float(win))

            win_rate = np.mean(self.win_history) if self.win_history else 0.0
            self._update_phase(win_rate)

            # Thêm snapshot vào pool
            if ep % self.cfg.pool_update_freq == 0:
                self._add_to_pool()
                print(f"  Pool cập nhật: {len(self.opponent_pool)} opponents")

            # Lưu checkpoint
            if ep % 500 == 0:
                self._save(ep)

            # Log
            if ep % self.cfg.log_freq == 0:
                elapsed = time.time() - start
                print(f"  Ep {ep:5d} | Phase {self.phase} | WinRate {win_rate:.3f} "
                      f"| LR {self.scheduler.get_last_lr()[0]:.6f} | {elapsed:.0f}s")

        self._save("final")
        print("✅ Training hoàn thành!")

    # ── Single episode ────────────────────────────────────────────────────────

    def _run_episode(self) -> bool:
        env       = self.env_factory()
        opponents = [self._sample_opponent() for _ in range(3)]
        obs_all   = env.reset()
        done      = False
        step      = 0
        self.buffer.reset()

        while not done and step < self.cfg.max_steps:
            obs    = obs_all[self.agent_id]
            sp, sc = encode_state(obs, self.agent_id)
            sp     = sp.unsqueeze(0).to(self.device)
            sc     = sc.unsqueeze(0).to(self.device)

            # Chọn action
            action, log_prob, _, value = self.net.get_action(sp, sc)

            # Opponent actions (greedy from pool)
            opp_actions = {}
            for i, opp_net in zip([j for j in range(4) if j != self.agent_id],
                                  opponents):
                opp_obs    = obs_all[i]
                osp, osc   = encode_state(opp_obs, i)
                a, _, _, _ = opp_net.get_action(
                    osp.unsqueeze(0).to(self.device),
                    osc.unsqueeze(0).to(self.device),
                    deterministic=False,
                )
                opp_actions[i] = a

            full_actions = {**{self.agent_id: action}, **opp_actions}
            obs_all, rewards, dones, infos = env.step(full_actions)

            raw_r = rewards[self.agent_id]
            info  = infos[self.agent_id]
            r     = self.shaper.shape(raw_r, info, self.phase)

            self.buffer.add(
                sp.squeeze(0).cpu(), sc.squeeze(0).cpu(),
                action, log_prob.cpu().item(), r, value.cpu().item(),
                float(dones[self.agent_id]),
            )

            done = dones.get('__all__', False) or dones.get(self.agent_id, False)
            step += 1

            # Cập nhật khi đủ n_steps
            if len(self.buffer.actions) >= self.cfg.n_steps:
                self._ppo_update(last_value=0.0)
                self.buffer.reset()

        # Cập nhật phần còn lại
        if self.buffer.actions:
            self._ppo_update(last_value=0.0)

        # Kết quả
        win = infos.get(self.agent_id, {}).get('win', False)
        env.close()
        return win

    # ── PPO update step ───────────────────────────────────────────────────────

    def _ppo_update(self, last_value: float):
        advantages, returns = self.buffer.compute_returns(
            last_value, self.cfg.gamma, self.cfg.gae_lambda)
        s, sc, a, old_lp, adv, ret = self.buffer.get_tensors(advantages, returns)
        s   = s.to(self.device)
        sc  = sc.to(self.device)
        a   = a.to(self.device)
        adv = adv.to(self.device)
        ret = ret.to(self.device)
        old_lp = old_lp.to(self.device)

        n = len(a)
        for _ in range(self.cfg.n_epochs):
            # Mini-batch shuffle
            idxs = torch.randperm(n)
            for start in range(0, n, self.cfg.batch_size):
                mb     = idxs[start:start + self.cfg.batch_size]
                lp, ent, val = self.net.evaluate_actions(s[mb], sc[mb], a[mb])

                ratio   = torch.exp(lp - old_lp[mb].detach())
                clip_r  = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps)
                pg_loss = -torch.min(ratio * adv[mb], clip_r * adv[mb]).mean()
                v_loss  = F.mse_loss(val, ret[mb])
                e_loss  = -ent.mean()

                loss = pg_loss + self.cfg.value_coef * v_loss + self.cfg.entropy_coef * e_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

        self.scheduler.step()

    # ── Checkpoint ───────────────────────────────────────────────────────────

    def _save(self, tag):
        path = os.path.join(self.cfg.save_dir, f"agent_{tag}.pt")
        torch.save({
            'model': self.net.state_dict(),
            'optim': self.optimizer.state_dict(),
            'phase': self.phase,
            'episode': self.global_episode,
        }, path)
        print(f"  💾 Saved: {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optim'])
        self.phase          = ckpt.get('phase', 0)
        self.global_episode = ckpt.get('episode', 0)
        print(f"  📂 Loaded: {path} | Phase {self.phase} | Ep {self.global_episode}")


import torch.nn.functional as F
