"""
ppo_trainer.py  (đã fix bugs)
------------------------------
Fix:
  1. Buffer lưu log_prob/value là Tensor (không .item()) để torch.stack được
  2. scheduler.step() chỉ gọi 1 lần mỗi episode (không phải mỗi mini-batch)
  3. import F lên đầu file
  4. phase_thresholds dùng list thay tuple (tránh warning dataclass)
"""

import os
import random
import time
import torch.nn.functional as F
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


@dataclass
class PPOConfig:
    # Môi trường
    grid_size     : int   = 13
    n_agents      : int   = 4
    max_steps     : int   = 800

    # PPO cốt lõi
    lr            : float = 3e-4
    gamma         : float = 0.99
    gae_lambda    : float = 0.95
    clip_eps      : float = 0.2
    entropy_coef  : float = 0.01
    value_coef    : float = 0.5
    max_grad_norm : float = 0.5

    # Rollout & update
    n_steps       : int   = 512
    n_epochs      : int   = 4
    batch_size    : int   = 64

    # Curriculum
    curriculum_phase  : int   = 0
    phase_thresh_0    : float = 0.30   # FIX: tách thành 2 float thay vì tuple
    phase_thresh_1    : float = 0.55

    # Reward shaping
    explore_reward_coef  : float = 0.5
    survival_reward_coef : float = 0.3
    win_reward           : float = 1.0
    death_penalty        : float = -0.5
    crate_reward         : float = 0.05
    item_reward          : float = 0.1
    step_penalty         : float = -0.001

    # Self-play pool
    pool_size       : int = 10
    pool_update_freq: int = 200

    # Lưu / log
    save_dir  : str = "checkpoints"
    log_freq  : int = 50
    save_freq : int = 500


class RolloutBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.spatials  = []
        self.scalars   = []
        self.actions   = []
        self.log_probs = []   # FIX: lưu Tensor, không .item()
        self.rewards   = []
        self.values    = []   # FIX: lưu float (scalar) - OK vì compute_returns dùng trực tiếp
        self.dones     = []

    def add(self, spatial, scalar, action, log_prob_tensor, reward, value_float, done):
        self.spatials.append(spatial)          # Tensor (5,H,W)
        self.scalars.append(scalar)            # Tensor (8,)
        self.actions.append(action)            # int
        self.log_probs.append(log_prob_tensor) # Tensor (1,) - FIX
        self.rewards.append(reward)            # float
        self.values.append(value_float)        # float
        self.dones.append(done)                # float

    def __len__(self):
        return len(self.actions)

    def compute_returns(self, last_value: float, gamma: float, gae_lambda: float):
        advantages = []
        gae        = 0.0
        next_val   = last_value
        for i in reversed(range(len(self.rewards))):
            delta = self.rewards[i] + gamma * next_val * (1 - self.dones[i]) - self.values[i]
            gae   = delta + gamma * gae_lambda * (1 - self.dones[i]) * gae
            advantages.insert(0, gae)
            next_val = self.values[i]
        returns = [a + v for a, v in zip(advantages, self.values)]
        return advantages, returns

    def get_tensors(self, advantages, returns):
        s   = torch.stack(self.spatials)
        sc  = torch.stack(self.scalars)
        a   = torch.tensor(self.actions, dtype=torch.long)
        lp  = torch.cat(self.log_probs)          # FIX: cat Tensors (không stack float)
        adv = torch.tensor(advantages, dtype=torch.float32)
        ret = torch.tensor(returns,    dtype=torch.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8) if len(adv) > 1 else adv * 0
        return s, sc, a, lp, adv, ret


class RewardShaper:
    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg

    def shape(self, raw_reward: float, info: dict, phase: int) -> float:
        r = raw_reward
        r += self.cfg.step_penalty
        r += self.cfg.crate_reward * info.get('crate_destroyed', 0)
        if info.get('item_picked', False):
            r += self.cfg.item_reward
        if info.get('dead', False):
            r += self.cfg.death_penalty
        if phase == 0:
            r += info.get('new_tiles_explored', 0) * self.cfg.explore_reward_coef * 0.01
        if phase <= 1:
            r += self.cfg.survival_reward_coef * 0.001
        return r


class PPOTrainer:
    def __init__(self, env_factory, cfg: PPOConfig = None,
                 agent_id: int = 0, device: str = "cpu"):
        self.cfg         = cfg or PPOConfig()
        self.agent_id    = agent_id
        self.device      = torch.device(device)
        self.env_factory = env_factory

        self.net       = BomberNet(self.cfg.grid_size).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.cfg.lr, eps=1e-5)
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1.0, end_factor=0.1,
            total_iters=5000   # total episodes
        )

        self.opponent_pool: List[BomberNet] = []
        self._add_to_pool()

        self.shaper      = RewardShaper(self.cfg)
        self.buffer      = RolloutBuffer()
        self.win_history = deque(maxlen=100)
        self.phase       = 0
        self.global_ep   = 0

        Path(self.cfg.save_dir).mkdir(exist_ok=True)

    # ── Pool ─────────────────────────────────────────────────────────────────

    def _add_to_pool(self):
        snap = BomberNet(self.cfg.grid_size).to(self.device)
        snap.load_state_dict(self.net.state_dict())
        snap.eval()
        self.opponent_pool.append(snap)
        if len(self.opponent_pool) > self.cfg.pool_size:
            self.opponent_pool.pop(0)

    def _sample_opponent(self):
        return random.choice(self.opponent_pool)

    # ── Curriculum ────────────────────────────────────────────────────────────

    def _update_phase(self, win_rate: float):
        thresholds = [self.cfg.phase_thresh_0, self.cfg.phase_thresh_1]
        if self.phase < 2 and win_rate > thresholds[self.phase]:
            self.phase += 1
            print(f"  🎓 Curriculum lên Phase {self.phase}!")
            self.cfg.explore_reward_coef  *= 0.5
            self.cfg.survival_reward_coef *= 0.5

    # ── Train ─────────────────────────────────────────────────────────────────

    def train(self, total_episodes: int = 5000):
        print(f"🚀 Training | Device: {self.device} | Episodes: {total_episodes}")
        t0 = time.time()

        for ep in range(1, total_episodes + 1):
            self.global_ep = ep
            win = self._run_episode()
            self.win_history.append(float(win))
            win_rate = float(np.mean(self.win_history))

            self._update_phase(win_rate)

            # FIX: scheduler.step() mỗi episode (không phải mỗi mini-batch)
            self.scheduler.step()

            if ep % self.cfg.pool_update_freq == 0:
                self._add_to_pool()

            if ep % self.cfg.save_freq == 0:
                self._save(ep)

            if ep % self.cfg.log_freq == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"  Ep {ep:5d} | Phase {self.phase} | WinRate {win_rate:.3f} "
                      f"| LR {lr:.2e} | {time.time()-t0:.0f}s")

        self._save("final")
        print("✅ Training hoàn thành!")

    # ── Episode ───────────────────────────────────────────────────────────────

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
            sp_d   = sp.unsqueeze(0).to(self.device)
            sc_d   = sc.unsqueeze(0).to(self.device)

            action, log_prob, _, value = self.net.get_action(sp_d, sc_d)

            # Opponent actions
            opp_actions = {}
            for i, opp in zip([j for j in range(4) if j != self.agent_id], opponents):
                osp, osc = encode_state(obs_all[i], i)
                a, _, _, _ = opp.get_action(
                    osp.unsqueeze(0).to(self.device),
                    osc.unsqueeze(0).to(self.device),
                )
                opp_actions[i] = a

            obs_all, rewards, dones, infos = env.step({**{self.agent_id: action}, **opp_actions})

            r    = self.shaper.shape(rewards[self.agent_id], infos[self.agent_id], self.phase)
            done = dones.get('__all__', False) or dones.get(self.agent_id, False)

            # FIX: lưu log_prob là Tensor, value là float
            self.buffer.add(
                sp.cpu(), sc.cpu(),
                action,
                log_prob.cpu(),               # Tensor (1,)
                r,                            # float
                value.cpu().item(),           # float
                float(dones[self.agent_id]),
            )
            step += 1

            if len(self.buffer) >= self.cfg.n_steps:
                self._ppo_update()
                self.buffer.reset()

        if len(self.buffer) > 0:
            self._ppo_update()

        win = infos.get(self.agent_id, {}).get('win', False)
        env.close()
        return win

    # ── PPO update ────────────────────────────────────────────────────────────

    def _ppo_update(self):
        adv_list, ret_list = self.buffer.compute_returns(
            0.0, self.cfg.gamma, self.cfg.gae_lambda)
        s, sc, a, old_lp, adv, ret = self.buffer.get_tensors(adv_list, ret_list)

        s, sc, a  = s.to(self.device), sc.to(self.device), a.to(self.device)
        adv, ret  = adv.to(self.device), ret.to(self.device)
        old_lp    = old_lp.to(self.device).detach()

        n = len(a)
        for _ in range(self.cfg.n_epochs):
            idxs = torch.randperm(n)
            for start in range(0, n, self.cfg.batch_size):
                mb = idxs[start:start + self.cfg.batch_size]
                lp, ent, val = self.net.evaluate_actions(s[mb], sc[mb], a[mb])

                ratio   = torch.exp(lp - old_lp[mb])
                clip_r  = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps)
                pg_loss = -torch.min(ratio * adv[mb], clip_r * adv[mb]).mean()
                v_loss  = F.mse_loss(val, ret[mb])
                loss    = pg_loss + self.cfg.value_coef * v_loss - self.cfg.entropy_coef * ent.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

        # FIX: scheduler KHÔNG gọi ở đây nữa (đã chuyển lên train())

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def _save(self, tag):
        path = os.path.join(self.cfg.save_dir, f"agent_{tag}.pt")
        torch.save({
            'model': self.net.state_dict(),
            'optim': self.optimizer.state_dict(),
            'phase': self.phase,
            'episode': self.global_ep,
        }, path)
        print(f"  💾 Saved: {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optim'])
        self.phase     = ckpt.get('phase', 0)
        self.global_ep = ckpt.get('episode', 0)
        print(f"  📂 Loaded: {path} | Phase {self.phase} | Ep {self.global_ep}")