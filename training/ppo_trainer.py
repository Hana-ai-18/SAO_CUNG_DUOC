"""
ppo_trainer.py  –  PPO + Curriculum + Self-Play cho BomIT thật
"""

import os, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List

from agent.network import BomberNet
from utils.state_encoder import encode_state
from agent.agent import compute_action_mask


@dataclass
class PPOConfig:
    # Env
    max_steps        : int   = 500    # BTC giới hạn 500 steps
    # PPO
    lr               : float = 3e-4
    gamma            : float = 0.99
    gae_lambda       : float = 0.95
    clip_eps         : float = 0.2
    entropy_coef     : float = 0.02   # cao hơn 1 chút để khám phá tốt hơn
    value_coef       : float = 0.5
    max_grad_norm    : float = 0.5
    n_steps          : int   = 512
    n_epochs         : int   = 4
    batch_size       : int   = 64
    # Curriculum phases: 0=explore 1=survive 2=fight
    phase_thresh_0   : float = 0.25
    phase_thresh_1   : float = 0.45
    # Reward
    win_reward       : float = 1.0
    kill_reward      : float = 0.3    # thưởng khi địch chết gần bom mình
    death_penalty    : float = -0.8
    box_reward       : float = 0.04
    item_reward      : float = 0.15
    step_penalty     : float = -0.001
    survival_bonus   : float = 0.002  # per step khi còn sống (phase<=1)
    explore_coef     : float = 0.3
    # Self-play
    pool_size        : int   = 10
    pool_update_freq : int   = 200
    # Save
    save_dir         : str   = 'checkpoints'
    save_freq        : int   = 500
    log_freq         : int   = 50


class RolloutBuffer:
    def reset(self):
        self.sp    = []   # spatial Tensor
        self.sc    = []   # scalar Tensor
        self.acts  = []   # int
        self.lps   = []   # log_prob Tensor
        self.rews  = []   # float
        self.vals  = []   # float
        self.dones = []   # float
        self.masks = []   # BoolTensor (1,6)

    def __init__(self):
        self.reset()

    def add(self, sp, sc, act, lp, rew, val, done, mask):
        self.sp.append(sp);    self.sc.append(sc)
        self.acts.append(act); self.lps.append(lp)
        self.rews.append(rew); self.vals.append(val)
        self.dones.append(done); self.masks.append(mask)

    def __len__(self): return len(self.acts)

    def compute_gae(self, last_val, gamma, lam):
        adv, gae, nv = [], 0.0, last_val
        for i in reversed(range(len(self.rews))):
            delta = self.rews[i] + gamma * nv * (1-self.dones[i]) - self.vals[i]
            gae   = delta + gamma * lam * (1-self.dones[i]) * gae
            adv.insert(0, gae)
            nv = self.vals[i]
        ret = [a+v for a,v in zip(adv, self.vals)]
        return adv, ret

    def tensors(self, adv, ret):
        s   = torch.stack(self.sp)
        sc  = torch.stack(self.sc)
        a   = torch.tensor(self.acts, dtype=torch.long)
        lp  = torch.cat(self.lps)
        adv = torch.tensor(adv, dtype=torch.float32)
        ret = torch.tensor(ret, dtype=torch.float32)
        msk = torch.cat(self.masks)   # (N,6)
        adv = (adv-adv.mean())/(adv.std()+1e-8) if len(adv)>1 else adv*0
        return s, sc, a, lp, adv, ret, msk


class RewardShaper:
    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg

    def shape(self, raw, info, phase, prev_alive_count, cur_alive_count):
        r = raw
        r += self.cfg.step_penalty

        # Thưởng phá box
        r += self.cfg.box_reward * info.get('boxes_destroyed', 0)

        # Thưởng nhặt item
        r += self.cfg.item_reward * info.get('items_collected', 0)

        # Thưởng kill (địch chết khi mình còn sống)
        kills = max(0, prev_alive_count - cur_alive_count)
        r += self.cfg.kill_reward * kills

        # Phạt chết
        if info.get('dead', False):
            r += self.cfg.death_penalty

        # Thưởng thắng
        if info.get('win', False):
            r += self.cfg.win_reward

        # Phase 0: thưởng khám phá
        if phase == 0:
            r += info.get('new_tiles', 0) * self.cfg.explore_coef * 0.01

        # Phase <=1: thưởng tồn tại
        if phase <= 1 and not info.get('dead', False):
            r += self.cfg.survival_bonus

        return r


class PPOTrainer:
    def __init__(self, env_factory, cfg=None, agent_id=0, device='cpu'):
        self.cfg         = cfg or PPOConfig()
        self.agent_id    = agent_id
        self.device      = torch.device(device)
        self.env_factory = env_factory

        self.net   = BomberNet().to(self.device)
        self.opt   = optim.Adam(self.net.parameters(), lr=self.cfg.lr, eps=1e-5)
        self.sched = optim.lr_scheduler.LinearLR(
            self.opt, start_factor=1.0, end_factor=0.1, total_iters=10000)

        self.pool: List[BomberNet] = []
        self._add_to_pool()

        self.buf        = RolloutBuffer()
        self.shaper     = RewardShaper(self.cfg)
        self.win_hist   = deque(maxlen=100)
        self.phase      = 0
        self.global_ep  = 0
        Path(self.cfg.save_dir).mkdir(exist_ok=True)

    def _add_to_pool(self):
        snap = BomberNet().to(self.device)
        snap.load_state_dict(self.net.state_dict())
        snap.eval()
        self.pool.append(snap)
        if len(self.pool) > self.cfg.pool_size:
            self.pool.pop(0)

    def _update_phase(self, wr):
        thresholds = [self.cfg.phase_thresh_0, self.cfg.phase_thresh_1]
        if self.phase < 2 and wr > thresholds[self.phase]:
            self.phase += 1
            print(f'  🎓 Phase → {self.phase}')
            self.cfg.explore_coef    *= 0.3
            self.cfg.survival_bonus  *= 0.3

    def train(self, total_episodes=5000):
        print(f'🚀 Training | device={self.device} | episodes={total_episodes}')
        t0 = time.time()
        for ep in range(1, total_episodes+1):
            self.global_ep = ep
            win = self._episode()
            self.win_hist.append(float(win))
            wr  = float(np.mean(self.win_hist))
            self._update_phase(wr)
            self.sched.step()

            if ep % self.cfg.pool_update_freq == 0:
                self._add_to_pool()
            if ep % self.cfg.save_freq == 0:
                self._save(ep)
            if ep % self.cfg.log_freq == 0:
                lr = self.opt.param_groups[0]['lr']
                print(f'  Ep {ep:5d} | Ph {self.phase} | WR {wr:.3f} '
                      f'| LR {lr:.2e} | {time.time()-t0:.0f}s')

        self._save('final')
        print('✅ Training hoàn thành!')

    def _episode(self):
        env     = self.env_factory()
        opps    = [random.choice(self.pool) for _ in range(3)]
        obs_all = env.reset()
        done    = False
        step    = 0
        self.buf.reset()

        # Đếm số địch sống để tính kill reward
        prev_alive = sum(1 for i,p in enumerate(np.array(obs_all[self.agent_id]['players']))
                         if i != self.agent_id and int(p[2]) == 1)

        while not done and step < self.cfg.max_steps:
            obs  = obs_all[self.agent_id]
            sp, sc = encode_state(obs, self.agent_id)
            mask = compute_action_mask(obs, self.agent_id)
            sp_d = sp.unsqueeze(0).to(self.device)
            sc_d = sc.unsqueeze(0).to(self.device)
            msk_d = mask.to(self.device)

            act, lp, _, val = self.net.get_action(sp_d, sc_d, mask=msk_d)

            # Opponent actions
            opp_acts = {}
            opp_ids  = [i for i in range(4) if i != self.agent_id]
            for i, opp in zip(opp_ids, opps):
                if int(np.array(obs_all[i]['players'])[i][2]) == 0:
                    opp_acts[i] = 0   # đã chết
                    continue
                osp, osc = encode_state(obs_all[i], i)
                om = compute_action_mask(obs_all[i], i).to(self.device)
                a, _, _, _ = opp.get_action(
                    osp.unsqueeze(0).to(self.device),
                    osc.unsqueeze(0).to(self.device), mask=om)
                opp_acts[i] = a

            full_acts = {**{self.agent_id: act}, **opp_acts}
            obs_all, rews, dones, infos = env.step(full_acts)

            # Đếm alive sau bước
            cur_alive = sum(1 for i,p in enumerate(np.array(obs_all[self.agent_id]['players']))
                            if i != self.agent_id and int(p[2]) == 1)

            r    = self.shaper.shape(rews[self.agent_id], infos[self.agent_id],
                                     self.phase, prev_alive, cur_alive)
            prev_alive = cur_alive
            done = dones.get('__all__', False) or dones.get(self.agent_id, False)

            self.buf.add(sp, sc, act, lp.cpu(), r,
                         val.cpu().item(), float(done), mask)
            step += 1

            if len(self.buf) >= self.cfg.n_steps:
                self._update()
                self.buf.reset()

        if len(self.buf) > 0:
            self._update()

        win = infos.get(self.agent_id, {}).get('win', False)
        env.close()
        return win

    def _update(self):
        adv_l, ret_l = self.buf.compute_gae(0.0, self.cfg.gamma, self.cfg.gae_lambda)
        s, sc, a, olp, adv, ret, msk = self.buf.tensors(adv_l, ret_l)
        s, sc, a = s.to(self.device), sc.to(self.device), a.to(self.device)
        adv, ret = adv.to(self.device), ret.to(self.device)
        olp = olp.to(self.device).detach()
        msk = msk.to(self.device)

        n = len(a)
        for _ in range(self.cfg.n_epochs):
            idx = torch.randperm(n)
            for st in range(0, n, self.cfg.batch_size):
                mb  = idx[st:st+self.cfg.batch_size]
                lp, ent, val = self.net.evaluate_actions(s[mb], sc[mb], a[mb], msk[mb])
                ratio   = torch.exp(lp - olp[mb])
                clip_r  = ratio.clamp(1-self.cfg.clip_eps, 1+self.cfg.clip_eps)
                pg_loss = -torch.min(ratio*adv[mb], clip_r*adv[mb]).mean()
                v_loss  = F.mse_loss(val, ret[mb])
                loss    = pg_loss + self.cfg.value_coef*v_loss - self.cfg.entropy_coef*ent.mean()

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
                self.opt.step()

    def _save(self, tag):
        path = os.path.join(self.cfg.save_dir, f'agent_{tag}.pt')
        torch.save({'model': self.net.state_dict(),
                    'optim': self.opt.state_dict(),
                    'phase': self.phase,
                    'ep'   : self.global_ep}, path)
        print(f'  💾 {path}')

    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ck['model'])
        self.opt.load_state_dict(ck['optim'])
        self.phase     = ck.get('phase', 0)
        self.global_ep = ck.get('ep', 0)
        print(f'  📂 Loaded {path} | phase={self.phase} | ep={self.global_ep}')