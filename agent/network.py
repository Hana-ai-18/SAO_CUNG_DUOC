"""
network.py  –  Actor-Critic CNN cho BomIT
-----------------------------------------
Input: spatial (9,13,13) + scalar (9,)
Output: logits (6,) + value (1,)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, ch), ch)
        self.norm2 = nn.GroupNorm(min(8, ch), ch)

    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return F.relu(out + x)


class BomberNet(nn.Module):
    N_ACTIONS = 6   # STOP LEFT RIGHT UP DOWN BOMB
    N_SPATIAL = 9   # kênh input
    N_SCALAR  = 9   # scalar dim

    def __init__(self):
        super().__init__()

        # ── CNN backbone ─────────────────────────────────────────────────
        self.cnn = nn.Sequential(
            nn.Conv2d(self.N_SPATIAL, 32, 3, padding=1),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            ResBlock(64),
            ResBlock(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),   # → (64,4,4) = 1024
        )

        # ── Scalar MLP ───────────────────────────────────────────────────
        self.scalar_net = nn.Sequential(
            nn.Linear(self.N_SCALAR, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # ── Fusion ───────────────────────────────────────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(1024 + 64, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.actor_head  = nn.Linear(256, self.N_ACTIONS)
        self.critic_head = nn.Linear(256, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)

    def forward(self, sp, sc):
        feat = self.cnn(sp).view(sp.size(0), -1)
        feat = torch.cat([feat, self.scalar_net(sc)], dim=1)
        feat = self.fusion(feat)
        return self.actor_head(feat), self.critic_head(feat)

    def get_action(self, sp, sc, mask=None, deterministic=False):
        """
        mask: BoolTensor (B, N_ACTIONS) – True = action bị cấm
        Returns (action_int, log_prob_tensor, entropy_tensor, value_tensor)
        """
        with torch.no_grad():
            logits, value = self.forward(sp, sc)
        if mask is not None:
            logits = logits.masked_fill(mask, -1e9)
        dist   = Categorical(logits=logits)
        action = logits.argmax(-1) if deterministic else dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy(), value

    def evaluate_actions(self, sp, sc, actions, mask=None):
        logits, value = self.forward(sp, sc)
        if mask is not None:
            logits = logits.masked_fill(mask, -1e9)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value.squeeze(-1)