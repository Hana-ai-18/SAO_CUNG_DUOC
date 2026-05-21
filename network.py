"""
network.py
----------
Kiến trúc mạng Actor-Critic dùng cho PPO.

Thiết kế:
  1. CNN backbone xử lý spatial features (5 kênh)
  2. MLP nhỏ xử lý scalar features
  3. Fusion layer kết hợp cả hai
  4. Tách thành 2 đầu: Actor (policy) + Critic (value)

Lý do chọn CNN: bản đồ Bomberman có cấu trúc không gian rõ ràng,
CNN học được pattern "bom gần → nguy hiểm" tốt hơn MLP thuần.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ResBlock(nn.Module):
    """Residual block nhỏ giúp gradient chảy tốt hơn khi train sâu."""
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LayerNorm([channels, 1, 1]),   # sẽ broadcast
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        # dùng LayerNorm trên channel dimension
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.norm1(self.net[0](x)))
        out = self.norm2(self.net[2](out))
        return F.relu(out + residual)


class BomberNet(nn.Module):
    """
    Actor-Critic network cho BomIT agent.

    Input:
      spatial : (B, 5, H, W)   – feature map 5 kênh
      scalar  : (B, 8)          – scalar stats của agent

    Output:
      logits  : (B, n_actions)  – log-probabilities cho Actor
      value   : (B, 1)          – state value cho Critic
    """

    N_ACTIONS = 6   # STOP, UP, DOWN, LEFT, RIGHT, BOMB

    def __init__(self, grid_size: int = 13, scalar_dim: int = 8):
        super().__init__()
        self.grid_size  = grid_size
        self.scalar_dim = scalar_dim

        # ── CNN backbone ──────────────────────────────
        self.cnn = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, padding=1),   # (B,32,13,13)
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (B,64,13,13)
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            ResBlock(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),                  # (B,64,4,4)
        )
        cnn_out = 64 * 4 * 4   # 1024

        # ── Scalar MLP ────────────────────────────────
        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # ── Fusion + heads ────────────────────────────
        fusion_dim = cnn_out + 64    # 1088

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512),
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
        # actor head khởi tạo nhỏ để tránh policy collapse
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)

    def forward(self, spatial: torch.Tensor, scalar: torch.Tensor):
        # CNN path
        cnn_feat = self.cnn(spatial)                          # (B,64,4,4)
        cnn_flat = cnn_feat.view(cnn_feat.size(0), -1)       # (B,1024)

        # Scalar path
        sc_feat  = self.scalar_mlp(scalar)                   # (B,64)

        # Fusion
        fused    = torch.cat([cnn_flat, sc_feat], dim=1)     # (B,1088)
        feat     = self.fusion(fused)                         # (B,256)

        logits   = self.actor_head(feat)                      # (B,6)
        value    = self.critic_head(feat)                     # (B,1)
        return logits, value

    def get_action(self, spatial: torch.Tensor, scalar: torch.Tensor,
                   deterministic: bool = False):
        """
        Sample 1 action (dùng lúc inference).
        Returns (action_int, log_prob, entropy, value)
        """
        with torch.no_grad():
            logits, value = self.forward(spatial, scalar)
        dist     = Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy(), value

    def evaluate_actions(self, spatial: torch.Tensor, scalar: torch.Tensor,
                         actions: torch.Tensor):
        """Dùng lúc update PPO."""
        logits, value = self.forward(spatial, scalar)
        dist          = Categorical(logits=logits)
        log_probs     = dist.log_prob(actions)
        entropy       = dist.entropy()
        return log_probs, entropy, value.squeeze(-1)
