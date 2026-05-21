"""
agent.py  (file nộp thi)
------------------------
Agent kết hợp:
  1. Neural network (PPO trained) – chiến thuật cao cấp
  2. Safety layer rule-based – phản xạ thoát bom tức thời
  3. Hybrid decision: nếu nguy hiểm → rule-based override,
                      ngược lại    → neural policy

Cách dùng:
    agent = BomITAgent(model_path="checkpoints/agent_final.pt")
    action = agent.act(obs)
"""

import os
import torch
import numpy as np
from collections import deque
from typing import Optional

from agent.network import BomberNet
from utils.state_encoder import encode_state, TILE_WALL, TILE_CRATE, TILE_EMPTY


# ── Hằng số hành động (điều chỉnh theo engine) ─────────────────────────────
ACTION_STOP  = 0
ACTION_UP    = 1
ACTION_DOWN  = 2
ACTION_LEFT  = 3
ACTION_RIGHT = 4
ACTION_BOMB  = 5

MOVE_DELTAS = {
    ACTION_UP:    (-1,  0),
    ACTION_DOWN:  ( 1,  0),
    ACTION_LEFT:  ( 0, -1),
    ACTION_RIGHT: ( 0,  1),
}


class SafetyLayer:
    """
    Rule-based safety: thoát vùng nguy hiểm bằng BFS tìm ô an toàn gần nhất.
    Chạy trước neural policy; nếu agent đang trong tầm nổ → override hành động.
    """

    DANGER_THRESHOLD = 0.4

    def safe_action(self, obs: dict, agent_id: int) -> Optional[int]:
        board   = np.array(obs['board'])
        H, W    = board.shape
        pos     = tuple(obs['agents'][agent_id]['position'])

        # Xây danger map
        danger  = np.zeros((H, W), dtype=np.float32)
        for bomb in obs.get('bombs', []):
            r, c  = bomb['position']
            timer = bomb.get('timer', 5)
            blast = bomb.get('blast_strength', 3)
            urgency = 1.0 - timer / 10.0
            danger[r][c] = max(danger[r][c], urgency)
            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                for step in range(1, blast + 1):
                    nr, nc = r + dr * step, c + dc * step
                    if not (0 <= nr < H and 0 <= nc < W):
                        break
                    if board[nr][nc] == TILE_WALL:
                        break
                    danger[nr][nc] = max(danger[nr][nc], urgency * 0.9)

        for flame in obs.get('flames', []):
            fr, fc = flame['position']
            danger[fr][fc] = 1.0

        if danger[pos[0]][pos[1]] < self.DANGER_THRESHOLD:
            return None  # an toàn, để neural policy quyết định

        # BFS tìm ô an toàn gần nhất
        from collections import deque as dq
        queue   = dq([(pos, [])])
        visited = {pos}

        while queue:
            (r, c), path = queue.popleft()
            if danger[r][c] < self.DANGER_THRESHOLD and (r, c) != pos:
                # trả về hành động đầu tiên trong path
                return path[0] if path else ACTION_STOP

            for action, (dr, dc) in MOVE_DELTAS.items():
                nr, nc = r + dr, c + dc
                if (0 <= nr < H and 0 <= nc < W and
                        board[nr][nc] not in [TILE_WALL, TILE_CRATE] and
                        (nr, nc) not in visited):
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [action]))

        return ACTION_STOP  # không tìm được đường → đứng yên


class BombDecisionLayer:
    """
    Quyết định có nên đặt bom không (override thêm).
    Tránh đặt bom tự sát = vừa đặt bom vừa không có đường thoát.
    """
    def should_bomb(self, obs: dict, agent_id: int) -> bool:
        board = np.array(obs['board'])
        H, W  = board.shape
        pos   = tuple(obs['agents'][agent_id]['position'])
        blast = obs['agents'][agent_id].get('blast_strength', 2)

        # Kiểm tra: sau khi đặt bom, có ô thoát không?
        # Giả lập bom tại pos
        safe_exits = self._count_safe_exits(board, pos, blast, H, W)
        return safe_exits >= 1

    def _count_safe_exits(self, board, pos, blast, H, W):
        from collections import deque as dq
        r0, c0 = pos
        # Tính vùng nổ
        blast_zone = {pos}
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            for step in range(1, blast + 1):
                nr, nc = r0 + dr * step, c0 + dc * step
                if not (0 <= nr < H and 0 <= nc < W):
                    break
                if board[nr][nc] == TILE_WALL:
                    break
                blast_zone.add((nr, nc))

        # BFS thoát ra ngoài blast_zone trong 3 bước
        queue   = dq([(pos, 0)])
        visited = {pos}
        exits   = 0
        while queue:
            (r, c), steps = queue.popleft()
            if (r, c) not in blast_zone:
                exits += 1
                if exits >= 1:
                    return exits
            if steps >= 3:
                continue
            for dr, dc in MOVE_DELTAS.values():
                nr, nc = r + dr, c + dc
                if (0 <= nr < H and 0 <= nc < W and
                        board[nr][nc] == TILE_EMPTY and
                        (nr, nc) not in visited):
                    visited.add((nr, nc))
                    queue.append(((nr, nc), steps + 1))
        return exits


# ── Main Agent ────────────────────────────────────────────────────────────────

class BomITAgent:
    """
    Agent kết hợp Neural Policy + Safety Rules.

    Parameters
    ----------
    model_path : đường dẫn đến file .pt đã train
    agent_id   : id của agent mình trên bản đồ (0-3)
    device     : 'cpu' hoặc 'cuda'
    """

    def __init__(self, model_path: Optional[str] = None,
                 agent_id: int = 0, device: str = "cpu"):
        self.agent_id    = agent_id
        self.device      = torch.device(device)
        self.net         = BomberNet().to(self.device)
        self.safety      = SafetyLayer()
        self.bomb_decide = BombDecisionLayer()

        if model_path and os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=self.device)
            self.net.load_state_dict(ckpt['model'])
            print(f"✅ Model loaded: {model_path}")
        else:
            print("⚠️  Không tìm thấy model, dùng random weights (cần train trước!)")

        self.net.eval()

    def act(self, obs: dict) -> int:
        """
        Trả về action int cho 1 bước.

        Parameters
        ----------
        obs : dict quan sát từ env.step() / env.reset() cho agent này

        Returns
        -------
        int : action (0-5)
        """
        # 1. Safety layer – kiểm tra nguy hiểm
        safe_action = self.safety.safe_action(obs, self.agent_id)
        if safe_action is not None:
            return safe_action

        # 2. Neural policy
        sp, sc = encode_state(obs, self.agent_id)
        sp = sp.unsqueeze(0).to(self.device)
        sc = sc.unsqueeze(0).to(self.device)

        action, _, _, _ = self.net.get_action(sp, sc, deterministic=False)

        # 3. Bomb safety check: nếu neural chọn đặt bom → kiểm tra tự sát
        if action == ACTION_BOMB:
            if not self.bomb_decide.should_bomb(obs, self.agent_id):
                # Không đặt bom, chọn di chuyển tốt nhất thay thế
                with torch.no_grad():
                    logits, _ = self.net(sp, sc)
                logits[0][ACTION_BOMB] = -1e9   # mask bom
                action = logits.argmax(dim=-1).item()

        return action


# ── Dùng thử không có env (smoke test) ─────────────────────────────────────
if __name__ == "__main__":
    # Tạo obs giả để test import
    fake_obs = {
        'board': np.zeros((13, 13), dtype=int),
        'bombs': [],
        'flames': [],
        'items': [],
        'agents': [
            {'position': (1, 1), 'alive': True, 'bomb_count': 1,
             'blast_strength': 2, 'can_kick': 0},
            {'position': (1, 11), 'alive': True, 'bomb_count': 1,
             'blast_strength': 2, 'can_kick': 0},
            {'position': (11, 1), 'alive': True, 'bomb_count': 1,
             'blast_strength': 2, 'can_kick': 0},
            {'position': (11, 11), 'alive': True, 'bomb_count': 1,
             'blast_strength': 2, 'can_kick': 0},
        ],
    }

    agent  = BomITAgent(agent_id=0)
    action = agent.act(fake_obs)
    print(f"✅ Smoke test passed | Action: {action}")
