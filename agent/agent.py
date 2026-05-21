"""
agent.py  –  File NỘP THI (đúng interface BTC)
------------------------------------------------
class Agent:
    def __init__(self, agent_id: int)
    def act(self, obs: dict) -> int   # trả về 0-5
"""

import os
import numpy as np
import torch
from typing import Optional

# Import tương đối (khi chạy standalone từ zip)
try:
    from agent.network import BomberNet
    from utils.state_encoder import (encode_state, ACTION_STOP, ACTION_LEFT,
                                     ACTION_RIGHT, ACTION_UP, ACTION_DOWN,
                                     ACTION_BOMB, MOVE_DELTAS,
                                     TILE_WALL, TILE_BOX, TILE_GRASS,
                                     TILE_ITEM_RAD, TILE_ITEM_CAP)
except ImportError:
    from network import BomberNet
    from state_encoder import (encode_state, ACTION_STOP, ACTION_LEFT,
                                ACTION_RIGHT, ACTION_UP, ACTION_DOWN,
                                ACTION_BOMB, MOVE_DELTAS,
                                TILE_WALL, TILE_BOX, TILE_GRASS,
                                TILE_ITEM_RAD, TILE_ITEM_CAP)


# ── Safety Layer ──────────────────────────────────────────────────────────────

class SafetyLayer:
    """
    BFS tìm đường thoát khỏi vùng nguy hiểm.
    Override neural policy khi đang trong tầm nổ.
    """
    DANGER_THRESH = 0.30   # timer <= ~5/7 → nguy hiểm

    def safe_action(self, obs: dict, agent_id: int) -> Optional[int]:
        board   = np.array(obs['map'],     dtype=np.int32)
        players = np.array(obs['players'], dtype=np.int32)
        bombs   = np.array(obs['bombs'],   dtype=np.int32)
        H, W    = board.shape
        pos     = (int(players[agent_id][0]), int(players[agent_id][1]))

        # Xây danger map
        danger = np.zeros((H, W), dtype=np.float32)
        if bombs.ndim == 2 and len(bombs) > 0:
            for b in bombs:
                br, bc, timer, owner = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                owner_bonus = int(players[owner][4]) if 0 <= owner < 4 else 0
                blast   = 1 + owner_bonus
                urgency = 1.0 - min(timer, 7) / 7.0
                danger[br][bc] = max(danger[br][bc], urgency)
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    for step in range(1, blast+1):
                        nr, nc = br+dr*step, bc+dc*step
                        if not (0 <= nr < H and 0 <= nc < W):
                            break
                        if board[nr][nc] == TILE_WALL:
                            break
                        danger[nr][nc] = max(danger[nr][nc], urgency * 0.9)
                        if board[nr][nc] == TILE_BOX:
                            break

        if danger[pos[0]][pos[1]] < self.DANGER_THRESH:
            return None   # an toàn → để neural quyết định

        # BFS tìm ô an toàn gần nhất
        from collections import deque
        queue   = deque([(pos, [])])
        visited = {pos}
        while queue:
            (r, c), path = queue.popleft()
            if danger[r][c] < self.DANGER_THRESH and (r, c) != pos:
                return path[0] if path else ACTION_STOP
            for act, (dr, dc) in MOVE_DELTAS.items():
                nr, nc = r+dr, c+dc
                if (0 <= nr < H and 0 <= nc < W
                        and board[nr][nc] not in [TILE_WALL, TILE_BOX]
                        and (nr, nc) not in visited):
                    # Tránh đi vào ô có bom đã tồn tại (theo rule BTC)
                    if not _has_existing_bomb(bombs, nr, nc):
                        visited.add((nr, nc))
                        queue.append(((nr, nc), path + [act]))

        return ACTION_STOP


def _has_existing_bomb(bombs, r, c):
    if bombs.ndim == 2 and len(bombs) > 0:
        for b in bombs:
            if int(b[0]) == r and int(b[1]) == c:
                return True
    return False


# ── Bomb Decision Layer ───────────────────────────────────────────────────────

class BombDecisionLayer:
    """
    Kiểm tra xem có nên đặt bom không.
    Từ chối nếu: không có đường thoát sau khi đặt bom.
    """
    MIN_EXITS = 1

    def should_bomb(self, obs: dict, agent_id: int) -> bool:
        board   = np.array(obs['map'],     dtype=np.int32)
        players = np.array(obs['players'], dtype=np.int32)
        bombs   = np.array(obs['bombs'],   dtype=np.int32)
        H, W    = board.shape

        me           = players[agent_id]
        bombs_left   = int(me[3])
        radius_bonus = int(me[4])
        blast        = 1 + radius_bonus
        pos          = (int(me[0]), int(me[1]))

        if bombs_left <= 0:
            return False

        # Tính blast zone nếu đặt bom tại pos
        blast_zone = {pos}
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            for step in range(1, blast+1):
                nr, nc = pos[0]+dr*step, pos[1]+dc*step
                if not (0 <= nr < H and 0 <= nc < W):
                    break
                if board[nr][nc] == TILE_WALL:
                    break
                blast_zone.add((nr, nc))
                if board[nr][nc] == TILE_BOX:
                    break

        # BFS tìm ô thoát khỏi blast_zone trong 7 bước (timer=7)
        from collections import deque
        queue   = deque([(pos, 0)])
        visited = {pos}
        exits   = 0
        while queue:
            (r, c), steps = queue.popleft()
            if (r, c) not in blast_zone:
                exits += 1
                if exits >= self.MIN_EXITS:
                    return True
            if steps >= 7:
                continue
            for dr, dc in MOVE_DELTAS.values():
                nr, nc = r+dr, c+dc
                if (0 <= nr < H and 0 <= nc < W
                        and board[nr][nc] not in [TILE_WALL, TILE_BOX]
                        and not _has_existing_bomb(bombs, nr, nc)
                        and (nr, nc) not in visited):
                    visited.add((nr, nc))
                    queue.append(((nr, nc), steps+1))
        return exits >= self.MIN_EXITS


# ── Action Mask ───────────────────────────────────────────────────────────────

def compute_action_mask(obs: dict, agent_id: int) -> torch.Tensor:
    """
    Trả về BoolTensor (1, 6): True = action không hợp lệ.
    Mask:
      - Đi vào tường/box/bom cũ
      - Đặt bom khi bombs_left=0 hoặc đang đứng trên bom
    """
    board   = np.array(obs['map'],     dtype=np.int32)
    players = np.array(obs['players'], dtype=np.int32)
    bombs   = np.array(obs['bombs'],   dtype=np.int32)
    H, W    = board.shape

    me         = players[agent_id]
    r, c       = int(me[0]), int(me[1])
    bombs_left = int(me[3])

    mask = [False] * 6   # False = hợp lệ

    # Mask hành động di chuyển
    for act, (dr, dc) in MOVE_DELTAS.items():
        nr, nc = r+dr, c+dc
        if not (0 <= nr < H and 0 <= nc < W):
            mask[act] = True
        elif board[nr][nc] in [TILE_WALL, TILE_BOX]:
            mask[act] = True
        elif _has_existing_bomb(bombs, nr, nc):
            # Không đi vào ô có bom đã tồn tại
            # (trừ ô mình đang đứng - nhưng mình chỉ check ô mới)
            mask[act] = True

    # Mask đặt bom
    if bombs_left <= 0 or _has_existing_bomb(bombs, r, c):
        mask[ACTION_BOMB] = True

    return torch.tensor([mask], dtype=torch.bool)   # (1,6)


# ── Main Agent (BTC interface) ────────────────────────────────────────────────

class Agent:
    """
    Interface chính xác theo yêu cầu BTC.
    File này đặt trong ZIP cùng model.pth.
    """

    MODEL_FILE = 'model.pth'   # tên file model trong ZIP

    def __init__(self, agent_id: int):
        self.agent_id   = agent_id
        self.device     = torch.device('cpu')   # máy chấm không có GPU
        self.net        = BomberNet().to(self.device)
        self.safety     = SafetyLayer()
        self.bomb_check = BombDecisionLayer()

        # Load model từ cùng thư mục với agent.py
        base_dir   = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, self.MODEL_FILE)
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=self.device)
            # hỗ trợ cả 2 format: dict có 'model' key, hoặc raw state_dict
            state = ckpt.get('model', ckpt)
            self.net.load_state_dict(state)
            print(f'[Agent {agent_id}] Model loaded: {model_path}')
        else:
            print(f'[Agent {agent_id}] WARNING: model.pth not found, using random weights')

        self.net.eval()

    def act(self, obs: dict) -> int:
        """
        Phải trả về trong 100ms (yêu cầu BTC).
        Pipeline: Safety → Neural (+ action mask) → Bomb check
        """
        # 1. Safety override khi đang trong tầm nổ
        safe_act = self.safety.safe_action(obs, self.agent_id)
        if safe_act is not None:
            return safe_act

        # 2. Neural policy với action mask
        sp, sc = encode_state(obs, self.agent_id)
        sp_d   = sp.unsqueeze(0).to(self.device)
        sc_d   = sc.unsqueeze(0).to(self.device)
        mask   = compute_action_mask(obs, self.agent_id).to(self.device)

        action, _, _, _ = self.net.get_action(sp_d, sc_d, mask=mask)

        # 3. Nếu chọn BOMB → kiểm tra tự sát
        if action == ACTION_BOMB:
            if not self.bomb_check.should_bomb(obs, self.agent_id):
                # Mask bom rồi chọn lại
                with torch.no_grad():
                    logits, _ = self.net(sp_d, sc_d)
                mask_bomb         = mask.clone()
                mask_bomb[0][ACTION_BOMB] = True
                logits = logits.masked_fill(mask_bomb, -1e9)
                action = logits.argmax(-1).item()

        return int(action)


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    board = np.zeros((13, 13), dtype=np.int32)
    # Thêm tường viền
    board[0, :] = board[12, :] = board[:, 0] = board[:, 12] = 1

    fake_obs = {
        'map'    : board,
        'players': np.array([
            [1, 1,  1, 1, 0],
            [11,11, 1, 1, 0],
            [1, 11, 1, 1, 0],
            [11, 1, 1, 1, 0],
        ], dtype=np.int32),
        'bombs'  : np.zeros((0, 4), dtype=np.int32),
    }

    agent  = Agent(agent_id=0)
    action = agent.act(fake_obs)
    names  = ['STOP','LEFT','RIGHT','UP','DOWN','BOMB']
    print(f'Smoke test OK → action={action} ({names[action]})')