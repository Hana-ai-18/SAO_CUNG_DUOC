"""
state_encoder.py
----------------
Chuyển đổi trạng thái game thô sang tensor đầu vào cho mạng neural.

Chiến lược: dùng 3 kênh (channels) riêng biệt như ảnh RGB:
  - Kênh bản đồ tĩnh  : tường, thùng, đường trống
  - Kênh nguy hiểm    : bom, lửa, timer nổ
  - Kênh thực thể      : agent mình, agent địch, vật phẩm

Ngoài ra có vector scalar bổ sung (stats của agent).
"""

import numpy as np
import torch


# ───────────────────────────────────────────────
# Hằng số bản đồ (điều chỉnh theo engine BomIT)
# ───────────────────────────────────────────────
TILE_EMPTY  = 0
TILE_WALL   = 1   # tường cứng, không phá được
TILE_CRATE  = 2   # thùng gỗ, phá bằng bom
TILE_BOMB   = 3
TILE_FIRE   = 4
TILE_ITEM_BLAST  = 5   # vật phẩm tăng sức nổ
TILE_ITEM_SPEED  = 6   # vật phẩm tăng tốc
TILE_ITEM_BOMB   = 7   # vật phẩm thêm bom
TILE_AGENT_SELF  = 8
TILE_AGENT_ENEMY = 9


GRID_SIZE   = 13    # kích thước bản đồ mặc định
N_CHANNELS  = 5     # số kênh feature map


def encode_state(obs: dict, agent_id: int) -> torch.Tensor:
    """
    Parameters
    ----------
    obs       : dict trả về từ env.step() / env.reset()
                Giả sử có các key: 'board', 'bombs', 'agents', 'items', 'flames'
    agent_id  : id agent hiện tại (0-3)

    Returns
    -------
    Tensor shape (N_CHANNELS, GRID_SIZE, GRID_SIZE) + scalar_vec (8,)
    Trả về tuple (spatial_tensor, scalar_tensor)
    """
    board = np.array(obs['board'], dtype=np.float32)   # (H, W)
    H, W  = board.shape

    channels = np.zeros((N_CHANNELS, H, W), dtype=np.float32)

    # --- Kênh 0: tường cứng ---
    channels[0] = (board == TILE_WALL).astype(np.float32)

    # --- Kênh 1: thùng (crates) ---
    channels[1] = (board == TILE_CRATE).astype(np.float32)

    # --- Kênh 2: nguy hiểm (bom + lửa) ---
    for bomb in obs.get('bombs', []):
        r, c   = bomb['position']
        timer  = bomb['timer']          # càng nhỏ càng nguy hiểm
        danger = 1.0 - timer / 10.0    # chuẩn hóa 0-1
        channels[2][r][c] = max(channels[2][r][c], danger)
        # đánh dấu bán kính nổ dự kiến
        blast  = bomb.get('blast_strength', 3)
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            for step in range(1, blast + 1):
                nr, nc = r + dr * step, c + dc * step
                if 0 <= nr < H and 0 <= nc < W:
                    if board[nr][nc] == TILE_WALL:
                        break
                    channels[2][nr][nc] = max(channels[2][nr][nc], danger * 0.8)

    for flame in obs.get('flames', []):
        fr, fc = flame['position']
        channels[2][fr][fc] = 1.0       # lửa = nguy hiểm tối đa

    # --- Kênh 3: vị trí agent mình ---
    my_pos = obs['agents'][agent_id]['position']
    channels[3][my_pos[0]][my_pos[1]] = 1.0

    # --- Kênh 4: agent địch + vật phẩm ---
    for aid, agent in enumerate(obs['agents']):
        if aid == agent_id:
            continue
        if agent.get('alive', True):
            r, c = agent['position']
            channels[4][r][c] = 1.0

    for item in obs.get('items', []):
        r, c = item['position']
        channels[4][r][c] = 0.5   # vật phẩm dùng cường độ 0.5

    spatial = torch.tensor(channels)  # (5, H, W)

    # --- Vector scalar ---
    my_info = obs['agents'][agent_id]
    scalar  = torch.tensor([
        my_pos[0] / H,
        my_pos[1] / W,
        my_info.get('bomb_count', 1)   / 5.0,
        my_info.get('blast_strength', 2) / 10.0,
        my_info.get('can_kick', 0),
        float(is_in_danger(channels[2], my_pos)),
        count_reachable_tiles(board, my_pos) / (H * W),
        count_destroyable_crates(board, my_pos, my_info.get('blast_strength', 2)) / 20.0,
    ], dtype=torch.float32)

    return spatial, scalar


# ─── Tiện ích phụ ───────────────────────────────

def is_in_danger(danger_channel: np.ndarray, pos: tuple, threshold: float = 0.3) -> bool:
    r, c = pos
    return bool(danger_channel[r][c] > threshold)


def count_reachable_tiles(board: np.ndarray, start: tuple, max_steps: int = 8) -> int:
    """BFS đếm ô đi được trong max_steps bước."""
    from collections import deque
    H, W    = board.shape
    visited = {start}
    queue   = deque([(start, 0)])
    count   = 0
    while queue:
        (r, c), steps = queue.popleft()
        count += 1
        if steps >= max_steps:
            continue
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                if board[nr][nc] == TILE_EMPTY and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), steps + 1))
    return count


def count_destroyable_crates(board: np.ndarray, pos: tuple, blast: int) -> int:
    """Đếm số thùng có thể phá nếu đặt bom tại pos."""
    H, W  = board.shape
    r0, c0 = pos
    count = 0
    for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
        for step in range(1, blast + 1):
            nr, nc = r0 + dr * step, c0 + dc * step
            if not (0 <= nr < H and 0 <= nc < W):
                break
            if board[nr][nc] == TILE_WALL:
                break
            if board[nr][nc] == TILE_CRATE:
                count += 1
                break
    return count
