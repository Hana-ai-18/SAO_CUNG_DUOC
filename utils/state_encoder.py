"""
state_encoder.py  –  khớp đúng format obs của BTC
--------------------------------------------------
obs = {
    "map":     np.ndarray (13,13)  0=Grass,1=Wall,2=Box,3=Item_Radius,4=Item_Capacity
    "players": np.ndarray (4,5)    [row, col, alive, bombs_left, bomb_radius_bonus]
    "bombs":   np.ndarray (N,4)    [row, col, timer, owner_id]
}
Actions: 0=STOP 1=LEFT 2=RIGHT 3=UP 4=DOWN 5=PLACE_BOMB
"""

import numpy as np
import torch

# ── Hằng số map ──────────────────────────────────────────────────────────────
TILE_GRASS    = 0
TILE_WALL     = 1
TILE_BOX      = 2
TILE_ITEM_RAD = 3
TILE_ITEM_CAP = 4

GRID   = 13
N_CH   = 9     # số kênh feature map
SCALAR = 9     # chiều vector scalar

# action constants (đúng theo BTC)
ACTION_STOP  = 0
ACTION_LEFT  = 1
ACTION_RIGHT = 2
ACTION_UP    = 3
ACTION_DOWN  = 4
ACTION_BOMB  = 5

MOVE_DELTAS = {
    ACTION_LEFT:  ( 0, -1),
    ACTION_RIGHT: ( 0,  1),
    ACTION_UP:    (-1,  0),
    ACTION_DOWN:  ( 1,  0),
}


def encode_state(obs: dict, agent_id: int):
    """
    Trả về (spatial Tensor (9,13,13), scalar Tensor (9,))

    Kênh spatial:
      0  wall
      1  box
      2  item_radius
      3  item_capacity
      4  danger map  (bom + lửa, cường độ theo timer)
      5  blast zone  (vùng nổ dự kiến của bom)
      6  vị trí mình
      7  vị trí địch (alive)
      8  escape map  (BFS: số bước thoát được – chuẩn hóa)
    """
    board   = np.array(obs['map'],     dtype=np.int32)    # (13,13)
    players = np.array(obs['players'], dtype=np.int32)    # (4,5)
    bombs   = np.array(obs['bombs'],   dtype=np.int32)    # (N,4)
    H, W    = board.shape

    ch = np.zeros((N_CH, H, W), dtype=np.float32)

    # ── Kênh 0-3: map tĩnh ──────────────────────────────────────────────
    ch[0] = (board == TILE_WALL).astype(np.float32)
    ch[1] = (board == TILE_BOX).astype(np.float32)
    ch[2] = (board == TILE_ITEM_RAD).astype(np.float32)
    ch[3] = (board == TILE_ITEM_CAP).astype(np.float32)

    # ── Kênh 4-5: bom ───────────────────────────────────────────────────
    if bombs.ndim == 2 and len(bombs) > 0:
        for b in bombs:
            br, bc, timer, owner = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            # blast_radius = 1 + bonus của owner
            owner_bonus = int(players[owner][4]) if 0 <= owner < 4 else 0
            blast       = 1 + owner_bonus

            # Kênh 4: danger (cường độ ~ urgency)
            urgency = 1.0 - min(timer, 7) / 7.0   # timer 7→0, urgency 0→1
            ch[4][br][bc] = max(ch[4][br][bc], urgency)

            # Kênh 5: blast zone dự kiến
            ch[5][br][bc] = 1.0
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                for step in range(1, blast + 1):
                    nr, nc = br + dr*step, bc + dc*step
                    if not (0 <= nr < H and 0 <= nc < W):
                        break
                    if board[nr][nc] == TILE_WALL:
                        break
                    ch[4][nr][nc] = max(ch[4][nr][nc], urgency * 0.9)
                    ch[5][nr][nc] = 1.0
                    if board[nr][nc] == TILE_BOX:
                        break   # dừng tại box (phá box nhưng không xuyên qua)

    # ── Kênh 6: vị trí mình ─────────────────────────────────────────────
    me = players[agent_id]
    mr, mc = int(me[0]), int(me[1])
    ch[6][mr][mc] = 1.0

    # ── Kênh 7: vị trí địch (alive) ─────────────────────────────────────
    for i, p in enumerate(players):
        if i == agent_id:
            continue
        if int(p[2]) == 1:   # alive
            ch[7][int(p[0])][int(p[1])] = 1.0

    # ── Kênh 8: escape map (BFS từ vị trí mình, chuẩn hóa) ──────────────
    ch[8] = _escape_map(board, ch[4], (mr, mc), H, W)

    spatial = torch.tensor(ch)   # (9,13,13)

    # ── Scalar vector ────────────────────────────────────────────────────
    alive        = int(me[2])
    bombs_left   = int(me[3])
    radius_bonus = int(me[4])
    blast_radius = 1 + radius_bonus

    in_danger    = float(ch[4][mr][mc] > 0.3)
    n_enemies    = int(sum(1 for i,p in enumerate(players)
                          if i != agent_id and int(p[2]) == 1))
    n_boxes      = int((board == TILE_BOX).sum())
    destroyable  = float(_count_destroyable(board, (mr,mc), blast_radius, H, W))
    escape_score = float(ch[8][mr][mc])

    scalar = torch.tensor([
        mr / H,
        mc / W,
        bombs_left   / 5.0,
        blast_radius / 6.0,
        in_danger,
        n_enemies    / 3.0,
        n_boxes      / 60.0,
        destroyable  / 8.0,
        escape_score,
    ], dtype=torch.float32)

    return spatial, scalar


# ── Helpers ──────────────────────────────────────────────────────────────────

def _escape_map(board, danger_ch, start, H, W):
    """
    BFS từ start, trả về ma trận (H,W):
      ô = 1.0 nếu là ô an toàn gần nhất (danger < 0.3)
      ô = 0.0 nếu không đi được hoặc chưa thăm
    Chuẩn hóa bằng khoảng cách (gần = cao hơn).
    """
    from collections import deque
    dist   = np.full((H, W), -1, dtype=np.float32)
    queue  = deque()
    dist[start[0]][start[1]] = 0
    queue.append(start)

    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < H and 0 <= nc < W:
                if dist[nr][nc] < 0 and board[nr][nc] in [TILE_GRASS, TILE_ITEM_RAD, TILE_ITEM_CAP]:
                    dist[nr][nc] = dist[r][c] + 1
                    queue.append((nr, nc))

    result = np.zeros((H, W), dtype=np.float32)
    # Đánh dấu ô an toàn với score tỷ lệ nghịch khoảng cách
    for r in range(H):
        for c in range(W):
            if dist[r][c] >= 0 and danger_ch[r][c] < 0.3:
                result[r][c] = 1.0 / (dist[r][c] + 1.0)
    return result


def _count_destroyable(board, pos, blast, H, W):
    r0, c0 = pos
    count  = 0
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        for step in range(1, blast+1):
            nr, nc = r0+dr*step, c0+dc*step
            if not (0 <= nr < H and 0 <= nc < W):
                break
            if board[nr][nc] == TILE_WALL:
                break
            if board[nr][nc] == TILE_BOX:
                count += 1
                break
    return count