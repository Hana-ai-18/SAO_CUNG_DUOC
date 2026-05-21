"""
mock_env.py  –  Mock environment khớp đúng BTC interface + logic
-----------------------------------------------------------------
Fix:
  1. Timer off-by-one: đặt timer=7 KHÔNG tick trong cùng step
  2. Chain reaction khi bom nổ chạm bom khác
  3. Không đặt bom đè lên ô đã có bom
"""
import numpy as np
import random
from collections import deque

TILE_GRASS    = 0
TILE_WALL     = 1
TILE_BOX      = 2
TILE_ITEM_RAD = 3
TILE_ITEM_CAP = 4

AGENT_STARTS = [(1,1), (11,11), (1,11), (11,1)]


class MockBomITEnv:
    def __init__(self, seed=None):
        self.seed      = seed or random.randint(0, 99999)
        self.max_steps = 500

    def reset(self):
        rng          = np.random.RandomState(self.seed)
        self.board   = self._gen_board(rng)
        self.players = np.array([[r,c,1,1,0] for r,c in AGENT_STARTS], dtype=np.int32)
        self.bombs   = []   # list of [row,col,timer,owner_id]
        self.step_n  = 0
        return {i: self._obs() for i in range(4)}

    def step(self, actions: dict):
        self.step_n += 1
        prev_alive = self.players[:,2].copy()

        # 1. Di chuyển
        for i, act in actions.items():
            if self.players[i][2] == 0:
                continue
            deltas = {1:(0,-1), 2:(0,1), 3:(-1,0), 4:(1,0)}
            if act in deltas:
                r,c  = self.players[i][0], self.players[i][1]
                dr,dc = deltas[act]
                nr,nc = r+dr, c+dc
                if (0<=nr<13 and 0<=nc<13
                        and self.board[nr][nc] not in [TILE_WALL, TILE_BOX]
                        and not self._bomb_at(nr,nc)):
                    self.players[i][0] = nr
                    self.players[i][1] = nc

        # 2. Đặt bom (FIX: không tick trong step này, timer=7 nguyên)
        for i, act in actions.items():
            if act == 5 and self.players[i][2]==1 and self.players[i][3]>0:
                r,c = self.players[i][0], self.players[i][1]
                if not self._bomb_at(r,c):   # FIX: không đặt lên ô đã có bom
                    self.bombs.append([r, c, 7, i])
                    self.players[i][3] -= 1

        # 3. Giảm timer
        for b in self.bombs:
            b[2] -= 1

        # 4. Kích nổ bom timer <= 0 (FIX: chain reaction)
        self._process_explosions()

        # 5. Loại agent (đã xử lý trong _explode)

        # 6. Nhặt item
        for i in range(4):
            if self.players[i][2] == 0:
                continue
            r,c = self.players[i][0], self.players[i][1]
            if self.board[r][c] == TILE_ITEM_RAD:
                self.players[i][4] = min(4, self.players[i][4]+1)
                self.board[r][c]   = TILE_GRASS
            elif self.board[r][c] == TILE_ITEM_CAP:
                self.players[i][3] = min(5, self.players[i][3]+1)
                self.board[r][c]   = TILE_GRASS

        obs_all   = {i: self._obs() for i in range(4)}
        alive_now = self.players[:,2]
        rewards, infos = {}, {}
        for i in range(4):
            dead = prev_alive[i]==1 and alive_now[i]==0
            win  = alive_now[i]==1 and alive_now.sum()==1
            infos[i]   = {'dead':bool(dead), 'win':bool(win),
                          'boxes_destroyed':0,'items_collected':0,'new_tiles':0}
            rewards[i] = -1.0 if dead else (1.0 if win else 0.0)

        done_all         = (alive_now.sum()<=1) or (self.step_n>=self.max_steps)
        dones            = {i: bool(alive_now[i]==0) for i in range(4)}
        dones['__all__'] = done_all
        return obs_all, rewards, dones, infos

    def close(self): pass

    # ── helpers ──────────────────────────────────────────────────────────

    def _bomb_at(self, r, c):
        return any(b[0]==r and b[1]==c for b in self.bombs)

    def _obs(self):
        bombs_arr = np.array(self.bombs, dtype=np.int32) if self.bombs \
                    else np.zeros((0,4), dtype=np.int32)
        return {'map':self.board.copy(), 'players':self.players.copy(), 'bombs':bombs_arr}

    def _gen_board(self, rng):
        board = np.zeros((13,13), dtype=np.int32)
        board[0,:]=board[12,:]=board[:,0]=board[:,12]=TILE_WALL
        safe = set()
        for r,c in AGENT_STARTS:
            for dr in range(-1,2):
                for dc in range(-1,2):
                    safe.add((r+dr,c+dc))
        for r in range(1,12):
            for c in range(1,12):
                if (r,c) not in safe and rng.random() < 0.35:
                    board[r][c] = TILE_BOX
        return board

    def _process_explosions(self):
        """FIX: chain reaction – bom nổ có thể kích bom khác nổ ngay."""
        to_explode = deque(i for i,b in enumerate(self.bombs) if b[2]<=0)
        exploded   = set()
        while to_explode:
            bi = to_explode.popleft()
            if bi in exploded:
                continue
            exploded.add(bi)
            b = self.bombs[bi]
            affected = self._blast_cells(b)
            # Phá box, loại agent
            for r,c in affected:
                if self.board[r][c] == TILE_BOX:
                    self.board[r][c] = TILE_GRASS
                for i in range(4):
                    if (self.players[i][2]==1
                            and self.players[i][0]==r
                            and self.players[i][1]==c):
                        self.players[i][2] = 0
                # Chain: kích bom khác trong vùng nổ
                for j,b2 in enumerate(self.bombs):
                    if j not in exploded and b2[0]==r and b2[1]==c:
                        to_explode.append(j)
            # Trả bombs_left cho chủ
            owner = b[3]
            if 0<=owner<4:
                self.players[owner][3] += 1

        # Xoá bom đã nổ
        self.bombs = [b for i,b in enumerate(self.bombs) if i not in exploded]

    def _blast_cells(self, bomb):
        br,bc  = bomb[0], bomb[1]
        owner  = bomb[3]
        blast  = 1 + int(self.players[owner][4]) if 0<=owner<4 else 1
        cells  = [(br,bc)]
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            for step in range(1, blast+1):
                nr,nc = br+dr*step, bc+dc*step
                if not (0<=nr<13 and 0<=nc<13): break
                if self.board[nr][nc]==TILE_WALL: break
                cells.append((nr,nc))
                if self.board[nr][nc]==TILE_BOX: break
        return cells