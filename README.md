# 🚀 GDGoC-HCMUS AI Challenge 2026 – BomIT Agent

## Chiến lược tổng thể (dựa theo SOTA 2024)

### Tại sao Hybrid PPO + Rule-based thắng?

Nghiên cứu mới nhất tại IJCAI 2024 (paper *"Multi-Agent Training for Pommerman"*)
chứng minh hướng mạnh nhất là **kết hợp**:

```
Neural Policy (PPO)   ←→   Safety Rule Layer
       ↓                          ↓
  Chiến lược dài hạn      Phản xạ tức thời
  (học từ dữ liệu)        (thoát bom, không tự sát)
```

---

## Cấu trúc project

```
bombit_agent/
├── agent/
│   ├── agent.py          # ⭐ Agent nộp thi (BomITAgent.act)
│   └── network.py        # CNN + Actor-Critic PPO network
├── training/
│   └── ppo_trainer.py    # PPO với Curriculum + Self-Play
├── utils/
│   └── state_encoder.py  # Chuyển obs → tensor (5 kênh)
└── train.py              # Entry point training
```

---

## Hướng dẫn nhanh

### 1. Cài đặt
```bash
pip install torch numpy
```

### 2. Train (thay MockEnv bằng engine BomIT thật)
```bash
python train.py --episodes 5000 --device cpu
# Nếu có GPU:
python train.py --episodes 10000 --device cuda
```

### 3. Tiếp tục train từ checkpoint
```bash
python train.py --resume checkpoints/agent_2000.pt --episodes 5000
```

### 4. Test agent
```python
from agent.agent import BomITAgent
agent = BomITAgent(model_path="checkpoints/agent_final.pt", agent_id=0)
action = agent.act(obs)  # obs từ env.reset() hoặc env.step()
```

---

## Kiến trúc mạng (network.py)

```
Input:
  spatial  (5, 13, 13)   ← 5 kênh feature map
  scalar   (8,)           ← stats: vị trí, số bom, blast, v.v.

CNN backbone:
  Conv(5→32) → GroupNorm → ReLU
  Conv(32→64) → GroupNorm → ReLU
  ResBlock(64)
  AdaptiveAvgPool → (64, 4, 4) → flatten (1024)

Scalar MLP:
  Linear(8→64) → ReLU → Linear(64→64)

Fusion:
  concat(1024+64=1088) → Linear(512) → Linear(256)

Output:
  Actor head  → logits (6 actions)
  Critic head → value  (1)
```

---

## 3 giai đoạn Curriculum Learning

| Phase | Mục tiêu | Thắng nếu win-rate > |
|-------|----------|----------------------|
| 0 – Explore | Khám phá bản đồ, nhặt items, phá thùng | 30% |
| 1 – Survive  | Tồn tại lâu, thoát bom, không tự sát   | 55% |
| 2 – Fight    | Tiêu diệt địch, chiến thuật cao cấp    | – |

---

## Reward Shaping

```python
r = game_reward          # thắng: +1.0, thua: -0.5
r += -0.001              # phạt mỗi bước (hành động nhanh)
r += 0.05 * crates       # thưởng phá thùng
r += 0.10 * item_picked  # thưởng nhặt vật phẩm
# Phase 0: thưởng ô mới khám phá
# Phase 1: thưởng tồn tại mỗi bước
```

Dense reward **giảm dần** (adaptive annealing) khi phase tăng
→ tránh agent "đào thùng mãi mà không đánh địch".

---

## Self-Play Pool

- Lưu 10 phiên bản agent cũ
- Mỗi trận: đấu với **random opponent từ pool** (không phải bản mới nhất)
- Tránh overfitting vào 1 chiến thuật cố định
- Cập nhật pool mỗi 200 episodes

---

## Tips nâng cao (nếu có thời gian)

1. **Imitation Learning** trước PPO: ghi lại tay 100 ván,
   pre-train network bằng supervised learning → học nhanh hơn 3x.

2. **Thêm LSTM**: thay AdaptiveAvgPool bằng GRU để nhớ lịch sử
   → hữu ích khi agent địch bị occlusion.

3. **Action masking**: mask các hành động không hợp lệ
   (đi vào tường, đặt bom khi đã hết) → training sạch hơn.

4. **Monte Carlo Tree Search (MCTS)** khi infer:
   mỗi bước thực hiện N simulations ngắn với policy network
   → chọn nước đi tốt nhất (tốn compute nhưng rất mạnh).

---

## Checklist nộp bài

- [ ] `agent.py` import được, `BomITAgent.act(obs)` trả về int 0-5
- [ ] Không có API call external
- [ ] Model `.pt` đi kèm
- [ ] Test smoke: `python agent/agent.py` chạy không lỗi
