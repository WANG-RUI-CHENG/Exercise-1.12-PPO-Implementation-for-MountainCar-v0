# Exercise-1.12-PPO-Implementation-for-MountainCar-v0

- 使用 PPO (Actor-Critic) 完整訓練流程
- 不需安裝 tensorboardX 或使用 SummaryWriter
- 相容 Gym (<=0.25) 與 Gymnasium / Gym (>=0.26)，已處理 reset()/step() 回傳格式差異
- 直接以終端機輸出 episode steps 與 return

## 環境說明 (MountainCar-v0)
- State: 2 維連續向量 (position, velocity)
- Action: 3 個離散動作
  - 0: push left
  - 1: do nothing
  - 2: push right
- Reward: 通常每一步為 -1，直到到達終點或回合結束
- Goal: 在步數上限內到達右側山頂旗幟。由於引擎動力不足，車子必須透過左右擺盪累積動能才有機會爬上山頂。

## 方法概述: PPO (Actor-Critic)
本實作使用 Actor-Critic 架構：
- Actor: 輸出離散動作機率分佈 pi(a|s)，收集資料時從該分佈抽樣動作
- Critic: 預測狀態價值 V(s)，作為 baseline 與 value regression 的目標

## 訓練訊號
1) Discounted Return
對每個時間步 t 計算折扣回報 G_t：
G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...

其中 gamma 為折扣因子 (0 < gamma < 1)。

2) Advantage (使用 critic 當 baseline)
A_t = G_t - V(s_t)

為提升訓練穩定性，可選擇做 advantage normalization：
A_t = (A_t - mean(A_t)) / (std(A_t) + 1e-8)

## PPO 核心: Clipped Objective
PPO 的重點是限制 policy 更新幅度，避免一次更新太大造成不穩定。

1) Importance Ratio (log-prob 版本，數值較穩定)
ratio = exp( log pi_new(a_t|s_t) - log pi_old(a_t|s_t) )

2) Clipped Surrogate Objective
surr1 = ratio * A_t
surr2 = clip(ratio, 1 - eps, 1 + eps) * A_t
L_actor = - mean( min(surr1, surr2) )

其中 eps 通常設為 0.2。Actor 以最小化 L_actor 來達到最大化 clipped objective 的效果。

3) Critic Loss (Value Regression)
L_critic = mean( (V(s_t) - G_t)^2 )

## 最佳化細節
- Mini-batch 更新：將 rollout 資料切成小 batch 更新
- Multiple epochs：同一批資料重複更新多次以提升 sample efficiency
- Gradient clipping：限制梯度範數避免梯度爆炸

## 執行方式
1) 安裝套件
如果使用 gym：
pip install gym torch numpy

如果使用 gymnasium：
pip install gymnasium torch numpy

2) 執行訓練
python PPO_MountainCar-v0_noTB.py

## 輸出與觀察方式
程式會定期印出 episode number、steps、return 等資訊。MountainCar-v0 通常每步 reward 為 -1，因此：
- steps 越少代表越快到終點
- return 越接近 0 (越不負) 表示表現越好
一開始常見 steps=200, return=-200；隨訓練進行，若能學到累積動能的策略，steps 會下降、return 會改善。

