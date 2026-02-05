# 强化学习 Reinforcement Learning, RL

目录:
- 强化学习到底是什么，用日常例子理解智能体如何“试错学习”
- 数学基础：MDP、状态、动作、奖励、折扣因子与回报
- Bellman 方程的直觉与推导
- 价值学习：Q-Learning 与 DQN 的原理与关键技巧
- 策略梯度与 REINFORCE，如何直接优化策略
- Actor-Critic：价值与策略的结合，低方差高效率
- 进阶算法：A3C / PPO 的思想与实现要点
- 实战：如何使用 Gym/Gymnasium 搭建与运行 RL 环境
- 深度强化学习：用深度网络逼近价值与策略的注意事项


## 1. 强化学习是什么

强化学习是让“智能体”（Agent）在“环境”（Environment）中通过“试错”（探索）与“反馈”（奖励）来学习“策略”（怎样行动）的过程。

日常类比：
- 学骑自行车：不断尝试（动作），摔倒是负奖励，稳住是正奖励，逐步形成骑行策略。
- 玩游戏闯关：每一步选择（动作）影响下一步的局面（状态），通关得到高奖励。
- 训练宠物：给对的行为以奖励（零食），让策略向期望行为靠拢。

关键元素：
- 状态 `s`：当前情境，例如游戏画面、传感器读数。
- 动作 `a`：智能体可执行的行为，例如向左/右、加速/减速。
- 奖励 `r`：环境对动作的即时反馈，衡量好坏。
- 策略 `π(a|s)`：在状态 `s` 下选择动作 `a` 的规则（可随机）。
- 目标：让长期累计奖励最大化。



## 2. 数学基础：MDP、回报与价值函数

强化学习通常建立在**马尔可夫决策过程（Markov Decision Process, MDP）** 之上：

\[ \text{MDP} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, r, \gamma \rangle \]

- \(\mathcal{S}\)：状态空间；\(\mathcal{A}\)：动作空间。
- \(\mathcal{P}(s'\mid s,a)\)：转移概率，执行动作 \(a\) 后由 \(s\) 到 \(s'\)。
- \(r(s,a)\) 或 \(r(s,a,s')\)：即时奖励函数。
- \(\gamma \in [0,1)\)：折扣因子，权衡“现在”与“未来”的重要性。

回报（从时间 \(t\) 开始的折扣累积奖励）：

\[ G_t = \sum_{k=0}^{\infty} \gamma^k \; R_{t+k+1} \]

策略 \(\pi(a\mid s)\) 定义了在状态 \(s\) 下选择动作 \(a\) 的概率。价值函数刻画“好坏”的期望：

\[ V^{\pi}(s) = \mathbb{E}_{\pi}[G_t \mid S_t = s] \]
\[ Q^{\pi}(s,a) = \mathbb{E}_{\pi}[G_t \mid S_t = s, A_t = a] \]
\[ A^{\pi}(s,a) = Q^{\pi}(s,a) - V^{\pi}(s) \]

优化目标（以起始状态分布 \(\rho\) 为例）：

\[ J(\pi) = \mathbb{E}_{s_0 \sim \rho,\; \tau \sim \pi} \Big[ \sum_{t=0}^{T} \gamma^t \; r(s_t, a_t) \Big] \]

---

## 3. Bellman 方程：原理与直觉

Bellman 方程揭示“当前价值 = 即时奖励 + 折扣后的未来价值”的递归结构。

期望形式（给定策略 \(\pi\)）：

\[ V^{\pi}(s) = \sum_{a} \pi(a\mid s) \sum_{s'} \mathcal{P}(s'\mid s,a) \big[ r(s,a,s') + \gamma V^{\pi}(s') \big] \]

\[ Q^{\pi}(s,a) = \sum_{s'} \mathcal{P}(s'\mid s,a) \Big[ r(s,a,s') + \gamma \sum_{a'} \pi(a'\mid s') Q^{\pi}(s',a') \Big] \]

最优形式（\(\pi^{*}\) 对应的 \(V^{*}, Q^{*}\)）：

\[ V^{*}(s) = \max_{a} \sum_{s'} \mathcal{P}(s'\mid s,a) \big[ r(s,a,s') + \gamma V^{*}(s') \big] \]

\[ Q^{*}(s,a) = \sum_{s'} \mathcal{P}(s'\mid s,a) \Big[ r(s,a,s') + \gamma \max_{a'} Q^{*}(s',a') \Big] \]

直觉：价值像“分期付款”的账单——现在拿到一笔奖励，加上未来可能获得的价值（打折）。

---

## 4. 价值学习：Q-Learning 与 DQN

### 4.1 Q-Learning（表格型）
- 思想：学习 \(Q^{*}(s,a)\) 的近似，使得在新状态选择 \(\arg\max_{a} Q(s,a)\)。
- 更新（时序差分，off-policy）：

\[ Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \Big( r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \Big) \]

- 探索：\(\varepsilon\)-greedy（以小概率随机动作）。
- 收敛（离散、表格）：在足够探索与合适的学习率递减下可收敛到 \(Q^{*}\)。

### 4.2 DQN（Deep Q-Network）
- 用神经网络 \(Q_{\theta}(s,a)\) 近似 Q 值，解决高维/连续观测。
- 目标值（bootstrap）：\( y = r + \gamma \max_{a'} Q_{\bar{\theta}}(s', a') \)。
- 损失：

\[ L(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \big[ \big( y - Q_{\theta}(s,a) \big)^2 \big] \]

- 两大稳定技巧：
  - 目标网络（\(\bar{\theta}\)）：延迟更新，减少非稳定的“追逐”效应。
  - 经验回放（Replay Buffer \(\mathcal{D}\)）：打乱相关性，提高样本利用率。
- Double DQN：避免过估计，用 \( a^{\*} = \arg\max_{a'} Q_{\theta}(s',a') \)，目标值改为 \( y = r + \gamma Q_{\bar{\theta}}(s', a^{\*}) \)。
- Dueling DQN：网络分支分别估计 \(V\) 与 \(A\)，再组合为 \(Q\)，提升学习效率。

---

## 5. 策略梯度：直接优化策略

将策略参数化为 \(\pi_{\theta}(a\mid s)\)，直接最大化 \(J(\theta)\)。策略梯度定理：

\[ \nabla_{\theta} J(\theta) = \mathbb{E}_{s \sim d^{\pi},\; a \sim \pi_{\theta}} \big[ \nabla_{\theta} \log \pi_{\theta}(a\mid s) \; Q^{\pi}(s,a) \big] \]

REINFORCE（蒙特卡洛）：

\[ \nabla_{\theta} J(\theta) \approx \frac{1}{N} \sum_{\text{episodes}} \sum_{t} \nabla_{\theta} \log \pi_{\theta}(a_t\mid s_t) \; G_t \]

基线（Baseline）减少方差（常选 \(b(s) = V^{\pi}(s)\)）：

\[ \nabla_{\theta} J(\theta) \approx \mathbb{E} \big[ \nabla_{\theta} \log \pi_{\theta}(a\mid s) \; (G_t - b(s_t)) \big] = \mathbb{E} \big[ \nabla_{\theta} \log \pi_{\theta}(a\mid s) \; A^{\pi}(s,a) \big] \]

熵正则（鼓励探索）：在目标中加入 \(\beta \cdot H(\pi_{\theta}(\cdot\mid s))\)。

---

## 6. Actor-Critic：价值与策略的结合

Actor（\(\pi_{\theta}\)）负责输出动作分布；Critic（\(V_{w}\) 或 \(Q_{w}\)）评估当前策略的价值，用于为 Actor 提供低方差的学习信号。

时序差分误差（TD error）：

\[ \delta_t = r_t + \gamma V_{w}(s_{t+1}) - V_{w}(s_t) \]

Actor 更新（用优势）：

\[ \nabla_{\theta} J(\theta) \approx \mathbb{E} \big[ \nabla_{\theta} \log \pi_{\theta}(a_t\mid s_t) \; \hat{A}_t \big] \]

广义优势估计（GAE）：

\[ \hat{A}_t = \sum_{\ell=0}^{\infty} (\gamma \lambda)^{\ell} \; \delta_{t+\ell} \]

Critic 可用 TD(\(\lambda\)) 或均方误差训练：\(\min_{w} \; \mathbb{E}[(\text{target} - V_{w}(s))^2]\)。

---

## 7. 进阶算法：A3C 与 PPO

### 7.1 A3C（Asynchronous Advantage Actor-Critic）
- 多个并行线程/进程独立与环境交互，异步地把梯度汇总到共享参数，缓解样本相关性与稳定性问题。
- 使用 n-step 回报与优势估计，提升学习速度。

### 7.2 PPO（Proximal Policy Optimization）
- 目标思想：限制每次策略更新的“步幅”，避免过大更新破坏已学到的策略。
- 定义概率比：\( r_t(\theta) = \frac{\pi_{\theta}(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)} \)
- 剪切目标：

\[ L^{\text{CLIP}}(\theta) = \mathbb{E} \Big[ \min\big( r_t(\theta) \; \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \; \hat{A}_t \big) \Big] \]

- 总损失常包含三项：策略损失（如上）、价值函数损失（MSE）、熵正则（提升探索）。
- 实现要点：轨迹采样、GAE 计算优势、mini-batch 多轮更新、KL 监控/早停、标准化优势。

（补充）其他常用：TRPO（约束 KL 的二阶优化）、SAC（连续动作的最大熵 RL，软价值与温度系数）。

---

## 8. 实战指南：使用 Gym/Gymnasium

安装与导入：

```bash
pip install gymnasium
```

基本使用（以 CartPole 为例，Gymnasium 接口）：

```python
import gymnasium as gym

env = gym.make("CartPole-v1")
obs, info = env.reset(seed=42)

for step in range(200):
    action = env.action_space.sample()  # 随机探索
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    if done:
        obs, info = env.reset()

env.close()
```

要点：
- `observation_space` 与 `action_space` 描述观测与动作的形状/类型。
- `reset()` 在 Gymnasium 返回 `(obs, info)`；`step()` 返回 5 元组。
- 离散动作适合 Q-Learning/DQN；连续动作更适合策略梯度/Actor-Critic（如 PPO/SAC）。
- 训练时需记录回报曲线、损失、策略熵等指标；固定随机种子以便复现实验。

---

## 9. 深度强化学习：与深度学习结合的注意事项

- 函数逼近：用深度网络近似 \(Q\)、\(V\)、\(\pi\)。设计网络结构需考虑状态特征（图像用 CNN，序列/POMDP 可用 RNN）。
- 稳定性：目标网络、经验回放、归一化与标准化、梯度裁剪、合适的学习率与优化器（如 Adam）。
- 数据分布漂移：策略在训练中不断变化，注意目标延迟与小步更新（PPO 的剪切、KL 约束）。
- 奖励设计：合理的奖励刻画任务目标，必要时做奖励缩放/平滑，防止梯度爆炸或探索崩溃。
- 观察处理：帧堆叠、状态标准化、图像下采样；连续控制任务常对动作加噪声以探索。
- 评估与泛化：区分训练回报与测试回报；用固定评估策略/随机种子进行横向比较。

---

## 10. 小结与学习建议

- 先在离散小环境（如 `CartPole-v1`, `MountainCar-v0`）用表格 Q-Learning 或 DQN 上手，再进入连续控制与策略梯度。
- 真正的差异：价值学习依赖“最大化未来价值”的 bootstrap；策略梯度直接优化策略，常配合 Actor-Critic 减少方差。
- 工程实践比公式更重要：稳定训练、良好监控、充分调参（学习率、折扣 \(\gamma\)、GAE \(\lambda\)、剪切 \(\varepsilon\) 等）。
- 前沿方向：分布式/自对弈、层级 RL、多任务泛化、模型为基础（Model-based RL）、离线 RL 等。

---

## 参考术语速览
- MDP：\(\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, r, \gamma \rangle\)
- 回报：\(G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}\)
- 价值：\(V^{\pi}(s)\), \(Q^{\pi}(s,a)\), \(A^{\pi}(s,a)\)
- Bellman（期望/最优）：见第 3 节公式
- Q-Learning 更新：\(Q \leftarrow Q + \alpha (r + \gamma \max Q' - Q)\)
- DQN 稳定技巧：目标网络、经验回放、Double/Dueling
- 策略梯度：\(\nabla J = \mathbb{E}[\nabla \log \pi \cdot Q]\)，REINFORCE 与 Baseline
- Actor-Critic：TD 误差、优势估计（GAE）
- PPO：剪切目标与 KL 监控

---

## 11. 多臂老虎机：探索与利用的入门示例

多臂老虎机（Multi-armed Bandit）是强化学习中最简单的形式，只有一个状态，但完美体现了"探索-利用权衡"（Exploration-Exploitation Trade-off）的核心问题。

### 问题描述
假设你有 K 台老虎机（臂），每台老虎机的奖励分布不同但未知：
- 第 i 台老虎机的期望奖励：\(\mu_i\)
- 目标：在 T 次尝试中最大化累计奖励

### 探索-利用权衡
- **利用（Exploitation）**：选择当前认为最好的老虎机
- **探索（Exploration）**：尝试其他老虎机，收集更多信息

### \(\varepsilon\)-Greedy 策略
最简单的解决方案，以概率 \(\varepsilon\) 随机探索，以概率 \(1-\varepsilon\) 利用：

\[ \pi(a) = \begin{cases} 
\arg\max_{a} \hat{\mu}_a & \text{以概率 } 1-\varepsilon \\
\text{随机动作} & \text{以概率 } \varepsilon
\end{cases} \]

其中 \(\hat{\mu}_a\) 是动作 a 的平均奖励估计。

### UCB（Upper Confidence Bound）
更智能的探索策略，平衡估计值和不确定性：

\[ \text{UCB}(a) = \hat{\mu}_a + c \sqrt{\frac{\ln t}{N_t(a)}} \]

- \(\hat{\mu}_a\)：平均奖励
- \(N_t(a)\)：动作 a 被选择的次数
- \(c\)：探索参数

UCB 选择上置信界最大的动作，既考虑平均表现，也考虑探索不足的动作。

### Thompson Sampling
贝叶斯方法，维护每个动作的奖励分布信念：

\[ \pi(a) = \mathbb{P}(\mu_a = \max_{a'} \mu_{a'}) \]

通过从后验分布采样，选择最可能最优的动作。

### 实际意义
多臂老虎机问题教会我们：
1. 盲目利用可能错过更好的选择
2. 盲目探索浪费机会成本  
3. 需要在信息收集和奖励获取间找到平衡

这个简单模型是理解所有RL算法中探索机制的基础。

---

## 12. MDP 求解：策略评估、改进与值迭代

### 12.1 策略评估（Policy Evaluation）
给定固定策略 \(\pi\)，计算其价值函数 \(V^{\pi}\)。使用迭代法：

\[ V_{k+1}(s) = \sum_{a} \pi(a\mid s) \sum_{s'} \mathcal{P}(s'\mid s,a) \big[ r(s,a,s') + \gamma V_k(s') \big] \]

直到 \(V_{k+1} \approx V_k\) 收敛。

### 12.2 策略改进（Policy Improvement）
基于当前价值函数，改进策略：

\[ \pi'(s) = \arg\max_{a} \sum_{s'} \mathcal{P}(s'\mid s,a) \big[ r(s,a,s') + \gamma V^{\pi}(s') \big] \]

策略改进定理保证 \(V^{\pi'} \geq V^{\pi}\)。

### 12.3 策略迭代（Policy Iteration）
交替进行策略评估和改进：

\[ \pi_0 \xrightarrow{\text{评估}} V^{\pi_0} \xrightarrow{\text{改进}} \pi_1 \xrightarrow{\text{评估}} V^{\pi_1} \xrightarrow{\text{改进}} \cdots \xrightarrow{} \pi^{*} \]

### 12.4 值迭代（Value Iteration）
直接迭代最优价值函数，更高效：

\[ V_{k+1}(s) = \max_{a} \sum_{s'} \mathcal{P}(s'\mid s,a) \big[ r(s,a,s') + \gamma V_k(s') \big] \]

值迭代本质是压缩映射，保证收敛到 \(V^{*}\)。

### 12.5 广义策略迭代（GPI）
大多数RL算法都可以看作策略评估和改进的交互过程，这是理解RL算法统一视角的关键。

---

## 13. 时序差分方法：n步TD与TD(λ)

### 13.1 n步TD方法
介于MC（使用整条轨迹）和TD(0)（使用单步）之间：

n步回报：
\[ G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n}) \]

n步TD更新：
\[ V(S_t) \leftarrow V(S_t) + \alpha [G_t^{(n)} - V(S_t)] \]

### 13.2 TD(λ)与资格迹
TD(λ)通过资格迹（Eligibility Trace）结合所有n步回报：

\[ G_t^{\lambda} = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)} \]

资格迹更新：
\[ E_t(s) = \begin{cases}
\gamma \lambda E_{t-1}(s) & \text{if } s \neq S_t \\
\gamma \lambda E_{t-1}(s) + 1 & \text{if } s = S_t
\end{cases} \]

TD(λ)更新：
\[ \delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \]
\[ V(s) \leftarrow V(s) + \alpha \delta_t E_t(s) \quad \forall s \]

### 13.3 实际意义
- \(\lambda = 0\)：TD(0)，只看一步
- \(\lambda = 1\)：等价于MC，看整条轨迹
- \(0 < \lambda < 1\)：折中，权衡偏差和方差

---

## 14. 离策略评估与重要性采样

### 14.1 重要性采样原理
用行为策略 \(\mu\) 的数据评估目标策略 \(\pi\)：

\[ \mathbb{E}_{x \sim p}[f(x)] = \mathbb{E}_{x \sim q}\left[ \frac{p(x)}{q(x)} f(x) \right] \]

在RL中，轨迹的重要性权重：

\[ \rho_t = \prod_{k=0}^{t} \frac{\pi(A_k\mid S_k)}{\mu(A_k\mid S_k)} \]

### 14.2 普通重要性采样
\[ V^{\pi}(s) \approx \frac{\sum_{i=1}^{n} \rho_t^{(i)} G_t^{(i)}}{\sum_{i=1}^{n} \rho_t^{(i)}} \]

### 14.3 加权重要性采样
\[ V^{\pi}(s) \approx \frac{\sum_{i=1}^{n} \rho_t^{(i)} G_t^{(i)}}{n} \]

### 14.4 Q(λ)和SARSA(λ)
将资格迹与离策略学习结合，用于Q-learning和SARSA的扩展。

---

## 15. 探索策略与最大熵强化学习

### 15.1 探索的重要性
探索是RL中的核心挑战，好的探索策略能：
- 发现环境中隐藏的高奖励区域
- 避免陷入局部最优
- 提高学习效率和稳定性

### 15.2 基于计数的探索
通过访问计数鼓励探索新状态：

\[ \text{Bonus}(s,a) = \frac{\beta}{\sqrt{N(s,a)}} \]

其中 \(N(s,a)\) 是状态-动作对的访问次数。

### 15.3 最大熵强化学习
传统RL只追求奖励最大化，最大熵RL同时最大化策略的熵：

\[ J(\pi) = \mathbb{E}_{\pi} \left[ \sum_{t} \gamma^t \big( r(s_t,a_t) + \alpha H(\pi(\cdot\mid s_t)) \big) \right] \]

其中 \(H(\pi(\cdot\mid s_t)) = -\sum_a \pi(a\mid s_t) \log \pi(a\mid s_t)\) 是策略熵。

### 15.4 Soft Actor-Critic (SAC)
最大熵RL的代表算法，结合了：
- 最大熵目标：鼓励探索和策略多样性
- Actor-Critic框架：高效学习
- 离线策略学习：样本高效

SAC的Q函数更新：

\[ J_Q(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \frac{1}{2} \big( Q_{\theta}(s,a) - (r + \gamma (V_{\bar{\phi}}(s') - \alpha \log \pi_{\phi}(a'\mid s')) ) \big)^2 \right] \]

### 15.5 实际应用
最大熵方法在连续控制任务中表现优异，特别是在需要多样化行为的场景中。

---

## 16. 连续控制：DPG、DDPG与TD3

### 16.1 确定性策略梯度（DPG）
对于连续动作空间，使用确定性策略：

\[ \mu_{\theta}(s) = a \]

确定性策略梯度定理：

\[ \nabla_{\theta} J(\theta) = \mathbb{E}_{s \sim \rho^{\mu}} \left[ \nabla_{\theta} \mu_{\theta}(s) \nabla_a Q^{\mu}(s,a) \big|_{a=\mu_{\theta}(s)} \right] \]

### 16.2 DDPG（Deep Deterministic Policy Gradient）
将DPG与深度网络结合：
- Actor网络：\(\mu_{\theta}(s)\)，输出确定性动作
- Critic网络：\(Q_{\phi}(s,a)\)，评估动作价值
- 目标网络：提高稳定性
- 经验回放：打破样本相关性

### 16.3 TD3（Twin Delayed DDPG）
改进DDPG，解决价值函数过估计问题：
1. **双Q学习**：使用两个Critic网络，取最小值作为目标
2. **延迟更新**：策略（Actor）更新频率低于Critic
3. **目标策略平滑**：对目标动作添加少量噪声

TD3的目标Q值：

\[ y = r + \gamma \min_{i=1,2} Q_{\bar{\phi}_i}(s', \tilde{a}') \]

其中 \(\tilde{a}' = \mu_{\bar{\theta}}(s') + \epsilon\)，\(\epsilon \sim \text{clip}(\mathcal{N}(0,\sigma), -c, c)\)

---

## 17. 部分可观测MDP（POMDP）与信念状态

### 17.1 POMDP定义
当智能体无法完全观测环境状态时：

\[ \text{POMDP} = \langle \mathcal{S}, \mathcal{A}, \mathcal{O}, \mathcal{P}, \mathcal{Z}, r, \gamma \rangle \]

- \(\mathcal{O}\)：观测空间
- \(\mathcal{Z}(o\mid s,a)\)：观测概率

### 17.2 信念状态（Belief State）
由于状态不完全可观测，维护状态的概率分布：

\[ b_t(s) = \mathbb{P}(s_t = s \mid o_0, a_0, \ldots, o_t) \]

信念状态更新：

\[ b_{t+1}(s') \propto \mathcal{Z}(o_{t+1}\mid s', a_t) \sum_{s} \mathcal{P}(s'\mid s, a_t) b_t(s) \]

### 17.3 解决方法
1. **直接学习策略**：输入观测序列，输出动作
2. **递归网络**：使用RNN或LSTM处理观测历史
3. **信念状态近似**：学习编码器将观测映射到潜在状态

### 17.4 实际意义
大多数现实世界问题都是部分可观测的，POMDP提供了更通用的建模框架。

---

## 18. 分层强化学习与Options框架

### 18.1 分层RL动机
解决长期信用分配问题，通过抽象提高学习效率：
- 高层策略选择子目标或选项
- 底层策略执行具体动作
- 时间抽象：不同层次在不同时间尺度上操作

### 18.2 Options框架
Option是一个三元组 \(\langle \mathcal{I}, \pi, \beta \rangle\)：
- \(\mathcal{I} \subseteq \mathcal{S}\)：初始状态集
- \(\pi\)：Option内部的策略
- \(\beta: \mathcal{S} \rightarrow [0,1]\)：终止条件

### 18.3 半马尔可夫决策过程（SMDP）
在Option层次上，环境变成半马尔可夫的：

\[ Q_{\mathcal{O}}(s,o) = \mathbb{E} \left[ r + \gamma^k \max_{o' \in \mathcal{O}} Q_{\mathcal{O}}(s',o') \mid s,o \right] \]

其中k是Option执行的时间步数。

### 18.4 实际应用
分层方法在需要长期规划和复杂行为的任务中特别有效。

---

## 19. 模型基础强化学习

### 19.1 基于模型 vs 无模型
- **无模型RL**：直接学习价值函数或策略
- **基于模型RL**：先学习环境模型，再利用模型进行规划

### 19.2 环境模型学习
学习转移动力学和奖励函数：

\[ \hat{\mathcal{P}}_{\phi}(s'\mid s,a), \quad \hat{r}_{\phi}(s,a) \]

### 19.3 模型预测控制（MPC）
使用学到的模型进行在线规划：
1. 从当前状态开始，用模型预测未来轨迹
2. 优化动作序列最大化预期回报
3. 执行第一个动作，重复过程

### 19.4 世界模型（World Models）
使用潜在变量模型学习环境表示：

\[ z_t \sim q_{\phi}(z_t\mid z_{t-1}, a_{t-1}, o_t) \]

在潜在空间中学习和规划。

### 19.5 优势与挑战
**优势**：样本效率高，可进行反事实推理
**挑战**：模型误差累积，规划计算成本高

---

## 20. 分布式强化学习与V-trace

### 20.1 分布式RL思想
不再估计期望回报，而是估计回报的完整分布：

\[ Z(s,a) = \sum_{t=0}^{\infty} \gamma^t R(s_t, a_t) \]

### 20.2 分布al Q学习
学习价值分布而不是标量价值：

\[ \mathcal{L} = \mathbb{E} \left[ D_{KL} \big( \mathcal{T} \hat{Z}_{\bar{\theta}}(s,a) \| Z_{\theta}(s,a) \big) \right] \]

其中 \(\mathcal{T}\) 是分布al Bellman算子。

### 20.3 V-trace
用于分布式Actor-Critic的离策略校正方法：

\[ v_s = V(x_s) + \sum_{t=s}^{s+n-1} \gamma^{t-s} \left( \prod_{i=s}^{t-1} c_i \right) \delta_t V \]

其中 \(c_i = \min(\bar{c}, \frac{\pi(a_i\mid x_i)}{\mu(a_i\mid x_i)})\)

### 20.4 实际意义
分布式方法能更好地捕捉风险态度和不确定性。

---

## 21. 经验回放改进与优先级采样

### 21.1 优先级经验回放
不是均匀采样，而是根据TD误差大小赋予优先级：

\[ p_i = |\delta_i| + \epsilon \]

采样概率：

\[ P(i) = \frac{p_i^{\alpha}}{\sum_k p_k^{\alpha}} \]

### 21.2 重要性采样权重
补偿优先级采样带来的偏差：

\[ w_i = \left( \frac{1}{N} \cdot \frac{1}{P(i)} \right)^{\beta} \]

### 21.3 Hindsight Experience Replay (HER)
对于稀疏奖励任务，重标签失败经验：
- 原本目标：G，实际达成：G'
- 重标签为：如果目标是G'，那么这次尝试是成功的

### 21.4 实际应用
优先级采样显著提高样本效率，特别是在复杂环境中。

---

## 22. 深度RL工程实践与超参数调优

### 22.1 网络架构设计
- 图像输入：CNN + 全连接
- 状态输入：多层感知机（MLP）
- 序列数据：RNN、LSTM、Transformer

### 22.2 输入预处理
- 帧堆叠：处理部分可观测性
- 状态标准化：\(s' = \frac{s - \mu}{\sigma}\)
- 图像处理：裁剪、缩放、灰度化

### 22.3 奖励工程
- 奖励缩放：避免梯度爆炸
- 奖励塑形：提供中间奖励信号
- 课程学习：从简单任务开始，逐步增加难度

### 22.4 关键超参数
- 学习率：\(10^{-3}\) 到 \(10^{-5}\)
- 折扣因子 \(\gamma\)：0.99 常见
- 批次大小：32-512
- 目标网络更新频率：每100-10000步

### 22.5 监控与调试
- 训练曲线：回报、损失、策略熵
- 可视化：价值函数、策略分布
- 敏感性分析：超参数影响

---

## 23. 安全强化学习与约束优化

### 23.1 安全RL目标
在追求高性能的同时，满足安全约束：

\[ \max_{\pi} J(\pi) \quad \text{s.t.} \quad C_i(\pi) \leq d_i \quad \forall i \]

### 23.2 约束策略优化（CPO）
在策略更新中强制执行约束：

\[ \pi_{k+1} = \arg\max_{\pi} \mathbb{E} \left[ \frac{\pi(a\mid s)}{\pi_k(a\mid s)} A^{\pi_k}(s,a) \right] \]
\[ \text{s.t.} \quad D_{KL}(\pi \| \pi_k) \leq \delta \]
\[ \quad \text{and} \quad J_C(\pi) \leq d \]

### 23.3 拉格朗日方法
将约束优化转化为无约束问题：

\[ \mathcal{L}(\pi, \lambda) = J(\pi) - \lambda (J_C(\pi) - d) \]

### 23.4 实际应用
在机器人、自动驾驶等安全关键领域尤为重要。

---

## 24. 离线强化学习

### 24.1 离线RL特点
从固定的数据集学习，不与环境交互：
- 数据来源：人类演示、其他策略的采样
- 挑战：分布外（OOD）动作的价值过估计

### 24.2 保守Q学习（CQL）
通过正则化避免价值过估计：

\[ \min_Q \max_\mu \alpha (\mathbb{E}_{s \sim \mathcal{D}}[\mathbb{E}_{a \sim \mu(a\mid s)}[Q(s,a)]] - \mathbb{E}_{s,a \sim \mathcal{D}}[Q(s,a)]]) + \frac{1}{2} \mathbb{E}_{s,a,s' \sim \mathcal{D}}[(Q(s,a) - \mathcal{B}^\pi \hat{Q}(s,a))^2] \]

### 24.3 策略约束
限制学习策略不要偏离数据分布太远：

\[ \pi_{\text{new}} = \arg\max_{\pi} \mathbb{E}_{s,a \sim \mathcal{D}}[Q(s,a)] - \beta D_{KL}(\pi(\cdot\mid s) \| \pi_{\beta}(\cdot\mid s)) \]

### 24.4 实际意义
使得RL能够利用历史数据，适用于真实世界应用。

---

## 25. 模仿学习与逆强化学习

### 25.1 模仿学习（Imitation Learning）
从专家演示中学习策略：
- 行为克隆（BC）：直接监督学习 \(\pi(a\mid s) \approx \pi_{\text{expert}}(a\mid s)\)
- 数据集聚合（DAgger）：交互式改进

### 25.2 逆强化学习（IRL）
从专家行为中推断奖励函数：

\[ \max_{r} \min_{\pi} \mathbb{E}_{\pi_{\text{expert}}}[r(s,a)] - \mathbb{E}_{\pi}[r(s,a)] - \psi(r) \]

### 25.3 对抗模仿学习
通过对抗训练同时学习策略和奖励：

\[ \min_{\pi} \max_{D} \mathbb{E}_{\pi}[\log D(s,a)] + \mathbb{E}_{\pi_{\text{expert}}}[\log(1-D(s,a))] \]

### 25.4 实际应用
当奖励函数难以设计时，从演示中学习是一种有效方法。

---

## 26. 多智能体强化学习概述

### 26.1 多智能体RL挑战
- 非平稳性：其他智能体也在学习
- 信用分配：哪个智能体贡献了多少
- 协调与通信：如何合作达成共同目标

### 26.2 主要设置
1. **完全合作**：所有智能体共享奖励函数
2. **完全竞争**：零和博弈
3. **混合动机**：既有合作又有竞争

### 26.3 学习方法
- 独立Q学习：每个智能体独立学习
- 集中训练分散执行：训练时知道全局信息，执行时只用局部信息
- 对手建模：学习其他智能体的策略

### 26.4 实际意义
多智能体系统更接近现实世界，但复杂度也大大增加。

---

## 27. 常见陷阱与理论提醒

### 27.1 训练不稳定的原因
1. 价值函数过估计
2. 探索不足
3. 超参数敏感
4. 奖励设计不合理

### 27.2 收敛性保证
- 表格Q学习：在足够探索下收敛到最优
- 函数逼近：一般没有收敛保证，可能发散
- 策略梯度：收敛到局部最优

### 27.3 偏差-方差权衡
- MC方法：无偏，高方差
- TD方法：有偏，低方差
- \(\lambda\)：调节权衡参数

### 27.4 实践建议
- 从小环境开始验证算法
- 多次运行取平均结果
- 仔细监控训练过程

---

## 28. 实际代码结构与最小示例

### 28.1 典型RL代码结构
```python
# 1. 环境接口
env = gym.make('EnvName')
obs = env.reset()

# 2. 智能体定义  
class Agent:
    def __init__(self, state_dim, action_dim):
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.value_net = ValueNetwork(state_dim)
        
    def act(self, state, deterministic=False):
        # 选择动作
        pass
        
    def update(self, batch):
        # 更新参数
        pass

# 3. 训练循环
for episode in range(total_episodes):
    state = env.reset()
    done = False
    
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        
        # 存储经验
        replay_buffer.push(state, action, reward, next_state, done)
        
        if len(replay_buffer) > batch_size:
            batch = replay_buffer.sample(batch_size)
            agent.update(batch)
            
        state = next_state
```

### 28.2 最小可运行示例
建议从CartPole环境的Q-learning开始，逐步扩展到更复杂算法。

---

## 29. 总结与进阶学习路径

### 29.1 学习路线建议
1. **基础**：多臂老虎机 → 表格Q-learning → DQN
2. **策略优化**：REINFORCE → Actor-Critic → PPO
3. **连续控制**：DDPG → TD3 → SAC
4. **高级主题**：分层RL → 基于模型RL → 离线RL

### 29.2 重要理论概念
- Bellman方程与动态规划
- 策略梯度定理  
- 探索-利用权衡
- 偏差-方差权衡

### 29.3 实践技能
- 环境接口使用（Gymnasium）
- 神经网络设计
- 超参数调优
- 训练监控与调试

### 29.4 资源推荐
- 经典教材：Sutton & Barton "Reinforcement Learning: An Introduction"
- 课程：CS234 (Stanford), CS285 (Berkeley)
- 代码库：Stable Baselines3, RLlib

---

（本文公式均按标准 RL 文献习惯书写，读者可结合代码实现进一步加深理解。）