import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import collections

# ========== 固定参数 ==========
random.seed(32)
np.random.seed(32)

n_rounds = 100
N = 1000
BA_M = 2

PAYOFF = {
    ('C', 'C'): (8, 8),
    ('C', 'D'): (0, 10),
    ('D', 'C'): (10, 0),
    ('D', 'D'): (5, 5)
}

# Q-learning 超参
QL_EPSILON = 0.05
QL_ALPHA   = 0.2
QL_GAMMA   = 0.5
QL_DECAY   = 0.9

# ========== 实验参数网格 ==========
PUNISH_PROB_LIST = [0.0, 0.25, 0.5, 0.75, 1]          # 惩罚概率
INIT_COOP_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]          # 初始合作者比例

# ========== 重复实验设置 ==========
N_REPS = 10                     # 每个参数点重复次数
BASE_SEED = 32                  # 基础种子，用于可复现性

# ========== 双层Q-learning Agent ==========
class TwoLayerQLAgent:
    def __init__(self, uid):
        self.uid = uid
        self.Q_rewire = collections.defaultdict(float)
        self.Q_action = collections.defaultdict(float)
        self.epsilon = QL_EPSILON
        self.alpha = QL_ALPHA
        self.gamma = QL_GAMMA

    def get_state(self, G, round_payoff, global_coop_rate):
        neigh = list(G.neighbors(self.uid))
        if len(neigh) == 0:
            neigh_payoff_state = 0
        else:
            neigh_payoffs = [round_payoff[v] for v in neigh]
            avg_neigh_payoff = sum(neigh_payoffs) / len(neigh_payoffs)
            if avg_neigh_payoff < 10:
                neigh_payoff_state = 0
            elif avg_neigh_payoff > 25:
                neigh_payoff_state = 2
            else:
                neigh_payoff_state = 1
        global_state = 0 if global_coop_rate < 0.5 else 1
        return (neigh_payoff_state, global_state)

    def choose_rewire_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        q0 = self.Q_rewire[(state[0], state[1], 0)]
        q1 = self.Q_rewire[(state[0], state[1], 1)]
        return 1 if q1 > q0 else 0 if q0 > q1 else random.randint(0, 1)

    def choose_strategy_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        q0 = self.Q_action[(state[0], state[1], 0)]
        q1 = self.Q_action[(state[0], state[1], 1)]
        return 1 if q1 > q0 else 0 if q0 > q1 else random.randint(0, 1)

    def update_rewire(self, state, action, reward, next_state):
        max_q = max(self.Q_rewire[(next_state[0], next_state[1], 0)],
                    self.Q_rewire[(next_state[0], next_state[1], 1)])
        td = reward + self.gamma * max_q - self.Q_rewire[(state[0], state[1], action)]
        self.Q_rewire[(state[0], state[1], action)] += self.alpha * td

    def update_strategy(self, state, action, reward, next_state):
        max_q = max(self.Q_action[(next_state[0], next_state[1], 0)],
                    self.Q_action[(next_state[0], next_state[1], 1)])
        td = reward + self.gamma * max_q - self.Q_action[(state[0], state[1], action)]
        self.Q_action[(state[0], state[1], action)] += self.alpha * td

    def decay_epsilon(self):
        self.epsilon *= QL_DECAY

# ========== 重连策略 ==========
def random_rewire(G, u, reputation, all_nodes):
    """随机断开最低声誉邻居，随机连接到非邻居"""
    neighbors = list(G.neighbors(u))
    if len(neighbors) == 0:
        return
    min_rep = min(reputation[v] for v in neighbors)
    low_rep_neighbors = [v for v in neighbors if reputation[v] == min_rep]
    target = random.choice(low_rep_neighbors)
    G.remove_edge(u, target)

    existing_neighbors = set(G.neighbors(u))
    existing_neighbors.add(u)
    candidates = [v for v in all_nodes if v not in existing_neighbors]
    if len(candidates) > 0:
        new_neighbor = random.choice(candidates)
        G.add_edge(u, new_neighbor)
    else:
        G.add_edge(u, target)

def best_neighbor_neighbor_rewire(G, u, reputation, all_nodes):
    """断开最低声誉邻居，连接到其邻居中声誉最高的非邻居节点"""
    neighbors = list(G.neighbors(u))
    if len(neighbors) == 0:
        return
    min_rep = min(reputation[v] for v in neighbors)
    low_rep_neighbors = [v for v in neighbors if reputation[v] == min_rep]
    target = random.choice(low_rep_neighbors)
    G.remove_edge(u, target)

    # 收集邻居的邻居（排除当前邻居和自身）
    neighbor_of_neighbors = set()
    for v in neighbors:
        for w in G.neighbors(v):
            if w != u and w not in neighbors:
                neighbor_of_neighbors.add(w)

    if len(neighbor_of_neighbors) > 0:
        best_rep = max(reputation[w] for w in neighbor_of_neighbors)
        best_candidates = [w for w in neighbor_of_neighbors if reputation[w] == best_rep]
        new_neighbor = random.choice(best_candidates)
        G.add_edge(u, new_neighbor)
    else:
        G.add_edge(u, target)

# ========== 运行单次实验的函数（修正奖励逻辑） ==========
def run_single_experiment(punish_prob, init_coop_ratio, rewire_func):
    G = nx.barabasi_albert_graph(N, BA_M)
    all_nodes = list(G.nodes)

    # 初始策略分配
    N_ALL_C = int(N * init_coop_ratio)
    N_ALL_D = N - N_ALL_C
    initial_strategies = ['ALL_C'] * N_ALL_C + ['ALL_D'] * N_ALL_D
    random.shuffle(initial_strategies)
    node_strategies = {u: {'strategy': initial_strategies[i]} for i, u in enumerate(all_nodes)}

    reputation = {u: 0 for u in G.nodes}
    agents = {u: TwoLayerQLAgent(u) for u in all_nodes}

    coop_rate_rec = []
    # 分别存储原始收益和最终收益
    raw_payoff = {n: 0 for n in all_nodes}
    final_payoff = {n: 0 for n in all_nodes}
    prev_final_payoff = final_payoff.copy()

    for rnd in range(n_rounds):
        temp_rewire = {}
        temp_actions = {}

        # 决策
        for u in all_nodes:
            if rnd == 0:
                temp_actions[u] = 1 if node_strategies[u]['strategy'] == 'ALL_D' else 0
                temp_rewire[u] = 0
            else:
                prev_coop_rate = coop_rate_rec[-1] if coop_rate_rec else 0.5
                state = agents[u].get_state(G, final_payoff, prev_coop_rate)   # 状态用最终收益
                temp_rewire[u] = agents[u].choose_rewire_action(state)
                temp_actions[u] = agents[u].choose_strategy_action(state)

        actions = {u: 'C' if temp_actions[u]==0 else 'D' for u in all_nodes}

        # 计算原始收益（无惩罚）
        raw_payoff = {n: 0 for n in all_nodes}
        for u, v in G.edges:
            a_u, a_v = actions[u], actions[v]
            p_u, p_v = PAYOFF[(a_u, a_v)]
            raw_payoff[u] += p_u
            raw_payoff[v] += p_v

        # 复制原始收益作为最终收益的初始值
        final_payoff = raw_payoff.copy()

        # 更新声誉（基于策略，不受惩罚影响）
        for u in all_nodes:
            if actions[u] == 'C':
                reputation[u] += 1

        # 惩罚机制：背叛者以概率 punish_prob 被清零收益
        for u in all_nodes:
            if actions[u] == 'D' and random.random() < punish_prob:
                final_payoff[u] = 0   # 惩罚后收益清零

        # 执行重连（根据传入的重连函数）
        for u in all_nodes:
            if temp_rewire[u] == 1:
                rewire_func(G, u, reputation, all_nodes)

        coop_rate = sum(1 for a in actions.values() if a=='C') / N
        coop_rate_rec.append(coop_rate)

        # Q-learning 更新
        if rnd > 0:
            for u in all_nodes:
                global_coop_rate = coop_rate
                state = agents[u].get_state(G, prev_final_payoff, global_coop_rate)
                # 重连更新使用原始收益（raw_payoff）
                next_state = agents[u].get_state(G, final_payoff, global_coop_rate)
                agents[u].update_rewire(state, temp_rewire[u], raw_payoff[u], next_state)
                # 策略更新使用最终收益（final_payoff）
                agents[u].update_strategy(state, temp_actions[u], final_payoff[u], next_state)
                agents[u].decay_epsilon()

        prev_final_payoff = final_payoff.copy()

    return coop_rate_rec[-1]

# ========== 网格搜索并记录最终合作率（两种重连方式，重复取平均） ==========
print("开始运行参数网格实验...")
print(f"每个参数点重复 {N_REPS} 次，基础种子 = {BASE_SEED}")

# 存储结果的两个矩阵
final_coop_random = np.zeros((len(PUNISH_PROB_LIST), len(INIT_COOP_RATIOS)))
final_coop_best_nn = np.zeros((len(PUNISH_PROB_LIST), len(INIT_COOP_RATIOS)))

# 对每种重连方式分别运行
for i, p in enumerate(PUNISH_PROB_LIST):
    for j, ic in enumerate(INIT_COOP_RATIOS):
        # ---- 随机重连 ----
        coop_random_reps = []
        for rep in range(N_REPS):
            # 为每次重复设置独立种子
            seed = BASE_SEED + i * len(INIT_COOP_RATIOS) * N_REPS + j * N_REPS + rep
            random.seed(seed)
            np.random.seed(seed)
            coop = run_single_experiment(p, ic, random_rewire)
            coop_random_reps.append(coop)
        final_coop_random[i, j] = np.mean(coop_random_reps)
        print(f"随机重连: p={p:.2f}, IC={ic:.1f} -> 平均合作率 = {final_coop_random[i, j]:.4f} (std={np.std(coop_random_reps):.4f})")

        # ---- 最优邻居-邻居重连 ----
        coop_best_reps = []
        for rep in range(N_REPS):
            # 与随机重连种子错开，避免序列重叠
            seed = BASE_SEED + i * len(INIT_COOP_RATIOS) * N_REPS + j * N_REPS + rep + 10000
            random.seed(seed)
            np.random.seed(seed)
            coop = run_single_experiment(p, ic, best_neighbor_neighbor_rewire)
            coop_best_reps.append(coop)
        final_coop_best_nn[i, j] = np.mean(coop_best_reps)
        print(f"最优邻居重连: p={p:.2f}, IC={ic:.1f} -> 平均合作率 = {final_coop_best_nn[i, j]:.4f} (std={np.std(coop_best_reps):.4f})")

print("所有实验完成。")

# ========== 绘制并排热力图（交换坐标轴） ==========
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 随机重连热力图（转置矩阵，交换坐标轴）
im1 = ax1.imshow(final_coop_random.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1,
                 origin='lower', extent=[PUNISH_PROB_LIST[0], PUNISH_PROB_LIST[-1],
                                         INIT_COOP_RATIOS[0], INIT_COOP_RATIOS[-1]])
ax1.set_xlabel('Punishment Probability (p)', fontsize=12)
ax1.set_ylabel('Initial Cooperation Ratio (IC)', fontsize=12)
ax1.set_title('Random Rewiring', fontsize=13, fontweight='bold')
ax1.set_xticks(PUNISH_PROB_LIST)
ax1.set_xticklabels([f'{p:.2f}' for p in PUNISH_PROB_LIST])
ax1.set_yticks(INIT_COOP_RATIOS)
ax1.set_yticklabels([f'{ic:.1f}' for ic in INIT_COOP_RATIOS])
cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label('Final Cooperation Rate', fontsize=10)

# 最优邻居-邻居重连热力图（转置矩阵，交换坐标轴）
im2 = ax2.imshow(final_coop_best_nn.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1,
                 origin='lower', extent=[PUNISH_PROB_LIST[0], PUNISH_PROB_LIST[-1],
                                         INIT_COOP_RATIOS[0], INIT_COOP_RATIOS[-1]])
ax2.set_xlabel('Punishment Probability (p)', fontsize=12)
ax2.set_ylabel('Initial Cooperation Ratio (IC)', fontsize=12)
ax2.set_title('Best Neighbor-of-Neighbor Rewiring', fontsize=13, fontweight='bold')
ax2.set_xticks(PUNISH_PROB_LIST)
ax2.set_xticklabels([f'{p:.2f}' for p in PUNISH_PROB_LIST])
ax2.set_yticks(INIT_COOP_RATIOS)
ax2.set_yticklabels([f'{ic:.1f}' for ic in INIT_COOP_RATIOS])
cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label('Final Cooperation Rate', fontsize=10)

plt.tight_layout()
plt.savefig('rewiring_comparison_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

