import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import collections
from itertools import product

# ========== 固定参数 ==========
random.seed(32)
np.random.seed(32)

n_rounds = 100
N = 1000
BA_M = 2

# 基础收益矩阵
BASE_PAYOFF = {
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

# 初始合作比例固定为 0.5
INIT_COOP_RATIO = 0.5

# 重复实验设置
N_REPS = 5
BASE_SEED = 32

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
    neighbors = list(G.neighbors(u))
    if len(neighbors) == 0:
        return
    min_rep = min(reputation[v] for v in neighbors)
    low_rep_neighbors = [v for v in neighbors if reputation[v] == min_rep]
    target = random.choice(low_rep_neighbors)
    G.remove_edge(u, target)

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

# ========== 仿真函数（支持重连策略和惩罚概率，修正奖励逻辑） ==========
def run_simulation_homogeneous(p, rewire_func):
    """无异质性版本（b=1），使用原始收益矩阵"""
    G = nx.barabasi_albert_graph(N, BA_M)
    all_nodes = list(G.nodes)

    N_ALL_C = int(N * INIT_COOP_RATIO)
    N_ALL_D = N - N_ALL_C
    initial_strategies = ['ALL_C'] * N_ALL_C + ['ALL_D'] * N_ALL_D
    random.shuffle(initial_strategies)
    node_strategies = {u: {'strategy': initial_strategies[i]} for i, u in enumerate(all_nodes)}

    reputation = {u: 0 for u in G.nodes}
    agents = {u: TwoLayerQLAgent(u) for u in all_nodes}

    coop_rate_rec = []
    raw_payoff = {n: 0 for n in all_nodes}
    final_payoff = {n: 0 for n in all_nodes}
    prev_final_payoff = final_payoff.copy()

    for rnd in range(n_rounds):
        temp_rewire = {}
        temp_actions = {}

        for u in all_nodes:
            if rnd == 0:
                temp_actions[u] = 1 if node_strategies[u]['strategy'] == 'ALL_D' else 0
                temp_rewire[u] = 0
            else:
                prev_coop_rate = coop_rate_rec[-1] if coop_rate_rec else 0.5
                state = agents[u].get_state(G, final_payoff, prev_coop_rate)
                temp_rewire[u] = agents[u].choose_rewire_action(state)
                temp_actions[u] = agents[u].choose_strategy_action(state)

        actions = {u: 'C' if temp_actions[u]==0 else 'D' for u in all_nodes}

        # 原始收益（无惩罚）
        raw_payoff = {n: 0 for n in all_nodes}
        for u, v in G.edges:
            a_u, a_v = actions[u], actions[v]
            p_u, p_v = BASE_PAYOFF[(a_u, a_v)]
            raw_payoff[u] += p_u
            raw_payoff[v] += p_v

        # 最终收益初始为原始收益
        final_payoff = raw_payoff.copy()

        # 声誉更新（仅合作者）
        for u in all_nodes:
            if actions[u] == 'C':
                reputation[u] += 1

        # 惩罚机制：背叛者以概率p被清零收益
        for u in all_nodes:
            if actions[u] == 'D' and random.random() < p:
                final_payoff[u] = 0

        # 执行重连
        for u in all_nodes:
            if temp_rewire[u] == 1:
                rewire_func(G, u, reputation, all_nodes)

        coop_rate = sum(1 for a in actions.values() if a=='C') / N
        coop_rate_rec.append(coop_rate)

        if rnd > 0:
            for u in all_nodes:
                global_coop_rate = coop_rate
                state = agents[u].get_state(G, prev_final_payoff, global_coop_rate)
                next_state = agents[u].get_state(G, final_payoff, global_coop_rate)
                # 重连使用原始收益，策略使用最终收益
                agents[u].update_rewire(state, temp_rewire[u], raw_payoff[u], next_state)
                agents[u].update_strategy(state, temp_actions[u], final_payoff[u], next_state)
                agents[u].decay_epsilon()

        prev_final_payoff = final_payoff.copy()

    return coop_rate_rec[-1]

def run_simulation_heterogeneous(b, p, rewire_func):
    """异质性版本（b>1），使用缩放收益"""
    G = nx.barabasi_albert_graph(N, BA_M)
    all_nodes = list(G.nodes)

    scale = {u: random.uniform(1, b) for u in all_nodes}
    def payoff_of(u, a_u, a_v):
        base_u, base_v = BASE_PAYOFF[(a_u, a_v)]
        return base_u * scale[u], base_v * scale[u]

    N_ALL_C = int(N * INIT_COOP_RATIO)
    N_ALL_D = N - N_ALL_C
    initial_strategies = ['ALL_C'] * N_ALL_C + ['ALL_D'] * N_ALL_D
    random.shuffle(initial_strategies)
    node_strategies = {u: {'strategy': initial_strategies[i]} for i, u in enumerate(all_nodes)}

    reputation = {u: 0 for u in G.nodes}
    agents = {u: TwoLayerQLAgent(u) for u in all_nodes}

    coop_rate_rec = []
    raw_payoff = {n: 0 for n in all_nodes}
    final_payoff = {n: 0 for n in all_nodes}
    prev_final_payoff = final_payoff.copy()

    for rnd in range(n_rounds):
        temp_rewire = {}
        temp_actions = {}

        for u in all_nodes:
            if rnd == 0:
                temp_actions[u] = 1 if node_strategies[u]['strategy'] == 'ALL_D' else 0
                temp_rewire[u] = 0
            else:
                prev_coop_rate = coop_rate_rec[-1] if coop_rate_rec else 0.5
                state = agents[u].get_state(G, final_payoff, prev_coop_rate)
                temp_rewire[u] = agents[u].choose_rewire_action(state)
                temp_actions[u] = agents[u].choose_strategy_action(state)

        actions = {u: 'C' if temp_actions[u]==0 else 'D' for u in all_nodes}

        # 原始收益（无惩罚）
        raw_payoff = {n: 0 for n in all_nodes}
        for u, v in G.edges:
            a_u, a_v = actions[u], actions[v]
            p_u, _ = payoff_of(u, a_u, a_v)
            _, p_v = payoff_of(v, a_v, a_u)
            raw_payoff[u] += p_u
            raw_payoff[v] += p_v

        # 最终收益初始为原始收益
        final_payoff = raw_payoff.copy()

        # 声誉更新
        for u in all_nodes:
            if actions[u] == 'C':
                reputation[u] += 1

        # 惩罚机制
        for u in all_nodes:
            if actions[u] == 'D' and random.random() < p:
                final_payoff[u] = 0

        # 执行重连
        for u in all_nodes:
            if temp_rewire[u] == 1:
                rewire_func(G, u, reputation, all_nodes)

        coop_rate = sum(1 for a in actions.values() if a=='C') / N
        coop_rate_rec.append(coop_rate)

        if rnd > 0:
            for u in all_nodes:
                global_coop_rate = coop_rate
                state = agents[u].get_state(G, prev_final_payoff, global_coop_rate)
                next_state = agents[u].get_state(G, final_payoff, global_coop_rate)
                agents[u].update_rewire(state, temp_rewire[u], raw_payoff[u], next_state)
                agents[u].update_strategy(state, temp_actions[u], final_payoff[u], next_state)
                agents[u].decay_epsilon()

        prev_final_payoff = final_payoff.copy()

    return coop_rate_rec[-1]

# ========== 参数网格 ==========
b_values = [1] + list(range(2, 6))   # b = 1,2,...,10
p_values = np.linspace(0, 1, 5)       # p = 0, 0.25, 0.5, 0.75, 1.0

# 存储结果（维度：len(p_values) x len(b_values)）
coop_random = np.zeros((len(p_values), len(b_values)))
coop_best_nn = np.zeros((len(p_values), len(b_values)))

print("开始网格搜索...")
print(f"每个参数点重复 {N_REPS} 次，基础种子 = {BASE_SEED}")

for i, p in enumerate(p_values):
    for j, b in enumerate(b_values):
        # 随机重连
        reps_random = []
        for rep in range(N_REPS):
            seed = BASE_SEED + i * len(b_values) * N_REPS + j * N_REPS + rep
            random.seed(seed)
            np.random.seed(seed)
            if b == 1:
                coop = run_simulation_homogeneous(p, random_rewire)
            else:
                coop = run_simulation_heterogeneous(b, p, random_rewire)
            reps_random.append(coop)
        coop_random[i, j] = np.mean(reps_random)

        # 最优邻居重连
        reps_best = []
        for rep in range(N_REPS):
            seed = BASE_SEED + i * len(b_values) * N_REPS + j * N_REPS + rep + 10000
            random.seed(seed)
            np.random.seed(seed)
            if b == 1:
                coop = run_simulation_homogeneous(p, best_neighbor_neighbor_rewire)
            else:
                coop = run_simulation_heterogeneous(b, p, best_neighbor_neighbor_rewire)
            reps_best.append(coop)
        coop_best_nn[i, j] = np.mean(reps_best)

        print(f"p={p:.2f}, b={b}: 随机={coop_random[i,j]:.4f} ± {np.std(reps_random):.4f}, 最优邻居={coop_best_nn[i,j]:.4f} ± {np.std(reps_best):.4f}")

print("所有仿真完成！")


# ========== 绘图：并排热力图（各自带颜色条） ==========
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 随机重连
im1 = ax1.imshow(coop_random, origin='lower', aspect='auto', cmap='RdYlGn',
                 extent=[p_values[0], p_values[-1], b_values[0], b_values[-1]],
                 vmin=0, vmax=1)
ax1.set_xlabel('Punishment Probability (p)', fontsize=12)
ax1.set_ylabel('Heterogeneity Strength (b)', fontsize=12)
ax1.set_title('Random Rewiring', fontsize=13, fontweight='bold')
ax1.set_xticks(p_values)
ax1.set_xticklabels([f'{p:.2f}' for p in p_values])
ax1.set_yticks(b_values[::2])
ax1.set_yticklabels(b_values[::2])
cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label('Final Cooperation Rate', fontsize=10)

# 最优邻居邻居重连
im2 = ax2.imshow(coop_best_nn, origin='lower', aspect='auto', cmap='RdYlGn',
                 extent=[p_values[0], p_values[-1], b_values[0], b_values[-1]],
                 vmin=0, vmax=1)
ax2.set_xlabel('Punishment Probability (p)', fontsize=12)
ax2.set_ylabel('Heterogeneity Strength (b)', fontsize=12)
ax2.set_title('Best Neighbor-of-Neighbor Rewiring', fontsize=13, fontweight='bold')
ax2.set_xticks(p_values)
ax2.set_xticklabels([f'{p:.2f}' for p in p_values])
ax2.set_yticks(b_values[::2])
ax2.set_yticklabels(b_values[::2])
cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label('Final Cooperation Rate', fontsize=10)

plt.tight_layout()
plt.savefig("b,p图.png")
plt.show()

# 打印统计
print("\n========== 平均合作率对比 ==========")
print(f"随机重连平均: {coop_random.mean():.4f}")
print(f"最优邻居邻居重连平均: {coop_best_nn.mean():.4f}")