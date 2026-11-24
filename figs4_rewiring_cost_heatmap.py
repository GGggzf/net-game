import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import collections

# ========== 参数 ==========
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

# ========== Q-learning 超参 ==========
QL_EPSILON = 0.05
QL_ALPHA   = 0.2
QL_GAMMA   = 0.5
QL_DECAY   = 0.9

# ========== 参数网格 ==========
F_VALUES = list(range(1, 11))          # 重连成本 F = 1..10
P_VALUES = [0.0, 0.25, 0.5, 0.75, 1]   # 惩罚概率 p

# ========== 重复实验设置 ==========
N_REPS = 5          # 每个参数点重复次数
BASE_SEED = 32       # 基础种子，用于可复现性

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
def random_rewire(G, u, reputation, all_nodes, F):
    """随机断开最低声誉邻居，随机连接到非邻居，返回成本（成功则F，否则0）"""
    neighbors = list(G.neighbors(u))
    if len(neighbors) == 0:
        return 0
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
        return F
    else:
        G.add_edge(u, target)
        return 0

def best_neighbor_neighbor_rewire(G, u, reputation, all_nodes, F):
    """断开最低声誉邻居，连接到其邻居中声誉最高的非邻居节点，返回成本"""
    neighbors = list(G.neighbors(u))
    if len(neighbors) == 0:
        return 0
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
        return F
    else:
        G.add_edge(u, target)
        return 0

# ========== 仿真函数（修正奖励逻辑） ==========
def run_simulation(F, p, rewire_func):
    G = nx.barabasi_albert_graph(N, BA_M)
    all_nodes = list(G.nodes)

    # 初始策略各半
    N_ALL_C = N // 2
    N_ALL_D = N - N_ALL_C
    initial_strategies = ['ALL_C'] * N_ALL_C + ['ALL_D'] * N_ALL_D
    random.shuffle(initial_strategies)
    node_strategies = {u: {'strategy': initial_strategies[i]} for i, u in enumerate(all_nodes)}
    reputation = {u: 0 for u in G.nodes}
    agents = {u: TwoLayerQLAgent(u) for u in all_nodes}

    # 收益变量
    raw_payoff = {n: 0 for n in all_nodes}
    punished_payoff = {n: 0 for n in all_nodes}
    net_payoff = {n: 0 for n in all_nodes}
    prev_net_payoff = net_payoff.copy()

    coop_rate_rec = []

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
                state = agents[u].get_state(G, net_payoff, prev_coop_rate)
                temp_rewire[u] = agents[u].choose_rewire_action(state)
                temp_actions[u] = agents[u].choose_strategy_action(state)

        actions = {u: 'C' if temp_actions[u]==0 else 'D' for u in all_nodes}

        # 1. 原始收益（无惩罚无成本）
        raw_payoff = {n: 0 for n in all_nodes}
        for u, v in G.edges:
            a_u, a_v = actions[u], actions[v]
            p_u, p_v = PAYOFF[(a_u, a_v)]
            raw_payoff[u] += p_u
            raw_payoff[v] += p_v

        # 2. 惩罚：背叛者以概率 p 清零
        punished_payoff = raw_payoff.copy()
        for u in all_nodes:
            if actions[u] == 'D' and random.random() < p:
                punished_payoff[u] = 0

        # 3. 执行重连，记录成本，计算净收益
        net_payoff = punished_payoff.copy()
        rewire_cost = {u: 0 for u in all_nodes}
        for u in all_nodes:
            if temp_rewire[u] == 1:
                cost = rewire_func(G, u, reputation, all_nodes, F)
                rewire_cost[u] = cost
                net_payoff[u] -= cost

        # 4. 声誉更新（合作者增加）
        for u in all_nodes:
            if actions[u] == 'C':
                reputation[u] += 1

        # 5. 合作率记录
        coop_rate = sum(1 for a in actions.values() if a=='C') / N
        coop_rate_rec.append(coop_rate)

        # 6. Q-learning 更新（仅更新实际选择的动作）
        if rnd > 0:
            for u in all_nodes:
                global_coop_rate = coop_rate
                # 重连更新：仅当实际选择了重连（temp_rewire[u]==1）
                if temp_rewire[u] == 1:
                    state_rewire = agents[u].get_state(G, prev_net_payoff, global_coop_rate)
                    next_state = agents[u].get_state(G, net_payoff, global_coop_rate)
                    # 奖励 = 原始收益 - 成本（不受惩罚影响）
                    reward_rewire = raw_payoff[u] - rewire_cost[u]
                    agents[u].update_rewire(state_rewire, 1, reward_rewire, next_state)
                # 策略更新：总是更新（因为策略选择总是发生）
                state_strat = agents[u].get_state(G, prev_net_payoff, global_coop_rate)
                next_state_strat = agents[u].get_state(G, net_payoff, global_coop_rate)
                # 策略奖励 = 惩罚后收益（不含重连成本）
                reward_strat = punished_payoff[u]
                agents[u].update_strategy(state_strat, temp_actions[u], reward_strat, next_state_strat)

                agents[u].decay_epsilon()

        prev_net_payoff = net_payoff.copy()

    # 返回最后 10 轮的平均合作率
    return np.mean(coop_rate_rec[-10:])

# ========== 网格搜索（重复 10 次取平均） ==========
print("开始网格搜索...")
print(f"每个参数点重复 {N_REPS} 次，基础种子 = {BASE_SEED}")

coop_random = np.zeros((len(F_VALUES), len(P_VALUES)))
coop_best_nn = np.zeros((len(F_VALUES), len(P_VALUES)))

for i, F in enumerate(F_VALUES):
    for j, p in enumerate(P_VALUES):
        # 随机重连
        reps_random = []
        for rep in range(N_REPS):
            seed = BASE_SEED + i * len(P_VALUES) * N_REPS + j * N_REPS + rep
            random.seed(seed)
            np.random.seed(seed)
            coop = run_simulation(F, p, random_rewire)
            reps_random.append(coop)
        coop_random[i, j] = np.mean(reps_random)
        std_random = np.std(reps_random)

        # 最优邻居重连
        reps_best = []
        for rep in range(N_REPS):
            seed = BASE_SEED + i * len(P_VALUES) * N_REPS + j * N_REPS + rep + 10000
            random.seed(seed)
            np.random.seed(seed)
            coop = run_simulation(F, p, best_neighbor_neighbor_rewire)
            reps_best.append(coop)
        coop_best_nn[i, j] = np.mean(reps_best)
        std_best = np.std(reps_best)

        print(f"F={F}, p={p:.2f} | 随机: {coop_random[i,j]:.4f} ± {std_random:.4f} | 最优: {coop_best_nn[i,j]:.4f} ± {std_best:.4f}")

print("所有实验完成。")

# ========== 绘图：并排热力图 ==========
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 随机重连
im1 = ax1.imshow(coop_random, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1,
                 origin='lower', extent=[P_VALUES[0], P_VALUES[-1], F_VALUES[0], F_VALUES[-1]])
ax1.set_xlabel('Punishment Probability (p)', fontsize=12)
ax1.set_ylabel('Rewiring Cost (F)', fontsize=12)
ax1.set_title('Random Rewiring', fontsize=13, fontweight='bold')
ax1.set_xticks(P_VALUES)
ax1.set_xticklabels([f'{p:.2f}' for p in P_VALUES])
ax1.set_yticks(F_VALUES[::2])
ax1.set_yticklabels(F_VALUES[::2])
cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label('Final Cooperation Rate', fontsize=10)

# 最优邻居-邻居重连
im2 = ax2.imshow(coop_best_nn, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1,
                 origin='lower', extent=[P_VALUES[0], P_VALUES[-1], F_VALUES[0], F_VALUES[-1]])
ax2.set_xlabel('Punishment Probability (p)', fontsize=12)
ax2.set_ylabel('Rewiring Cost (F)', fontsize=12)
ax2.set_title('Best Neighbor-of-Neighbor Rewiring', fontsize=13, fontweight='bold')
ax2.set_xticks(P_VALUES)
ax2.set_xticklabels([f'{p:.2f}' for p in P_VALUES])
ax2.set_yticks(F_VALUES[::2])
ax2.set_yticklabels(F_VALUES[::2])
cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label('Final Cooperation Rate', fontsize=10)

plt.tight_layout()
plt.savefig('rewiring_cost_punishment_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()