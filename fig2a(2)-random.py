import networkx as nx
import matplotlib.pyplot as plt
import random, numpy as np, collections

# ========== 参数 ==========
random.seed(32);
np.random.seed(32)
n_rounds = 200
STRATEGIES = ['ALL_C', 'ALL_D']

N = 1000
BA_M = 2

PAYOFF = {
    ('C', 'C'): (8, 8),
    ('C', 'D'): (0, 10),
    ('D', 'C'): (10, 0),
    ('D', 'D'): (5, 5)
}

# ========== 惩罚机制参数列表（用于对比） ==========
PUNISH_PROB_LIST = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# ========== Q-learning 超参 ==========
QL_EPSILON = 0.05
QL_ALPHA = 0.2
QL_GAMMA = 0.5
QL_DECAY = 0.9


# ========== 双层Q-learning Agent ==========
class TwoLayerQLAgent:
    """
    双层决策机制：
    - 第一层：学习 "维持(0)" 或 "重连(1)"
    - 第二层：学习 "合作(0)" 或 "背叛(1)"
    """

    def __init__(self, uid):
        self.uid = uid
        # 第一层Q表：重连决策 (维持=0, 重连=1)
        self.Q_rewire = collections.defaultdict(float)
        # 第二层Q表：策略决策 (合作=0, 背叛=1)
        self.Q_action = collections.defaultdict(float)

        self.epsilon = QL_EPSILON
        self.alpha = QL_ALPHA
        self.gamma = QL_GAMMA

    def get_state(self, G, round_payoff, global_coop_rate):
        """获取当前状态，包含邻居信息和全局信息"""
        neigh = list(G.neighbors(self.uid))

        # 邻居收益状态
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

        # 全局合作率状态
        global_state = 0 if global_coop_rate < 0.5 else 1

        return (neigh_payoff_state, global_state)

    def choose_rewire_action(self, state):
        """第一层：选择维持(0)或重连(1)"""
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        q0 = self.Q_rewire[(state[0], state[1], 0)]
        q1 = self.Q_rewire[(state[0], state[1], 1)]
        return 1 if q1 > q0 else 0 if q0 > q1 else random.randint(0, 1)

    def choose_strategy_action(self, state):
        """第二层：选择合作(0)或背叛(1)"""
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        q0 = self.Q_action[(state[0], state[1], 0)]
        q1 = self.Q_action[(state[0], state[1], 1)]
        return 1 if q1 > q0 else 0 if q0 > q1 else random.randint(0, 1)

    def update_rewire(self, state, action, reward, next_state):
        """更新重连决策Q表"""
        max_q = max(self.Q_rewire[(next_state[0], next_state[1], 0)],
                    self.Q_rewire[(next_state[0], next_state[1], 1)])
        td = reward + self.gamma * max_q - self.Q_rewire[(state[0], state[1], action)]
        self.Q_rewire[(state[0], state[1], action)] += self.alpha * td

    def update_strategy(self, state, action, reward, next_state):
        """更新策略决策Q表"""
        max_q = max(self.Q_action[(next_state[0], next_state[1], 0)],
                    self.Q_action[(next_state[0], next_state[1], 1)])
        td = reward + self.gamma * max_q - self.Q_action[(state[0], state[1], action)]
        self.Q_action[(state[0], state[1], action)] += self.alpha * td

    def decay_epsilon(self):
        self.epsilon *= QL_DECAY


def run_simulation(PUNISH_PROB):
    """运行单次模拟，返回合作率记录"""
    # 初始化网络
    G = nx.barabasi_albert_graph(N, BA_M)
    all_nodes = list(G.nodes)

    # 初始化策略
    N_ALL_C = 500
    N_ALL_D = 500
    initial_strategies_list = ['ALL_C'] * N_ALL_C + ['ALL_D'] * N_ALL_D
    random.shuffle(initial_strategies_list)

    node_strategies = {u: {'strategy': initial_strategies_list[i]} for i, u in enumerate(all_nodes)}
    reputation = {u: 0 for u in G.nodes}

    # 初始化所有agent
    agents = {u: TwoLayerQLAgent(u) for u in all_nodes}

    coop_rate_rec = []
    actions = None
    round_payoff = {n: 0 for n in all_nodes}
    prev_round_payoff = round_payoff.copy()

    for rnd in range(n_rounds):
        # 1. 双层决策
        temp_rewire = {}
        temp_actions = {}

        for u in all_nodes:
            if rnd == 0:
                temp_actions[u] = 1 if node_strategies[u]['strategy'] == 'ALL_D' else 0
                temp_rewire[u] = 0
            else:
                prev_coop_rate = coop_rate_rec[-1] if coop_rate_rec else 0.5
                state = agents[u].get_state(G, prev_round_payoff, prev_coop_rate)
                temp_rewire[u] = agents[u].choose_rewire_action(state)
                temp_actions[u] = agents[u].choose_strategy_action(state)

        # 2. 先执行博弈（使用当前网络结构）
        actions = {u: 'C' if temp_actions[u] == 0 else 'D' for u in all_nodes}

        # 3. 收益计算（原始收益）
        round_payoff = {n: 0 for n in all_nodes}
        for u, v in G.edges:
            a_u, a_v = actions[u], actions[v]
            p_u, p_v = PAYOFF[(a_u, a_v)]
            round_payoff[u] += p_u
            round_payoff[v] += p_v

        # 保存原始收益（未受惩罚）
        base_payoff = round_payoff.copy()

        # 4. 声誉更新
        for u in all_nodes:
            if actions[u] == 'C':
                reputation[u] += 1

        # 5. 惩罚机制（修改最终收益）
        for u in all_nodes:
            if actions[u] == 'D' and random.random() < PUNISH_PROB:
                round_payoff[u] = 0

        # 6. 执行重连操作
        for u in all_nodes:
            if temp_rewire[u] == 1:
                neighbors = list(G.neighbors(u))
                if len(neighbors) > 0:
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

        # 7. Q-learning 更新
        coop_rate = sum(1 for a in actions.values() if a == 'C') / N
        coop_rate_rec.append(coop_rate)

        if rnd > 0:
            for u in all_nodes:
                global_coop_rate = coop_rate
                state = agents[u].get_state(G, prev_round_payoff, global_coop_rate)

                # 重连使用原始收益，策略使用最终收益
                rewire_reward = base_payoff[u]
                strategy_reward = round_payoff[u]

                next_state = agents[u].get_state(G, round_payoff, global_coop_rate)
                agents[u].update_rewire(state, temp_rewire[u], rewire_reward, next_state)
                agents[u].update_strategy(state, temp_actions[u], strategy_reward, next_state)
                agents[u].decay_epsilon()

        prev_round_payoff = round_payoff.copy()

    return coop_rate_rec


# ========== 主程序：运行多组对比实验 ==========
print("开始多参数对比模拟...")

results = {}
final_coop_rates = {}

for punish_prob in PUNISH_PROB_LIST:
    print(f"正在模拟 PUNISH_PROB = {punish_prob}...")
    # 重置随机种子以保证可比性（每个实验用相同初始条件）
    random.seed(32)
    np.random.seed(32)
    coop_rates = run_simulation(punish_prob)
    results[punish_prob] = coop_rates
    final_coop_rates[punish_prob] = coop_rates[-1]  # 最终合作率

print("模拟完成！")
print("最终合作率:", final_coop_rates)

# ========== 绘图 ==========
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

colors = ['royalblue', 'darkorange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

# 1. 合作率对比（10条曲线）
for i, (punish_prob, coop_rates) in enumerate(results.items()):
    axes[0, 0].plot(range(1, n_rounds + 1), coop_rates,
                    marker='o', lw=2, color=colors[i],
                    markersize=2, label=f'P={punish_prob}')
axes[0, 0].set_xlabel('Round')
axes[0, 0].set_ylabel('Cooperation rate')
axes[0, 0].set_title('Cooperation Rate vs Punishment Probability')
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim([0, 1])

# 2. 最终合作率折线图
p_values = list(final_coop_rates.keys())
final_rates = list(final_coop_rates.values())
axes[0, 1].plot(p_values, final_rates, marker='o', lw=2, color='steelblue', markersize=8)
axes[0, 1].set_xlabel('Punishment Probability (p)')
axes[0, 1].set_ylabel('Final Cooperation Rate')
axes[0, 1].set_title('Final Cooperation Rate vs Punishment Probability')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_ylim([0, 1])

# 3. 重连率（占位）
axes[0, 2].text(0.5, 0.5, 'Rewire Rate\n(Placeholder)',
                ha='center', va='center', transform=axes[0, 2].transAxes, fontsize=12)
axes[0, 2].set_title('Rewire Rate per Round')

# 4. 边类型统计（占位）
axes[1, 0].text(0.5, 0.5, 'Edge Type\n(Placeholder)',
                ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=12)
axes[1, 0].set_title('Edge Type Distribution')

# 5. 惩罚统计（占位）
axes[1, 1].text(0.5, 0.5, 'Punishment Rate\n(Placeholder)',
                ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
axes[1, 1].set_title('Punishment Rate per Round')

# 6. 网络度分布（占位）
axes[1, 2].text(0.5, 0.5, 'Degree Distribution\n(Placeholder)',
                ha='center', va='center', transform=axes[1, 2].transAxes, fontsize=12)
axes[1, 2].set_title('Final Network Degree Distribution')

plt.tight_layout()
plt.show()