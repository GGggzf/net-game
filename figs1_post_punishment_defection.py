import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import collections

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# ========== 参数 ==========
random.seed(32)
np.random.seed(32)

n_rounds = 200
N = 1000
BA_M = 2

PAYOFF = {
    ('C', 'C'): (8, 8),
    ('C', 'D'): (0, 10),
    ('D', 'C'): (10, 0),
    ('D', 'D'): (5, 5)
}

# ========== 惩罚机制参数 ==========
PUNISH_PROB = 0.5  # 背叛后被惩罚的概率

# ========== Q-learning 超参 ==========
QL_EPSILON = 0.05
QL_ALPHA = 0.2
QL_GAMMA = 0.5
QL_DECAY = 0.9

# ========== 初始策略 ==========
N_ALL_C = 500
N_ALL_D = 500
assert N_ALL_C + N_ALL_D == N, f"策略总数 {N_ALL_C + N_ALL_D} 必须等于 N={N}"

# ========== 初始化网络 ==========
G = nx.barabasi_albert_graph(N, BA_M)
all_nodes = list(G.nodes)

initial_strategies_list = ['ALL_C'] * N_ALL_C + ['ALL_D'] * N_ALL_D
random.shuffle(initial_strategies_list)

node_strategies = {u: {'strategy': initial_strategies_list[i]} for i, u in enumerate(all_nodes)}
reputation = {u: 0 for u in all_nodes}
cum_payoff = {u: 0 for u in all_nodes}


# ========== 双层 Q-learning Agent ==========
class TwoLayerQLAgent:
    """
    双层决策机制：
    - 第一层：学习 "维持(0)" 或 "重连(1)"
    - 第二层：学习 "合作(0)" 或 "背叛(1)"
    """

    def __init__(self, uid):
        self.uid = uid
        self.Q_rewire = collections.defaultdict(float)  # 第一层Q表：重连决策
        self.Q_action = collections.defaultdict(float)  # 第二层Q表：策略决策
        self.epsilon = QL_EPSILON
        self.alpha = QL_ALPHA
        self.gamma = QL_GAMMA

    def get_state(self, G, round_payoff, global_coop_rate):
        """
        状态 = (邻居平均收益状态, 全局合作率状态)
        """
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
        """第一层：选择维持(0)或重连(1)"""
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        q0 = self.Q_rewire[(state[0], state[1], 0)]
        q1 = self.Q_rewire[(state[0], state[1], 1)]
        if q1 > q0:
            return 1
        elif q0 > q1:
            return 0
        else:
            return random.randint(0, 1)

    def choose_strategy_action(self, state):
        """第二层：选择合作(0)或背叛(1)"""
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        q0 = self.Q_action[(state[0], state[1], 0)]
        q1 = self.Q_action[(state[0], state[1], 1)]
        if q1 > q0:
            return 1
        elif q0 > q1:
            return 0
        else:
            return random.randint(0, 1)

    def update_rewire(self, state, action, reward, next_state):
        """更新重连Q表"""
        max_q = max(
            self.Q_rewire[(next_state[0], next_state[1], 0)],
            self.Q_rewire[(next_state[0], next_state[1], 1)]
        )
        td = reward + self.gamma * max_q - self.Q_rewire[(state[0], state[1], action)]
        self.Q_rewire[(state[0], state[1], action)] += self.alpha * td

    def update_strategy(self, state, action, reward, next_state):
        """更新策略Q表"""
        max_q = max(
            self.Q_action[(next_state[0], next_state[1], 0)],
            self.Q_action[(next_state[0], state[1], 1)]
        )
        td = reward + self.gamma * max_q - self.Q_action[(state[0], state[1], action)]
        self.Q_action[(state[0], state[1], action)] += self.alpha * td

    def decay_epsilon(self):
        self.epsilon *= QL_DECAY


# 初始化 agents
agents = {u: TwoLayerQLAgent(u) for u in all_nodes}

# ========== 宏观统计 ==========
coop_rate_rec = []
rewire_count_rec = [0] * (n_rounds + 1)
rewire_rate_rec = []
punish_count_rec = [0] * (n_rounds + 1)
punish_rate_rec = []
edge_type_count = {'CC': [0] * (n_rounds + 1), 'CD': [0] * (n_rounds + 1), 'DD': [0] * (n_rounds + 1)}

# ========== 微观行为跟踪 ==========
# 每个节点每轮动作、收益、是否被惩罚
action_history = {u: [] for u in all_nodes}  # 'C' / 'D'
payoff_history = {u: [] for u in all_nodes}  # 每轮收益
punished_flag_history = {u: [] for u in all_nodes}  # 0 / 1
rewire_history = {u: [] for u in all_nodes}  # 0 / 1

# ========== 主循环 ==========
print("初始化完成，开始模拟...")
print(f"网络节点数: {N}, 边数: {G.number_of_edges()}")

actions = None
round_payoff = {u: 0 for u in all_nodes}
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

    actions = {u: 'C' if temp_actions[u] == 0 else 'D' for u in all_nodes}

    # 2. 收益计算
    round_payoff = {u: 0 for u in all_nodes}
    for u, v in G.edges:
        a_u, a_v = actions[u], actions[v]
        p_u, p_v = PAYOFF[(a_u, a_v)]
        round_payoff[u] += p_u
        round_payoff[v] += p_v

    # 3. 声誉更新：合作则声誉 +1
    for u in all_nodes:
        if actions[u] == 'C':
            reputation[u] += 1

    # 4. 惩罚机制
    punished_nodes = []
    for u in all_nodes:
        if actions[u] == 'D' and random.random() < PUNISH_PROB:
            round_payoff[u] = 0
            punished_nodes.append(u)

    punish_count_rec[rnd + 1] = len(punished_nodes)
    d_count = sum(1 for a in actions.values() if a == 'D')
    punish_rate_rec.append(len(punished_nodes) / d_count if d_count > 0 else 0)

    # 5. 执行重连
    rewire_count = 0
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
                    rewire_count += 1
                else:
                    G.add_edge(u, target)

    rewire_count_rec[rnd + 1] = rewire_count
    rewire_rate_rec.append(rewire_count / N)

    # 6. 计算合作率
    coop_rate = sum(1 for a in actions.values() if a == 'C') / N
    coop_rate_rec.append(coop_rate)

    # 7. Q-learning 更新
    if rnd > 0:
        for u in all_nodes:
            prev_coop_rate = coop_rate_rec[rnd - 1] if rnd - 1 >= 0 else 0.5
            state = agents[u].get_state(G, prev_round_payoff, prev_coop_rate)
            next_state = agents[u].get_state(G, round_payoff, coop_rate)

            reward = round_payoff[u]
            agents[u].update_rewire(state, temp_rewire[u], reward, next_state)
            agents[u].update_strategy(state, temp_actions[u], reward, next_state)
            agents[u].decay_epsilon()

    prev_round_payoff = round_payoff.copy()

    # 8. 累计收益
    for u in all_nodes:
        cum_payoff[u] += round_payoff[u]

    # 9. 记录边类型
    cc = cd = dd = 0
    for u, v in G.edges:
        a1, a2 = actions[u], actions[v]
        if a1 == 'C' and a2 == 'C':
            cc += 1
        elif a1 == 'D' and a2 == 'D':
            dd += 1
        else:
            cd += 1
    edge_type_count['CC'][rnd + 1] = cc
    edge_type_count['CD'][rnd + 1] = cd
    edge_type_count['DD'][rnd + 1] = dd

    # 10. 微观历史记录
    punished_set = set(punished_nodes)
    for u in all_nodes:
        action_history[u].append(actions[u])
        payoff_history[u].append(round_payoff[u])
        punished_flag_history[u].append(1 if u in punished_set else 0)
        rewire_history[u].append(temp_rewire[u])

    if (rnd + 1) % 50 == 0:
        print(f"Round {rnd + 1}: 合作率={coop_rate:.2%}, 重连次数={rewire_count}, 被惩罚D节点={len(punished_nodes)}")

print("\n模拟完成！")

# ========== 绘制：首次惩罚后的再次背叛次数分布 ==========
first_punish = {}
for u in all_nodes:
    punish_rounds = [r for r in range(n_rounds) if punished_flag_history[u][r] == 1]
    if punish_rounds:
        first_punish[u] = punish_rounds[0]  # 第一次被惩罚的轮次

relapse_counts = []
for u, first_round in first_punish.items():
    # 从第一次惩罚的下一轮开始，统计背叛次数
    relapse_cnt = sum(1 for r in range(first_round + 1, n_rounds) if action_history[u][r] == 'D')
    relapse_counts.append(relapse_cnt)

if relapse_counts:
    fig, ax = plt.subplots(figsize=(10, 6))
    # 直方图
    bins = range(max(relapse_counts) + 2)
    ax.hist(relapse_counts, bins=bins, alpha=0.7, label='节点数', color='steelblue', edgecolor='black')
    ax.set_xlabel('首次惩罚后再次背叛次数', fontsize=12)
    ax.set_ylabel('节点数', fontsize=12)
    ax.set_title('节点第一次被惩罚后后续背叛次数分布', fontsize=14)

    # 累积分布
    ax2 = ax.twinx()
    counts, edges = np.histogram(relapse_counts, bins=bins)
    cum_counts = np.cumsum(counts)
    cum_frac = cum_counts / cum_counts[-1]
    ax2.plot(edges[:-1], cum_frac, 'ro-', label='累积比例')
    ax2.set_ylabel('累积比例', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='lower right')

    ax.legend()
    plt.tight_layout()
    plt.savefig("再次背叛次数分布.png", dpi=300)
    plt.show()
else:
    print("没有节点被惩罚，无法绘制再次背叛次数分布图。")