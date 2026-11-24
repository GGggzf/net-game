# 完整的双层决策机制代码（奖励分离版）
import networkx as nx
import matplotlib.pyplot as plt
import random, numpy as np, collections

# ========== 参数 ==========
random.seed(32);
np.random.seed(32)
n_rounds = 500
STRATEGIES = ['ALL_C', 'ALL_D']

N = 1000
BA_M = 2

PAYOFF = {
    ('C', 'C'): (8, 8),
    ('C', 'D'): (0, 10),
    ('D', 'C'): (10, 0),
    ('D', 'D'): (5, 5)
}

# ========== 惩罚机制参数 ==========
PUNISH_PROB = 0.1  # 背叛后被惩罚的概率

# ========== Q-learning 超参 ==========
QL_EPSILON = 0.05
QL_ALPHA = 0.2
QL_GAMMA = 0.5
QL_DECAY = 0.9

# ========== 初始化网络 ==========
G = nx.barabasi_albert_graph(N, BA_M)
all_nodes = list(G.nodes)

# ========== 初始化策略（仅第 0 轮用） ==========
N_ALL_C = 500
N_ALL_D = 500

assert N_ALL_C + N_ALL_D == N, f"策略总数 {N_ALL_C + N_ALL_D} 必须等于 N={N}"
initial_strategies_list = ['ALL_C'] * N_ALL_C + ['ALL_D'] * N_ALL_D
random.shuffle(initial_strategies_list)

node_strategies = {u: {'strategy': initial_strategies_list[i]} for i, u in enumerate(all_nodes)}
reputation = {u: 0 for u in G.nodes}
cum_payoff = {u: 0 for u in G.nodes}


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


# 初始化所有agent
agents = {u: TwoLayerQLAgent(u) for u in all_nodes}

# ========== 统计容器 ==========
strat_count = {s: [0] * (n_rounds + 1) for s in STRATEGIES}
for s in STRATEGIES:
    strat_count[s][0] = sum(1 for u in all_nodes if node_strategies[u]['strategy'] == s)

coop_rate_rec = []
edge_type_count = {'CC': [0] * (n_rounds + 1), 'CD': [0] * (n_rounds + 1), 'DD': [0] * (n_rounds + 1)}
punish_count_rec = [0] * (n_rounds + 1)
punish_rate_rec = []

# 重连相关统计
rewire_count_rec = [0] * (n_rounds + 1)
rewire_rate_rec = []

# 网络结构统计
cooperator_degree_rec = []
defector_degree_rec = []
clustering_coeff_rec = []
assortativity_rec = []
modularity_rec = []
num_communities_rec = []
coop_community_ratio_rec = []

print("初始化完成，开始模拟...")
print(f"网络节点数: {N}, 边数: {G.number_of_edges()}")

# ========== 主循环 ==========
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

    # 转换为行动
    actions = {u: 'C' if temp_actions[u] == 0 else 'D' for u in all_nodes}

    # 2. 收益计算（原始收益）
    round_payoff = {n: 0 for n in all_nodes}
    for u, v in G.edges:
        a_u, a_v = actions[u], actions[v]
        p_u, p_v = PAYOFF[(a_u, a_v)]
        round_payoff[u] += p_u
        round_payoff[v] += p_v

    # 保存原始收益（未受惩罚）
    base_payoff = round_payoff.copy()

    # 3. 声誉更新
    for u in all_nodes:
        if actions[u] == 'C':
            reputation[u] += 1

    # 4. 惩罚机制（修改最终收益）
    punished_nodes = []
    for u in all_nodes:
        if actions[u] == 'D' and random.random() < PUNISH_PROB:
            round_payoff[u] = 0
            punished_nodes.append(u)

    punish_count_rec[rnd + 1] = len(punished_nodes)
    d_count = sum(1 for a in actions.values() if a == 'D')
    punish_rate_rec.append(len(punished_nodes) / d_count if d_count > 0 else 0)

    # 5. 执行重连操作
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

    # 6. Q-learning 更新
    coop_rate = sum(1 for a in actions.values() if a == 'C') / N
    coop_rate_rec.append(coop_rate)

    if rnd > 0:
        for u in all_nodes:
            global_coop_rate = coop_rate
            state = agents[u].get_state(G, prev_round_payoff, global_coop_rate)

            # 重连决策使用原始收益（base_payoff）
            rewire_reward = base_payoff[u]
            # 策略决策使用最终收益（round_payoff，受惩罚影响）
            strategy_reward = round_payoff[u]

            next_state = agents[u].get_state(G, round_payoff, global_coop_rate)

            agents[u].update_rewire(state, temp_rewire[u], rewire_reward, next_state)
            agents[u].update_strategy(state, temp_actions[u], strategy_reward, next_state)

            agents[u].decay_epsilon()

    prev_round_payoff = round_payoff.copy()

    # 7. 累计收益
    for n in all_nodes:
        cum_payoff[n] += round_payoff[n]

    # 8. 统计
    for s in STRATEGIES:
        strat_count[s][rnd + 1] = sum(1 for u in all_nodes if node_strategies[u]['strategy'] == s)

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

    # 网络结构统计
    coop_degrees = [G.degree(u) for u in all_nodes if actions[u] == 'C']
    defect_degrees = [G.degree(u) for u in all_nodes if actions[u] == 'D']
    avg_coop_degree = sum(coop_degrees) / len(coop_degrees) if coop_degrees else 0
    avg_defect_degree = sum(defect_degrees) / len(defect_degrees) if defect_degrees else 0
    cooperator_degree_rec.append(avg_coop_degree)
    defector_degree_rec.append(avg_defect_degree)

    try:
        clustering = nx.average_clustering(G)
    except:
        clustering = 0
    clustering_coeff_rec.append(clustering)

    node_types = {u: 0 if actions[u] == 'C' else 1 for u in all_nodes}
    nx.set_node_attributes(G, node_types, 'type')
    try:
        assort = nx.attribute_assortativity_coefficient(G, 'type')
    except:
        assort = 0
    assortativity_rec.append(assort)

    try:
        from networkx.algorithms.community import greedy_modularity_communities, modularity
        communities = list(greedy_modularity_communities(G))
        num_communities = len(communities)
        modularity_score = modularity(G, communities)
        community_coop_ratio = {}
        for comm_id, comm_nodes in enumerate(communities):
            coop_count = sum(1 for u in comm_nodes if actions[u] == 'C')
            community_coop_ratio[comm_id] = coop_count / len(comm_nodes) if comm_nodes else 0
        coop_dominant_comms = sum(1 for ratio in community_coop_ratio.values() if ratio > 0.5)
        coop_comm_ratio = coop_dominant_comms / num_communities if num_communities > 0 else 0
    except Exception as e:
        modularity_score = 0
        num_communities = 0
        coop_comm_ratio = 0
    modularity_rec.append(modularity_score)
    num_communities_rec.append(num_communities)
    coop_community_ratio_rec.append(coop_comm_ratio)

    if (rnd + 1) % 50 == 0:
        print(f"Round {rnd + 1}: 合作率={coop_rate:.2%}, 重连次数={rewire_count}, 被惩罚D节点={len(punished_nodes)}")

print("\n模拟完成！")

# ========== 绘图 ==========
fig, axes = plt.subplots(3, 3, figsize=(20, 18))

# 第一行：原有图表
# 1. 合作率
axes[0, 0].plot(range(1, n_rounds + 1), coop_rate_rec, marker='o', lw=2, color='royalblue', markersize=3)
axes[0, 0].set_xlabel('Round');
axes[0, 0].set_ylabel('Cooperation rate')
axes[0, 0].set_title(f'Cooperation Rate (Punish prob={PUNISH_PROB})');
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim([0, 1])

# 2. 重连次数统计
axes[0, 1].plot(range(1, n_rounds + 1), rewire_count_rec[1:], marker='s', lw=2, color='darkorange', markersize=3)
axes[0, 1].set_xlabel('Round');
axes[0, 1].set_ylabel('Rewire Count')
axes[0, 1].set_title('Rewire Count per Round');
axes[0, 1].grid(True, alpha=0.3)

# 3. 重连率
axes[0, 2].plot(range(1, n_rounds + 1), rewire_rate_rec, marker='^', lw=2, color='green', markersize=3)
axes[0, 2].set_xlabel('Round');
axes[0, 2].set_ylabel('Rewire Rate')
axes[0, 2].set_title('Rewire Rate per Round');
axes[0, 2].grid(True, alpha=0.3)

# 第二行：网络结构分析
# 4. 度中心性差异：合作者 vs 背叛者
axes[1, 0].plot(range(1, n_rounds + 1), cooperator_degree_rec, label='Cooperators (C)',
                lw=2, color='blue', marker='o', markersize=3)
axes[1, 0].plot(range(1, n_rounds + 1), defector_degree_rec, label='Defectors (D)',
                lw=2, color='red', marker='s', markersize=3)
axes[1, 0].fill_between(range(1, n_rounds + 1), cooperator_degree_rec, defector_degree_rec,
                        alpha=0.3, color='gray', label='Difference')
axes[1, 0].set_xlabel('Round');
axes[1, 0].set_ylabel('Average Degree')
axes[1, 0].set_title('Degree Centrality: Cooperators vs Defectors\n(Who is in the center?)')
axes[1, 0].legend();
axes[1, 0].grid(True, alpha=0.3)

# 5. 聚类系数
axes[1, 1].plot(range(1, n_rounds + 1), clustering_coeff_rec, marker='D', lw=2,
                color='teal', markersize=3)
axes[1, 1].set_xlabel('Round');
axes[1, 1].set_ylabel('Average Clustering Coefficient')
axes[1, 1].set_title('Clustering Coefficient\n(Does rewiring create "cliques"?)')
axes[1, 1].grid(True, alpha=0.3)

# 6. 同配性（Assortativity）
axes[1, 2].plot(range(1, n_rounds + 1), assortativity_rec, marker='h', lw=2,
                color='purple', markersize=3)
axes[1, 2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[1, 2].set_xlabel('Round');
axes[1, 2].set_ylabel('Assortativity Coefficient')
axes[1, 2].set_title('Assortativity (Homophily)\n(Do cooperators connect with cooperators?)')
axes[1, 2].set_ylim([-1, 1])
axes[1, 2].grid(True, alpha=0.3)

# 第三行：社区结构分析
# 7. 模块度（Modularity）
axes[2, 0].plot(range(1, n_rounds + 1), modularity_rec, marker='*', lw=2,
                color='darkgreen', markersize=5)
axes[2, 0].set_xlabel('Round');
axes[2, 0].set_ylabel('Modularity (Q)')
axes[2, 0].set_title('Network Modularity\n(Community structure strength)')
axes[2, 0].grid(True, alpha=0.3)

# 8. 社区数量与合作者主导社区比例
ax_twin = axes[2, 1].twinx()
line1 = axes[2, 1].plot(range(1, n_rounds + 1), num_communities_rec, marker='p', lw=2,
                        color='navy', markersize=3, label='# Communities')
line2 = ax_twin.plot(range(1, n_rounds + 1), coop_community_ratio_rec, marker='H', lw=2,
                     color='crimson', markersize=3, label='Coop-dominated Ratio')
axes[2, 1].set_xlabel('Round')
axes[2, 1].set_ylabel('Number of Communities', color='navy')
ax_twin.set_ylabel('Cooperator-dominated Communities Ratio', color='crimson')
axes[2, 1].set_title('Community Structure\n(Coop vs Defector communities)')
axes[2, 1].grid(True, alpha=0.3)
lines = line1 + line2
labels = [l.get_label() for l in lines]
axes[2, 1].legend(lines, labels, loc='upper right')

# 9. 边类型统计（原有）
axes[2, 2].plot(range(n_rounds + 1), edge_type_count['CC'], label='CC', lw=2, color='blue')
axes[2, 2].plot(range(n_rounds + 1), edge_type_count['CD'], label='CD', lw=2, color='orange')
axes[2, 2].plot(range(n_rounds + 1), edge_type_count['DD'], label='DD', lw=2, color='red')
axes[2, 2].set_xlabel('Round');
axes[2, 2].set_ylabel('Edge Count')
axes[2, 2].set_title('Edge Type Distribution (CC/CD/DD)')
axes[2, 2].legend();
axes[2, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ========== 最终网络可视化 ==========
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7))

final_cooperators = [u for u in all_nodes if actions[u] == 'C']
final_defectors = [u for u in all_nodes if actions[u] == 'D']

coop_degrees_final = [G.degree(u) for u in final_cooperators]
defect_degrees_final = [G.degree(u) for u in final_defectors]

axes2[0].hist([coop_degrees_final, defect_degrees_final], bins=20,
              label=['Cooperators', 'Defectors'],
              color=['blue', 'red'], alpha=0.7, edgecolor='black')
axes2[0].set_xlabel('Degree');
axes2[0].set_ylabel('Frequency')
axes2[0].set_title('Final Degree Distribution by Strategy')
axes2[0].legend();
axes2[0].grid(True, alpha=0.3)

sample_size = min(200, N)
sample_nodes = random.sample(all_nodes, sample_size)
subG = G.subgraph(sample_nodes)
pos = nx.spring_layout(subG, seed=42)
node_colors = ['blue' if actions[u] == 'C' else 'red' for u in subG.nodes()]
node_sizes = [G.degree(u) * 20 for u in subG.nodes()]

nx.draw_networkx_nodes(subG, pos, node_color=node_colors, node_size=node_sizes,
                       alpha=0.7, ax=axes2[1])
nx.draw_networkx_edges(subG, pos, alpha=0.2, ax=axes2[1])
axes2[1].set_title(f'Network Snapshot (Sample {sample_size} nodes)\nBlue=Coop, Red=Defect, Size=Degree')
axes2[1].axis('off')

plt.tight_layout()
plt.savefig("复杂网络.png")
plt.show()

# ========== 打印最终统计摘要 ==========
print("\n" + "=" * 60)
print("网络结构分析摘要")
print("=" * 60)
print(f"最终合作率: {coop_rate_rec[-1]:.2%}")
print(f"最终平均聚类系数: {clustering_coeff_rec[-1]:.4f}")
print(f"最终同配性系数: {assortativity_rec[-1]:.4f} (正值表示同质性)")
print(f"最终模块度: {modularity_rec[-1]:.4f}")
print(f"最终社区数量: {num_communities_rec[-1]}")
print(f"合作者主导社区比例: {coop_community_ratio_rec[-1]:.2%}")
print(f"\n度中心性对比:")
print(f"  - 合作者平均度: {cooperator_degree_rec[-1]:.2f}")
print(f"  - 背叛者平均度: {defector_degree_rec[-1]:.2f}")
degree_diff = cooperator_degree_rec[-1] - defector_degree_rec[-1]
print(f"  - 差异 (C-D): {degree_diff:+.2f} ({'合作者更中心' if degree_diff > 0 else '背叛者更中心'})")
print("=" * 60)