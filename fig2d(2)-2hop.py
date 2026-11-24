import networkx as nx
import matplotlib.pyplot as plt
import random, numpy as np, collections
# ========== 参数 ==========
random.seed(32);
np.random.seed(32)
n_rounds = 400
STRATEGIES = ['ALL_C', 'ALL_D']
N = 1000
BA_M = 2
PAYOFF = {
    ('C', 'C'): (8, 8),
    ('C', 'D'): (0, 10),
    ('D', 'C'): (10, 0),
    ('D', 'D'): (5, 5)
}
# ========== 实验参数组合 ==========
# 更新为10种惩罚概率
PUNISH_PROB_LIST = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
INIT_COOP_RATIO = 0.5  # 固定初始合作者比例为0.5
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
# ========== 运行单次实验的函数 ==========
def run_single_experiment(punish_prob, init_coop_ratio):
    G = nx.barabasi_albert_graph(N, BA_M)
    all_nodes = list(G.nodes)
    N_ALL_C = int(N * init_coop_ratio)
    N_ALL_D = N - N_ALL_C
    initial_strategies_list = ['ALL_C'] * N_ALL_C + ['ALL_D'] * N_ALL_D
    random.shuffle(initial_strategies_list)
    node_strategies = {u: {'strategy': initial_strategies_list[i]} for i, u in enumerate(all_nodes)}
    reputation = {u: 0 for u in G.nodes}
    agents = {u: TwoLayerQLAgent(u) for u in all_nodes}
    coop_rate_rec = []
    round_payoff = {n: 0 for n in all_nodes}
    prev_round_payoff = round_payoff.copy()
    for rnd in range(n_rounds):
        temp_rewire = {}
        temp_actions = {}
        for u in all_nodes:
            if rnd == 0:
                temp_actions[u] = 1 if node_strategies[u]['strategy'] == 'ALL_D' else 0
                temp_rewire[u] = 0
            else:
                prev_coop_rate = coop_rate_rec[-1] if coop_rate_rec else 0.5
                state = agents[u].get_state(G, round_payoff, prev_coop_rate)
                temp_rewire[u] = agents[u].choose_rewire_action(state)
                temp_actions[u] = agents[u].choose_strategy_action(state)
        actions = {u: 'C' if temp_actions[u] == 0 else 'D' for u in all_nodes}
        # 收益计算
        round_payoff = {n: 0 for n in all_nodes}
        for u, v in G.edges:
            a_u, a_v = actions[u], actions[v]
            p_u, p_v = PAYOFF[(a_u, a_v)]
            round_payoff[u] += p_u
            round_payoff[v] += p_v
        # 声誉更新
        for u in all_nodes:
            if actions[u] == 'C':
                reputation[u] += 1
        # 惩罚机制
        for u in all_nodes:
            if actions[u] == 'D' and random.random() < punish_prob:
                round_payoff[u] = 0
        # 执行重连操作
        for u in all_nodes:
            if temp_rewire[u] == 1:
                neighbors = list(G.neighbors(u))
                if len(neighbors) > 0:
                    min_rep = min(reputation[v] for v in neighbors)
                    low_rep_neighbors = [v for v in neighbors if reputation[v] == min_rep]
                    target = random.choice(low_rep_neighbors)

                    # =========【修改核心】预判：删除u‑target后u会不会变成孤立节点 =========
                    if G.degree[u] - 1 <= 0:
                        # 删除这条边会导致u孤立，直接放弃本次重连，保持原有连接不做任何改动
                        continue

                    G.remove_edge(u, target)
                    existing_neighbors = set(G.neighbors(u))
                    existing_neighbors.add(u)
                    # 获取二阶邻居
                    second_order_neighbors = set()
                    for neighbor in G.neighbors(u):
                        for nn in G.neighbors(neighbor):
                            if nn != u and nn not in G.neighbors(u):
                                second_order_neighbors.add(nn)
                    candidates = [v for v in second_order_neighbors
                                  if v != u and v not in existing_neighbors]

                    if len(candidates) > 0:
                        max_rep = max(reputation[v] for v in candidates)
                        best_candidates = [v for v in candidates if reputation[v] == max_rep]
                        new_neighbor = random.choice(best_candidates)
                        G.add_edge(u, new_neighbor)
                    else:
                        # 找不到候选新伙伴，把刚才删掉的旧边恢复回来
                        G.add_edge(u, target)
        # ========= 删除原来的孤立节点兜底修复代码，源头阻止产生孤立节点 =========

        # Q-learning 更新
        coop_rate = sum(1 for a in actions.values() if a == 'C') / N
        coop_rate_rec.append(coop_rate)
        if rnd > 0:
            for u in all_nodes:
                global_coop_rate = coop_rate
                state = agents[u].get_state(G, prev_round_payoff, global_coop_rate)
                rewire_reward = round_payoff[u]
                next_state = agents[u].get_state(G, round_payoff, global_coop_rate)
                agents[u].update_rewire(state, temp_rewire[u], rewire_reward, next_state)
                strategy_reward = round_payoff[u]
                agents[u].update_strategy(state, temp_actions[u], strategy_reward, next_state)
                agents[u].decay_epsilon()
        prev_round_payoff = round_payoff.copy()
    return coop_rate_rec
# ========== 运行所有实验 ==========
print("开始运行10组实验...")
results = {}
for p in PUNISH_PROB_LIST:
    print(f"运行实验: p={p}, 初始合作率={INIT_COOP_RATIO}")
    coop_rec = run_single_experiment(p, INIT_COOP_RATIO)
    results[p] = coop_rec
print("所有实验完成！")
# ========== 绘图 ==========
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
# 定义10种不同颜色（使用tab10配色方案）
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
# 1. 合作率对比图（10条曲线，对应不同p值）
ax1 = axes[0, 0]
for i, p in enumerate(PUNISH_PROB_LIST):
    ax1.plot(range(1, n_rounds + 1), results[p],
             color=colors[i],
             linewidth=2.5,
             marker='o',
             markersize=3,
             alpha=0.8,
             label=f'p={p}')
ax1.set_xlabel('Round', fontsize=12)
ax1.set_ylabel('Cooperation rate', fontsize=12)
ax1.set_title(f'Cooperation Rate (Initial Coop Ratio={INIT_COOP_RATIO})', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 1])
ax1.legend(loc='best', fontsize=9, ncol=2)
# 2. 最终合作率对比（折线图，折点用圆圈显示）
ax2 = axes[0, 1]
final_rates = [results[p][-1] for p in PUNISH_PROB_LIST]
ax2.plot(PUNISH_PROB_LIST, final_rates,
         marker='o',
         linewidth=2.5,
         markersize=10,
         color='darkblue',
         markerfacecolor='white',
         markeredgewidth=2,
         markeredgecolor='darkblue')
ax2.set_xlabel('Punishment Probability (p)', fontsize=12)
ax2.set_ylabel('Final Cooperation Rate', fontsize=12)
ax2.set_title('Final Cooperation Rate by Punishment Probability', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 1])
# 在折点添加数值标签
for i, (p, rate) in enumerate(zip(PUNISH_PROB_LIST, final_rates)):
    ax2.text(p, rate + 0.03, f'{rate:.3f}', ha='center', va='bottom', fontsize=9)
# 3. 收敛速度分析
ax3 = axes[0, 2]
def get_convergence_round(coop_rec, threshold=0.05):
    if len(coop_rec) < 10:
        return n_rounds
    for i in range(10, len(coop_rec)):
        window = coop_rec[i - 10:i]
        if max(window) - min(window) < threshold:
            return i
    return n_rounds
conv_rounds = [get_convergence_round(results[p]) for p in PUNISH_PROB_LIST]
ax3.plot(PUNISH_PROB_LIST, conv_rounds, marker='s', linewidth=2.5, markersize=10, color='darkgreen')
ax3.set_xlabel('Punishment Probability (p)', fontsize=12)
ax3.set_ylabel('Convergence Round', fontsize=12)
ax3.set_title('Convergence Speed vs Punishment Probability', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
# 4. 合作率演化热力图
ax4 = axes[1, 0]
data_matrix = np.array([results[p] for p in PUNISH_PROB_LIST])
im = ax4.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax4.set_xticks([0, 25, 50, 75, 99])
ax4.set_xticklabels(['1', '25', '50', '75', '99'])
ax4.set_yticks(range(len(PUNISH_PROB_LIST)))
ax4.set_yticklabels([f'p={p}' for p in PUNISH_PROB_LIST])
ax4.set_xlabel('Round', fontsize=12)
ax4.set_ylabel('Punishment Probability', fontsize=12)
ax4.set_title('Cooperation Rate Evolution Heatmap', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax4, label='Cooperation Rate')
# 5. 特定轮次的合作率对比
ax5 = axes[1, 1]
rounds_to_check = [10, 30, 50, 70, 99]
x_pos = np.arange(len(PUNISH_PROB_LIST))
width = 0.15
for i, r in enumerate(rounds_to_check):
    ax5.bar(x_pos + i * width, [results[p][r] for p in PUNISH_PROB_LIST], width, label=f'Round {r + 1}', alpha=0.8)
ax5.set_xlabel('Punishment Probability (p)', fontsize=12)
ax5.set_ylabel('Cooperation Rate', fontsize=12)
ax5.set_title('Cooperation Rate at Different Rounds', fontsize=14, fontweight='bold')
ax5.set_xticks(x_pos + width * 2)
ax5.set_xticklabels([f'{p}' for p in PUNISH_PROB_LIST])
ax5.legend(loc='best', fontsize=9)
ax5.grid(True, alpha=0.3, axis='y')
ax5.set_ylim([0, 1])
# 6. 合作率标准差分析（稳定性）
ax6 = axes[1, 2]
# 计算每10轮的标准差
window_size = 10
std_data = []
for p in PUNISH_PROB_LIST:
    stds = []
    for i in range(0, n_rounds - window_size + 1, window_size):
        window = results[p][i:i + window_size]
        stds.append(np.std(window))
    std_data.append(stds)
for i, p in enumerate(PUNISH_PROB_LIST):
    x_vals = range(5, n_rounds + 1, window_size)
    ax6.plot(x_vals, std_data[i], marker='o', label=f'p={p}', color=colors[i], linewidth=2)
ax6.set_xlabel('Round', fontsize=12)
ax6.set_ylabel('Standard Deviation (10‑round window)', fontsize=12)
ax6.set_title('Cooperation Rate Stability Analysis', fontsize=14, fontweight='bold')
ax6.legend(loc='best', fontsize=8, ncol=2)
ax6.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('节点保护-邻居.png', dpi=150, bbox_inches='tight')
plt.show()
# 打印汇总统计
print("\n========== 实验结果汇总 ==========")
print(f"{'Punish Prob':<12} {'Init Coop':<10} {'Final Coop Rate':<15} {'Converge Round':<15}")
print("-" * 55)
for p in PUNISH_PROB_LIST:
    final_rate = results[p][-1]
    conv_round = get_convergence_round(results[p])
    print(f"{p:<12.1f} {INIT_COOP_RATIO:<10.1f} {final_rate:<15.4f} {conv_round:<15}")
