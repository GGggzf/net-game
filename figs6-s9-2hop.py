import networkx as nx
import matplotlib.pyplot as plt
import random, numpy as np, collections
# ========== 参数 ==========
random.seed(32); np.random.seed(32)
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
PUNISH_PROB_LIST = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
INIT_COOP_RATIO = 0.5
# ========== Q-learning 超参 ==========
QL_EPSILON = 0.05
QL_ALPHA = 0.2
QL_GAMMA = 0.5
QL_DECAY = 0.9
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
        return td
    def update_strategy(self, state, action, reward, next_state):
        max_q = max(self.Q_action[(next_state[0], next_state[1], 0)],
                    self.Q_action[(next_state[0], next_state[1], 1)])
        td = reward + self.gamma * max_q - self.Q_action[(state[0], state[1], action)]
        self.Q_action[(state[0], state[1], action)] += self.alpha * td
        return td
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
    rewire_rate_rec = []
    rewire_success_rate_rec = []
    punish_rate_rec = []
    edge_cc_rec = []
    edge_cd_rec = []
    edge_dd_rec = []

    # 四类行为：ReC, ReD, KC, KD
    rec_rec = []
    red_rec = []
    kc_rec = []
    kd_rec = []

    # Q更新、策略稳定性
    td_mean_rec = []
    strategy_switch_rec = []

    round_payoff = {n: 0 for n in all_nodes}
    prev_round_payoff = round_payoff.copy()
    prev_strategy_action = {u: None for u in all_nodes}

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

        # 统计4类行为占比
        ReC = sum(1 for u in all_nodes if temp_rewire[u]==1 and temp_actions[u]==0)
        ReD = sum(1 for u in all_nodes if temp_rewire[u]==1 and temp_actions[u]==1)
        KC  = sum(1 for u in all_nodes if temp_rewire[u]==0 and temp_actions[u]==0)
        KD  = sum(1 for u in all_nodes if temp_rewire[u]==0 and temp_actions[u]==1)
        rec_rec.append(ReC / N)
        red_rec.append(ReD / N)
        kc_rec.append(KC / N)
        kd_rec.append(KD / N)

        # 统计策略切换比例
        switch_cnt = 0
        if rnd > 0:
            for u in all_nodes:
                if prev_strategy_action[u] is not None and prev_strategy_action[u] != temp_actions[u]:
                    switch_cnt += 1
        strategy_switch_rec.append(switch_cnt / N)
        for u in all_nodes:
            prev_strategy_action[u] = temp_actions[u]

        actions = {u: 'C' if temp_actions[u] == 0 else 'D' for u in all_nodes}
        # 收益计算
        round_payoff = {n: 0 for n in all_nodes}
        for u, v in G.edges:
            a_u, a_v = actions[u], actions[v]
            p_u, p_v = PAYOFF[(a_u, a_v)]
            round_payoff[u] += p_u
            round_payoff[v] += p_v
        base_payoff = round_payoff.copy()
        # 声誉更新
        for u in all_nodes:
            if actions[u] == 'C':
                reputation[u] += 1
        # 惩罚机制
        punish_count = 0
        for u in all_nodes:
            if actions[u] == 'D' and random.random() < punish_prob:
                round_payoff[u] = 0
                punish_count += 1
        # 执行重连操作
        rewire_launch_count = sum(temp_rewire.values())
        rewire_success_count = 0
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
                        rewire_success_count += 1
                    else:
                        G.add_edge(u, target)
        # 重连成功率
        if rewire_launch_count > 0:
            success_rate = rewire_success_count / rewire_launch_count
        else:
            success_rate = 0.0

        # 统计边类型 CC / CD / DD
        cc, cd, dd = 0, 0, 0
        for u, v in G.edges:
            au, av = actions[u], actions[v]
            if au == 'C' and av == 'C':
                cc += 1
            elif au == 'D' and av == 'D':
                dd += 1
            else:
                cd += 1
        # Q-learning 更新
        coop_rate = sum(1 for a in actions.values() if a == 'C') / N
        coop_rate_rec.append(coop_rate)
        rewire_rate_rec.append(rewire_launch_count / N)
        rewire_success_rate_rec.append(success_rate)
        punish_rate_rec.append(punish_count / N)
        edge_cc_rec.append(cc)
        edge_cd_rec.append(cd)
        edge_dd_rec.append(dd)

        td_abs_list = []
        if rnd > 0:
            for u in all_nodes:
                global_coop_rate = coop_rate
                state = agents[u].get_state(G, prev_round_payoff, global_coop_rate)
                rewire_reward = base_payoff[u]
                next_state = agents[u].get_state(G, round_payoff, global_coop_rate)
                td_r = agents[u].update_rewire(state, temp_rewire[u], rewire_reward, next_state)
                strategy_reward = round_payoff[u]
                td_s = agents[u].update_strategy(state, temp_actions[u], strategy_reward, next_state)
                td_abs_list.append(abs(td_r))
                td_abs_list.append(abs(td_s))
                agents[u].decay_epsilon()
        if len(td_abs_list) > 0:
            td_mean_rec.append(np.mean(td_abs_list))
        else:
            td_mean_rec.append(0.0)

        prev_round_payoff = round_payoff.copy()
    return {
        "coop": coop_rate_rec,
        "rewire": rewire_rate_rec,
        "rewire_success": rewire_success_rate_rec,
        "punish": punish_rate_rec,
        "cc": edge_cc_rec,
        "cd": edge_cd_rec,
        "dd": edge_dd_rec,
        "rec": rec_rec,
        "red": red_rec,
        "kc": kc_rec,
        "kd": kd_rec,
        "td_mean": td_mean_rec,
        "strategy_switch": strategy_switch_rec,
        "final_graph": G
    }

# ========== 运行所有实验 ==========
print("开始运行10组实验...")
results = {}
for p in PUNISH_PROB_LIST:
    print(f"运行实验: p={p}, 初始合作率={INIT_COOP_RATIO}")
    res_dict = run_single_experiment(p, INIT_COOP_RATIO)
    results[p] = res_dict
print("所有实验完成！")

sample_p = 0.5
s = results[sample_p]
round_axis = np.arange(1, n_rounds + 1)

# ========== 原图：2行3列子图，修改ax3增加重连成功率 ==========
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# 1. ax[0,0] 全部10条：合作率时序
ax1 = axes[0, 0]
for i, p in enumerate(PUNISH_PROB_LIST):
    ax1.plot(round_axis, results[p]["coop"],
             color=colors[i], linewidth=2.5, marker='o', markersize=3, alpha=0.8, label=f'p={p}')
ax1.set_xlabel('Round', fontsize=12)
ax1.set_ylabel('Cooperation rate', fontsize=12)
#ax1.set_title(f'Cooperation Rate (Initial Coop Ratio={INIT_COOP_RATIO})', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 1])
ax1.legend(loc='best', fontsize=9, ncol=2)

# 2. ax[0,1] 全部10个p：最终合作率折线
ax2 = axes[0, 1]
final_rates = [results[p]["coop"][-1] for p in PUNISH_PROB_LIST]
ax2.plot(PUNISH_PROB_LIST, final_rates,
         marker='o', linewidth=2.5, markersize=10, color='darkblue',
         markerfacecolor='white', markeredgewidth=2, markeredgecolor='darkblue')
ax2.set_xlabel('Punishment Probability (p)', fontsize=12)
ax2.set_ylabel('Final Cooperation Rate', fontsize=12)
#ax2.set_title('Final Cooperation Rate by Punishment Probability', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 1])
for i, (p, rate) in enumerate(zip(PUNISH_PROB_LIST, final_rates)):
    ax2.text(p, rate + 0.03, f'{rate:.3f}', ha='center', va='bottom', fontsize=9)

# ======3. ax[0,2]【修改】重连发起率 + 重连成功率 ======
ax3 = axes[0, 2]
ax3.plot(round_axis, s["rewire"], color='#d62728', lw=2.5, marker='o', markersize=3, label='Number of rewiring agents')
ax3.plot(round_axis, s["rewire_success"], color='darkgreen', lw=2.5, linestyle='--', label='Rewiring success rate')
ax3.set_xlabel('Round', fontsize=12)
ax3.set_ylabel('Rate', fontsize=12)
#ax3.set_title(f'Rewiring‑attempt Rate and Rewiring Success Rate (p={sample_p})', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_ylim(0,1)
ax3.legend(fontsize=10)

# ======4. ax[1,0] 边类型CC/CD/DD时序，仅 p=0.5======
ax4 = axes[1, 0]
ax4.plot(round_axis, s["cc"], label="CC", lw=2, color="green")
ax4.plot(round_axis, s["cd"], label="CD", lw=2, color="orange")
ax4.plot(round_axis, s["dd"], label="DD", lw=2, color="red")
ax4.set_xlabel('Round', fontsize=12)
ax4.set_ylabel('Edge Count', fontsize=12)
#ax4.set_title(f'Edge Type Distribution over Rounds (p={sample_p})', fontsize=14, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

# ======5. ax[1,1] 惩罚率时序，仅 p=0.5======
ax5 = axes[1, 1]
ax5.plot(round_axis, s["punish"], color='#2ca02c', lw=2.5, marker='o', markersize=3, label=f'p={sample_p}')
ax5.set_xlabel('Round', fontsize=12)
ax5.set_ylabel('Punishment Rate', fontsize=12)
#ax5.set_title(f'Punishment Rate per Round (p={sample_p})', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.set_ylim(0,1)
ax5.legend(fontsize=10)

# ======6. ax[1,2] 最终网络度分布直方图，仅 p=0.5======
ax6 = axes[1, 2]
G_final = s["final_graph"]
degree_seq = [d for n, d in G_final.degree()]
ax6.hist(degree_seq, bins=range(0, max(degree_seq)+2), color="darkgreen", alpha=0.7, edgecolor='black')
ax6.set_xlabel('Node Degree', fontsize=12)
ax6.set_ylabel('Count', fontsize=12)
#ax6.set_title(f'Final Network Degree Distribution (p={sample_p})', fontsize=14, fontweight='bold')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('邻居1.png', dpi=150, bbox_inches='tight')

# ========== 新增图2：四类行为组合 ReC ReD KC KD ==========
plt.figure(figsize=(10,6))
plt.plot(round_axis, s["rec"], label="Re,C", lw=2)
plt.plot(round_axis, s["red"], label="Re,D", lw=2)
plt.plot(round_axis, s["kc"], label="K,C", lw=2)
plt.plot(round_axis, s["kd"], label="K,D", lw=2)
plt.xlabel("Round", fontsize=12)
plt.ylabel("Fraction of population", fontsize=12)
#plt.title(f"Four‑behavior composition ratio (p={sample_p})", fontsize=13, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.ylim(0,1)
plt.tight_layout()
plt.savefig('邻居2.png', dpi=300, bbox_inches='tight')

# ========== 新增图3：TD更新幅度 + 策略切换振荡 ==========
fig2, (ax_q, ax_switch) = plt.subplots(2,1, figsize=(10,8), sharex=True)
ax_q.plot(round_axis, s["td_mean"], color='darkred', lw=2)
ax_q.set_ylabel("Mean |TD error|", fontsize=12)
ax_q.set_title("Q‑learning update magnitude (average absolute TD error)", fontsize=13, fontweight='bold')
ax_q.grid(alpha=0.3)

ax_switch.plot(round_axis, s["strategy_switch"], color='darkblue', lw=2)
ax_switch.set_xlabel("Round", fontsize=12)
ax_switch.set_ylabel("Strategy switch fraction", fontsize=12)
#ax_switch.set_title("Strategy switching rate (higher = more oscillation/unstable)", fontsize=13, fontweight='bold')
ax_switch.grid(alpha=0.3)
ax_switch.set_ylim(0,1)
plt.tight_layout()
plt.savefig('邻居3.png', dpi=300, bbox_inches='tight')

plt.show()

# ========== 收敛轮次函数 & 结果打印 ==========
def get_convergence_round(coop_rec, threshold=0.05):
    if len(coop_rec) < 10:
        return n_rounds
    for i in range(10, len(coop_rec)):
        window = coop_rec[i - 10:i]
        if max(window) - min(window) < threshold:
            return i
    return n_rounds
print("\n========== 实验结果汇总 ==========")
print(f"{'Punish Prob':<12} {'Init Coop':<10} {'Final Coop Rate':<15} {'Converge Round':<15}")
print("-" * 55)
for p in PUNISH_PROB_LIST:
    final_rate = results[p]["coop"][-1]
    conv_round = get_convergence_round(results[p]["coop"])
    print(f"{p:<12.1f} {INIT_COOP_RATIO:<10.1f} {final_rate:<15.4f} {conv_round:<15}")
