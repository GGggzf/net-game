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
        """更新重连决策Q表，返回TD误差"""
        max_q = max(self.Q_rewire[(next_state[0], next_state[1], 0)],
                    self.Q_rewire[(next_state[0], next_state[1], 1)])
        td = reward + self.gamma * max_q - self.Q_rewire[(state[0], state[1], action)]
        self.Q_rewire[(state[0], state[1], action)] += self.alpha * td
        return td
    def update_strategy(self, state, action, reward, next_state):
        """更新策略决策Q表，返回TD误差"""
        max_q = max(self.Q_action[(next_state[0], next_state[1], 0)],
                    self.Q_action[(next_state[0], next_state[1], 1)])
        td = reward + self.gamma * max_q - self.Q_action[(state[0], state[1], action)]
        self.Q_action[(state[0], state[1], action)] += self.alpha * td
        return td
    def decay_epsilon(self):
        self.epsilon *= QL_DECAY

def run_simulation(PUNISH_PROB):
    """运行单次模拟，返回合作率记录 + 新增各项时序统计、最终网络"""
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
    # ===== 新增记录容器 =====
    coop_rate_rec = []
    rewire_rate_rec = []        # 每轮重连节点比例(发起重连)
    rewire_success_rate_rec = []# 重连成功率：成功重连 / 发起重连
    punish_rate_rec = []        # 每轮惩罚节点比例
    edge_cc_rec = []
    edge_cd_rec = []
    edge_dd_rec = []

    # 四类行为组合 ReC,ReD,KC,KD
    rec_rec = []
    red_rec = []
    kc_rec = []
    kd_rec = []

    # Q更新幅度、策略稳定性
    td_mean_rec = []            # 每轮平均TD误差绝对值
    strategy_switch_rec = []    # 策略切换节点占比（振荡指标）

    actions = None
    round_payoff = {n: 0 for n in all_nodes}
    prev_round_payoff = round_payoff.copy()
    prev_strategy_action = {u: None for u in all_nodes} # 记录上一轮策略动作，用于统计切换

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

        # 统计四类行为组合
        ReC = sum(1 for u in all_nodes if temp_rewire[u]==1 and temp_actions[u]==0)
        ReD = sum(1 for u in all_nodes if temp_rewire[u]==1 and temp_actions[u]==1)
        KC  = sum(1 for u in all_nodes if temp_rewire[u]==0 and temp_actions[u]==0)
        KD  = sum(1 for u in all_nodes if temp_rewire[u]==0 and temp_actions[u]==1)
        rec_rec.append(ReC/N)
        red_rec.append(ReD/N)
        kc_rec.append(KC/N)
        kd_rec.append(KD/N)

        # 统计策略切换比例（稳定性）
        switch_cnt = 0
        if rnd>0:
            for u in all_nodes:
                if prev_strategy_action[u] is not None and prev_strategy_action[u] != temp_actions[u]:
                    switch_cnt +=1
        strategy_switch_rec.append(switch_cnt / N)
        # 更新上一轮策略动作
        for u in all_nodes:
            prev_strategy_action[u] = temp_actions[u]

        # 2. 先执行博弈（使用当前网络结构）
        actions = {u: 'C' if temp_actions[u] == 0 else 'D' for u in all_nodes}
        # 3. 收益计算（原始收益）
        round_payoff = {n: 0 for n in all_nodes}
        for u, v in G.edges:
            a_u, a_v = actions[u], actions[v]
            p_u, p_v = PAYOFF[(a_u, a_v)]
            round_payoff[u] += p_u
            round_payoff[v] += p_v
        base_payoff = round_payoff.copy()
        # 4. 声誉更新
        for u in all_nodes:
            if actions[u] == 'C':
                reputation[u] += 1
        # 5. 惩罚机制（修改最终收益）
        punish_count = 0
        for u in all_nodes:
            if actions[u] == 'D' and random.random() < PUNISH_PROB:
                round_payoff[u] = 0
                punish_count += 1
        # 6. 执行重连操作 & 统计重连数量、重连成功数量
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
                    candidates = [v for v in all_nodes if v not in existing_neighbors]
                    if len(candidates) > 0:
                        new_neighbor = random.choice(candidates)
                        G.add_edge(u, new_neighbor)
                        rewire_success_count +=1
                    else:
                        G.add_edge(u, target)
        # 计算重连成功率
        if rewire_launch_count > 0:
            success_rate = rewire_success_count / rewire_launch_count
        else:
            success_rate = 0.0

        # ===== 统计边类型 CC / CD / DD =====
        cc, cd, dd = 0,0,0
        for u, v in G.edges:
            au, av = actions[u], actions[v]
            if au == 'C' and av == 'C':
                cc +=1
            elif au == 'D' and av == 'D':
                dd +=1
            else:
                cd +=1
        # 7. Q-learning 更新
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
                # 重连使用原始收益，策略使用最终收益
                rewire_reward = base_payoff[u]
                strategy_reward = round_payoff[u]
                next_state = agents[u].get_state(G, round_payoff, global_coop_rate)
                td_r = agents[u].update_rewire(state, temp_rewire[u], rewire_reward, next_state)
                td_s = agents[u].update_strategy(state, temp_actions[u], strategy_reward, next_state)
                td_abs_list.append(abs(td_r))
                td_abs_list.append(abs(td_s))
                agents[u].decay_epsilon()
        if len(td_abs_list)>0:
            td_mean_rec.append(np.mean(td_abs_list))
        else:
            td_mean_rec.append(0.0)

        prev_round_payoff = round_payoff.copy()
    # 返回全部时序数据 + 最终网络用于度分布
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
# ========== 主程序：运行多组对比实验 ==========
print("开始多参数对比模拟...")
results = {}
final_coop_rates = {}
for punish_prob in PUNISH_PROB_LIST:
    print(f"正在模拟 PUNISH_PROB = {punish_prob}...")
    random.seed(32)
    np.random.seed(32)
    res_dict = run_simulation(punish_prob)
    results[punish_prob] = res_dict
    final_coop_rates[punish_prob] = res_dict["coop"][-1]
print("模拟完成！")
print("最终合作率:", final_coop_rates)

sample_p = 0.5
sample_data = results[sample_p]
round_axis = np.arange(1, n_rounds+1)

# ========== 第一张大图：原有6个子图，修改子图3增加重连成功率 ==========
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
colors = ['royalblue', 'darkorange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
# 1. 合作率对比（10条曲线）
for i, (p, data) in enumerate(results.items()):
    axes[0,0].plot(round_axis, data["coop"], lw=2, color=colors[i],
                   marker='o', markersize=2, label=f'P={p}')
axes[0,0].set_xlabel('Round')
axes[0,0].set_ylabel('Cooperation rate')
axes[0,0].set_title('Cooperation Rate vs Punishment Probability')
axes[0,0].legend(fontsize=8)
axes[0,0].grid(True, alpha=0.3)
axes[0,0].set_ylim([0, 1])
# 2. 最终合作率折线图
p_values = list(final_coop_rates.keys())
final_rates = list(final_coop_rates.values())
axes[0,1].plot(p_values, final_rates, marker='o', lw=2, color='steelblue', markersize=8)
axes[0,1].set_xlabel('Punishment Probability (p)')
axes[0,1].set_ylabel('Final Cooperation Rate')
axes[0,1].set_title('Final Cooperation Rate vs Punishment Probability')
axes[0,1].grid(True, alpha=0.3)
axes[0,1].set_ylim([0, 1])
# 3.【修改】重连发起率 + 重连成功率两条曲线
axes[0,2].plot(round_axis, sample_data["rewire"], lw=2, color='crimson', label='Number of rewiring agents')
axes[0,2].plot(round_axis, sample_data["rewire_success"], lw=2, color='darkgreen', linestyle='--', label='Rewiring success rate')
axes[0,2].set_xlabel('Round')
axes[0,2].set_ylabel('Rate')
axes[0,2].set_title(f'Rewire Launch & Success Rate (P={sample_p})')
axes[0,2].legend()
axes[0,2].grid(True, alpha=0.3)
axes[0,2].set_ylim(0, 1)
# 4. 边类型数量时序（CC/CD/DD）
axes[1,0].plot(round_axis, sample_data["cc"], label="CC", lw=2, color="green")
axes[1,0].plot(round_axis, sample_data["cd"], label="CD", lw=2, color="orange")
axes[1,0].plot(round_axis, sample_data["dd"], label="DD", lw=2, color="red")
axes[1,0].set_xlabel('Round')
axes[1,0].set_ylabel('Edge Count')
axes[1,0].set_title(f'Edge Type Distribution over Rounds (P={sample_p})')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)
# 5. 惩罚率时序
axes[1,1].plot(round_axis, sample_data["punish"], lw=2, color="darkblue")
axes[1,1].set_xlabel('Round')
axes[1,1].set_ylabel('Punishment Rate')
axes[1,1].set_title(f'Punishment Rate per Round (P={sample_p})')
axes[1,1].grid(True, alpha=0.3)
axes[1,1].set_ylim(0,1)
# 6. 网络度分布直方图
G_final = sample_data["final_graph"]
degree_seq = [d for n, d in G_final.degree()]
axes[1,2].hist(degree_seq, bins=range(0, max(degree_seq)+2), color="darkgreen", alpha=0.7, edgecolor='black')
axes[1,2].set_xlabel('Node Degree')
axes[1,2].set_ylabel('Count')
axes[1,2].set_title(f'Final Network Degree Distribution (P={sample_p})')
axes[1,2].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('测试1.png', dpi=300, bbox_inches='tight')

# ========== 第二张新增图：四类行为组合占比 ReC,ReD,KC,KD ==========
plt.figure(figsize=(10,6))
plt.plot(round_axis, sample_data["rec"], label="Re,C", lw=2)
plt.plot(round_axis, sample_data["red"], label="Re,D", lw=2)
plt.plot(round_axis, sample_data["kc"], label="K,C", lw=2)
plt.plot(round_axis, sample_data["kd"], label="K,D", lw=2)
plt.xlabel("Round")
plt.ylabel("Fraction of population")
plt.title(f"Four behavior composition ratio (P={sample_p})")
plt.legend()
plt.grid(alpha=0.3)
plt.ylim(0,1)
plt.tight_layout()
plt.savefig('测试2.png', dpi=300, bbox_inches='tight')

# ========== 第三张新增图：Q更新幅度(TD绝对值均值) + 策略切换率（振荡指标） ==========
fig2, (ax1, ax2) = plt.subplots(2,1, figsize=(10,8), sharex=True)
ax1.plot(round_axis, sample_data["td_mean"], color='darkred', lw=2)
ax1.set_ylabel("Mean |TD error|")
ax1.set_title("Q‑learning update magnitude (average absolute TD error)")
ax1.grid(alpha=0.3)

ax2.plot(round_axis, sample_data["strategy_switch"], color='darkblue', lw=2)
ax2.set_xlabel("Round")
ax2.set_ylabel("Strategy switch fraction")
ax2.set_title("Strategy switching rate (higher = more oscillation/unstable)")
ax2.grid(alpha=0.3)
ax2.set_ylim(0,1)
plt.tight_layout()
plt.savefig('测试3.png', dpi=300, bbox_inches='tight')

plt.show()
