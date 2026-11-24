import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import warnings
# 屏蔽exp溢出警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ========== 参数 ==========
random.seed(32)
np.random.seed(32)
n_rounds = 400
N = 1000
BA_M = 2
PAYOFF = {
    ('C', 'C'): (8, 8),
    ('C', 'D'): (0, 10),
    ('D', 'C'): (10, 0),
    ('D', 'D'): (5, 5)
}
# 惩罚机制参数列表（用于对比）
PUNISH_PROB_LIST = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
INIT_COOP_RATIO = 0.5
# Fermi学习参数
FERMI_K = 0.1
REWIRE_PROB = 0.47  # 固定重连概率

def fermi_prob(p_u, p_v, k):
    """Fermi模仿概率，增加clip防止exp溢出"""
    z = (p_u - p_v) / k
    z = np.clip(z, -700, 700)
    return 1.0 / (1.0 + np.exp(z))


def run_single_experiment(punish_prob, init_coop_ratio):
    """
    Fermi策略更新 + 保留原代码2的重连逻辑（二阶邻居优先、选高声誉新邻居）
    """
    # 初始化BA网络
    G = nx.barabasi_albert_graph(N, BA_M)
    all_nodes = list(G.nodes)
    # 初始化策略
    N_ALL_C = int(N * init_coop_ratio)
    N_ALL_D = N - N_ALL_C
    initial_strategies_list = ['C'] * N_ALL_C + ['D'] * N_ALL_D
    random.shuffle(initial_strategies_list)
    node_strategy = {u: initial_strategies_list[i] for i, u in enumerate(all_nodes)}

    reputation = {u: 0 for u in G.nodes}

    # 记录容器
    coop_rate_rec = []
    rewire_rate_rec = []
    punish_rate_rec = []
    edge_cc_rec = []
    edge_cd_rec = []
    edge_dd_rec = []

    for rnd in range(n_rounds):
        # 1. 当前策略作为博弈行动
        actions = {u: node_strategy[u] for u in all_nodes}

        # 2. 原始收益计算
        round_payoff = {n: 0 for n in all_nodes}
        for u, v in G.edges:
            a_u, a_v = actions[u], actions[v]
            p_u, p_v = PAYOFF[(a_u, a_v)]
            round_payoff[u] += p_u
            round_payoff[v] += p_v

        # 3. 更新声誉：合作则声誉+1
        for u in all_nodes:
            if actions[u] == 'C':
                reputation[u] += 1

        # 4. 惩罚机制：背叛者以punish_prob概率收益置0
        punish_count = 0
        for u in all_nodes:
            if actions[u] == 'D' and random.random() < punish_prob:
                round_payoff[u] = 0
                punish_count += 1

        # 5. 执行重连：每个节点以REWIRE_PROB概率重连（保留代码2重连逻辑：删最低声誉邻居，优先二阶邻居，选最高声誉候选）
        rewire_count = 0
        for u in all_nodes:
            if random.random() < REWIRE_PROB:
                rewire_count += 1
                neighbors = list(G.neighbors(u))
                if len(neighbors) > 0:
                    # 删除声誉最小的老邻居
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
                    else:
                        G.add_edge(u, target)

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

        # 6. Fermi策略更新（第一轮rnd=0不更新，从第二轮开始）
        if rnd > 0:
            for u in all_nodes:
                neigh = list(G.neighbors(u))
                if len(neigh) == 0:
                    continue
                # 随机选一个邻居作为模仿对象
                v = random.choice(neigh)
                pu = round_payoff[u]
                pv = round_payoff[v]
                w = fermi_prob(pu, pv, FERMI_K)
                if random.random() < w:
                    node_strategy[u] = node_strategy[v]

        # 记录本轮统计
        coop_rate = sum(1 for a in actions.values() if a == 'C') / N
        coop_rate_rec.append(coop_rate)
        rewire_rate_rec.append(rewire_count / N)
        punish_rate_rec.append(punish_count / N)
        edge_cc_rec.append(cc)
        edge_cd_rec.append(cd)
        edge_dd_rec.append(dd)

    return {
        "coop": coop_rate_rec,
        "rewire": rewire_rate_rec,
        "punish": punish_rate_rec,
        "cc": edge_cc_rec,
        "cd": edge_cd_rec,
        "dd": edge_dd_rec,
        "final_graph": G
    }


# ========== 运行所有实验 ==========
print("开始运行10组实验...")
results = {}
for p in PUNISH_PROB_LIST:
    print(f"运行实验: p={p}, 初始合作率={INIT_COOP_RATIO}")
    random.seed(32)
    np.random.seed(32)
    res_dict = run_single_experiment(p, INIT_COOP_RATIO)
    results[p] = res_dict
print("所有实验完成！")

# ========== 绘图 ==========
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
round_axis = np.arange(1, n_rounds + 1)
sample_p = 0.5
s = results[sample_p]

# 1. ax[0,0] 全部10条：合作率时序
ax1 = axes[0, 0]
for i, p in enumerate(PUNISH_PROB_LIST):
    ax1.plot(round_axis, results[p]["coop"],
             color=colors[i], linewidth=2.5, marker='o', markersize=3, alpha=0.8, label=f'p={p}')
ax1.set_xlabel('Round', fontsize=12)
ax1.set_ylabel('Cooperation rate', fontsize=12)
ax1.set_title(f'Cooperation Rate (Initial Coop Ratio={INIT_COOP_RATIO})', fontsize=14, fontweight='bold')
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
ax2.set_title('Final Cooperation Rate by Punishment Probability', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 1])
for i, (p, rate) in enumerate(zip(PUNISH_PROB_LIST, final_rates)):
    ax2.text(p, rate + 0.03, f'{rate:.3f}', ha='center', va='bottom', fontsize=9)

# 3. ax[0,2] 重连率时序，仅 p=0.5
ax3 = axes[0, 2]
ax3.plot(round_axis, s["rewire"], color='#d62728', lw=2.5, marker='o', markersize=3, label=f'p={sample_p}')
ax3.set_xlabel('Round', fontsize=12)
ax3.set_ylabel('Rewire Rate', fontsize=12)
ax3.set_title(f'Rewire Rate per Round (p={sample_p})', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_ylim(0,1)
ax3.legend(fontsize=10)

# 4. ax[1,0] 边类型CC/CD/DD时序，仅 p=0.5
ax4 = axes[1, 0]
ax4.plot(round_axis, s["cc"], label="CC", lw=2, color="green")
ax4.plot(round_axis, s["cd"], label="CD", lw=2, color="orange")
ax4.plot(round_axis, s["dd"], label="DD", lw=2, color="red")
ax4.set_xlabel('Round', fontsize=12)
ax4.set_ylabel('Edge Count', fontsize=12)
ax4.set_title(f'Edge Type Distribution over Rounds (p={sample_p})', fontsize=14, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

# 5. ax[1,1] 惩罚率时序，仅 p=0.5
ax5 = axes[1, 1]
ax5.plot(round_axis, s["punish"], color='#2ca02c', lw=2.5, marker='o', markersize=3, label=f'p={sample_p}')
ax5.set_xlabel('Round', fontsize=12)
ax5.set_ylabel('Punishment Rate', fontsize=12)
ax5.set_title(f'Punishment Rate per Round (p={sample_p})', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.set_ylim(0,1)
ax5.legend(fontsize=10)

# 6. ax[1,2] 最终网络度分布直方图，仅 p=0.5
ax6 = axes[1, 2]
G_final = s["final_graph"]
degree_seq = [d for n, d in G_final.degree()]
ax6.hist(degree_seq, bins=range(0, max(degree_seq)+2), color="darkgreen", alpha=0.7, edgecolor='black')
ax6.set_xlabel('Node Degree', fontsize=12)
ax6.set_ylabel('Count', fontsize=12)
ax6.set_title(f'Final Network Degree Distribution (p={sample_p})', fontsize=14, fontweight='bold')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fermi_邻居.png', dpi=150, bbox_inches='tight')
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
