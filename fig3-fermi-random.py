import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import warnings

# 屏蔽exp溢出警告（我们已经clip处理，警告只是提示，不是错误）
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
# Fermi学习参数
FERMI_K = 0.1
REWIRE_PROB = 0.47  # 固定重连概率

def fermi_prob(p_u, p_v, k):
    """Fermi模仿概率，增加clip防止exp溢出"""
    z = (p_u - p_v) / k
    # 截断指数输入，防止溢出，np.exp(709)是float64安全上限
    z = np.clip(z, -700, 700)
    return 1.0 / (1.0 + np.exp(z))


def run_simulation(PUNISH_PROB):
    """运行单次模拟，Fermi学习 + 固定概率重连"""
    # 初始化BA网络
    G = nx.barabasi_albert_graph(N, BA_M)
    all_nodes = list(G.nodes)

    # 初始化策略：一半C一半D
    N_ALL_C = 500
    N_ALL_D = 500
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
        # 1. 使用当前节点策略进行博弈
        actions = {u: node_strategy[u] for u in all_nodes}

        # 2. 原始收益计算
        round_payoff = {n: 0 for n in all_nodes}
        for u, v in G.edges:
            a_u, a_v = actions[u], actions[v]
            p_u, p_v = PAYOFF[(a_u, a_v)]
            round_payoff[u] += p_u
            round_payoff[v] += p_v
        base_payoff = round_payoff.copy()

        # 3. 更新声誉：合作则声誉+1
        for u in all_nodes:
            if actions[u] == 'C':
                reputation[u] += 1

        # 4. 惩罚机制：背叛者以PUNISH_PROB概率收益置0
        punish_count = 0
        for u in all_nodes:
            if actions[u] == 'D' and random.random() < PUNISH_PROB:
                round_payoff[u] = 0
                punish_count += 1

        # 5. 执行重连：每个节点以REWIRE_PROB=0.5概率重连
        rewire_count = 0
        for u in all_nodes:
            if random.random() < REWIRE_PROB:
                rewire_count += 1
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
                    continue  # 无邻居，无法模仿
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

# ========== 绘图 ==========
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
colors = ['royalblue', 'darkorange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
round_axis = np.arange(1, n_rounds + 1)

# 1. 合作率对比（10条曲线）
for i, (p, data) in enumerate(results.items()):
    axes[0, 0].plot(round_axis, data["coop"], lw=2, color=colors[i],
                    marker='o', markersize=2, label=f'P={p}')
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

# 3. 重连率时序（取P=0.5作为示例曲线）
sample_p = 0.5
sample_data = results[sample_p]
axes[0, 2].plot(round_axis, sample_data["rewire"], lw=2, color='crimson')
axes[0, 2].set_xlabel('Round')
axes[0, 2].set_ylabel('Rewire Rate')
axes[0, 2].set_title(f'Rewire Rate per Round (P={sample_p})')
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].set_ylim(0, 1)

# 4. 边类型数量时序（CC/CD/DD，同样用P=0.5示例）
axes[1, 0].plot(round_axis, sample_data["cc"], label="CC", lw=2, color="green")
axes[1, 0].plot(round_axis, sample_data["cd"], label="CD", lw=2, color="orange")
axes[1, 0].plot(round_axis, sample_data["dd"], label="DD", lw=2, color="red")
axes[1, 0].set_xlabel('Round')
axes[1, 0].set_ylabel('Edge Count')
axes[1, 0].set_title(f'Edge Type Distribution over Rounds (P={sample_p})')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 5. 惩罚率时序（P=0.5示例）
axes[1, 1].plot(round_axis, sample_data["punish"], lw=2, color="darkblue")
axes[1, 1].set_xlabel('Round')
axes[1, 1].set_ylabel('Punishment Rate')
axes[1, 1].set_title(f'Punishment Rate per Round (P={sample_p})')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_ylim(0, 1)

# 6. 网络度分布直方图（最后一轮网络，P=0.5）
G_final = sample_data["final_graph"]
degree_seq = [d for n, d in G_final.degree()]
axes[1, 2].hist(degree_seq, bins=range(0, max(degree_seq) + 2), color="darkgreen", alpha=0.7, edgecolor='black')
axes[1, 2].set_xlabel('Node Degree')
axes[1, 2].set_ylabel('Count')
axes[1, 2].set_title(f'Final Network Degree Distribution (P={sample_p})')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fermi随机重连.png', dpi=150, bbox_inches='tight')
plt.show()
