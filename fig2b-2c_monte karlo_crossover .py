import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import collections
from dataclasses import dataclass
from typing import List, Dict, Tuple
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings('ignore')

# ========== 【路线2：固定p网格蒙特卡洛配置】 ==========
PUNISH_GRID = np.linspace(0, 1, 11)   # 惩罚网格 [0,0.1,0.2 ... 1.0]
N_REPEAT_PER_P = 20                   # 每个p下面独立重复次数
STRATEGY_RANDOM = 'random'
STRATEGY_NN_HIGH = 'nn_high'

# ========== Q-learning固定参数 ==========
QL_EPSILON = 0.05
QL_ALPHA = 0.2
QL_GAMMA = 0.5
QL_DECAY = 0.9

@dataclass
class MCParams:
    """单次运行的参数组合"""
    punish_prob: float
    n_all_c_ratio: float
    seed: int


def run_single_simulation(params: MCParams, rewire_func) -> Dict:
    """执行单次仿真，指定重连函数"""
    random.seed(params.seed)
    np.random.seed(params.seed)
    # 初始化网络
    N = 1000
    BA_M = 2
    G = nx.barabasi_albert_graph(N, BA_M)
    all_nodes = list(G.nodes)
    # 初始化策略
    n_all_c = int(N * params.n_all_c_ratio)
    n_all_d = N - n_all_c
    initial_strategies_list = ['ALL_C'] * n_all_c + ['ALL_D'] * n_all_d
    random.shuffle(initial_strategies_list)
    node_strategies = {u: {'strategy': initial_strategies_list[i]} for i, u in enumerate(all_nodes)}
    reputation = {u: 0 for u in G.nodes}

    # agent初始化
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

    agents = {u: TwoLayerQLAgent(u) for u in all_nodes}
    PAYOFF = {
        ('C', 'C'): (8, 8),
        ('C', 'D'): (0, 10),
        ('D', 'C'): (10, 0),
        ('D', 'D'): (5, 5)
    }
    n_rounds = 400
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

        base_payoff = round_payoff.copy()
        # 声誉更新
        for u in all_nodes:
            if actions[u] == 'C':
                reputation[u] += 1
        # 惩罚
        for u in all_nodes:
            if actions[u] == 'D' and random.random() < params.punish_prob:
                round_payoff[u] = 0
        # 重连
        def rewire_random(G, u, reputation, all_nodes):
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
                return 1
            else:
                G.add_edge(u, target)
                return 0

        def rewire_nn_high(G, u, reputation, all_nodes):
            neighbors = list(G.neighbors(u))
            if len(neighbors) == 0:
                return 0
            min_rep = min(reputation[v] for v in neighbors)
            low_rep_neighbors = [v for v in neighbors if reputation[v] == min_rep]
            target = random.choice(low_rep_neighbors)
            G.remove_edge(u, target)
            neighbors_of_neighbors = set()
            for v in neighbors:
                for w in G.neighbors(v):
                    if w != u and w not in neighbors:
                        neighbors_of_neighbors.add(w)
            candidates = list(neighbors_of_neighbors)
            if len(candidates) > 0:
                max_rep = max(reputation[v] for v in candidates)
                high_rep_candidates = [v for v in candidates if reputation[v] == max_rep]
                new_neighbor = random.choice(high_rep_candidates)
                G.add_edge(u, new_neighbor)
                return 1
            else:
                G.add_edge(u, target)
                return 0

        rewire_count = 0
        for u in all_nodes:
            if temp_rewire[u] == 1:
                if rewire_func == "random":
                    rewire_count += rewire_random(G, u, reputation, all_nodes)
                elif rewire_func == "nn_high":
                    rewire_count += rewire_nn_high(G, u, reputation, all_nodes)

        coop_rate = sum(1 for a in actions.values() if a == 'C') / N
        coop_rate_rec.append(coop_rate)

        if rnd > 0:
            for u in all_nodes:
                global_coop_rate = coop_rate
                state = agents[u].get_state(G, prev_round_payoff, global_coop_rate)
                rewire_reward = base_payoff[u]
                next_state = agents[u].get_state(G, round_payoff, global_coop_rate)
                agents[u].update_rewire(state, temp_rewire[u], rewire_reward, next_state)
                strategy_reward = round_payoff[u]
                agents[u].update_strategy(state, temp_actions[u], strategy_reward, next_state)
                agents[u].decay_epsilon()
        prev_round_payoff = round_payoff.copy()

    # ✅取最后的平均合作率作为稳态值
    steady_coop = np.mean(coop_rate_rec[-100:])
    return {"steady_coop": steady_coop, "params": params}


# ======================主循环：固定p网格，每个p做N_REPEAT_PER_P次配对重复======================
all_records = []
seed_base = 2000
for p in PUNISH_GRID:
    print(f"\n===== 开始惩罚p={p:.2f}, 每个p重复{N_REPEAT_PER_P}次 =====")
    for rep_id in range(N_REPEAT_PER_P):
        seed = seed_base + int(p*1000) + rep_id
        params = MCParams(punish_prob=p, n_all_c_ratio=0.5, seed=seed)
        # 同一seed，分别跑两种重连，配对
        res_rand = run_single_simulation(params, rewire_func="random")
        res_nn = run_single_simulation(params, rewire_func="nn_high")
        delta = res_rand["steady_coop"] - res_nn["steady_coop"]
        all_records.append({
            "p": p,
            "rep_id": rep_id,
            "rho_random": res_rand["steady_coop"],
            "rho_2hop": res_nn["steady_coop"],
            "delta_rho": delta
        })

df = pd.DataFrame(all_records)
df.to_csv("grid_mc_result.csv", index=False)
print("仿真完成，结果保存 grid_mc_result.csv")

# =====================绘图1：两种重连策略随惩罚p的均值±std====================
p_list = []
rand_mean = []
rand_std = []
nn_mean = []
nn_std = []
delta_mean_list = []
ci_low_list = []
ci_high_list = []

for p in PUNISH_GRID:
    sub = df[df["p"] == p]
    p_list.append(p)
    rand_mean.append(np.mean(sub["rho_random"]))
    rand_std.append(np.std(sub["rho_random"], ddof=1))
    nn_mean.append(np.mean(sub["rho_2hop"]))
    nn_std.append(np.std(sub["rho_2hop"], ddof=1))
    # 该p点内配对差值做t‑95%CI
    delta_arr = np.array(sub["delta_rho"])
    m_delta = np.mean(delta_arr)
    sd_delta = np.std(delta_arr, ddof=1)
    n = len(delta_arr)
    se = sd_delta / np.sqrt(n)
    tval = stats.t.ppf(0.975, df=n-1)
    ci_l = m_delta - tval * se
    ci_h = m_delta + tval * se
    delta_mean_list.append(m_delta)
    ci_low_list.append(ci_l)
    ci_high_list.append(ci_h)

p_arr = np.array(p_list)
rand_mean_arr = np.array(rand_mean)
rand_std_arr = np.array(rand_std)
nn_mean_arr = np.array(nn_mean)
nn_std_arr = np.array(nn_std)
delta_mean_arr = np.array(delta_mean_list)
ci_low_arr = np.array(ci_low_list)
ci_high_arr = np.array(ci_high_list)

# 图1：两种重连合作率曲线
plt.rcParams['font.family'] = 'Times New Roman'
fig1, ax1 = plt.subplots(figsize=(6, 4.5), dpi=400)
ax1.plot(p_arr, rand_mean_arr, "bo-", lw=2, label="random select")
ax1.fill_between(p_arr, np.clip(rand_mean_arr-rand_std_arr,0,1), np.clip(rand_mean_arr+rand_std_arr,0,1), alpha=0.25, color="blue")
ax1.plot(p_arr, nn_mean_arr, "rs-", lw=2, label="neighbors of neighbors")
ax1.fill_between(p_arr, np.clip(nn_mean_arr-nn_std_arr,0,1), np.clip(nn_mean_arr+nn_std_arr,0,1), alpha=0.25, color="red")
ax1.set_xlabel("Punishment Probability (p)")
ax1.set_ylabel("Final Cooperation Rate")
ax1.set_ylim(0, 1.02)
ax1.grid(alpha=0.3)
ax1.legend()
plt.tight_layout()
plt.savefig("蒙特卡洛p新的.png", dpi=400, bbox_inches="tight")
plt.show()

# 图2 Δρ_C(p)，每个固定p点，使用该p内部多次配对样本计算的t‑CI
fig2, ax2 = plt.subplots(figsize=(6,4.5), dpi=400)
ax2.fill_between(p_arr, ci_low_arr, ci_high_arr, color="#606060", alpha=0.28, label=r"95% Confidence Interval")
ax2.plot(p_arr, delta_mean_arr, color="#1a1a1a", lw=1.5, marker='o', markersize=4, label=r"$\Delta \rho_C(p)$")
ax2.axhline(y=0, color="#c82423", linestyle="--", lw=1.2, label="Equal performance")
ax2.set_xlabel(r"Punishment probability $p$")
ax2.set_ylabel(r"$\Delta \rho_C(p) = \bar{\rho}_C^\mathrm{random} - \bar{\rho}_C^\mathrm{2‑hop}$")
ax2.grid(alpha=0.3, lw=0.45)
ax2.legend(fontsize=9)
plt.tight_layout()
plt.savefig("delta_ρ_新的.png", dpi=400, bbox_inches="tight")
plt.show()


# =================打印每个p的统计表格，用于补充材料================
print("\n" + "="*80)
print(f"{'p':<8}{'delta_mean':<14}{'CI_low':<14}{'CI_high':<14}")
for pi, dm, cl, ch in zip(p_arr, delta_mean_arr, ci_low_arr, ci_high_arr):
    print(f"{pi:<8.2f}{dm:<14.4f}{cl:<14.4f}{ch:<14.4f}")
