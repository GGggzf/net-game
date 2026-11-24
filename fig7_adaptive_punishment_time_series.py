import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import collections
import warnings
warnings.filterwarnings('ignore')

# ========== 参数配置 ==========
N_RUNS = 1                 # 每种模式独立运行次数（建议后续改成20或30）
N_ROUNDS = 500             # 每轮运行的回合数
N = 1000                   # 节点数
BA_M = 2                   # BA网络参数
INIT_COOP_RATIO = 0.5      # 初始合作者比例

PAYOFF = {
    ('C', 'C'): (8, 8),
    ('C', 'D'): (0, 10),
    ('D', 'C'): (10, 0),
    ('D', 'D'): (5, 5)
}

# ========== Q-learning固定参数 ==========
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
            elif avg_neigh_payoff < 20:
                neigh_payoff_state = 1
            elif avg_neigh_payoff < 30:
                neigh_payoff_state = 2
            else:
                neigh_payoff_state = 3

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

# ========== 随机重连策略 ==========
def rewire_random(G, u, reputation, all_nodes):
    """
    随机断开最低声誉邻居，并随机连接到一个非邻居
    """
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

# ========== 连续型动态惩罚函数（线性递减） ==========
def get_rule_based_punish_prob(prev_coop_rate, p_min=0.0, p_max=0.8):
    """
    连续型惩罚概率：合作率越低，惩罚越高（线性关系）
    公式: punish_prob = p_max - (p_max - p_min) * prev_coop_rate
    结果裁剪到 [p_min, p_max]
    """
    punish_prob = p_max - (p_max - p_min) * prev_coop_rate
    return np.clip(punish_prob, p_min, p_max)

# ========== 单次仿真运行 ==========
def run_single_simulation(punish_mode, run_seed):
    """
    punish_mode 可以是:
        - ('fixed', p)
        - ('uniform', low, high)
        - ('beta', alpha, beta)
        - ('discrete', values, probs)
        - ('rule_based', p_min, p_max)   # 连续型动态惩罚
    """
    random.seed(run_seed)
    np.random.seed(run_seed)

    # 初始化网络
    G = nx.barabasi_albert_graph(N, BA_M)
    all_nodes = list(G.nodes)

    # 初始策略
    n_all_c = int(N * INIT_COOP_RATIO)
    n_all_d = N - n_all_c
    initial_strategies = ['ALL_C'] * n_all_c + ['ALL_D'] * n_all_d
    random.shuffle(initial_strategies)
    node_strategies = {u: {'strategy': initial_strategies[i]} for i, u in enumerate(all_nodes)}

    reputation = {u: 0 for u in G.nodes}
    agents = {u: TwoLayerQLAgent(u) for u in all_nodes}

    coop_rate_rec = []
    punish_prob_rec = []

    raw_payoff = {n: 0 for n in all_nodes}       # 原始收益（不受惩罚影响）
    round_payoff = {n: 0 for n in all_nodes}     # 惩罚后的收益（用于状态/记录）
    prev_raw_payoff = raw_payoff.copy()
    prev_round_payoff = round_payoff.copy()

    for rnd in range(N_ROUNDS):
        # ========== 计算本轮惩罚概率 ==========
        if punish_mode[0] == 'fixed':
            punish_prob = punish_mode[1]

        elif punish_mode[0] == 'uniform':
            low, high = punish_mode[1], punish_mode[2]
            punish_prob = random.uniform(low, high)

        elif punish_mode[0] == 'beta':
            alpha, beta = punish_mode[1], punish_mode[2]
            punish_prob = np.random.beta(alpha, beta)

        elif punish_mode[0] == 'discrete':
            values, probs = punish_mode[1], punish_mode[2]
            punish_prob = random.choices(values, weights=probs, k=1)[0]

        elif punish_mode[0] == 'rule_based':
            p_min, p_max = punish_mode[1], punish_mode[2]
            # 第0轮没有上一轮合作率，用初始合作率替代
            prev_coop_rate = coop_rate_rec[-1] if len(coop_rate_rec) > 0 else INIT_COOP_RATIO
            punish_prob = get_rule_based_punish_prob(prev_coop_rate, p_min, p_max)

        else:
            raise ValueError("Invalid punish_mode")

        punish_prob_rec.append(punish_prob)

        # ========== 决策 ==========
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

        # ========== 计算原始收益 ==========
        raw_payoff = {n: 0 for n in all_nodes}
        for u, v in G.edges:
            a_u, a_v = actions[u], actions[v]
            p_u, p_v = PAYOFF[(a_u, a_v)]
            raw_payoff[u] += p_u
            raw_payoff[v] += p_v

        # ========== 声誉更新 ==========
        for u in all_nodes:
            if actions[u] == 'C':
                reputation[u] += 1

        # ========== 惩罚：背叛者以 punish_prob 概率被惩罚，收益清零 ==========
        round_payoff = raw_payoff.copy()
        for u in all_nodes:
            if actions[u] == 'D' and random.random() < punish_prob:
                round_payoff[u] = 0

        # ========== 重连 ==========
        for u in all_nodes:
            if temp_rewire[u] == 1:
                rewire_random(G, u, reputation, all_nodes)

        # ========== 记录合作率 ==========
        coop_rate = sum(1 for a in actions.values() if a == 'C') / N
        coop_rate_rec.append(coop_rate)

        # ========== Q-learning 更新 ==========
        if rnd > 0:
            for u in all_nodes:
                global_coop_rate = coop_rate

                state = agents[u].get_state(G, prev_round_payoff, global_coop_rate)
                next_state = agents[u].get_state(G, round_payoff, global_coop_rate)

                # 重连层：原始收益
                agents[u].update_rewire(state, temp_rewire[u], raw_payoff[u], next_state)

                # 策略层：惩罚后收益
                agents[u].update_strategy(state, temp_actions[u], round_payoff[u], next_state)

                agents[u].decay_epsilon()

        prev_raw_payoff = raw_payoff.copy()
        prev_round_payoff = round_payoff.copy()

    return coop_rate_rec, punish_prob_rec

# ========== 多次运行，取平均和标准差 ==========
def run_multiple(punish_mode, n_runs=N_RUNS):
    all_series = []
    all_punish_series = []

    for i in range(n_runs):
        seed = 1000 + i
        coop_series, punish_series = run_single_simulation(punish_mode, seed)
        all_series.append(coop_series)
        all_punish_series.append(punish_series)

    all_series = np.array(all_series)             # runs × rounds
    all_punish_series = np.array(all_punish_series)

    mean_series = np.mean(all_series, axis=0)
    std_series = np.std(all_series, axis=0)

    mean_punish = np.mean(all_punish_series, axis=0)
    std_punish = np.std(all_punish_series, axis=0)

    return mean_series, std_series, mean_punish, std_punish

# ========== 主程序 ==========
print("开始运行仿真...")
print(f"每种模式独立运行次数: {N_RUNS}，每轮回合数: {N_ROUNDS}")
print(f"Q-learning参数: epsilon={QL_EPSILON}, alpha={QL_ALPHA}, gamma={QL_GAMMA}")

# ========== 定义惩罚模式 ==========
mode_fixed = ('fixed', 0.3)
mode_uniform = ('uniform', 0.0, 0.6)
mode_beta = ('beta', 2.0, 8.0)
mode_discrete = ('discrete', [0.1, 0.2, 0.3], [0.25, 0.5, 0.25])

# 连续型动态惩罚：合作率越低惩罚越高，p_min=0.1（高合作率时），p_max=0.5（低合作率时）
# 实际使用 p_min=0.1, p_max=0.35（如用户示例）
mode_rule_based = ('rule_based', 0.0, 0.8)

# ========== 运行 ==========
print("计算固定概率模式...")
mean_fixed, std_fixed, mean_p_fixed, std_p_fixed = run_multiple(mode_fixed)

print("计算均匀分布模式...")
mean_uniform, std_uniform, mean_p_uniform, std_p_uniform = run_multiple(mode_uniform)

print("计算Beta分布模式...")
mean_beta, std_beta, mean_p_beta, std_p_beta = run_multiple(mode_beta)

print("计算离散分布模式...")
mean_discrete, std_discrete, mean_p_discrete, std_p_discrete = run_multiple(mode_discrete)

print("计算连续动态惩罚模式...")
mean_rule, std_rule, mean_p_rule, std_p_rule = run_multiple(mode_rule_based)

# ========== 绘制合作率对比图 ==========
plt.figure(figsize=(10, 6))
rounds = np.arange(1, N_ROUNDS + 1)

plt.plot(rounds, mean_fixed, 'b-', lw=2, label='Fixed p')
plt.fill_between(rounds,
                 np.clip(mean_fixed - std_fixed, 0, 1),
                 np.clip(mean_fixed + std_fixed, 0, 1),
                 alpha=0.2, color='blue')

plt.plot(rounds, mean_uniform, 'r-', lw=2, label='Uniform p')
plt.fill_between(rounds,
                 np.clip(mean_uniform - std_uniform, 0, 1),
                 np.clip(mean_uniform + std_uniform, 0, 1),
                 alpha=0.2, color='red')

# 可选：取消注释以显示其他模式
# plt.plot(rounds, mean_beta, 'g-', lw=2, label='Beta(2,8)')
# plt.fill_between(rounds,
#                  np.clip(mean_beta - std_beta, 0, 1),
#                  np.clip(mean_beta + std_beta, 0, 1),
#                  alpha=0.2, color='green')
#
# plt.plot(rounds, mean_discrete, 'm-', lw=2, label='Discrete {0.1,0.2,0.3}')
# plt.fill_between(rounds,
#                  np.clip(mean_discrete - std_discrete, 0, 1),
#                  np.clip(mean_discrete + std_discrete, 0, 1),
#                  alpha=0.2, color='magenta')

plt.plot(rounds, mean_rule, 'k-', lw=2, label='Dynamic punishment mechanism')
plt.fill_between(rounds,
                 np.clip(mean_rule - std_rule, 0, 1),
                 np.clip(mean_rule + std_rule, 0, 1),
                 alpha=0.2, color='black')

plt.xlabel('Round', fontsize=12)
plt.ylabel('Cooperation Rate', fontsize=12)
#plt.title('Cooperation Rate Evolution under Different Punishment Modes', fontsize=13)
plt.grid(True, alpha=0.3)
plt.legend(loc='best')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("cooperation_five_modes.png", dpi=150)
plt.show()

# ========== 单独绘制连续动态惩罚的惩罚概率变化 ==========
plt.figure(figsize=(10, 5))
plt.plot(rounds, mean_p_rule, 'k-', lw=2, label='Continuous dynamic punishment probability')
plt.fill_between(rounds,
                 np.clip(mean_p_rule - std_p_rule, 0, 1),
                 np.clip(mean_p_rule + std_p_rule, 0, 1),
                 alpha=0.2, color='black')

plt.xlabel('Round', fontsize=12)
plt.ylabel('Punishment Probability', fontsize=12)
plt.title('Punishment Probability under Continuous Dynamic Mechanism (Linear Decay)', fontsize=13)
plt.grid(True, alpha=0.3)
plt.legend(loc='best')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("rule_based_punishment_prob.png", dpi=150)
plt.show()

print("\n运行完成！")
print("已保存图片：")
print("1. cooperation_five_modes.png")
print("2. rule_based_punishment_prob.png")