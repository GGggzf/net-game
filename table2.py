import networkx as nx
import matplotlib.pyplot as plt
import random, numpy as np, collections
from scipy.stats import bootstrap

# ========== 全局参数 ==========
n_rounds = 400
N = 1000
BA_M = 2
PAYOFF = {
    ('C', 'C'): (8, 8),
    ('C', 'D'): (0, 10),
    ('D', 'C'): (10, 0),
    ('D', 'D'): (5, 5)
}
# ========== Q‑learning 超参 ==========
QL_EPSILON = 0.05
QL_ALPHA = 0.2
QL_GAMMA = 0.5
QL_DECAY = 0.9

#==== 蒙特卡洛设置====
n_mc = 20

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


def run_simulation(PUNISH_PROB, rewire_mode="random", drop_prob=0.0):
    G = nx.barabasi_albert_graph(N, BA_M)
    all_nodes = list(G.nodes)
    N_ALL_C = 500
    N_ALL_D = 500
    initial_strategies_list = ['ALL_C'] * N_ALL_C + ['ALL_D'] * N_ALL_D
    random.shuffle(initial_strategies_list)
    node_strategies = {u: {'strategy': initial_strategies_list[i]} for i, u in enumerate(all_nodes)}
    reputation = {u: 0 for u in G.nodes}
    agents = {u: TwoLayerQLAgent(u) for u in all_nodes}

    coop_rate_rec = []
    rewire_rate_rec = []
    rewire_success_rate_rec = []

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
                state = agents[u].get_state(G, prev_round_payoff, prev_coop_rate)
                temp_rewire[u] = agents[u].choose_rewire_action(state)
                temp_actions[u] = agents[u].choose_strategy_action(state)

        for u in all_nodes:
            prev_strategy_action[u] = temp_actions[u]

        actions = {u: 'C' if temp_actions[u] == 0 else 'D' for u in all_nodes}
        round_payoff = {n: 0 for n in all_nodes}
        for u, v in G.edges:
            a_u, a_v = actions[u], actions[v]
            p_u, p_v = PAYOFF[(a_u, a_v)]
            round_payoff[u] += p_u
            round_payoff[v] += p_v
        base_payoff = round_payoff.copy()

        for u in all_nodes:
            if actions[u] == 'C':
                reputation[u] += 1

        for u in all_nodes:
            if actions[u] == 'D' and random.random() < PUNISH_PROB:
                round_payoff[u] = 0

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

                    if rewire_mode == "random":
                        candidates = [v for v in all_nodes if v not in existing_neighbors]
                    elif rewire_mode == "2hop":
                        two_hop = set()
                        for nb in G.neighbors(u):
                            two_hop.update(G.neighbors(nb))
                        two_hop.discard(u)
                        two_hop -= existing_neighbors
                        candidates = list(two_hop)
                    else:
                        raise ValueError("rewire_mode must be random or 2hop")

                    if len(candidates) > 0:
                        if rewire_mode == "2hop":
                            candidates = sorted(candidates, key=lambda x:-reputation[x])
                            new_neighbor = candidates[0]
                        else:
                            new_neighbor = random.choice(candidates)

                        if random.random() < drop_prob:
                            G.add_edge(u, target)
                        else:
                            G.add_edge(u, new_neighbor)
                            rewire_success_count +=1
                    else:
                        G.add_edge(u, target)

        if rewire_launch_count > 0:
            success_rate = rewire_success_count / rewire_launch_count
        else:
            success_rate = 0.0

        coop_rate = sum(1 for a in actions.values() if a == 'C') / N
        coop_rate_rec.append(coop_rate)
        rewire_rate_rec.append(rewire_launch_count / N)
        rewire_success_rate_rec.append(success_rate)

        if rnd > 0:
            for u in all_nodes:
                global_coop_rate = coop_rate
                state = agents[u].get_state(G, prev_round_payoff, global_coop_rate)
                rewire_reward = base_payoff[u]
                strategy_reward = round_payoff[u]
                next_state = agents[u].get_state(G, round_payoff, global_coop_rate)
                agents[u].update_rewire(state, temp_rewire[u], rewire_reward, next_state)
                agents[u].update_strategy(state, temp_actions[u], strategy_reward, next_state)
                agents[u].decay_epsilon()
        prev_round_payoff = round_payoff.copy()

    start_idx = 200
    coop_ss = np.mean(np.array(coop_rate_rec[start_idx:]))
    rew_launch_ss = np.mean(np.array(rewire_rate_rec[start_idx:]))
    rew_suc_ss = np.mean(np.array(rewire_success_rate_rec[start_idx:]))
    eff_ss = rew_launch_ss * rew_suc_ss

    return {
        "coop_steady": float(coop_ss),
        "mean_effective_edge": float(eff_ss),
    }

def bootstrap_mean_ci(arr, n_resample=10000):
    """计算bootstrap 95%置信区间，返回(mean, ci_low, ci_high)"""
    arr = np.asarray(arr)
    res = bootstrap((arr,), np.mean, n_resamples=n_resample, confidence_level=0.95)
    mu = np.mean(arr)
    return mu, res.confidence_interval.low, res.confidence_interval.high


if __name__ == "__main__":
    p_list = [0.15,0.40,0.60]
    #存储每个工况多次mc结果列表
    mc_raw = dict()
    mc_ctrl = dict()
    for p in p_list:
        for mode in ["random","2hop"]:
            mc_raw[(p,mode)] = []
            mc_ctrl[(p,mode)] = []

    for mc_idx in range(n_mc):
        print(f"\n======== Monte‑Carlo run {mc_idx+1}/{n_mc} ========")
        for p in p_list:
            #原始
            res_rand_raw = run_simulation(p, "random", drop_prob=0.0)
            res_2hop_raw = run_simulation(p, "2hop", drop_prob=0.0)
            eff_rand_raw = res_rand_raw["mean_effective_edge"]
            eff_2hop_raw = res_2hop_raw["mean_effective_edge"]

            if eff_2hop_raw > eff_rand_raw:
                drop_prob = 1.0 - (eff_rand_raw / eff_2hop_raw)
            else:
                drop_prob = 0.0
            #受控
            res_rand_ctrl = run_simulation(p,"random",drop_prob=0.0)
            res_2hop_ctrl = run_simulation(p,"2hop",drop_prob=drop_prob)

            mc_raw[(p,"random")].append(res_rand_raw)
            mc_raw[(p,"2hop")].append(res_2hop_raw)
            mc_ctrl[(p,"random")].append(res_rand_ctrl)
            mc_ctrl[(p,"2hop")].append(res_2hop_ctrl)

    #==== 聚合统计，计算bootstrap均值与95%CI ====
    print("\n\n########## A1蒙特卡洛汇总表（{}次MC，稳态200‑400轮）##########".format(n_mc))
    header = "| p | Raw‑random coop | Raw‑2hop coop | Ctrl‑random coop | Ctrl‑2hop coop | Raw‑eff(random) | Raw‑eff(2hop) | Ctrl‑eff(random) | Ctrl‑eff(2hop) |"
    sep = "|---|---|---|---|---|---|---|---|---|"
    print(header)
    print(sep)
    for p in p_list:
        def get_arr(key,field):
            lst = key[0]
            return [d[field] for d in lst]
        rr_list = mc_raw[(p,"random")]
        r2_list = mc_raw[(p,"2hop")]
        cr_list = mc_ctrl[(p,"random")]
        c2_list = mc_ctrl[(p,"2hop")]

        rr_coop, rr_coop_low, rr_coop_high = bootstrap_mean_ci([x["coop_steady"] for x in rr_list])
        r2_coop, r2_coop_low, r2_coop_high = bootstrap_mean_ci([x["coop_steady"] for x in r2_list])
        cr_coop, cr_coop_low, cr_coop_high = bootstrap_mean_ci([x["coop_steady"] for x in cr_list])
        c2_coop, c2_coop_low, c2_coop_high = bootstrap_mean_ci([x["coop_steady"] for x in c2_list])

        rr_eff, rr_eff_low, rr_eff_high = bootstrap_mean_ci([x["mean_effective_edge"] for x in rr_list])
        r2_eff, r2_eff_low, r2_eff_high = bootstrap_mean_ci([x["mean_effective_edge"] for x in r2_list])
        cr_eff, cr_eff_low, cr_eff_high = bootstrap_mean_ci([x["mean_effective_edge"] for x in cr_list])
        c2_eff, c2_eff_low, c2_eff_high = bootstrap_mean_ci([x["mean_effective_edge"] for x in c2_list])

        def fmt(mu,low,high):
            return f"{mu:.4f} [{low:.4f}, {high:.4f}]"

        print(f"| {p:.2f} | {fmt(rr_coop,rr_coop_low,rr_coop_high)} | {fmt(r2_coop,r2_coop_low,r2_coop_high)} | {fmt(cr_coop,cr_coop_low,cr_coop_high)} | {fmt(c2_coop,c2_coop_low,c2_coop_high)} | {fmt(rr_eff,rr_eff_low,rr_eff_high)} | {fmt(r2_eff,r2_eff_low,r2_eff_high)} | {fmt(cr_eff,cr_eff_low,cr_eff_high)} | {fmt(c2_eff,c2_eff_low,c2_eff_high)} |")

    print("\n说明：单元格格式为 mean [95% bootstrap CI low, high]；放在补充材料Table S")
