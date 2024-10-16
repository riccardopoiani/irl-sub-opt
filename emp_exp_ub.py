import mdptoolbox.example
import numpy as np
from math import ceil
import scipy.sparse as _sp
import matplotlib.pyplot as plt

import mdp


def gaussian_ci(n: int, std: float):
    return 1.96 * std / np.sqrt(n)


def compute_discounted_visitation_matrix(T, pol, num_s, num_a, discount):
    Ppolicy = np.empty((num_s, num_s))
    for aa in range(num_a):  # avoid looping over S
        # the rows that use action a.
        ind = (pol == aa).nonzero()[0]
        # if no rows use action a, then no need to assign this
        if ind.size > 0:
            try:
                Ppolicy[ind, :] = T[aa][ind, :]
            except ValueError:
                Ppolicy[ind, :] = T[aa][ind, :].todense()
    return np.linalg.solve((_sp.eye(num_s, num_s) - discount * Ppolicy), np.eye(num_s))


g = 0.9
n_states = 10
n_actions = 2
n_runs = 20
eps = 0.1
delta = 0.1
n_samples = (1 / (1 - g) ** 2) * (1 / eps ** 2) * np.log(1 / delta) * n_states
n_samples = ceil(n_samples)

P, R = mdptoolbox.example.forest(S=10)
R /= R.max()
pi = mdp.PolicyIteration(P, R, g)

# Build
policy_list, value_list = pi.run()
policy_list = policy_list[:-1]
value_list = value_list[:-1]

expert_policy = policy_list[-1]

# Compute xi bounds
xi_list = []
for v in value_list:
    xi_list.append(np.abs(v - value_list[-1]).max())
n_experts = len(xi_list) - 1

results = []
for _ in range(n_runs):
    # Compute empirical model
    hat_P = np.zeros((n_actions, n_states, n_states))
    for state in range(n_states):
        for action in range(n_actions):
            next_states = np.random.choice(hat_P.shape[2], replace=True, size=n_samples, p=P[action, state, :])
            unique, counts = np.unique(next_states, return_counts=True)
            for i in range(len(unique)):
                hat_P[action, state, unique[i]] += counts[i] / n_samples

    # Compute for each of the sub-optimal agents the discounted occupancy matrix
    d_pi_list = []
    for pol in policy_list[:-1]:
        d_pi_list.append(compute_discounted_visitation_matrix(hat_P, pol, n_states, n_actions, g))

    # Compute empirical max-bound
    max_bounds = [[] for _ in range(n_states)]
    for s in range(n_states):
        for a in range(n_actions):
            temp = []
            for exp in range(n_experts):
                if policy_list[exp][s] == a:
                    for s_new in range(n_states):
                        if d_pi_list[exp][s_new, s] > 0:
                            temp.append(xi_list[exp] / d_pi_list[exp][s_new, s])
            if len(temp) == 0:
                max_bounds[s].append(1 / (1 - g))
            else:
                max_bounds[s].append(min(temp))

    max_bounds = np.array(max_bounds)
    max_bounds = max_bounds.flatten()
    results.append(max_bounds)

results = np.array(results)
mean_ub = results.mean(axis=0)
std_ub = results.std(axis=0)
ci_ub = gaussian_ci(n_runs, std_ub)

ax = plt.figure().gca()
ax.set_xticks([i for i in range(n_states * n_actions)])
plt.ylabel(r"Upper bound on $\zeta(s,a)$")
plt.xlabel("(s,a) index")
plt.plot(mean_ub, marker='o', linestyle='None')
plt.savefig("figures/emp_ub.png")
