import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

class Environment:
    def __init__(self, probs):
        self.probs = probs

    def step(self, action):
        return 1 if (np.random.random() < self.probs[action]) else 0

class EpsilonGreedyAgent:
    def __init__(self, num_actions, eps):
        self.num_actions = num_actions
        self.eps = eps
        self.n_A = np.zeros((num_actions), dtype=int)
        self.Q_A = np.zeros((num_actions), dtype=float)

    def update_knowledge(self, action, reward):
        self.n_A[action] += 1
        self.Q_A[action] += (1.0/self.n_A[action]) * (reward - self.Q_A[action])

    def act(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_actions)

        else:
            return np.random.choice(np.flatnonzero(self.Q_A == self.Q_A.max()))

def run_experiment(probs, num_episodes, eps):
    env = Environment(probs)
    agent = EpsilonGreedyAgent(len(probs), eps)
    actions, rewards = [], []
    for ep in range(num_episodes):
        action = agent.act()
        rew = env.step(action)
        agent.update_knowledge(action, rew)
        actions.append(action)
        rewards.append(rew)
        """if ep % 500 == 499:
            print("Finished {} episodes".format(ep))"""
    #print("Done running one experiment")
    #print(actions[0])
    return np.array(actions), np.array(rewards)


PROBS = [0.10, 0.50, 0.60, 0.80, 0.10,
         0.25, 0.60, 0.45, 0.75, 0.65]
EPS = 0.2
EPISODES = 1000
EXPERIMENTS = 10000
save_fig = True
out_dir = os.path.join(os.getcwd(), "figures/MAB")

print("Running eps-greedy agent with num_actions = {} and eps = {}".format(len(PROBS), EPS))
R = np.zeros((EPISODES, ))
A = np.zeros((EPISODES, len(PROBS)))

for i in tqdm(range(EXPERIMENTS), desc="Running agent", ascii=False, ncols=75):
    actions, rewards = run_experiment(PROBS, EPISODES, EPS)  # perform experiment
    #print(rewards)
    if (i + 1) % (EXPERIMENTS / 100) == 0:
        print("[Experiment {}/{}] ".format(i + 1, EXPERIMENTS) +
              "n_steps = {}, ".format(EPISODES) +
              "reward_avg = {}".format(np.sum(rewards) / len(rewards)))
    R += rewards
    for j, a in enumerate(actions):
        A[j][a] += 1

# Plot reward results
R_avg = R / float(EXPERIMENTS)
plt.plot(R_avg, ".")
plt.xlabel("Step")
plt.ylabel("Average Reward")
plt.grid()
ax = plt.gca()
plt.xlim([1, EPISODES])
if save_fig:
    if not os.path.exists(out_dir): os.mkdir(out_dir)
    plt.savefig(os.path.join(out_dir, "rewards.png"), bbox_inches="tight")
else:
    plt.show()
plt.close()

# Plot action results
for i in range(len(PROBS)):
    A_pct = 100 * A[:,i] / EXPERIMENTS
    steps = list(np.array(range(len(A_pct)))+1)
    plt.plot(steps, A_pct, "-",
             linewidth=4,
             label="Arm {} ({:.0f}%)".format(i+1, 100*PROBS[i]))
plt.xlabel("Step")
plt.ylabel("Count Percentage (%)")
leg = plt.legend(loc='upper left', shadow=True)
plt.xlim([1, EPISODES])
plt.ylim([0, 100])
for legobj in leg.legendHandles:
    legobj.set_linewidth(4.0)
if save_fig:
    if not os.path.exists(out_dir): os.mkdir(out_dir)
    plt.savefig(os.path.join(out_dir, "actions.png"), bbox_inches="tight")
else:
    plt.show()
plt.close()

