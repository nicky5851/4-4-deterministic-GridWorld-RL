import numpy as np
from dataclasses import dataclass

# ----------------------------
# GridWorld (4x4 deterministic)
# ----------------------------
# States: 0..15 (row-major)
# Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
ACTIONS = ["U", "R", "D", "L"]
ARROWS = {0: "↑", 1: "→", 2: "↓", 3: "←"}

@dataclass
class GridWorld4x4:
    n_rows: int = 4
    n_cols: int = 4
    start_state: int = 0
    goal_state: int = 15
    step_penalty: float = -1.0
    goal_reward: float = 10.0

    def reset(self):
        self.s = self.start_state
        return self.s

    def _to_rc(self, s):
        return divmod(s, self.n_cols)

    def _to_s(self, r, c):
        return r * self.n_cols + c

    def step(self, a):
        r, c = self._to_rc(self.s)

        if a == 0:   # UP
            r2, c2 = max(r - 1, 0), c
        elif a == 1: # RIGHT
            r2, c2 = r, min(c + 1, self.n_cols - 1)
        elif a == 2: # DOWN
            r2, c2 = min(r + 1, self.n_rows - 1), c
        elif a == 3: # LEFT
            r2, c2 = r, max(c - 1, 0)
        else:
            raise ValueError("Invalid action")

        s2 = self._to_s(r2, c2)

        # Reward logic
        done = (s2 == self.goal_state)
        reward = self.goal_reward if done else self.step_penalty

        self.s = s2
        return s2, reward, done, {}

# ----------------------------
# Utilities
# ----------------------------
def epsilon_greedy(Q, s, eps, rng):
    if rng.random() < eps:
        return rng.integers(0, Q.shape[1])
    return int(np.argmax(Q[s]))

def print_policy(Q, env: GridWorld4x4):
    grid = []
    for s in range(env.n_rows * env.n_cols):
        if s == env.goal_state:
            grid.append(" G ")
        elif s == env.start_state:
            a = int(np.argmax(Q[s]))
            grid.append(f"S{ARROWS[a]} ")
        else:
            a = int(np.argmax(Q[s]))
            grid.append(f" {ARROWS[a]} ")
    # pretty print 4x4
    for r in range(env.n_rows):
        row = grid[r*env.n_cols:(r+1)*env.n_cols]
        print("".join(row))

def evaluate_greedy(Q, env, max_steps=100):
    s = env.reset()
    total = 0.0
    for _ in range(max_steps):
        a = int(np.argmax(Q[s]))
        s, r, done, _ = env.step(a)
        total += r
        if done:
            break
    return total, done

# ----------------------------
# Step 0: Random agent demo
# ----------------------------
env = GridWorld4x4()
rng = np.random.default_rng(0)

print("=== Step 0: Random agent run ===")
s = env.reset()
total = 0.0
for t in range(30):
    a = rng.integers(0, 4)
    s, r, done, _ = env.step(a)
    total += r
    if done:
        break
print(f"Random run: steps={t+1}, total_reward={total:.1f}, reached_goal={done}")

# ----------------------------
# Step 1: Q-learning
# ----------------------------
print("\n=== Step 1: Q-learning training ===")

n_states = env.n_rows * env.n_cols
n_actions = 4
Q = np.zeros((n_states, n_actions), dtype=float)

alpha = 0.2     # learning rate
gamma = 0.95    # discount factor
eps_start = 1.0
eps_end = 0.05
episodes = 2000
max_steps = 100

for ep in range(episodes):
    s = env.reset()
    # linear epsilon decay
    eps = eps_end + (eps_start - eps_end) * (1 - ep / episodes)

    for _ in range(max_steps):
        a = epsilon_greedy(Q, s, eps, rng)
        s2, r, done, _ = env.step(a)

        # Q-learning update:
        # Q[s,a] <- Q[s,a] + alpha * (r + gamma*max_a' Q[s',a'] - Q[s,a])
        td_target = r + (0.0 if done else gamma * np.max(Q[s2]))
        Q[s, a] += alpha * (td_target - Q[s, a])

        s = s2
        if done:
            break

# ----------------------------
# Step 2: Inspect learned policy
# ----------------------------
print("\n=== Step 2: Learned greedy policy (arrows) ===")
print_policy(Q, env)

score, reached = evaluate_greedy(Q, env)
print(f"\nGreedy evaluation: total_reward={score:.1f}, reached_goal={reached}")