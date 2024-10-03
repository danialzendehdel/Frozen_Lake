
import numpy as np
from SFL_Environment import CustomSFLEnv

def policy_evaluation(pi, P, gamma=1.0, theta=1e-10, terminal_states=None):
    prev_V = np.zeros(len(P))
    while True:
        V = np.zeros(len(P))
        for s in range(len(P)):
            if terminal_states is not None and s in terminal_states:
                V[s] = 0
                continue
            a = pi[s]  # Get the action from the policy for state s
            for prob, next_state, reward, done in P[s][a]:
                V[s] += prob * (reward + gamma * prev_V[next_state] * (not done))
        if np.max(np.abs(prev_V - V)) < theta:
            break
        prev_V = V.copy()
    return V


