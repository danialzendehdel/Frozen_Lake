from SFL_Environment import CustomSFLEnv
from policy_evaluation import policy_evaluation



if __name__ == "__main__":
    env = CustomSFLEnv()
    P = env.env_dynamic()

    terminal_states = env.holes + [env.goal]
    pi = [2, 0, 1, 3, 0, 1, 2, 3, 3, 1, 3, 3, 0, 2, 2, 3]
    V = policy_evaluation(pi, P, gamma=0.9, terminal_states=terminal_states)
    print(V.reshape(env.grid_size, env.grid_size))