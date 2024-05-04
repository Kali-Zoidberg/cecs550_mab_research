import numpy as np
class Bandit:
    """
    See https://hackernoon.com/contextual-multi-armed-bandit-problems-in-reinforcement-learning
    """
    def __init__(self, n_actions, n_features):
        self.n_actions = n_actions
        self.n_features = n_features
        self.theta = np.random.rand(n_actions, n_features)

    def get_reward(self, action, x):
        return x @ self.theta[action] + np.random.normal()
    def get_optimal_reward(self, x):
        return np.max(x @ self.theta.T)