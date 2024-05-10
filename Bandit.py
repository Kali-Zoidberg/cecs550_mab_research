import numpy as np
class Bandit:
    """
    See https://hackernoon.com/contextual-multi-armed-bandit-problems-in-reinforcement-learning
    """
    def __init__(self, n_actions, n_features):
        self.n_actions = n_actions
        self.n_features = n_features
        self.theta = np.random.rand(n_actions, n_features)
        print(self.theta)

    def get_reward(self, action, x):
        return x @ self.theta[action] + np.random.normal()
    def get_optimal_reward(self, x):
        """
        TODO, this needs work in terms of selecting best 'flight' of wine
        :param x:
        :return:
        """
        return np.max(x @ self.theta.T) #@ is matrix multiplication