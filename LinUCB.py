import numpy as np

class LinUCB:
    """
    See https://hackernoon.com/contextual-multi-armed-bandit-problems-in-reinforcement-learnings
    """
    def __init__(self, n_actions, n_features, alpha=1.0):
        self.n_actions = n_actions
        self.n_features = n_features
        self.alpha = alpha
        self.A = np.array(
            [np.identity(n_features) for _ in range(n_actions)]
        ) #action covariance matrix
        self.b = np.array(
            [np.zeros(n_features) for _ in range(n_actions)]
        )
        self.theta = np.array(
            [np.zeros(n_features) for _ in range(n_actions)]
        )

    def predict(self, context):
        context = np.array(context) #convert list to ndarray
        context = context.reshape(-1, 1) #reshape context to a single-column matrix
        p = np.zeros(self.n_actions)
        for a in range(0, self.n_actions):
            theta = np.dot(
                np.linalg.inv(self.A[a]), self.b[a]
            ) #theta_a = A_a^-1 * b_a
            theta = theta.reshape(-1, 1)
            p[a] = np.dot(theta.T, context) + self.alpha * np.sqrt(
                np.dot(context.T, np.dot(np.linalg.inv(self.A[a]), context))
            )

        return p
    def update(self, action, context, reward):
        """
        Occurs after an action is selected
        :param action:
        :param context:
        :param reward:
        :return:
        """
        self.A[action] += np.outer(context, context) #A_a = A_a + x_t * x_t^T
        self.b[action] += reward * context #b_a = b_a + r_t * x_tx