import numpy as np
"""
Implementation thanks to:
https://github.com/kfoofw/bandit_simulations/blob/master/python/contextual_bandits/analysis/linUCB%20disjoint%20implementation%20and%20analysis.md

"""

# Create class object for a single linear ucb disjoint arm
class linucb_disjoint_arm():

    def __init__(self, arm_index, d, alpha):
        # Track arm index
        self.arm_index = arm_index

        # Keep track of alpha
        self.alpha = alpha

        # A: (d x d) matrix = D_a.T * D_a + I_d.
        # The inverse of A is used in ridge regression
        self.A = np.identity(d)

        # b: (d x 1) corresponding response vector.
        # Equals to D_a.T * c_a in ridge regression formulation
        self.b = np.zeros([d, 1])

    def calc_UCB(self, x_array):
        # Find A inverse for ridge regression
        A_inv = np.linalg.inv(self.A)

        # Perform ridge regression to obtain estimate of covariate coefficients theta
        # theta is (d x 1) dimension vector
        self.theta = np.dot(A_inv, self.b)


        # Find ucb based on p formulation (mean + std_dev)
        # p is (1 x 1) dimension vector
        p = np.dot(self.theta.T, x_array) + self.alpha * np.sqrt(np.dot(x_array.T, np.dot(A_inv, x_array)))

        return p

    def reward_update(self, reward, x):

        # Update A which is (d * d) matrix.
        self.A += np.dot(x, x.T)

        # Update b which is (d x 1) vector
        # reward is scalar
        self.b += reward * x

class linucb_policy():

    def __init__(self, K_arms, d, alpha):
        self.K_arms = K_arms
        self.linucb_arms = [linucb_disjoint_arm(arm_index=1, d=d, alpha=alpha) for i in range(K_arms)]

    def select_arm(self, x_array):
        # Initiate ucb to be 0
        highest_ucb = -1

        # Track index of arms to be selected on if they have the max UCB.
        candidate_arms = []

        for arm_index in range(self.K_arms):
            # Calculate ucb based on each arm using current covariates at time t
            arm_ucb = self.linucb_arms[arm_index].calc_UCB(x_array)

            # If current arm is higher than current highest_ucb
            if arm_ucb > highest_ucb:
                # Set new max ucb
                highest_ucb = arm_ucb

                # Reset candidate_arms list with new entry based on current arm
                candidate_arms = [arm_index]

            # If there is a tie, append to candidate_arms
            if arm_ucb == highest_ucb:
                candidate_arms.append(arm_index)

        # Choose based on candidate_arms randomly (tie breaker)
        chosen_arm = np.random.choice(candidate_arms)

        return chosen_arm

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
        print("printing A")
        print(self.A.shape)
        print(self.A)
        self.b = np.array(
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
            print(context.shape)
            print(theta.T.shape)
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