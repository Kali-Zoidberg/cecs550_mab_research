import numpy as np
from Bandit import Bandit
from LinUCB import LinUCB
class Experiment:
    def __init__(self, epochs:int, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.epochs = epochs
        self.results_by_mab_name = {}
    def run_experiments(self):

        #Dictionray containing the name of the algorithm and the function it pertains to
        mab_algs = {
            'linUCB':self.generic_mab_alg_test
        }

        for mab_name, mab_func in mab_algs.items():
            n_actions = 10
            n_features = 5
            #define bandit and algorithm
            bandit = Bandit(n_actions, n_features)
            model_agent = LinUCB(n_actions, n_features, alpha=1.0)

            #define cntext (features)
            x = np.random.randn(n_features)

            #make predictions
            pred_rewards = np.array([model_agent.predict(x) for i in range(n_actions)])

            #Choose action with highest reward
            action = np.argmax(pred_rewards)

            #Give reward to bandit
            reward = bandit.get_reward(action, x)

            #update agent (algorithm) params
            model_agent.update(x, reward)

            self.results_by_mab_name[mab_name] = mab_func()

        return self.results_by_mab_name

    def generic_mab_alg_test(self):
        """
        Generic setup for running an experiment on a unique MAB alg
        :return:
        """
        results = []
        for i in range(0,self.epochs):
            result = []
            results.append(result)
        pass