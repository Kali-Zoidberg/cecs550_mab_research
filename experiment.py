import numpy as np
from Bandit import Bandit
from LinUCB import LinUCB
class Experiment:
    def __init__(self, timesteps:int, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.timesteps = timesteps
        self.results_by_mab_name = {}

    def run_experiments(self, x):
        n_actions = 10
        print(x.shape)
        n_features = x.shape[1] #n_features on a 2d array is # of cols for the first row.

        #Dictionary containing the name of the algorithm and the function it pertains to
        mab_algs = {
            'linUCB':LinUCB(n_actions, n_features, alpha=1.0)
        }

        for mab_name, mab_agent in mab_algs.items():
            #initialize results for agent
            self.results_by_mab_name[mab_name] = {
                'cumulative_reward_by_timestep' : [],
                'cumulative_regret_by_timestep' : []
            }

            for t in range(0, self.timesteps):
                #define bandit and algorithm
                bandit = Bandit(n_actions, n_features)

                #make predictions
                pred_rewards = np.array([mab_agent.predict(x) for i in range(n_actions)])

                #Choose action with highest reward
                action = np.argmax(pred_rewards)

                #Give reward to bandit
                reward = bandit.get_reward(action, x)
                #Update cumulative reward (accumulation as t -> T)
                if t == 0 :
                    cumulative_reward = reward
                else:
                    cumulative_reward = self.results_by_mab_name[mab_name]['cumulative_reward_by_timestep'][
                                            t - 1] + reward

                self.results_by_mab_name[mab_name]['cumulative_reward_by_timestep'].append(cumulative_reward)

                #update agent (algorithm) params
                mab_agent.update(action, x, reward)

                #compare with optimal reward to calculate regret
                optimal_reward = bandit.get_optimal_reward(x)

                #calculate regret and cumulativbe regret
                regret = optimal_reward - reward
                cumulative_regret = None
                if t == 0:
                    cumulative_regret = regret
                else:
                    cumulative_regret = self.results_by_mab_name[mab_name]['cumulative_regret_by_timestep'][t-1] + regret

                #Append cumulative regret to our results
                self.results_by_mab_name[mab_name]['cumulative_regret_by_timestep'].append(cumulative_regret)

        return self.results_by_mab_name

    def generic_mab_alg_test(self):
        """
        Generic setup for running an experiment on a unique MAB alg
        :return:
        """
        results = []
        for i in range(0,self.timesteps):
            result = []
            results.append(result)
        pass