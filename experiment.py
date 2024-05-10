import numpy as np
from Bandit import Bandit
from LinUCB import linucb_policy
from context_lin_bandits import real_environments
from context_lin_bandits import context_lin_bandit_algorithms as algorithms
class Experiment:
    def __init__(self, timesteps:int, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.timesteps = timesteps
        self.results_by_mab_name = {}

    def run_experiments(self, x):
        k_arms = 10
        print(x.shape)
        n_features = x.shape[1] #n_features on a 2d array is # of cols for the first row.

        #Dictionary containing the name of the algorithm and the function it pertains to
        mab_algs = {
            'linUCB':linucb_policy(k_arms, n_features, alpha=1.0)
        }
        parser = argparse.ArgumentParser(description="Process Configs", argument_default=argparse.SUPPRESS)
        args = utils.parse_args(parser)
        print(args.__dict__)

        # Run Algorithms
        ENV = real_environments.env_ml100k.ml100k_Env

        ENV.reset()
        CLBBF = algorithms.CLBBF(args.T, ENV)
        CLBBF_rewards = CLBBF.rewards()
        CLBBF_cumrewards = np.cumsum(CLBBF_rewards)
        CLBBF_ctr = CLBBF_cumrewards / (np.arange(args.T) + 1)

        ENV.reset()
        OFUL = algorithms.OFUL(args.T, ENV)
        OFUL_rewards = OFUL.rewards()
        OFUL_cumrewards = np.cumsum(OFUL_rewards)
        OFUL_ctr = OFUL_cumrewards / (np.arange(args.T) + 1)
        for mab_name, mab_agent in mab_algs.items():
            #initialize results for agent
            self.results_by_mab_name[mab_name] = {
                'cumulative_reward_by_timestep' : [],
                'cumulative_regret_by_timestep' : []
            }
            # Initiate policy
            linucb_policy_object = linucb_policy(K_arms=K_arms, d=d, alpha=alpha)

            # Instantiate trackers
            aligned_time_steps = 0
            cumulative_rewards = 0
            aligned_ctr = []
            unaligned_ctr = []  # for unaligned time steps

            #Randomly sample a menu f wine, say 40
            #we caculate a flight as 4 wines
            #the optimal permutation is the maximum rating you can get from a permutation of 4 from the menu
            #that is, if the highest ratings are 10, 9, 9, 9, then the max score is 37 and is the optimal
            #Now each round the bandit will be able to choose from these 40 wines and will calculate
            #the regret and reward based on the max score!

            optimal_reward = 0
            #This means that the size of the arms is equal to the size of the menu or the size of the flight.

            flight_size = 4
            # Open data
            chosen_flights = []
            for t in range(0, self.timesteps):

                #select menu
                menu_size = 40
                x_menu = x[:x.shape[0]/menu_size]
                x_flight = []
                for i in range(0, flight_size):
                    x_flight.append(x_menu[np.random.rand(0,menu_size)])

                #What arm corresponds with what?
                #Does it correspond with a permutation of a flight?
                # Find policy's chosen arm based on input features
                # maybe this needs to be pairs of flights?
                best_arm = linucb_policy_object.select_arm(x_flight)
                chosen_flights.append(best_arm)

                y_chosen = []
                data_reward = np.sum(y)
                # Use reward information for the chosen arm to update
                linucb_policy_object.linucb_arms[arm_index].reward_update(data_reward, x_single)

                # For CTR calculation
                aligned_time_steps += 1
                cumulative_rewards += data_reward
                aligned_ctr.append(cumulative_rewards / aligned_time_steps)

                return (aligned_time_steps, cumulative_rewards, aligned_ctr, linucb_policy_object)

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