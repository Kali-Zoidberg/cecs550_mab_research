class Experiment:
    def __init__(self, epochs:int, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.epochs = epochs
        self.results_by_mab_name = {}
    def run_experiments(self):

        #Dictionray containing the name of the algorithm and the function it pertains to
        mab_algs = {
            'generic_mab':self.generic_mab_alg_test
        }

        for mab_name, mab_func in mab_algs.items():
            self.results_by_mab_name[mab_name] = mab_func()

        #Write result to CSV
        #show graphs via visualizer


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