import checkpoint as checkpoint
import visualizer
from experiment import Experiment
from ucimlrepo import fetch_ucirepo
import numpy as np

if __name__ == '__main__':
    #checkpointer.py
    #save checkpoint
    print("hello")
    epochs = 1000
    checkpoint_dir = checkpoint.make_checkpoint_dir()

    #save python files in checkpoint
    checkpoint.save_python_files(checkpoint_dir)
    #experiment.py
    experiments = Experiment(epochs, checkpoint_dir)
    #load data
    #run experiment

    # fetch dataset
    wine_quality = fetch_ucirepo(id=186)
    # data (as pandas dataframes)
    # define cntext (features)
    x = wine_quality.data.features
    y = wine_quality.data.targets
    results_by_mab = experiments.run_experiments(x)
    print(results_by_mab)
    #log experiment results in checkpoint

    """                                                                                                                                                                                                                                                                             
    Naive MAB setup: Chooses based on alcohol content
    Has to choose drinks that are good but won't get it extremely drunk.
    """
    #visualizer.py
    #visualize results
    visualizer.plot_multiline_graph("some graph", "x-axis", "y-axis", 
                                    results_by_mab)
    #save