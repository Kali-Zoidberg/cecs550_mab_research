import checkpoint as checkpoint
if __name__ == '__main__':
    #checkpointer.py
    #save checkpoint
    print("hello")
    checkpoint_dir = checkpoint.make_checkpoint_dir()

    #save python files in checkpoint
    checkpoint.save_python_files(checkpoint_dir)
    #experiment.py
    #load data
    #run experiment
    #log experiment results in checkpoint

    """                                                                                                                                                                                                                                                                             
    Naive MAB setup: Chooses based on alcohol content
    Has to choose drinks that are good but won't get it extremely drunk.
    """
    #visualizer.py
    #visualize
    #save