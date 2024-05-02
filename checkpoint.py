import datetime as datetime
import os
import shutil


def make_checkpoint_dir():
    """
    This makes a checkpoint directory to save the state of the experiment and its results.
    :return: Returns the checkpoint directory to save graphs, excel results, logs etc.
    """
    now = datetime.datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H-%M-%S")
    checkpoint_path = os.path.join('checkpoints', dt_string)
    os.mkdir(checkpoint_path)
    os.mkdir(os.path.join(checkpoint_path, 'python'))
    os.mkdir(os.path.join(checkpoint_path, 'results'))
    return checkpoint_path

def save_python_files(checkpoint_dir):
    for file in os.listdir():
        print(file)
        if file.find(".py") != -1:
            shutil.copyfile(os.path.join(os.getcwd(), file), os.path.join(os.path.join('python',checkpoint_dir), file))
            #save_file
            pass
    else:
        pass