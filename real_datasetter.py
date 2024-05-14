import os,sys

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

createDirectory('./real_models')
createDirectory('./real_preprocess')

createDirectory('./real_datasets/ml100k/raw/')
os.system('unzip ./real_datasets/ml-100k.zip -d ./real_datasets/ml100k/raw/')
createDirectory('./real_datasets/ml100k/preprocess/')