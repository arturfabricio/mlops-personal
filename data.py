import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def mnist():
    train_dir = r"C:\Users\artur\Documents\GitHub\dtu_mlops\data\corruptmnist\train_1.npz"
    train_dataset = np.load(train_dir)

    test_dir = r"C:\Users\artur\Documents\GitHub\dtu_mlops\data\corruptmnist\test.npz"
    test_dataset = np.load(test_dir)
    # exchange with the corrupted mnist dataset
    # train = torch.randn(50000, 784)
    # test = torch.randn(10000, 784) 

    train = pd.DataFrame.from_dict({item: train_dataset[item] for item in train_dataset.files}, orient='index') 
    new_train = train.T
    proper_train = pd.DataFrame(columns=['images','labels'])

    for i in range(5000):
        proper_train.at[i,'images'] = new_train['images'][0][i]
        proper_train.at[i,'labels'] = new_train['labels'][0][i]

    test = pd.DataFrame.from_dict({item: test_dataset[item] for item in test_dataset.files}, orient='index') 
    new_test = test.T
    proper_test = pd.DataFrame(columns=['images','labels'])

    for i in range(5000):
        proper_test.at[i,'images'] = new_test['images'][0][i]
        proper_test.at[i,'labels'] = new_test['labels'][0][i]

    return proper_train, proper_test