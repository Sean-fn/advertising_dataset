import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.utils.tensorboard

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



class AdDataset(Dataset):
    def __init__(self, path, mode):
        with open(path, 'r') as f:
            data = pd.read_csv(f)

        def gettime(point):
            t = point.split(' ')

            temp_date = t[0].split('-')
            temptp = t[1].split(':')

            def makeint(target):
                target_accu = 0
                for i, j in enumerate(target):
                    j = int(j)
                    if i == 0:
                        target_accu +=(j*10000)
                    elif i == 1:
                        target_accu += (j*100)
                    elif i == 2:
                        target_accu += j
                return target_accu
            
            date = makeint(temp_date)
            time = makeint(temptp)
            result = (date * 1000000) + time

            return result

        data['Country'] = data['Country'].astype('category').cat.codes
        data['City'] = data['City'].astype('category').cat.codes
        data.drop('Ad Topic Line', axis=1, inplace=True)
        
        data['NewTimestamp'] = data['Timestamp'].apply(gettime)
        data.drop('Timestamp', axis=1, inplace=True)

        print(data.info())
        print(data.head())


AdDataset(path='data/advertising.csv', mode='train')