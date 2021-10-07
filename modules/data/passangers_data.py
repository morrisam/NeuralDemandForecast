
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler

"""## Data Plot"""

training_set = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv')
#training_set = pd.read_csv('shampoo.csv')
training_set.head()
training_set.to_csv('modules/data/airline-passengers.csv',index=False)
training_set = training_set.iloc[:,1:2].values