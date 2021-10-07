import argparse
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt


class model_lstm_manyto1(nn.Module):
    def __init__(self,input_size=1,output_size=1,dropout=0.4):
        super(model_lstm_manyto1,self).__init__()
        self.hidden_size=24
        self.num_layers=1
        self.Dropout=nn.Dropout(dropout)

        self.rnn=nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
         #   dropout=dropout
        )

        print(f"dropout: {dropout}")

        self.fc=nn.Linear(self.hidden_size,output_size)

    def forward(self,x,prev_state):
        output, (h_out,c_out) = self.rnn(x, prev_state)
        h_out.shape
        h_out_last=h_out[h_out.shape[0]-1,:,:]
        h_out_last.shape
        h_temp=h_out_last.view(-1,self.hidden_size)
        h_temp2=self.Dropout(h_temp)
        # self.fc(h_temp).shape
        units = self.fc(h_temp2)

        return units,(h_out,c_out)

    def init_state(self,batch_size=1):
        h_0= torch.zeros(self.num_layers,batch_size,self.hidden_size).to(torch.float32)
        c_0= torch.zeros(self.num_layers,batch_size,self.hidden_size).to(torch.float32)
        return h_0,c_0
