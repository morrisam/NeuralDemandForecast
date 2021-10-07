import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt

from modules.lstm_model import model_lstm_manyto1
from modules.time_series_dataset import dataset_time_series

from sklearn.preprocessing import MinMaxScaler


class forecast_model():
        def __init__(self,train,val,test,train_raw,val_raw,test_raw,label_columns,scaler,args):
            self.early_stopping=True
            self.train=train
            self.val=val
            self.test=test
            self.train_raw=train_raw
            self.val_raw = val_raw
            self.test_raw = test_raw
            self.args=args
            self.label_columns=label_columns
            self.scaler=scaler
            x, _ = train
            self.input_size = x.shape[2]

            torch.manual_seed(15)
            self.model=model_lstm_manyto1(input_size=x.shape[2],output_size=1,dropout=args.dropout)
            print(f"args: {args}")


        def fit(self,dataloader):

            self.loss=dict(epoch=[],loss=[],rmse_training=[],mape_training=[],rmse_val=[],mape_val=[])
            self.model.train()
            # best_model=self.model

            criterion = nn.MSELoss()
            # optimizer=optim.SGD(model.parameters(),lr=0.01)
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            for epoch in range(self.args.max_epochs):

                for batch, (x, y) in enumerate(dataloader):
                    optimizer.zero_grad()
                    state = self.model.init_state(batch_size=x.shape[0])
                    # y.shape
                    # y.view(x.shape[0],-1).shape
                    y_pred, _ = self.model(x, state)
                    # y_pred.view(-1,y_pred.shape[0]).shape
                    # loss = criterion(y_pred.flatten(), y.flatten())
                    loss = criterion(y_pred, y.view(x.shape[0], -1))

                    for s in state:
                        s.detach()
                    loss.backward()
                    optimizer.step()
                    if epoch % 100 == 0:
                        print({'epoch': epoch, 'batch': batch, 'loss': loss.item()})
                if epoch % 10 == 0:
                    rmse_score,mape_score=self.score()
                    self.loss['epoch'].append(epoch)
                    self.loss['loss'].append(loss.item())
                    self.loss['rmse_training'].append(rmse_score[0])
                    self.loss['mape_training'].append(mape_score[0])
                    self.loss['rmse_val'].append(rmse_score[1])
                    self.loss['mape_val'].append(mape_score[1])
                    print({'epoch': epoch, 'rmse': rmse_score, 'mape': mape_score})
                    self.model.train()
                    #early stopping
                    lag=2 # how many epoch 'signals' we look back
                    min_change=+0.05 #what is the minimun improvement was -0.02
                    it=len(self.loss['mape_val'])
                    if it >= 7: #since which epoch signal we consider early stopping
                        change= ( self.loss['mape_val'][it-1] - self.loss['mape_val'][it-1 - lag] ) / self.loss['mape_val'][it-1 - lag]
                        if change > min_change:
                            print (f"Early stopping in epoch: {epoch} due to change of: {change:.5f}")
                            break
            # rmse_score, mape_score = self.score()
            # print('last model:')
            # print({'epoch': epoch, 'rmse': rmse_score, 'mape': mape_score})





        def predict(self,data,scaler,label_columns):
            self.model.eval()
            x,y=data
            state = self.model.init_state(x.shape[0])
            o, s = self.model(x, state)
            y_hat = o.detach().numpy() * np.ones((1, self.input_size))
            if scaler is not None:
                y_hat = self.scaler.inverse_transform(y_hat)
            return y_hat[:,label_columns]

        def baseline(self,lag=1):
            xv, yv = self.val_raw
            xt, yt= self.test_raw
            #last one from val and
            y_hat=np.vstack([yv[-lag-1:],yt[:-lag-1]])
            return y_hat

        def score_baseline(self,lag=1):
            x,y=self.test_raw
            y_hat=self.baseline(lag)
            # np.hstack([ self.baseline(), y])
            return self.rmse(y, y_hat),self.mape(y, y_hat)

        def rmse(self,expected, predictions):
            mse = mean_squared_error(expected, predictions)
            rmse = sqrt(mse)
            return rmse

        def mape(self,expected, predictions):
            mask = expected != 0
            return (np.fabs(expected - predictions) / expected)[mask].mean()

        def score(self,sets_list=None):
            """train and val errors"""
            if sets_list==None:
                sets_list=[(self.train,self.train_raw),(self.val,self.val_raw)]
            rmse_scores=[]
            mape_scores=[]
            for train,train_raw in sets_list:
                pred = self.predict(train, scaler=self.scaler, label_columns=self.label_columns)
                xt, yt = train_raw
                #since there an option to have a shifted prediction (we move the vector yt)
                yt_adj=yt[-pred.shape[0]:]
                rmse_scores.append(self.rmse(yt_adj, pred))
                mape_scores.append(self.mape(yt_adj, pred))

            return rmse_scores,mape_scores

        def show_forecast_curve(self,length):
            return 1


if __name__=="__main__":
    print('program starting')

    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--sequence-length', type=int, default=10)
    parser.add_argument('--dropout', type=int, default=0.5)
    args = parser.parse_args()

    ##############             load dataset             ##############



    dataset = dataset_time_series(sequence_length=args.sequence_length, label_columns=[2],scaler=MinMaxScaler())
    input_size = dataset.data.shape[1]  # this is the number of features in the data

    dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True
        )

    #initialization
    fmodel=forecast_model(train=dataset.train,val=dataset.val,test=dataset.test,args=args,
                          train_raw=dataset.train_raw,val_raw=dataset.val_raw,test_raw=dataset.test_raw,label_columns=dataset.label_columns,
                          scaler=dataset.sc)
    #trianing
    fmodel.fit(dataloader)
    fmodel.predict(dataset.train,scaler=dataset.sc,label_columns=dataset.label_columns)

    #test error on test
    pred_test=fmodel.predict(dataset.test,scaler=dataset.sc,label_columns=dataset.label_columns)
    xt,yt=dataset.test_raw
    fmodel.rmse(yt,pred_test)

    rmse_score,mape_score=fmodel.score(sets_list=[(dataset.test,dataset.test_raw)])
    print(f"TESTS SCORES: rmse: {rmse_score}, mape: {mape_score}")

    # fmodel.baseline()
    print(fmodel.score_baseline())


