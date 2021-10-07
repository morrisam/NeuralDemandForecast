import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt

from modules.lstm_model import model_lstm_manyto1
from modules.time_series_dataset import dataset_time_series

from modules.plots import plot_2_lines,plot_simple_curve

from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA

from time_series_dataset import dataset_time_series


class benchmark_model():
        def __init__(self,train,val,test,train_raw,val_raw,test_raw,label_columns,scaler,args):
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
            self.model=None



        def fit(self,order=(0,0,0)):
            self.model_type=None
            xt,yt=self.train_raw
            xt=np.delete(xt,self.label_columns,1)

            params=dict(start_p=1, start_q=1,
                                        max_p=6, max_q=3, m=52,max_order=20,
                                        start_P=0, seasonal=True,
                                        d=0, D=0, trace=True,
                                        error_action='ignore',
                                        suppress_warnings=True,
                                        stepwise=True)
            params2 = dict(start_p=1, start_q=1,
                          max_p=1, max_q=1, m=52, max_order=2,
                          start_P=0, seasonal=True,
                          d=0, D=0, trace=True,
                          error_action='ignore',
                          suppress_warnings=True,
                          stepwise=False)
            if self.input_size==1:
                #means we are not predicting multivariate data
                xt=None
            if order == (0,0,0):
                self.model = auto_arima(y=yt,X=xt, **params2)
                self.model_type='pmdarima'
            else:
                self.model = ARIMA(yt,exog=xt,order=order).fit()

        def predict(self,n_periods):

            if self.input_size==1:
                #means we are not predicting multivariate data
                x=None
            else:
                # we are predicting val and test sets
                xv, _ = self.val_raw
                xts, _ = self.test_raw
                x = np.vstack([xv, xts])[0:n_periods,:]
                x=np.delete(x,self.label_columns,1)

            #predict function based on model
            if self.model_type=='pmdarima':
                pred = self.model.predict(n_periods=n_periods,X=x)
            else:
                pred = self.model.forecast(n_periods,exog=x)
            return pred

        def rmse(self,expected, predictions):
            mse = mean_squared_error(expected, predictions)
            rmse = sqrt(mse)
            return rmse

        def mape(self,expected, predictions):
            mask = expected != 0
            return (np.fabs(expected - predictions) / expected)[mask].mean()



if __name__=="__main__":
    print('program starting')

    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence-length', type=int, default=10)
    args = parser.parse_args()

    ##############             load dataset             ##############

    # file='data/airline-passengers.csv'
    # file='../retail_data/46_store_code.csv'
    file='data/timestamp_tests.csv'

    df1=pd.read_csv(file)
    df2=df1.copy()
    # df2=df1.iloc[:,1:2].copy()
    # df2=df1.iloc[0:81,1:3].copy()

    plot_simple_curve(df2.iloc[:,0])

    dataset = dataset_time_series(sequence_length=args.sequence_length,df=df2, label_columns=[0],cycle_signal=False,winsorization=True)
    input_size = dataset.data.shape[1]  # this is the number of features in the data

    #initialization
    bmodel=benchmark_model(train=dataset.train,val=dataset.val,test=dataset.test,args=args,
                          train_raw=dataset.train_raw,val_raw=dataset.val_raw,test_raw=dataset.test_raw,label_columns=dataset.label_columns,
                          scaler=dataset.sc)
    #Training
    bmodel.fit(order=(0,0,0))
    _ , y_train=dataset.train_raw
    _ , y_val=dataset.val_raw
    _ , y_test = dataset.test_raw

    horizon=y_val.shape[0]+y_test.shape[0]
    pred_val = bmodel.predict(horizon)
    pred = np.concatenate((np.zeros(shape=(y_train.shape[0],1)),pred_val.reshape(-1,1)))
    start=dataset.data.shape[0] - pred.shape[0]
    datax=np.vstack([y_train,y_val,y_test])
    # datax.shape
    # dataset.data[start:].shape
    plot_2_lines(datax=datax, pred=pred, cut_num=y_train.shape[0], text='txt',save_image=False)

    print('------------------------------------------------------')
    print(bmodel.model.summary())
    print('------------------------------------------------------')

    #test error on test
    pred_test=pred[-y_test.shape[0]:]
    xt,yt=dataset.test_raw
    rmse_score,mape_score=bmodel.rmse(yt,pred_test),bmodel.mape(yt.flatten(),pred_test)
    print(f"TESTS SCORES: rmse: {rmse_score}, mape: {mape_score}")





