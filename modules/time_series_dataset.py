import argparse
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize

class WindowGen():
    def __init__(self,input_width, label_width, shift,
                 data=np.arange(200,250),label_columns=None):

        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.label_columns=label_columns

        self.total_window_size = input_width + shift + label_width

        self.input_slice=slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.input_width + self.shift
        self.labels_slice = slice(self.label_start, self.label_start+self.label_width)
        print(self.labels_slice)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
                        ])
    def split_window(self, features):
        inputs = features[:,self.input_slice,:]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels=np.stack([labels[:,:,i] for i in self.label_columns],axis=-1)
        return inputs,labels

class dataset_time_series(torch.utils.data.Dataset):
    def __init__(self,df=None,
                 sequence_length=4,train_quant=0.7,validation_quant=0.15,
                 label_columns=None,
                 cycle_signal=True,winsorization=False,scaler=None):
        self.sequence_length=sequence_length
        self.shift=2
        # self.scale=scale
        self.sc=scaler
        if isinstance(df, pd.DataFrame):
            self.data = self.process_df_data(df)
        else:
            #single
            # self.data=self.get_random_data()
            # self.data = self.get_stock_data()
            # self.data=self.get_airline_data()
            #multi
            # self.data = self.get_random_data_multi()
            # self.data = self.get_weather_data()
            self.data = self.get_stocks_data_multi()

        self.label_columns=label_columns

        if winsorization:
            self.winsorize_data(limits=[0.10, 0.10])

        if cycle_signal:
            self.add_cycle()

        if self.sc!=None:
            self.scaled_data =  self.scale_data(self.data,train_quant)
        else:
            self.scaled_data=self.data

        #Since we predict only the after n instances, lets calcualte what is n:
        self.first_case_idx=self.sequence_length+self.shift
        raw=(self.data[self.first_case_idx:, :], self.data[self.first_case_idx:, label_columns])
        self.train_raw,self.val_raw,self.test_raw = self.split_train_val_test( raw,train_quant,validation_quant)

        print(f"creating window object with sequence: {self.sequence_length} and shift: {self.shift}")
        self.wi = WindowGen(input_width=self.sequence_length, label_width=1, shift=self.shift,label_columns=label_columns)
        self.windowed_data=self.window_data(self.scaled_data)

        self.train,self.val,self.test=self.split_train_val_test(self.windowed_data,train_quant,validation_quant)

    def add_cycle(self,cycle_length=52):
        t=np.arange(self.data.shape[0])
        cycle_features=np.vstack([
            np.sin(t*2*np.pi/cycle_length),
            np.cos(t*2*np.pi/cycle_length) ]).reshape(-1,2)
        # cycle_features[0:2,:]
        # cycle_features[50:52+1,:]
        self.data=np.hstack([self.data,cycle_features])


    def winsorize_data(self,limits):
        """https://en.wikipedia.org/wiki/Winsorizing"""
        for j in np.arange(self.data.shape[1]):
            self.data[:,j] = winsorize(self.data[:,j], limits=limits)

    def process_df_data(self,df):
        data=np.array(df)
        return data

    def get_airline_data(self):
        link='https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
        df=pd.read_csv(link)
        data_arr=np.array(df['Passengers']).reshape(-1,1)

        # self.sc = MinMaxScaler()
        # scale_arr = self.sc.fit_transform(data_arr)
        return data_arr

    def get_stocks_data_multi(self):
        df=pd.read_excel('data/stocks_1.xlsx',sheet_name='Sheet1',engine='openpyxl')
        data=df.iloc[:,1:]
        return np.array(data)

    def get_weather_data(self):
        df_data=pd.read_csv("data/weather_1.csv")
        #it is a large data set we take every 10th row
        # df_data.head()
        df_data=df_data[0::20].copy()
        # df_data.columns
        # df_data.iloc[1]
        # df_data.columns
        data=np.array(df_data)
        # self.sc = MinMaxScaler()
        # scale_data = self.sc.fit_transform(data)
        # scale_data[1,:]
        return data

    def get_random_data(self):
        i=np.arange(0+1, 1000+1)
        y=i*1+0
        # self.sc = MinMaxScaler()
        # scale_y = self.sc.fit_transform(y.reshape(-1,1))
        # scale_y.shape
        # y.reshape((len(i), 1))
        return y.reshape(-1,1)

    def get_random_data_multi(self):
        x1 = np.arange(0 + 1, 1000 + 1)*2
        x2= np.arange(2000 + 1, 4000 + 1,2)*2
        x3 = 2*x1**(1.2)+3*x2+5
        data=np.hstack([x1.reshape(-1,1),x2.reshape(-1,1),x3.reshape(-1,1)])
        return data

    def scale_data(self,data,train_quant):
        n_training=int(len(data)*train_quant)
        self.sc = MinMaxScaler()

        #fitting only on training
        self.sc.fit(data[:n_training])
        #transforming the entire data
        scaled_data=self.sc.transform(data)

        #check
        # transformed_data=self.sc.inverse_transform(scaled_data)
        # t=abs(transformed_data-data)<=0.00000001
        # t.sum(axis=1)
        return scaled_data


    def window_data(self,data):
        """
        this function takes a time series in n dimentions and return x,y which x has the n dimentions too
        :return:
        """

        d = np.stack([data])

        x, y = self.wi.split_window(d)

        for row in np.arange(1,len(data)-self.sequence_length-self.shift):
            #print(row)
            d = np.stack([data[row:]])
            sample_inputs, sample_output = self.wi.split_window(d)
            #this bring back a batch, but we do not need it now
            # sample_inputs.reshape(sample_inputs.shape[1],sample_inputs.shape[2])
            # le_output=sample_output.reshape(sample_output.shape[1],sample_output.shape[2])
            x=np.vstack([x,sample_inputs])
            y=np.vstack([y, sample_output])
        return torch.tensor(x,dtype=torch.float32),torch.tensor(y,dtype=torch.float32)


    def split_train_val_test(self,data,train_quant,validation_quant):
        x,y=data
        train_size = int(len(y) * train_quant)
        val_size=int(len(y) * validation_quant)
        test_size=len(y)-val_size-train_size

        train = x[0:train_size],y[0:train_size]
        val = x[train_size:train_size+val_size],y[train_size:train_size+val_size]
        test = x[train_size+val_size:len(y)],y[train_size+val_size:len(y)]
        # tx,ty=test
        return train,val,test




    def get_stock_data(self):
        df=pd.read_csv("data/HistoricalData_1625363658135.csv")
        df.head()
        closing=np.array(df['Close/Last'].str.replace("$","").astype(float))
        data_arr=np.flip(closing).reshape(-1,1)
        self.sc = MinMaxScaler()
        scale_arr = self.sc.fit_transform(data_arr)
        return scale_arr

    def plot(self):
        plt.plot(self.data)

    def __len__(self):
        x,y=self.train
        return len(y)

    def __getitem__(self,index):
        # option 1
        # return (
        #         torch.tensor(self.data[index:index+self.sequence_length]).to(torch.float32),
        #         torch.tensor(self.data[index+1:index+self.sequence_length+1]).to(torch.float32)
        # )

        # option 2
        # d=np.stack([self.train[index:]])
        # sample_inputs, sample_output = self.wi.split_window(d) #this is bringing back a batch, but I dont need in _getitem_ maybe in later useage
        # return (
        #     torch.tensor(sample_inputs[0,:,:]).to(torch.float32),
        #     torch.tensor(sample_output[0, :, :]).to(torch.float32)
        # )

        ##option 3
        x,y=self.train
        return (x[index],y[index])

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--sequence-length', type=int, default=10)
    args = parser.parse_args()


    print(1)

    file='data/timestamp_tests.csv'
    df=pd.read_csv(file).iloc[:,1:]

    dataset = dataset_time_series(sequence_length=args.sequence_length, df=df, label_columns=[1],
                                  train_quant=0.6, validation_quant=0.2, cycle_signal=False, winsorization=False)


    # dataset = dataset_time_series(sequence_length=args.sequence_length,label_columns=[2])
    dataset.data
    x,y=dataset.train_raw
    x, y = dataset.train
    y[0:10]
    x.shape

    x[0,:, :]
    y[0]
    x.shape

    x, y = dataset.train_raw

    x.shape
    x, y = dataset.val_raw

    x.shape
    x, y = dataset.test_raw


    x, y = dataset.test_raw
    x,y = dataset.train
    y
    dataset.windowed_data
    y.max()
    y.min()
    print(2)
