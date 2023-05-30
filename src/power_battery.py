# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: Power Battery dataset utilities
Date:    2022/03/10
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from prepare import prep_env_tsf
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt


# Turn off the SettingWithCopyWarning
pd.set_option('mode.chained_assignment', None)


class PowerBatteryData(Dataset):
    """
    Desc: Wind turbine power generation data
          Here, e.g.    15 days for training,
                        3 days for validation
    """

    def __init__(self, data_path='./data/local_data_structure',
                 flag='all',
                 size=None,
                 scaler=None,         
                 train_cars=15,     # 15 days
                 val_cars=3,        # 3 days
                predict_input=None,
                 is_predict=False
                 ):
        super().__init__()
        if size is None:
            self.input_len = 24
            self.output_len = 12
        else:
            self.input_len = size[0]
            self.output_len = size[1]
        # initialization
        self.flag = flag
        self.data_path = data_path
        self.is_predict = is_predict
        self.scaler = scaler
        self.train_cars = train_cars
        self.val_cars = val_cars
      
        self.__read_data__()
        self.feature_columns =['mileage','cycle_flag','soc'] 

        if self.scaler is None:
            self.scaler = self.standard_scaler()    
        for key in self.df_raw.keys():
            self.df_raw[key][self.feature_columns] = self.scaler['feature_scaler'].transform(self.df_raw[key][self.feature_columns])
            self.df_raw[key]['charge_energy'] = self.scaler['target_scaler'].transform(self.df_raw[key]['charge_energy'].values.reshape(-1,1))
        if self.is_predict:
            self.predict_decoder_input =predict_input
        self.__get_battery_pair__()


    def __read_data__(self):
        self.df_raw = {}
        file_list = os.listdir(self.data_path)
        if self.flag == 'train':
            file_list = file_list[:self.train_cars]
        elif self.flag == 'val':
            file_list = file_list[self.train_cars:self.train_cars+self.val_cars]

        # 遍历指定目录下的所有文件
        for filename in file_list:
            # 判断是否为CSV文件
            if filename.endswith(".csv"):
                # 读取CSV文件并存储到字典中，键为文件名（不包含.csv后缀），值为对应的DataFrame
                data = pd.read_csv(os.path.join(self.data_path, filename),index_col=0)
                
                #筛选出soc在40-90之间的数据
                data = data[(data['soc']>=40)&(data['soc']<=90)]
                data = data[(data['charge_energy']>=40)&(data['charge_energy']<=80)]
                ##根据index排序
                data = data.sort_index()
                ##file
                self.df_raw[filename[:-11]] = data
                
        # for key,data_frame in self.df_raw.items():
        #     data_frame.plot(x='mileage',y='charge_energy')
        #     plt.savefig('./data/visual_data/{}.png'.format(key))

    def standard_scaler(self):
        ##把字典中所有self.df_raw放在一个dataframe中
        df = pd.concat(self.df_raw.values(), ignore_index=True)
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
        mileage_scaler = StandardScaler()

        ##fit charge_energy
        target_scaler.fit(df['charge_energy'].values.reshape(-1, 1))
        ##fit 除了charge_energy的所有特征
        feature_scaler.fit(df.drop(['charge_energy'],axis=1))
        mileage_scaler.fit(df['mileage'].values.reshape(-1, 1))
        self.scaler ={}
        self.scaler['feature_scaler'] = feature_scaler
        self.scaler['target_scaler'] = target_scaler
        self.scaler['mileage_scaler'] = mileage_scaler
        #保存scaler到./data下
        joblib.dump(feature_scaler, './data/feature_scaler.pkl')
        joblib.dump(target_scaler, './data/target_scaler.pkl')
    
        return self.scaler
        

    def __get_battery_pair__(self):
        self.data_list = []
        
        #读取dict中的每一个dataframe
        for key,data_frame in self.df_raw.items():
                ##将datafarme里的datatime的index转化为月分和日分
            data_frame_index = pd.to_datetime(data_frame.index)
            data_frame['month'] = data_frame_index.month
            data_frame['day'] = data_frame_index.day
            
            if self.flag != 'predict'  :
                for i in range(len(data_frame)-self.input_len-self.output_len+1):
                    data = {}
                    s_begin = i
                    s_end = s_begin + self.input_len
                    r_begin = s_end
                    r_end = r_begin + self.output_len
                    data['encoder_input']  = data_frame[['month','day','soc','mileage','charge_energy']].iloc[s_begin:s_end].values
                    data['decoder_input'] = data_frame[['month','day','mileage']].iloc[r_begin:r_end].values
                    data['decoder_output'] = data_frame[['charge_energy']].iloc[r_begin:r_end].values
                    self.data_list.append(data)

            if self.flag == 'predict':
                    data = {}
                    decoder_input = self.predict_decoder_input
                    #选出 car_id = key的数据
                    decoder_input = decoder_input[decoder_input['car_id']==key][['month','day','mileage']]
                    decoder_input['mileage'] = self.scaler['mileage_scaler'].transform(decoder_input['mileage'].values.reshape(-1,1))
                    data['decoder_input'] = decoder_input[['month','day','mileage']].values
                    data['encoder_input'] = data_frame[['month','day','soc','mileage','charge_energy']].iloc[-self.input_len:].values
                    self.data_list.append(data)


    def __get_data__(self, turbine_id):
        data_x = self.__get_battery(turbine_id)
        data_y = data_x
        return data_x, data_y



    def __getitem__(self, index):
        encoder_input = self.data_list[index]['encoder_input']
        decoder_input = self.data_list[index]['decoder_input']
        if self.is_predict:
            return encoder_input, decoder_input
        else:
            label = self.data_list[index]['decoder_output']
            return encoder_input, decoder_input,label

    def __len__(self):
        return len(self.data_list) 

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


if __name__ == '__main__':

    settings = prep_env_tsf().__dict__

    data_set = PowerBatteryData(size=(24,3))

    data_loader = DataLoader(
        data_set,
        batch_size=settings["batch_size"],
        shuffle=True,
        num_workers=settings["num_workers"],
        drop_last=True
    )

    for i, data in enumerate(data_loader):
        encoder_in, decoder_in,decoder_out = data
        print('{}:encoder_in{}__decoder_in{}__decoder_out{}'.format(i, encoder_in.shape, decoder_in.shape,decoder_out.shape))
    pass

    # predict_input =pd.read_csv('./data/result.csv')
    # # 重命名列名
    # predict_input = predict_input.rename(columns={
    #     '车辆号': 'car_id',
    #     '拟充电时间': 'index',
    #     '拟充电时刻里程': 'mileage',
    #     '估计的充电量': 'charge_energy'
    # })
    # # 将 'index' 列设置为索引，并将其解析为 datetime
    # predict_input['index'] = pd.to_datetime(predict_input['index'])
    # predict_input = predict_input.set_index('index')
    # #读出month和day作为新的两列
    # predict_input_index = pd.to_datetime(predict_input.index)
    # predict_input['month'] = predict_input_index.month
    # predict_input['day'] = predict_input_index.day


    # data_set = PowerBatteryData(size=(24,3),data_path='./data/test_data_structure',
    #                             flag='predict',is_predict=True,predict_input=predict_input)

    # for i in range(len(data_set)):
    #     encoder_input, decoder_input = data_set.__getitem__(i)
    #     print('{}:encoder_in{}__decoder_in{}'.format(i, encoder_input.shape, decoder_input.shape))
