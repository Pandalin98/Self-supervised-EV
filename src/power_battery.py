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
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import torch.nn.functional as F
#去除警告
import warnings
from torch.nn.utils.rnn import pad_sequence
import joblib
from datetime import timedelta
import random
import math
# from src.callback.patch_mask import create_patch 

warnings.filterwarnings('ignore')

# Turn off the SettingWithCopyWarning
pd.set_option('mode.chained_assignment', None)


class PowerBatteryData(Dataset):
    """
    Desc: Wind turbine power generation data
          Here, e.g.    15 days for training,
                        3 days for validation
    """

    def __init__(self, data_path='./data/local_data_structure',
                 split='pretrain',
                 size=None,
                 scale=None,         
                 train_cars=18,     # 15 days
                 val_cars=2,        # 3 days
                predict_input=None,
                visual_data=False,
                 ):
        super().__init__()
        if size is None:
            self.input_len = 24
            self.output_len = 12
        else:
            self.input_len = size[0]
            self.output_len = size[1]
        # initialization
        self.split = split
        self.data_path = data_path
        self.scaler = scale
        self.train_cars = train_cars
        self.val_cars = val_cars
      
        self.__read_data__()
        if visual_data:
            self.visual_data()
        self.feature_columns =['vehicle_speed', 'vehicle_status', 'charge_status', 'mileage',
       'total_voltage', 'total_current', 'soc',
       'max_single_cell_voltage', 'min_single_cell_voltage', 'max_temperature',
       'min_temperature', 'drive_motor_controller_temperature',
       'drive_motor_speed', 'drive_motor_torque', 'drive_motor_temperature',
       'motor_controller_input_voltage', 'motor_controller_dc_bus_current',
       'rechargeable_energy_storage_device_current']+['cycle_flag']

        if self.scaler is None:
            self.scaler = self.standard_scaler()    
        else:
            self.scaler ={}
            self.scaler['feature_scaler'] = joblib.load('./data/feature_scaler.pkl')
            self.scaler['target_scaler'] = joblib.load('./data/target_scaler.pkl')
            self.scaler['mileage_scaler'] = joblib.load('./data/mileage_scaler.pkl')
        for key in self.df_raw.keys():
            self.df_raw[key][self.feature_columns] = self.scaler['feature_scaler'].transform(self.df_raw[key][self.feature_columns])
            self.df_raw[key]['charge_energy'] = self.scaler['target_scaler'].transform(self.df_raw[key]['charge_energy'].values.reshape(-1,1))
            self.charge_up_bound =   self.scaler['target_scaler'].transform([[60]])[0][0]                      
            self.charge_low_bound = self.scaler['target_scaler'].transform([[50]])[0][0]
        if self.split == 'predict':
            self.predict_decoder_input =predict_input
        self.__get_battery_pair__()


    def __read_data__(self):
        self.df_raw = {}
        file_list = os.listdir(self.data_path)
        # todo 真实运行时改回来遍历指定目录下的所有文件
        for filename in file_list:
            # 判断是否为parquet文件
            if filename.endswith(".parquet"):
                # 读取parquet文件并存储到字典中，键为文件名（不包含.parquet后缀），值为对应的DataFrame
                data = pd.read_parquet(os.path.join(self.data_path, filename))
                #筛选出charge_energy在50-60之间的数据
                data = data[(data['charge_energy']>=50)]
                ##根据index排序
                data = data.sort_index()
        ##todo 大规模数据前更改
                vehicle_number = filename.split('_')[0]
                #如果是CL1则改为CL01
                if len(vehicle_number) == 3:
                    vehicle_number = 'CL0'+vehicle_number[2]
                self.df_raw[vehicle_number] = data
                print('读取{}成功'.format(vehicle_number))
        # for key,data_frame in self.df_raw.items():
        #     data_frame.plot(x='mileage',y='charge_energy')
        #     plt.savefig('./data/visual_data/{}.png'.format(key))
    def visual_data(self):
        for key,data_frame in self.df_raw.items():
            data = data_frame[data_frame['begin_charge_flag']==1]
            data_frame.plot(x='mileage',y='charge_energy')
            if not os.path.exists('./data/visual_data'):
                os.makedirs('./data/visual_data')
            plt.savefig('./data/visual_data/{}.png'.format(key))
            print('保存{}图片成功'.format(key))
    
    def standard_scaler(self):
        ##把字典中所有self.df_raw放在一个dataframe中
        df = pd.concat(self.df_raw.values(), ignore_index=True)
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
        mileage_scaler = StandardScaler()

        ##fit charge_energy
        target_scaler.fit(df['charge_energy'].values.reshape(-1, 1))
        ##fit 除了feature_scaler的所有列
        feature_scaler.fit(df[self.feature_columns])
        mileage_scaler.fit(df['mileage'].values.reshape(-1, 1))
        self.scaler ={}
        self.scaler['feature_scaler'] = feature_scaler
        self.scaler['target_scaler'] = target_scaler
        self.scaler['mileage_scaler'] = mileage_scaler
        #保存scaler到./data下
        joblib.dump(feature_scaler, './data/feature_scaler.pkl')
        joblib.dump(target_scaler, './data/target_scaler.pkl')
        joblib.dump(mileage_scaler, './data/mileage_scaler.pkl')
        return self.scaler
    
    def __pad_stack__(self, encoder_data):
        max_rows = max(df.shape[0] for df in encoder_data)  # 找到最大的行数

        # 创建零值填充数组
        padding_array = np.zeros_like(encoder_data[0].iloc[-1])

        # 填充或截断每个 DataFrame，使其具有相同的行数
        encoder_data_processed = []
        for df in encoder_data:
            if df.shape[0] < max_rows:
                # 填充不足的行数
                padding_rows = max_rows - df.shape[0]
                padded_rows = np.tile(padding_array, (padding_rows, 1))
                padded_df = pd.concat([df, pd.DataFrame(padded_rows, columns=df.columns)])
                encoder_data_processed.append(padded_df)
            elif df.shape[0] > max_rows:
                # 截断多余的行数
                truncated_df = df.iloc[:max_rows]
                encoder_data_processed.append(truncated_df)
            else:
                # 行数已经相同，无需处理
                encoder_data_processed.append(df)

        # 对处理后的 encoder_data 进行堆叠操作
        encoder_array = np.stack(encoder_data_processed)
        return encoder_array
##重构数据
    def __get_battery_pair__(self):
        self.data_list = []
        time_feature =[ 'month', 'weekday','day', 'hour']
        #读取dict中的每一个dataframe
        ##按照key的顺序读取
        keys_ori = list(self.df_raw.keys())
        # keys_ori.sort()
        if self.split == 'train':
            keys = keys_ori[:self.train_cars]
        elif self.split == 'val':
            keys = keys_ori[self.train_cars:self.train_cars+self.val_cars]
        elif self.split == 'test':
            keys = keys_ori[self.train_cars:self.train_cars+self.val_cars]
        else:
            keys = keys_ori
        for key in keys:
                data_frame_all = self.df_raw[key]
                data_frame_grobyed = data_frame_all.groupby(['cycle_flag'])
                #把每一个 group 的 frame 按照 cycle_flag 的顺序存在一个 list 中
                group_keys = list(data_frame_grobyed.groups.keys())  # 将groups转换为列表
                group_keys.sort()  # 对列表进行排序
                data_frame_list = [data_frame_grobyed.get_group(x) for x in group_keys]
                print('车辆{}的有效充电循环数为{}'.format(key,len(data_frame_list)))
                ##对于 data_frame_list 中的每一个 frame，进行日期的提取
                time_list = []
                data_time_index = []
                for data_frame in data_frame_list:
                    data_frame_index = pd.to_datetime(data_frame.index[0])
                    time_table = np.array([data_frame_index.month,
                    data_frame_index.weekday(),
                    data_frame_index.day,
                    data_frame_index.hour])
                    time_list.append(time_table)
                    data_time_index.append(data_frame_index)
                if self.split != 'predict'  :
                    if len(data_frame_list)-self.input_len-self.output_len+1 <=0:
                        continue
                    total_len = len(data_frame_list)
                    for i in range(total_len-self.input_len-self.output_len+1):
                        data = {}
                        s_begin = i
                        s_end = s_begin + self.input_len
                        r_begin = s_end
                        r_end = r_begin + self.output_len
                        encoder_data = data_frame_list[s_begin:s_end]
                        prior = [df['charge_energy'].values[-1] for df in encoder_data]
                        encoder_data = [df[self.feature_columns] for df in encoder_data]
                        prior = np.array(prior).reshape(-1,1)
                        encoder_data = self.__pad_stack__(encoder_data)
                        #读取data_frame_list中的每一个frame的['begin_charge_flag']==1
                        decoder_data_all = pd.concat(df[(df['begin_charge_flag'] == 1)] for df in data_frame_list[r_begin:])
                        #选出时间超过last_time 15天的数据
                        last_time = data_time_index[s_end]
                        begin_predict = last_time + timedelta(days=15)
                        last_predcit = last_time + timedelta(days=30*6)
                        decoder_data_s = decoder_data_all[(decoder_data_all.index>=begin_predict)&(decoder_data_all.index<=last_predcit)]
                        #随机从decoder_data_s顺序采样n个数据，并返回采样的index
                        data['encoder_input']  = encoder_data
                        #把列表中的值转化为numpy数组
                        data['encoder_mark'] = np.array(time_list[s_begin:s_end])    
                        data['prior'] = prior
                        if len(decoder_data_s)<self.output_len:
                            break
                        if self.split == 'pretrain':
                            sample_times =1
                        else:
                            sample_times = min(5,len(decoder_data_s)-self.output_len+1)
                        for j in range(sample_times):
                            sampled_indices = random.sample(range(len(decoder_data_s)), self.output_len)
                            sampled_indices.sort()
                            decoder_data = decoder_data_s.iloc[sampled_indices,:]
                            time_list_sampled = [time_list[idx] for idx in sampled_indices]  # 访问 time_list 中的元素
                            data['decoder_input'] =np.concatenate([np.array(time_list_sampled) ,decoder_data[['mileage']].values],axis=1) 
                            data['decoder_output'] = decoder_data['charge_energy'].values
                            self.data_list.append(data)

                if self.split == 'predict':
                        data = {}
                        decoder_input = self.predict_decoder_input

                        #选出 car_id = key的数据
                        decoder_input = decoder_input[decoder_input['car_id']==key][time_feature+['mileage']]
                        decoder_input['mileage'] = self.scaler['mileage_scaler'].transform(decoder_input['mileage'].values.reshape(-1,1))
                        encoder_data = data_frame_list[-self.input_len:]
                        prior = [df['charge_energy'].values[-1] for df in encoder_data]
                        prior = np.array(prior).reshape(-1,1)
                        encoder_data = [df[self.feature_columns] for df in encoder_data]
                        encoder_data = self.__pad_stack__(encoder_data)
                        data['decoder_input'] = decoder_input.values
                        data['encoder_input'] = encoder_data
                        data['encoder_mark'] = np.array(time_list[-self.input_len:])
                        data['prior'] = prior
                        self.data_list.append(data)


    def __getitem__(self, index):
        dict = {}
        dict['encoder_input'] = torch.tensor(self.data_list[index]['encoder_input'],dtype=torch.float32)
        dict['prior'] = torch.tensor(self.data_list[index]['prior'],dtype=torch.float32)
        dict['decoder_input'] = torch.tensor(self.data_list[index]['decoder_input'],dtype=torch.float32)
        dict['encoder_mark'] = torch.tensor(self.data_list[index]['encoder_mark'],dtype=torch.float32)
        if self.split != 'predict':
            dict['label'] = torch.tensor(self.data_list[index]['decoder_output'],dtype=torch.float32)
        return dict

    def __len__(self):
        return len(self.data_list) 

    def inverse_transform( data):
        return self.scaler.inverse_transform(data)


def Vari_len_collate_func(batch_dic):
    batch_len = len(batch_dic)  # 批尺寸
    sorted_batch = sorted(batch_dic, key=lambda x: x['encoder_input'].shape[0], reverse=True)
    #batch_dic (cl,tl,fl)转化为 (tl,cl,fl)
    x_batch = [dic['encoder_input'].transpose(0,1) for dic in sorted_batch]
    label_batch = [dic['label'] for dic in sorted_batch if 'label' in dic.keys()]
    prior_batch = [dic['prior'] for dic in sorted_batch]
    x_dec_batch = [dic['decoder_input'] for dic in sorted_batch]
    x_mark_batch = [dic['encoder_mark'] for dic in sorted_batch]
    res = {}
    res['encoder_input'] = pad_sequence(x_batch, batch_first=True, padding_value=0).transpose(1,2)
    res['decoder_input'] = pad_sequence(x_dec_batch, batch_first=True, padding_value=0)
    res['prior'] = torch.stack(prior_batch, dim=0)
    res['encoder_mark'] = torch.stack(x_mark_batch, dim=0)
    if len(label_batch) > 0:
        res['label'] = torch.stack(label_batch, dim=0)

    return res

if __name__ == '__main__':


    data_set = PowerBatteryData(size=(32,3),split='train')

    data_loader = DataLoader(
        data_set,
        batch_size=64,
        shuffle=True,
        num_workers=32,
        drop_last=True,
        collate_fn=Vari_len_collate_func
    )

    for i, dict in enumerate(data_loader):
        encoder_in,encoder_mark, decoder_in,decoder_out,prior = dict['encoder_input'],dict['encoder_mark'],dict['decoder_input'],dict['label'],dict['prior']
        print('{}:encoder_in{}__encoder_mark{}__decoder_in{}__decoder_out{}__prior{}'.format(i, encoder_in.shape,encoder_mark.shape, decoder_in.shape,decoder_out.shape,prior.shape))
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
    # predict_input['weekday'] = predict_input_index.weekday
    # predict_input['hour'] = predict_input_index.hour    
    # predict_input['day'] = predict_input_index.day
    # data_set = PowerBatteryData(size=(24,3),        predict_input=predict_input,
    #     split='predict',
    #     data_path = './data/test_data_structure')

    # data_loader = DataLoader(
    #     data_set,
    #     batch_size=64,
    #     shuffle=True,
    #     num_workers=32,
    #     drop_last=False,
    #     collate_fn=Vari_len_collate_func
    # )

    # for i, dict in enumerate(data_loader):
    #     encoder_in,encoder_mark, decoder_in,prior = dict['encoder_input'],dict['encoder_mark'],dict['decoder_input'],dict['prior']
    #     print('{}:encoder_in{}__encoder_mark{}__decoder_in{}__prior{}'.format(i, encoder_in.shape,encoder_mark.shape, decoder_in.shape,prior.shape))
    # pass
