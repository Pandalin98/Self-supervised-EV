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
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import joblib
import matplotlib.pyplot as plt
import torch.nn.functional as F
#去除警告
import warnings
from torch.nn.utils.rnn import pad_sequence
from datetime import timedelta
import random
import math
import pickle
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
                 scale=True,         
                 train_cars=15,     
                 val_cars=3,        
                predict_input=None,
                visual_data=False,
                sort = False,
                down_task = 'point_predict',
                read_data_list = False
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
        self.sort = sort
        self.feature_columns =['vehicle_speed', 'vehicle_status', 'charge_status', 'mileage',
       'total_voltage', 'total_current', 'soc',
       'max_single_cell_voltage', 'min_single_cell_voltage', 'max_temperature',
       'min_temperature', 'drive_motor_controller_temperature',
       'drive_motor_speed', 'drive_motor_torque', 'drive_motor_temperature',
       'motor_controller_input_voltage', 'motor_controller_dc_bus_current',
       'rechargeable_energy_storage_device_current']+['cycle_flag']
        self.down_task = down_task

        
        self.__read_data__()
        if self.scaler is None:
            self.scaler = self.get_standard_scaler()    
        else:
            self.scaler ={}
            self.scaler['feature_scaler'] = joblib.load('./data/feature_scaler.pkl')
            self.scaler['target_scaler'] = joblib.load('./data/target_scaler.pkl')
            self.scaler['mileage_scaler'] = joblib.load('./data/mileage_scaler.pkl')
            self.scaler['prior_scaler'] = joblib.load('./data/prior_scaler.pkl')
            print('读取scaler成功')
        
        if visual_data:
            self.visual_data()
        self.__get_battery_pin__()
        del self.df_raw



    def __read_data__(self):
        self.df_raw = {}
        file_list = os.listdir(self.data_path)
        if self.sort == True:
            file_list.sort()
        for filename in file_list:
            # 判断是否为parquet文件
            if filename.endswith(".parquet"):
                # 读取parquet文件并存储到字典中，键为文件名（不包含.parquet后缀），值为对应的DataFrame
                data = pd.read_parquet(os.path.join(self.data_path, filename))
                #筛选出charge_energy在50-60之间的数据
                # data = data[(data['charge_energy']>=50)]
                ##根据index排序
                data = data.sort_index()
                vehicle_number = filename.split('_')[0]
                #如果是CL1则改为CL01
                if len(vehicle_number) == 3:
                    vehicle_number = 'CL0'+vehicle_number[2]
                self.df_raw[vehicle_number] = data
                print('读取{}成功'.format(vehicle_number))
                # # 调试选项
                # break
                    # for key,data_frame in self.df_raw.items():
        #     data_frame.plot(x='mileage',y='charge_energy')
        #     plt.savefig('./data/visual_data/{}.png'.format(key))
    def visual_data(self):
        file_path  = './data/visual_data/visual_data.pkl'
        visual_data = {}
        for key,data_frame in self.df_raw.items():
            data = data_frame[data_frame['begin_charge_flag']==1]
            visual_data[key] = data
            #绘制散点图
            data.plot.scatter(x='mileage',y='charge_energy')
            # data.scatter(x='mileage',y='charge_energy')
            if not os.path.exists('./data/visual_data'):
                os.makedirs('./data/visual_data')
            plt.savefig('./data/visual_data/{}.png'.format(key))
            print('保存{}图片成功'.format(key))
        with open(file_path,'wb') as f:
            pickle.dump(visual_data,f)
    
    def get_standard_scaler(self):
        ##把字典中所有self.df_raw放在一个dataframe中
        df = pd.concat(self.df_raw.values(), ignore_index=True)
        feature_scaler = MinMaxScaler()
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
        
        keys = list(self.df_raw.keys())
        prior_total = []
        for key in keys:
            data_frame_all = self.df_raw[key]
            data_frame_grobyed = data_frame_all.groupby(['cycle_flag'])
            group_keys = list(data_frame_grobyed.groups.keys())  # 将groups转换为列表
            group_keys.sort()  # 对列表进行排序
            data_frame_list = [data_frame_grobyed.get_group(x) for x in group_keys]
            prior_list = self.get_prior(data_frame_list)
            prior_total.append(prior_list)
        prior_total = pd.DataFrame(np.concatenate(prior_total,axis=0))
        prior_scaler = StandardScaler()
        prior_scaler.fit(prior_total)
        self.scaler['prior_scaler'] = prior_scaler
        joblib.dump(prior_scaler, './data/prior_scaler.pkl')
        print('保存scaler成功')

        
        return self.scaler
    

    
    def get_prior(self,df_list):
        
        ##参照论文A state of health estimation framework 
        # based on real-world electric vehicles operating data中的五个prior；
        # 同时增加第六个prior：真实充电容量
        def get_cell_voltage_diff(df_list,ui):
            ##1.内部特征:最大电压差
            df_list = [df[(df['charge_status']==1)] for df in df_list]
            ##将total_voltage一项全部取整数
            df_list = [df.assign(total_voltage=df['total_voltage'].astype(int)) for df in df_list]
            df_list = [df[df['total_voltage'] == ui ] for df in df_list]
            hf1 = [(abs(df['max_single_cell_voltage']-df['min_single_cell_voltage'])).min() for df in df_list]
            return hf1
        
        def get_charge_time(df_list,up):
            ##2.内部特征:恒压充电时间
            df_list = [df[(df['charge_status']==1)] for df in df_list]
            ##将total_voltage一项全部取整数
            df_list = [df.assign(total_voltage=df['total_voltage'].astype(int)) for df in df_list]
            df_list = [df[df['total_voltage'] == up ] for df in df_list]
            hf2 = [len(df) for df in df_list]
            return hf2
        
        def get_temp_percentage(df_list,Th,Tl):
            #找出极端充电温度的比例
            df_list = [df[(df['charge_status']==1)] for df in df_list]
            df_list = [(df['min_temperature']+df['max_temperature'])/2 for df in df_list]
            #找出温度在Th和Tl之间的数据
            hf4 = [1 - len(df[(df >= Tl) & (df <= Th)]) / len(df_list[i]) for i, df in enumerate(df_list)]
            return hf4
        
        def soc_propotion(df_list,SOCv):
            ## 5.内部特征:低SOC充电比例
            df_list = [df[(df['charge_status']==1)]['soc'] for df in df_list]
            hf5 = [len(df[df<=SOCv])/len(df) for df in df_list]
            return hf5
    
        hf1 = get_cell_voltage_diff(df_list,ui=366)
        
        hf2 = get_charge_time(df_list,up=365)
        
        ##3.外部特征:行驶里程数
        hf3 = [(df['mileage'].max()) for df in df_list]
        
        ##4.外部特征:平均温度
        hf4 = get_temp_percentage(df_list,Th=35,Tl=10)
        
        ##5.外部特征:低soc充电比例
        hf5 = soc_propotion(df_list,SOCv=19)
            
        ##6.充电容量
        hf6 = [df['charge_energy'][0] for df in df_list]
        
        #将hf1-hf6横向拼接
        HF = np.column_stack((hf1,hf2,hf3,hf4,hf5,hf6))
        
        ##将NAN替换为对应列的均值
        HF = np.nan_to_num(HF, nan=np.nanmean(HF, axis=0))
        
        return HF
    
    def get_standarded(self,df):
            df[self.feature_columns] = self.scaler['feature_scaler'].transform(df[self.feature_columns])
            df['charge_energy'] = self.scaler['target_scaler'].transform(df['charge_energy'].values.reshape(-1,1))
            df['fixed_capacity'] = self.scaler['target_scaler'].transform(df['fixed_capacity'].values.reshape(-1,1))

            return df
    
    ##重构数据
    def __get_battery_pin__(self):
        ###
        # 构造数据指针
        ###
        pin_list = []
        out_pin = 0
        self.groups = {}
        self.global_feature = {}
        #读取dict中的每一个dataframe
        ##按照key的顺序读取
        keys_ori = list(self.df_raw.keys())
        # keys_ori.sort()
        if self.split == 'train':
            keys = keys_ori[:self.train_cars] 
            # keys = keys_ori[:self.train_cars+self.val_cars]
        elif self.split == 'val':
            keys = keys_ori[self.train_cars:self.train_cars+self.val_cars] 
        elif self.split == 'test':  
            keys = keys_ori[self.train_cars+self.val_cars:] 
        elif self.split == 'pretrain':
            keys = keys_ori
        #逐个车辆开始构造
        for key in keys:
            in_pin = 0
            global_list = []
            data_frame_all = self.df_raw[key]
            data_frame_grobyed = data_frame_all.groupby(['cycle_flag'])
            group_keys = list(data_frame_grobyed.groups.keys())  # 将groups转换为列表
            group_keys.sort()  # 对列表进行排序
            #创建原始大的dataframe
            data_frame_list = [data_frame_grobyed.get_group(x) for x in group_keys]
            prior_list = self.scaler['prior_scaler'].transform(self.get_prior(data_frame_list))
            print('车辆{}的有效充电循环数为{}'.format(key,len(data_frame_list)))
            #构建global_feature
            data_frame_list = [self.get_standarded(df) for df in data_frame_list]
            global_df = self.get_global_feature(data_frame_list)        
            global_df['prior'] = prior_list.tolist()
            self.global_feature[key] = global_df  
            ##构建encoder_data
            encoder_data = [torch.from_numpy(df[self.feature_columns].values[:3000,:]) for df in data_frame_list]
            total_len = len(encoder_data)
            if total_len-self.input_len-self.output_len+1 <=0:
                continue
            self.groups[key] = encoder_data
            for i in range(total_len-self.input_len-self.output_len+1):
                pin_list.append({
                    'out_pin': out_pin,
                    'in_pin': in_pin,
                    'vehicle_number': key,
                })
                in_pin = in_pin + 1
                out_pin = out_pin + 1
                
        pin_df = pd.DataFrame(pin_list).set_index('out_pin')
        self.pin_df = pin_df    
    
    def get_global_feature(self,df_list):
        global_list = []
        for cycle_data in df_list:
            decoder_data = cycle_data[(cycle_data['begin_charge_flag'] == 1)]
            charge_energy = decoder_data['charge_energy'].values
            fixed_capacity = decoder_data['fixed_capacity'].values
            time_table = np.array([decoder_data.index.month,
                decoder_data.index.weekday,
                decoder_data.index.day,
                decoder_data.index.hour])
            mileage = decoder_data['mileage'].values
            global_list.append({
                'time_table': time_table,
                'mileage': mileage,
                'charge_energy': charge_energy,
                'fixed_capacity': fixed_capacity})
        global_df = pd.DataFrame(global_list)
        return global_df

    def __len__(self):
        return len(self.pin_df)

    def __get_battery_pair__(self, index):
        ###
        # 提取数据对
        ###
        pin = self.pin_df.loc[index]
        
        data = {}
        data_frame_list = self.groups[pin['vehicle_number']]
        global_feature = self.global_feature[pin['vehicle_number']]
        
        i  = pin['in_pin']
        s_begin = i
        s_end = s_begin + self.input_len
        r_begin = s_end
        r_end = r_begin + self.output_len
        encoder_data = data_frame_list[s_begin:s_end]
        # encoder_data = [torch.from_numpy(df[self.feature_columns].values[:3000,:]) for df in encoder_data]
        padded_encoder_tensor = pad_sequence(encoder_data, batch_first=True)
        data['encoder_input'] = torch.tensor(padded_encoder_tensor,dtype=torch.float32)
        data['prior'] = torch.tensor(np.stack(global_feature['prior'].iloc[s_begin:s_end].values),dtype=torch.float32)
        data['encoder_mark']=torch.tensor(np.stack(global_feature['time_table'].iloc[s_begin:s_end].values),dtype=torch.float32)
        data['decoder_input'] = torch.tensor(np.stack(global_feature['mileage'].iloc[s_begin:r_end].values),dtype=torch.float32)
        data['decoder_mark'] =torch.tensor(np.stack(global_feature['time_table'].iloc[s_begin:r_end].values),dtype=torch.float32)
        data['fixed_capacity'] = torch.tensor(np.stack(global_feature['fixed_capacity'].iloc[r_begin:r_end].values),dtype=torch.float32)
        data['charge_energy'] = torch.tensor(np.stack(global_feature['charge_energy'].iloc[r_begin:r_end].values),dtype=torch.float32)
        return data
        
    def __getitem__(self, index):
        
        data = self.__get_battery_pair__(index)
        
        dict = {}
        dict['encoder_input'] = data['encoder_input']
        dict['prior'] = data['prior']
        dict['encoder_mark'] = data['encoder_mark'].squeeze(-1)
        dict['decoder_input'] = data['decoder_input']
        dict['decoder_mark'] = data['decoder_mark'].squeeze(-1)
        # if self.down_task == 'point_predict':
        #     dict['label'] = data['fixed_capacity'].squeeze(-1)
        # if self.down_task == 'interval_predict':
        #     dict['label'] = data['charge_energy'].squeeze(-1)
        # dict['label'] = data['charge_energy'].squeeze(-1)
        dict['label'] = data['charge_energy'].squeeze(-1)

        return dict


    def inverse_transform(self,data):
        return self.scaler.inverse_transform(data)


def Vari_len_collate_func(batch_dic):
    #batch_dic (cl,tl,fl)转化为 (tl,cl,fl)
    x_batch = [dic['encoder_input'].transpose(0,1) for dic in batch_dic] #(cl,tl,fl)
    label_batch = [dic['label'] for dic in batch_dic if 'label' in dic.keys()] #
    prior_batch = [dic['prior'] for dic in batch_dic]
    x_dec_batch = [dic['decoder_input'] for dic in batch_dic]
    x_mark_batch = [dic['encoder_mark'] for dic in batch_dic]
    dec_mark_batch = [dic['decoder_mark'] for dic in batch_dic]
    res = {}
    res['encoder_input'] = pad_sequence(x_batch, batch_first=True, padding_value=0).transpose(1,2)
    res['decoder_input'] = pad_sequence(x_dec_batch, batch_first=True, padding_value=0)
    res['decoder_mark'] = pad_sequence(dec_mark_batch, batch_first=True, padding_value=0)
    res['prior'] = torch.stack(prior_batch, dim=0)
    res['encoder_mark'] = torch.stack(x_mark_batch, dim=0)
    if len(label_batch) > 0:
        res['label'] = pad_sequence(label_batch, batch_first=True, padding_value=-1)

    return res

if __name__ == '__main__':


    data_set_pretrain = PowerBatteryData(size=(32,3),split='pretrain',visual_data=False,scale=None,read_data_list=False)
    # with open('./data/list_data/pretrain_list.pkl','wb') as f:
    #     pickle.dump(data_set_pretrain,f)
    # data_set_train = PowerBatteryData(size=(32,3),split='train',visual_data=False,scale=True,read_data_list=False)
    # # with open('./data/list_data/train_list.pkl','wb') as f:
    # #     pickle.dump(data_set_train,f)
    # data_set_val = PowerBatteryData(size=(32,3),split='val',visual_data=False,scale=True,read_data_list=False)
    # with open('./data/list_data/val_list.pkl','wb') as f:
    #     pickle.dump(data_set_val,f)
    
    data_loader = DataLoader(
        data_set_pretrain,
        batch_size=16,
        shuffle=True,
        drop_last=True,
        collate_fn=Vari_len_collate_func
    )

    for i, dict in enumerate(data_loader):
        encoder_in,encoder_mark, decoder_in,decoder_out,prior,decoder_mark = dict['encoder_input'],dict['encoder_mark'],dict['decoder_input'],dict['label'],dict['prior'],dict['decoder_mark']
        print('{}:encoder_in{}__encoder_mark{}__decoder_in{}__decoder_out{}__prior{}__decoder_mark{}'.format(i, encoder_in.shape,encoder_mark.shape, decoder_in.shape,decoder_out.shape,prior.shape,decoder_mark.shape))
    # pass

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
    #     batch_size=1,
    #     shuffle=True,
    #     num_workers=32,
    #     drop_last=False,
    #     collate_fn=Vari_len_collate_func
    # )

    # for i, dict in enumerate(data_loader):
    #     encoder_in,encoder_mark, decoder_in,prior,input_mileage = dict['encoder_input'],dict['encoder_mark'],dict['decoder_input'],dict['prior'],dict['input_mileage']
    #     print('{}:encoder_in{}__encoder_mark{}__decoder_in{}__prior{}__input_mileage{}'.format(i, encoder_in.shape,encoder_mark.shape, decoder_in.shape,prior.shape,input_mileage.shape))
    # pass
