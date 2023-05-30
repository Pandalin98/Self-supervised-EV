import numpy as np
import pandas as pd
import os
## 忽略警告
import warnings
warnings.filterwarnings('ignore')


def data_processing(read_path,write_path):
# 读取目标文件夹下所有文件名
    file_list = os.listdir(read_path)

    # 创建一个空字典
    data_dict = {}

    # 循环遍历所有文件
    for file_name in file_list:
        # 读取数据集
        dispose_path = os.path.join(read_path , file_name)
        data = pd.read_excel(dispose_path, index_col=0)
        print("正在处理文件：{}".format(dispose_path))
        # 根据文件名和 CLX 字段确定车辆编号
        car_id = file_name.split('_')[0]
        
        # 修改列名
        data.rename(columns={
            '数据时间':'date',
            '车速': 'speed',
            '车辆状态': 'vehicle_status',
            '充电状态': 'charge_status',
            '累计里程': 'mileage',
            '总电压': 'total_voltage',
            '总电流': 'total_current',
            'SOC': 'soc',
            '电池单体电压最高值': 'highest_voltage',
            '电池单体电压最低值': 'lowest_voltage',
            '最高温度值': 'highest_temperature',
            '最低温度值': 'lowest_temperature'
        }, inplace=True)

        ## 填充 nan 值
        data.fillna(method='ffill', inplace=True)
        data.fillna(method='bfill', inplace=True)

        # 如果该车辆的数据在字典中已存在，则将新的数据添加到其中
        if car_id in data_dict:
            data_dict[car_id] = pd.concat([data_dict[car_id], data])
        # 如果该车辆的数据不存在，则将新的数据添加为一个新的 DataFrame
        else:
            data_dict[car_id] = data
    feature_columns = ['mileage', 'cycle_flag', 'charge_energy','soc']
    # 对每个车辆的数据进行处理
    for car_id, data in data_dict.items():
        #设置循环标记
        mask = (data['charge_status'].shift(2) == 3) & (data['charge_status'].shift(1) == 3) & (data['charge_status'] == 1)
        data['cycle_flag'] = mask.cumsum()
        
        feature_data = pd.DataFrame(columns=feature_columns)
        # 对每个充放电循环进行处理
        groups = data.groupby(data['cycle_flag'])
        for group, frame in groups:
            charge_data = frame[frame['charge_status'] == 1]
            charge_energy = -1/180 * charge_data['total_current'].sum()
            if len(charge_data) >0:
                #取充电数据的最后一行
                charge_data['charge_energy'] = charge_energy
                charge_data_slice =charge_data.iloc[-1]
                # frame['charge_energy'] = charge_energy
                
                #feature_data增加一行
                feature_data = feature_data.append(charge_data_slice[feature_columns])
            # print("车辆{}，循环{}中充电状态为:{}".format(car_id, group, charge_energy))

        # 将feature_data保存到 csv 文件中
        #判断文件夹是否存在，不存在则创建
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        output_path = os.path.join(write_path, '{}_output.csv'.format(car_id))
        # date_fmt = 'yyyy-mm-dd hh:mm:ss'
        feature_data.to_csv(output_path, index=True)

 

if __name__ == '__main__':
    #载入数据
        print(os.getcwd())
        local_read_path="local_data/"
        local_write_path="local_data_structure/"
        test_read_path="test_data/"
        test_write_path="test_data_structure/"
        data_processing(local_read_path,local_write_path)
        data_processing(test_read_path,test_write_path)