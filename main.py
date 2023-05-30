# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: Training and Validation
Date:    2022/03/10
"""
import os
import time
import numpy as np
from typing import Callable
import paddle
import random
from paddle.io import DataLoader
from common import EarlyStopping
from common import adjust_learning_rate
from common import Experiment
from prepare import prep_env_tsf
from st_transformer import WPFModel
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib
from power_battery import PowerBatteryData
import pandas as pd
def val(experiment, model, data_loader, criterion):
    # type: (Experiment, WPFModel, DataLoader, Callable) -> np.array
    """
    Desc:
        Validation function
    Args:
        experiment:
        model:
        data_loader:
        criterion:
    Returns:
        The validation loss
    """
    validation_loss = []
    for i, (batch_enc, batch_dec,batch_y) in enumerate(data_loader):
        sample, truth = experiment.process_one_batch(model, batch_enc, batch_dec,batch_y)
        loss = criterion(sample, truth)
        validation_loss.append(loss.item())
    validation_loss = np.average(validation_loss)
    return validation_loss

def visual_val(sample,batch_y,id):
    target_scaler = joblib.load('./data/target_scaler.pkl')  
    sample = sample.numpy()
    batch_y = batch_y.numpy()
    sample = target_scaler.inverse_transform(sample.reshape(-1,1))
    batch_y = target_scaler.inverse_transform(batch_y.reshape(-1,1))
    preds = sample
    gt = batch_y
    batch_size = sample.shape[0]

    # 可视化张量
    fig, axs = plt.subplots(batch_size, figsize=(10, batch_size))

    for i in range(batch_size):
        axs[i].plot(preds[i, :],label='preds')
        axs[i].plot(gt[i, :],label='gt')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('./visual/sample'+str(id)+'.png')

def train_and_val(experiment, model, model_folder, is_debug=False):
    # type: (Experiment, WPFModel, str, bool) -> None
    """
    Desc:
        Training and validation
    Args:
        experiment:
        model:
        model_folder: folder name of the model
        is_debug:
    Returns:
        None
    """
    args = experiment.get_args()
    train_data, train_loader = experiment.get_data(flag='train')
    val_data, val_loader = experiment.get_data(flag='val')

    path_to_model = os.path.join(args["checkpoints"], model_folder)
    if not os.path.exists(path_to_model):
        os.makedirs(path_to_model)

    early_stopping = EarlyStopping(patience=args["patience"], verbose=True)
    optimizer = experiment.get_optimizer(model, args["lr"])
    criterion = Experiment.get_criterion()

    epoch_start_time = time.time()
    for epoch in range(args["train_epochs"]):
        iter_count = 0
        train_loss = []
        model.train()
        for i, (batch_enc, batch_dec,batch_y) in enumerate(train_loader):
            iter_count += 1
            sample, truth = experiment.process_one_batch(model, batch_enc, batch_dec,batch_y)
            loss = criterion(sample, truth)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.minimize(loss)
            optimizer.step()
        val_loss = val(experiment, model, val_loader, criterion)
        if is_debug:
            train_loss = np.average(train_loss)
            epoch_end_time = time.time()
            print("Epoch: {}, \nTrain Loss: {}, \nValidation Loss: {}".format(epoch, train_loss, val_loss))
            print("Elapsed time for epoch-{}: {}".format(epoch, epoch_end_time - epoch_start_time))
            epoch_start_time = epoch_end_time

        # Early Stopping if needed
        early_stopping(val_loss, model, path_to_model)
        if early_stopping.early_stop:
            print("Early stopped! ")
            print("input_len: {}, \noutput_len: {}, \nValidation Loss: {}".format(experiment.args['input_len'], experiment.args['output_len'], val_loss))
            break
        adjust_learning_rate(optimizer, epoch + 1, args)
    return val_loss

def predict(model,config):
    predict_input =pd.read_csv('./data/result.csv')
    # 重命名列名
    predict_input = predict_input.rename(columns={
        '车辆号': 'car_id',
        '拟充电时间': 'index',
        '拟充电时刻里程': 'mileage',
        '估计的充电量': 'charge_energy'
    })
    # 将 'index' 列设置为索引，并将其解析为 datetime
    predict_input['index'] = pd.to_datetime(predict_input['index'])
    predict_input = predict_input.set_index('index')
    #读出month和day作为新的两列
    predict_input_index = pd.to_datetime(predict_input.index)
    predict_input['month'] = predict_input_index.month
    predict_input['day'] = predict_input_index.day
    

    data_set = PowerBatteryData(size=[config.input_len, config.output_len],data_path='./data/test_data_structure',
                                flag='predict',is_predict=True,predict_input=predict_input)
    car_id = list(predict_input.car_id.unique())
    target_scaler = joblib.load('./data/target_scaler.pkl')
    predict_ah = []
    for i in range(len(data_set)):
        encoder_input, decoder_input = data_set.__getitem__(i)
        encoder_input = paddle.to_tensor(encoder_input)
        decoder_input = paddle.to_tensor(decoder_input)
        encoder_input = paddle.unsqueeze(encoder_input, axis=0).astype('float32')
        decoder_input = paddle.unsqueeze(decoder_input, axis=0).astype('float32')

        #深度学习模型进入eval模式
        model.eval()
        sample = model(encoder_input, decoder_input)
        sample = sample.numpy()
        sample = target_scaler.inverse_transform(sample.reshape(-1,1))
        predict_ah.append(sample)
    
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>预测值为：>>>>>>>>>>>>>>>>>>>>>>>>>>')
    for i in range(len(predict_ah)):
        print(predict_ah[i])


if __name__ == "__main__":
    fix_seed = 3407
    random.seed(fix_seed)
    paddle.seed(fix_seed)
    np.random.seed(fix_seed)

    settings = prep_env_tsf()
    ##config转化为dict
    settings = settings.__dict__
    #
    # Set up the initial environment
    # Current settings for the model
    # cur_setup = '{}_t{}_i{}_o{}_ls{}_train{}_val{}'.format(
    #     settings["filename"], settings["task"], settings["input_len"], settings["output_len"], settings["lstm_layer"],
    #     settings["train_size"], settings["val_size"]
    # )
    cur_setup = 'i{}_o{}_enc{}_dec{}_train_epochs{}_train_cars{}'.format(
        settings["input_len"], settings["output_len"], settings["model"]['encoder_layers'],settings["model"]['decoder_layers'],
        settings["train_epochs"],
        settings["train_cars"]
    )
    start_train_time = time.time()
    end_train_time = start_train_time
    start_time = start_train_time

    exp = Experiment(settings)
    print('\n>>>>>>> Training  >>>>>>>>>>>>>>>>>>>>>>>>>>\n')
    config = prep_env_tsf()
    size = [config.input_len, config.output_len]
    model = WPFModel(config=config)
    val_loss_data=train_and_val(exp, model=model, model_folder=cur_setup,is_debug=settings["is_debug"])
    paddle.device.cuda.empty_cache()
    if settings["is_debug"]:
        end_time = time.time()
        print("\nTraining  in {} secs".format( end_time - start_time))
        start_time = end_time
        end_train_time = end_time
    if settings["is_debug"]:
        print("\nTotal time in training  "
              "{} secs".format(end_train_time - start_train_time))
    print(val_loss_data)

    print('\n>>>>>>>Pridicting>>>>>>>>>>>>>>>>>>>>>>>>\n')
    dataset = predict(model,config)