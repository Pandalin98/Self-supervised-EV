import numpy as np
import pandas as pd
import os
import torch
from torch import nn
from torch.cuda import memory_allocated
from src.models.NervFormer import NervFormer
from src.learner import Learner, transfer_weights
from src.callback.tracking import *
from src.callback.patch_mask import *
from src.callback.transforms import *
from src.metrics import *
from src.metrics import rmse,mae,REP
from src.basics import set_device,set_seed
from src.datautils import *

import datetime 
import argparse
import joblib
from src.callback.patch_mask import create_patch
from src.power_battery import PowerBatteryData



class Args:
    def __init__(self):
        self.device = 0
        self.is_pretrain = 0
        self.is_finetune = 0
        self.is_linear_probe = 1
        self.explain = 0
        self.test = True
        self.predict = True
        self.project_name = 'power_battery'
        self.dset_pretrain = 'Power-Battery'
        self.dset_finetune = 'Power-Battery'
        self.data_path = './data/local_data_structure'
        self.batch_size = 512
        self.num_workers = 0
        self.scale = True
        self.dist = False
        self.revin = 0
        self.n_layers = 3
        self.n_layers_dec = 1
        self.prior_dim = 6
        self.n_heads = 16
        self.d_model = 512
        self.dropout = 0.15
        self.head_dropout = 0.05
        self.input_len = 96
        self.output_len = 192
        self.patch_len = 1000
        self.stride = True
        self.stride_ratio = 1.0
        self.mask_ratio = 0.4
        self.recon_weight = 0.5
        self.kl_temperature = 0.1
        self.n_epochs_pretrain = 300
        self.lr = 1e-4
        self.n_epochs_finetune = 200
        self.head_epochs_ratio = 0.05
        self.pretrained_model_id = 1
        self.model_type = 'based_model'
        self.finetuned_model_id = 1

args = Args()
args.stride = int(args.patch_len * args.stride_ratio)
args.dist = False
args.pretrained_model = 'patchtst_pretrained_dataset'+str(args.dset_pretrain)+'_patch'+str(args.patch_len) + '_stride'+str(args.stride) + '_epochs-pretrain' + str(args.n_epochs_pretrain) + '_mask' + str(args.mask_ratio)  + '_model' + str(args.pretrained_model_id)
args.save_path = 'saved_models/' + args.dset_pretrain + '/masked_patchtst/' + args.model_type + '/'
if not os.path.exists(args.save_path): os.makedirs(args.save_path)

time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
# args.save_finetuned_model = '_cw'+str(args.context_points)+'_tw'+str(args.output_len) + '_patch'+str(args.patch_len) + '_stride'+str(args.stride) + '_epochs-finetune' + str(args.n_epochs_finetune) + '_mask' + str(args.mask_ratio)  + '_model' + str(args.finetuned_model_id)
suffix_name = '_time'+str(time)+'_ol'+str(args.output_len) + '_patch'+str(args.patch_len) + '_stride'+str(args.stride) + '_epochs-finetune' + str(args.n_epochs_finetune) + '_model' + str(args.finetuned_model_id)+ '_n_layer'+str(args.n_layers)+'_n_dec'+str(args.n_layers_dec)+'_n_head'+str(args.n_heads)+'_d_model'+str(args.d_model)+'_dropout'+str(args.dropout)+'_head_dropout'+str(args.head_dropout)
args.d_ff = 4 * args.d_model
args.save_pretrained_model = args.pretrained_model
args.save_linear_probe_model = 'linear_probe' + suffix_name
args.save_finetuned_model = 'finetuned' + suffix_name
print('args:', args)
# ... rest of your code ...
def get_model(c_in, head_type,args):
    """
    c_in: number of variables
    """
    # get number of patches
    
    # get modelxb, yb
    model = NervFormer(c_in=c_in,head_type=head_type,
                target_dim=args.output_len,
                patch_len=args.patch_len,
                stride=args.stride,
                n_layers=args.n_layers,
                n_layers_dec=args.n_layers_dec,
                n_heads=args.n_heads,
                d_model=args.d_model,
                shared_embedding=True,
                d_ff=args.d_ff,                        
                dropout=args.dropout,
                head_dropout=args.head_dropout,
                act='relu',
                res_attention=False,
                prior_dim=args.prior_dim,
                input_len=args.input_len,
                output_len=args.output_len,
                output_representation=args.output_representation           )        
    # print out the model size
    print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model


input_len = args.input_len
finetune_strie_ratio = args.stride_ratio
args.dset = args.dset_pretrain
args.initialize_wandb = False
args.output_representation = False
head_type = 'prior_pooler'

model = get_model(19, head_type, args)
model = transfer_weights("saved_models/Power-Battery/masked_patchtst/based_model/linear_probe_time2024-03-03-03-52-01_ol192_patch1000_stride1000_epochs-finetune200_model1_n_layer3_n_dec1_n_head16_d_model512_dropout0.2_head_dropout0.05.pth",model)
## 模型进入eval模式
# model.to(args.device)
model.eval()

def get_dataset_dict(args):
    dataset_dict = {}
    if args.task_flag =='pretrain':
        dataset_dict['pretrain'] = PowerBatteryData(**{
                'data_path': args.data_path,
                'scale': args.scale,
                'size':[args.input_len,args.output_len],
                }, split='pretrain')
    if args.task_flag == 'finetune' or args.task_flag == 'linear_probe':
        "make dataset: {}\n".format(args.task_flag)
        dataset_dict['train'] = PowerBatteryData(**{
                'data_path': args.data_path,
                'scale': args.scale,
                'size':[args.input_len,args.output_len],
                }, split='train')
        dataset_dict['val'] =  PowerBatteryData(**{
                'data_path': args.data_path,
                'scale': args.scale,
                'size':[args.input_len,args.output_len],
                }, split='val')
        dataset_dict['test'] =  PowerBatteryData(**{
                'data_path': args.data_path,
                'scale': args.scale,
                'size':[args.input_len,args.output_len],
                }, split='test')
    if args.task_flag == 'test':
        dataset_dict['test'] = PowerBatteryData(**{
                'data_path': args.data_path,
                'scale': args.scale,
                'size':[args.input_len,args.output_len],
                }, split='test')
    return dataset_dict

args.task_flag = 'pretrain'

dataset_dict = get_dataset_dict(args)
dls = get_dls(args,dataset_dict) 

model.debug = True
##
def set_patch(xb,args):
    """
    take xb from learner and convert to patch: [bs x seq_len × nvars] -> [bs x num_patch x patch_len*nvars]
    """
    bs,cl,tl,fl = xb.shape
    xb = xb.reshape(bs*cl,tl,fl)
    xb_patch, num_patch = create_patch(xb, args.patch_len, args.stride)    # xb: [bs x seq_len × nvars ]
    xb = xb_patch.reshape(bs,cl,num_patch,-1)     
    return xb  
pred_list = []
label_list = []
capcacity_list = []
feature_list = []
with torch.no_grad():
    for i, batch_dict in enumerate(dls.train):
        encoder_input,label,prior,encoder_mark,decoder_input,decoder_mark = batch_dict['encoder_input'],batch_dict['label'],batch_dict['prior'],batch_dict['encoder_mark'],batch_dict['decoder_input'],batch_dict['decoder_mark']
        endocer_input = set_patch(encoder_input,args)
        # encoder_input = encoder_input.to(args.device)
        # prior = prior.to(args.device)
        # encoder_mark = encoder_mark.to(args.device)
        # decoder_input = decoder_input.to(args.device)
        # decoder_mark = decoder_mark.to(args.device)
        preds = model(endocer_input,prior,decoder_input,encoder_mark,decoder_mark)
        #得到表征
        pred = preds[0]
        feature = preds[1].numpy()
        capcacity = preds[2].numpy()
        pred_list.append(pred)
        label_list.append(label)
        capcacity_list.append(capcacity)
        feature_list.append(feature)

pred_data = np.concatenate(pred_list,axis=0)
label_data = np.concatenate(label_list,axis=0)
capcacity_data = np.concatenate(capcacity_list,axis=0)
feature_data = np.concatenate(feature_list,axis=0)


import matplotlib.pyplot as plt

input_len = 96
# 计算每行的误差
error_test = np.abs(pred_data - label_data)

# 计算每行的最大误差
max_error = np.mean(error_test, axis=1)

# 找到最大误差最小的10个行索引
min_max_error_row_indices = np.argsort(max_error)[-399:-319]

# 对每个最小误差的行进行绘图
for i, min_max_error_row_index in enumerate(min_max_error_row_indices, 1):
    # 获取最大误差最小的一行数组
    min_error_row_pred = pred_data[min_max_error_row_index]
    min_error_row_true = label_data[min_max_error_row_index]
    min_error_capcacity = capcacity_data[min_max_error_row_index]
    min_error_feature = feature_data[min_max_error_row_index]
    
    # 创建一个新的图形
    plt.figure()
    # plt.plot(range(input_len),input, 'black', label='Past values')
    # 绘制区间预测
    plt.plot(range(len(min_error_row_true)),min_error_row_pred, 'b-', label='Predicted')
    # 绘制实际观测值
    plt.plot(range(len(min_error_row_true)),min_error_row_true, 'r-', label='True values')
    # plt.plot(range(len(min_error_row_true)),min_error_capcacity, 'g-', label='Capcacity')
    # plt.plot(range(len(min_error_row_true)),min_error_feature, 'y-', label='Feature')   
    
    # plt.fill_between(range(len(min_error_row_true)),min_error_row_pred- quantile_lower, min_error_row_pred+quantile_lower, color='gray', alpha=0.5, label='Predicted interval')
    
    # 添加图例
    plt.legend()
    
    # 显示图形
    plt.show()