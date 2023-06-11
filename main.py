

import numpy as np
import pandas as pd
import os
import torch
from torch import nn

from src.models.fusing_patchTST import Fusing_PatchTST
from src.learner import Learner, transfer_weights
from src.callback.tracking import *
from src.callback.patch_mask import *
from src.callback.transforms import *
from src.metrics import *
from src.basics import set_device,set_seed
from datautils import *
# import multiprocessing as mp
# mp.set_start_method('spawn', force=True)
import datetime
import argparse
import joblib
from src.callback.patch_mask import create_patch
#设计随机种子



parser = argparse.ArgumentParser()

parser.add_argument('--device',type=int,default=0,help='device id')
# Pretraining and Finetuning
parser.add_argument('--is_pretrain', type=int, default=1, help='do pretraining or not')
parser.add_argument('--is_finetune', type=int, default=0, help='do finetuning or not')
parser.add_argument('--is_linear_probe', type=int, default=1, help='if linear_probe: only finetune the last layer')
parser.add_argument('--test',type=bool,default=False,help='fest model identification')
parser.add_argument('--predcit',type=bool,default=True,help='predict')
parser.add_argument('--project_name',type=str,default='power_battery',help='project name')
# Dataset and dataloader
parser.add_argument('--dset_pretrain', type=str, default='Power-Battery', help='pretrain dataset name')
parser.add_argument('--dset_finetune', type=str, default='Power-Battery', help='finetune dataset name')
parser.add_argument('--data_path', type=str, default='./data/local_data_structure', help='data path')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_workers', type=int, default=8, help='number of workers for DataLoader')
parser.add_argument('--scale', type=str, default=None, help='scale the input data')
parser.add_argument('--dist',type=bool,default=False,help='distrubuted training')
# Patch
parser.add_argument('--patch_len', type=int, default=500, help='patch length')
parser.add_argument('--stride', type=int, default=True, help='stride between patch')
parser.add_argument('--stride_ratio', type=float, default=1.5, help='stride between patch')
# RevIN
parser.add_argument('--revin', type=int, default=0, help='reversible instance normalization')
# Model args
parser.add_argument('--n_layers', type=int, default=5, help='number of Transformer layers')
parser.add_argument('--n_layers_dec', type=int, default=3, help='Transformer d_ff')
parser.add_argument('--prior_dim', type=int, default=1, help='dim of prior information')
parser.add_argument('--n_heads', type=int, default=16, help='number of Transformer heads')
parser.add_argument('--d_model', type=int, default=1024, help='Transformer d_model')
parser.add_argument('--dropout', type=float, default=0.15, help='Transformer dropout')
parser.add_argument('--head_dropout', type=float, default=0.2, help='head dropout')
parser.add_argument('--input_len', type=int, default=16, help='input time series length')
parser.add_argument('--output_len', type=int, default=3, help='output dimension')
#head args
# Pretrain task
parser.add_argument('--mask_ratio', type=float, default=0.4, help='masking ratio for the input')
parser.add_argument('--recon_weight', type=float, default=0.7, help='input dimension')
parser.add_argument('--kl_temperature', type=float, default=0.01, help='input dimension')
# Optimization args
parser.add_argument('--n_epochs_pretrain', type=int, default=10, help='number of pre-training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--n_epochs_finetune', type=int, default=10, help='number of finetuning epochs')
parser.add_argument('--head_epochs_ratio',type=float,default=0.2,help='ratio of head epochs')
# model id to keep track of the number of models saved
parser.add_argument('--pretrained_model_id', type=int, default=1, help='id of the saved pretrained model')
parser.add_argument('--model_type', type=str, default='based_model', help='for multivariate model or univariate model')
parser.add_argument('--finetuned_model_id', type=int, default=1, help='id of the saved finetuned model')
# parser.add_argument('--pretrained_model', type=str, default='saved_models/CALCE_Battery/masked_patchtst/based_model/patchtst_pretrained_datasetCALCE_Battary_patch12_stride3_epochs-pretrain200_mask0.4_model1.pth', 
#                     help='path of the pretrained model')



args = parser.parse_args()
args.stride = int(args.patch_len * args.stride_ratio)

print('args:', args)
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
# get available GPU devide
if not args.dist:
    set_device(device_id=args.device)
set_seed(3407)

def get_model(c_in, head_type,args):
    """
    c_in: number of variables
    """
    # get number of patches
    
    # get modelxb, yb
    model = Fusing_PatchTST(c_in=c_in,head_type=head_type,
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
                )        
    # print out the model size
    print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model


def find_lr(dls,head_type='pretrain'):
    # get dataloader
    model = get_model(dls.vars, head_type,args)
    if args.task_flag != 'pretrain':
        # weight_path = args.save_path + args.pretrained_model + '.pth'
        model = transfer_weights(args.save_path+args.save_pretrained_model+ '.pth', model)
    # get loss
    loss_func = torch.nn.MSELoss(reduction='mean')    
    # get callbacks
    if args.task_flag == 'pretrain':
        cbs = [PatchMaskCB(patch_len=args.patch_len, stride=args.stride, mask_ratio=args.mask_ratio)]
    else:
        cbs = [PatchCB(patch_len=args.patch_len, stride=args.stride)]
    loss_func = torch.nn.MSELoss(reduction='mean')
    # define learner
    learn = Learner(dls, model, 
                        lr=args.lr, 
                        cbs=cbs,
                        flag = args.task_flag,
                        args = args
                        )                        
    # fit the data to the model
    suggested_lr = learn.lr_finder(args.task_flag,loss_func = loss_func)
    print('suggested_lr', suggested_lr)
    torch.cuda.empty_cache()
    return suggested_lr  


def pretrain_func(dls,lr=args.lr):
    # get model     
    model = get_model(dls.vars, 'pretrain',args)
    # get loss
    # get callbacks
    #如果arg.save_pretrained_model为空，则创建模型

    cbs = [
         PatchMaskCB(patch_len=args.patch_len, stride=args.stride, mask_ratio=args.mask_ratio),
         SaveModelCB(monitor='train_loss', fname=args.save_pretrained_model,                       
                        path=args.save_path)
        ]
                        
    #把namespace转换成dict
    # define learner
    learn = Learner(dls, model, 
                        lr=lr, 
                        cbs=cbs,flag=args.task_flag
                        ,args = args
                        #metrics=[mse]
                        )                        
    # fit the data to the model
    learn.fit_pretrain(n_epochs=args.n_epochs_pretrain, lr_max=lr)

    train_loss = learn.recorder['train_loss']
    df = pd.DataFrame(data={'train_loss': train_loss})
    df.to_csv(args.save_path + args.save_pretrained_model + '_losses.csv', float_format='%.6f', index=False)

def save_recorders(learn,save_path):
    train_loss = learn.recorder['train_loss']
    valid_loss = learn.recorder['valid_loss']
    df = pd.DataFrame(data={'train_loss': train_loss, 'valid_loss': valid_loss})
    df.to_csv(args.save_path + save_path + '_losses.csv', float_format='%.6f', index=False)


def finetune_func(dls,lr=args.lr,head_type='regression'):
    print('end-to-end finetuning')

    # get model 
    model = get_model(dls.vars, head_type, args)
    # transfer weight
    # weight_path = args.pretrained_model + '.pth'
    model =transfer_weights(args.save_path+args.save_pretrained_model+ '.pth', model)

    loss_func = torch.nn.MSELoss(reduction='mean')

    # get loss
    # get callbacks
    cbs = [
                 PatchCB(patch_len=args.patch_len, stride=args.stride),
         SaveModelCB(monitor='valid_loss', fname=args.save_finetuned_model, path=args.save_path)
        ]
    # define learner
    learn = Learner(dls, model, 
                        lr=lr, 
                        cbs=cbs,
                         args = args,
                        metrics=[rmse],flag=args.task_flag
                        )                            
    # fit the data to the model
    #learn.fit_one_cycle(n_epochs=args.n_epochs_finetune, lr_max=lr)
    
    learn.fine_tune(n_epochs=args.n_epochs_finetune, base_lr=lr, freeze_epochs=10,loss_func=loss_func)
    save_recorders(learn,save_path=args.save_finetuned_model)
    return learn.model


def linear_probe_func(dls,lr=args.lr,head_type='regression'):
    print('linear probing')
    # get dataloader
    # get model 
    model = get_model(dls.vars, head_type, args)
    # transfer weight
    # weight_path = args.save_path + args.pretrained_model + '.pth'
    model = transfer_weights(args.save_path+args.pretrained_model+ '.pth', model)
    loss_func = torch.nn.MSELoss(reduction='mean')
    # get loss
    # get callbacks
    cbs = [
         PatchCB(patch_len=args.patch_len, stride=args.stride),
         SaveModelCB(monitor='valid_loss', fname=args.save_linear_probe_model, path=args.save_path)
        ]
    # define learner
    learn = Learner(dls, model, 
                        lr=lr, 
                        cbs=cbs, args = args,
                        metrics=[rmse],flag=args.task_flag, 
                        )                            
    # fit the data to the model
    learn.linear_probe(n_epochs=args.n_epochs_finetune, base_lr=lr,loss_func=loss_func)
    save_recorders(learn,save_path=args.save_linear_probe_model)
    return learn.model


def test_func(model,dls):
    # get callbacks
    args.test = True
    cbs = [PatchCB(patch_len=args.patch_len, stride=args.stride)]
    learn = Learner(dls, model,cbs=cbs,flag=args.task_flag,args=args)
    score = None
    out  = learn.test(dls.test,model, scores=score) 
    if score is None:
            # out: a list of [pred, targ, score]
        print('pred:', out[0])
        print('targ:', out[1])
    else:
        print('score:', out[2])

    # save results
    #读取当前时间，存为str
    now = datetime.datetime.now()
    now = now.strftime('%Y-%m-%d-%H-%M-%S')
    pd.DataFrame(np.array(out[2]).reshape(1,-1), columns=['rmse','mae']).to_csv(now + '_acc.csv', float_format='%.6f', index=False)
    
    return out

def cal_gpu(module):
    if isinstance(module, torch.nn.DataParallel):
        module = module.module
    for submodule in module.children():
        if hasattr(submodule, "_parameters"):
            parameters = submodule._parameters
            if "weight" in parameters:
                return parameters["weight"].device

if __name__ == '__main__':
    ##保持预训练stride_ratio=1，finetune时再恢复
    finetune_strie_ratio = args.stride_ratio
    args.dset = args.dset_pretrain
    if args.is_pretrain:
        args.stride_ratio = 1
        args.task_flag = 'pretrain'
        args.batch_size = int(32*32/args.input_len)
        # suggested_lr = 1e-4
        # get dataloader
        dls = get_dls(args)    
        suggested_lr = find_lr(dls)
        # Pretrain
        pretrain_func(dls,suggested_lr)
        print('pretraining completed')
        del dls
    
    if args.is_finetune:
        # args.dset = args.dset_finetune
        # Finetune
        args.stride_ratio = finetune_strie_ratio
        args.task_flag = 'finetune'
        head_type = 'prior_pooler'
        # get dataloader
        dls = get_dls(args)
        suggested_lr = find_lr(dls,head_type=head_type)        
        # suggested_lr = 1e-4
        model=finetune_func(dls,suggested_lr,head_type=head_type)        
        print('finetune completed')
        # # Test
        # out = test_func(model=model)         
        # print('----------- Complete! -----------')
        del dls

    if args.is_linear_probe:
        print('begin linear_probe')
        args.stride_ratio = finetune_strie_ratio
        args.batch_size = 4*args.batch_size
        # args.dset = args.dset_finetune
        # Finetune
        args.task_flag = 'linear_probe'
        head_type = 'prior_pooler'

        dls = get_dls(args)
        suggested_lr = find_lr(dls,head_type=head_type)        
        model =  linear_probe_func(dls,suggested_lr,head_type=head_type)        
        print('linear_probe completed')
        del dls
        # # Test
        # out = test_func(model)        
        # print('----------- Complete! -----------')


    if args.predcit:
        # Test
        predict_input_orin =pd.read_csv('./data/result.csv')
        # 重命名列名
        predict_input = predict_input_orin.rename(columns={
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
        predict_input['weekday'] = predict_input_index.weekday
        predict_input['day'] = predict_input_index.day
        predict_input['hour'] = predict_input_index.hour
        data_set = PowerBatteryData(size=[args.input_len, args.output_len],data_path='./data/test_data_structure',
                                    scale = True,
                                    split='predict',
                                    predict_input=predict_input,
                                    sort = True)
        car_id = list(predict_input.car_id.unique())
        target_scaler = joblib.load('./data/target_scaler.pkl')
        try :
            moedl = model
        except:
            if args.is_finetune:
                model_path = args.save_path+args.save_finetuned_model+ '.pth'
            if args.is_linear_probe:
                model_path = args.save_path+args.save_linear_probe_model+ '.pth'
            # model_path = "saved_models/Power-Battery/masked_patchtst/based_model/linear_probe_time2023-06-10-14-30-32_ol24_patch1000_stride1250_epochs-finetune10_model1_n_layer4_n_dec1_n_head16_d_model768_dropout0.15_head_dropout1.pth"
            model = get_model(19, 'prior_pooler', args)
            # model_path ="saved_models/Power-Battery/masked_patchtst/based_model/linear_probe_time2023-06-10-16-07-57_ol24_patch500_stride750_epochs-finetune400_model1_n_layer5_n_dec3_n_head16_d_model1024_dropout0.15_head_dropout0.2.pth"
            model.load_state_dict(torch.load(model_path))
            model.to(args.device)
            model.eval()
        print('成功加载finetuned模型')
            
        model.eval()
        device = cal_gpu(model)

        predict_ah = []
        for i in range(len(data_set)):
            dict = data_set.__getitem__(i)
            encoder_input = dict['encoder_input']
            decoder_input = dict['decoder_input']
            encoder_mark = dict['encoder_mark']
            prior = dict['prior']
            encoder_input = torch.tensor(encoder_input)
            encoder_mark = torch.tensor(encoder_mark)
            decoder_input = torch.tensor(decoder_input)
            prior = torch.tensor(prior)
            encoder_input,_ = create_patch(encoder_input, args.patch_len, args.stride)
            encoder_input = torch.unsqueeze(encoder_input, axis=0).float().to(device)
            encoder_mark = torch.unsqueeze(encoder_mark, axis=0).float().to(device)
            prior = torch.unsqueeze(prior, axis=0).float().to(device)
            decoder_input = torch.unsqueeze(decoder_input, axis=0).float().to(device)

            #深度学习模型进入eval模式
            sample = model(encoder_input,prior, decoder_input,encoder_mark)
            sample = sample.detach().cpu().numpy()
            sample = target_scaler.inverse_transform(sample.reshape(-1,1))
            predict_ah.append(sample)
        
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>预测值为：>>>>>>>>>>>>>>>>>>>>>>>>>>')
        #predict_ah 的元素内可能包含多个值，将这些值全部展平，并拼成一维数组
        for i in range(len(predict_ah)):
            print(predict_ah[i])
        # ##将预测值写入csv文件
        # flatten_predict_ah = np.array(predict_ah).flatten()
        # predict_input_orin['估计的充电量'] = flatten_predict_ah
        # #读取当前时间，存为str
        # now = datetime.datetime.now()
        # predict_input_orin.to_csv('./data/result'+now+'.csv',index=False)
        # print('>>>>>>>>>>>>>>>>>>>>>>>>>>预测值写入csv文件完成>>>>>>>>>>>>>>>>>>>>>>>>>>'+now)