

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
parser.add_argument('--is_finetune', type=int, default=1, help='do finetuning or not')
parser.add_argument('--is_linear_probe', type=int, default=0, help='if linear_probe: only finetune the last layer')
parser.add_argument('--test',type=bool,default=False,help='fest model identification')
parser.add_argument('--predcit',type=bool,default=True,help='predict')
parser.add_argument('--project_name',type=str,default='power_battery',help='project name')
# Dataset and dataloader
parser.add_argument('--dset_pretrain', type=str, default='Power-Battery', help='pretrain dataset name')
parser.add_argument('--dset_finetune', type=str, default='Power-Battery', help='finetune dataset name')
parser.add_argument('--data_path', type=str, default='./data/local_data_structure', help='data path')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--num_workers', type=int, default=8, help='number of workers for DataLoader')
parser.add_argument('--scale', type=str, default=None, help='scale the input data')
parser.add_argument('--dist',type=bool,default=False,help='distrubuted training')
# Patch
parser.add_argument('--patch_len', type=int, default=1000, help='patch length')
parser.add_argument('--stride', type=int, default=None, help='stride between patch')
parser.add_argument('--stride_ratio', type=float, default=1.25, help='stride between patch')
# RevIN
parser.add_argument('--revin', type=int, default=0, help='reversible instance normalization')

# Model args
parser.add_argument('--n_layers', type=int, default=4, help='number of Transformer layers')
parser.add_argument('--n_layers_dec', type=int, default=1, help='Transformer d_ff')
parser.add_argument('--prior_dim', type=int, default=1, help='dim of prior information')
parser.add_argument('--n_heads', type=int, default=16, help='number of Transformer heads')
parser.add_argument('--d_model', type=int, default=768, help='Transformer d_model')
parser.add_argument('--dropout', type=float, default=0.15, help='Transformer dropout')
parser.add_argument('--head_dropout', type=float, default=0.3, help='head dropout')
parser.add_argument('--input_len', type=int, default=32, help='input time series length')
parser.add_argument('--output_len', type=int, default=3, help='output dimension')
#head args

# Pretrain task
parser.add_argument('--mask_ratio', type=float, default=0.4, help='masking ratio for the input')
parser.add_argument('--recon_weight', type=float, default=0.7, help='input dimension')
parser.add_argument('--kl_temperature', type=float, default=0.01, help='input dimension')
# Optimization args
parser.add_argument('--n_epochs_pretrain', type=int, default=200, help='number of pre-training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--n_epochs_finetune', type=int, default=50, help='number of finetuning epochs')
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
args.save_pretrained_model = args.pretrained_model


# args.save_finetuned_model = '_cw'+str(args.context_points)+'_tw'+str(args.output_len) + '_patch'+str(args.patch_len) + '_stride'+str(args.stride) + '_epochs-finetune' + str(args.n_epochs_finetune) + '_mask' + str(args.mask_ratio)  + '_model' + str(args.finetuned_model_id)
suffix_name = '_tw'+str(args.output_len) + '_patch'+str(args.patch_len) + '_stride'+str(args.stride) + '_epochs-finetune' + str(args.n_epochs_finetune) + '_model' + str(args.finetuned_model_id)
args.save_finetuned_model = args.dset_finetune+'_patchtst_finetuned'+suffix_name
if args.is_linear_probe: args.save_linear_probe_model = args.dset_finetune+'_patchtst_linear-probe'+suffix_name
args.d_ff = 4 * args.d_model
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
                )        
    # print out the model size
    print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model


def find_lr(head_type='pretrain'):
    # get dataloader
    dls = get_dls(args)    
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
    return suggested_lr  


def pretrain_func(lr=args.lr):
    # get dataloader
    dls = get_dls(args)
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


def finetune_func(lr=args.lr,head_type='regression'):
    print('end-to-end finetuning')
    # get dataloader
    dls = get_dls(args)
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


def linear_probe_func(lr=args.lr,head_type='regression'):
    print('linear probing')
    # get dataloader
    dls = get_dls(args)
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


if __name__ == '__main__':
    ##保持预训练stride_ratio=1，finetune时再恢复
    finetune_strie_ratio = args.stride_ratio
    args.dset = args.dset_pretrain
    if args.is_pretrain:
        args.stride_ratio = 1
        args.task_flag = 'pretrain'
        # suggested_lr = 1e-4

        suggested_lr = find_lr()
        # Pretrain
        pretrain_func(suggested_lr)
        print('pretraining completed')
    
    if args.is_finetune:
        # args.dset = args.dset_finetune
        # Finetune
        args.stride_ratio = finetune_strie_ratio
        args.task_flag = 'finetune'
        head_type = 'prior_pooler'
        suggested_lr = find_lr(head_type=head_type)        
        # suggested_lr = 1e-4
        model = finetune_func(suggested_lr,head_type=head_type)        
        print('finetune completed')
        # # Test
        # out = test_func(model=model)         
        # print('----------- Complete! -----------')

    if args.is_linear_probe:
        args.stride_ratio = finetune_strie_ratio
        # args.dset = args.dset_finetune
        # Finetune
        args.task_flag = 'linear_probe'
        head_type = 'prior_pooler'
        suggested_lr = find_lr(head_type=head_type)        
        moedl_1 =  linear_probe_func(suggested_lr)        
        print('linear_probe completed')
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
                                    predict_input=predict_input)
        car_id = list(predict_input.car_id.unique())
        target_scaler = joblib.load('./data/target_scaler.pkl')
        try :
            model_path = args.save_path+args.save_finetuned_model+ '.pth'
            model = get_model(19, 'prior_pooler', args)
            model.load_state_dict(torch.load(model_path))
            model.to(args.device)
            model.eval()
            print('成功加载finetuned模型')
        except:
            moedl = model_1
        model.eval()


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
            encoder_input = torch.unsqueeze(encoder_input, axis=0).float().to(args.device)
            encoder_mark = torch.unsqueeze(encoder_mark, axis=0).float().to(args.device)
            prior = torch.unsqueeze(prior, axis=0).float().to(args.device)
            decoder_input = torch.unsqueeze(decoder_input, axis=0).float().to(args.device)

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