
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
from src.callback.scheduler import LRFinderCB
from src.metrics import *
from src.metrics import rmse,mae,REP
from src.basics import set_device,set_seed
from src.datautils import *
import datetime 
import argparse
import joblib
from src.callback.patch_mask import create_patch
from src.power_battery import PowerBatteryData
# from src.explain import explain_funciton
#设计随机种子
set_seed(42)

parser = argparse.ArgumentParser()

parser.add_argument('--device',type=int,default=0,help='device id')
# Pretraining and Finetuning
parser.add_argument('--is_pretrain', type=int, default=0, help='do pretraining or not')
parser.add_argument('--is_finetune', type=int, default=0, help='do finetuning or not')
parser.add_argument('--is_linear_probe', type=int, default=1, help='if linear_probe: only finetune the last layer')
parser.add_argument('--explain', type=int, default=0, help='explain the model')
parser.add_argument('--test',type=bool,default=True,help='fest model identification')
parser.add_argument('--predcit',type=bool,default=True,help='predict')
parser.add_argument('--project_name',type=str,default='power_battery',help='project name')
# Dataset and dataloader
parser.add_argument('--dset_pretrain', type=str, default='Power-Battery', help='pretrain dataset name')
parser.add_argument('--dset_finetune', type=str, default='Power-Battery', help='finetune dataset name')
parser.add_argument('--data_path', type=str, default='./data/local_data_structure', help='data path')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for DataLoader')
parser.add_argument('--scale', type=str, default=True, help='scale the input data')
parser.add_argument('--dist',type=bool,default=False,help='distrubuted training')
# RevIN
parser.add_argument('--revin', type=int, default=0, help='reversible instance normalization')
# Model args
parser.add_argument('--n_layers', type=int, default=3, help='number of Transformer layers')
parser.add_argument('--n_layers_dec', type=int, default=1, help='Transformer d_ff')
parser.add_argument('--prior_dim', type=int, default=6, help='dim of prior information')
parser.add_argument('--n_heads', type=int, default=16, help='number of Transformer heads')
parser.add_argument('--d_model', type=int, default=512, help='Transformer d_model')
parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
parser.add_argument('--head_dropout', type=float, default=0.05, help='head dropout')
parser.add_argument('--input_len', type=int, default=96, help='input time series length')
parser.add_argument('--output_len', type=int, default=192, help='output time series length')
# Patch
parser.add_argument('--patch_len', type=int, default=1000, help='patch length')
parser.add_argument('--stride', type=int, default=1000, help='stride between patch')
parser.add_argument('--stride_ratio', type=float, default=1.0, help='stride between patch')#head args
# Pretrain task
parser.add_argument('--mask_ratio', type=float, default=0.5, help='masking ratio for the input')
parser.add_argument('--recon_weight', type=float, default=0.3, help='input dimension')
parser.add_argument('--kl_temperature', type=float, default=0.5, help='input dimension')
# Optimization args
parser.add_argument('--n_epochs_pretrain', type=int, default=400, help='number of pre-training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--n_epochs_finetune', type=int, default=400, help='number of finetuning epochs')
parser.add_argument('--head_epochs_ratio',type=float,default=0.05,help='ratio of head epochs')
# model id to keep track of the number of models saved
parser.add_argument('--pretrained_model_id', type=int, default=1, help='id of the saved pretrained model')
parser.add_argument('--model_type', type=str, default='based_model', help='for multivariate model or univariate model')
parser.add_argument('--finetuned_model_id', type=int, default=1, help='id of the saved finetuned model')
# parser.add_argument('--pretrained_model', type=str, default='saved_models/CALCE_Battery/masked_patchtst/based_model/patchtst_pretrained_datasetCALCE_Battary_patch12_stride3_epochs-pretrain200_mask0.4_model1.pth', 
#                     help='path of the pretrained model')
# parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')



args = parser.parse_args()
args.stride = int(args.patch_len * args.stride_ratio)
args.dist = False
print('args:', args)
args.pretrained_model = 'patchtst_pretrained_dataset'+str(args.dset_pretrain)+'_patch'+str(args.patch_len) + '_stride'+str(args.stride) + '_epochs-pretrain' + str(args.n_epochs_pretrain) + '_mask' + str(args.mask_ratio)+'_kl'+str(args.kl_temperature) + '_recon_weight'+str(args.recon_weight)+'_model' + str(args.pretrained_model_id)
args.save_path = 'saved_models/' + args.dset_pretrain + '/masked_patchtst/' + args.model_type +'/'
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
                )        
    # print out the model size
    print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model


def find_lr(dls,head_type='pretrain',find_bs=False):
    
    # get dataloader
    model = get_model(dls.vars, args.head_type,args)
    if args.task_flag != 'pretrain':
        ##TODO 已经修改
        # args.save_pretrained_model ="patchtst_pretrained_datasetPower-Battery_patch500_stride500_epochs-pretrain100_mask0.1_model1"
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
    # find the best learning rate
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
                        path=args.save_path),
         EarlyStoppingCB(monitor='train_loss',patient=20)
        ]
                        
    #把namespace转换成dict
    # define learner
    learn = Learner(dls, model, 
                        lr=lr, 
                        cbs=cbs,flag=args.task_flag
                        ,args = args
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


def finetune_func(dls,lr=args.lr):
    print('end-to-end finetuning')
    head_type = args.head_type
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
         SaveModelCB(monitor='valid_loss', fname=args.save_finetuned_model, path=args.save_path),
                  EarlyStoppingCB(monitor='valid_loss',patient=20)

        ]
    # define learner
    learn = Learner(dls, model, 
                        lr=lr, 
                        cbs=cbs,
                         args = args,
                        metrics=[rmse,mae,REP],flag=args.task_flag
                        )                            
    # fit the data to the model
    #learn.fit_one_cycle(n_epochs=args.n_epochs_finetune, lr_max=lr)
    
    learn.fine_tune(n_epochs=args.n_epochs_finetune, base_lr=lr, freeze_epochs=10,loss_func=loss_func)
    save_recorders(learn,save_path=args.save_finetuned_model)
    return learn.model


def linear_probe_func(dls,lr=args.lr):
    print('linear probing')
    # get dataloader
    # get model 
    head_type = args.head_type
    model = get_model(dls.vars, head_type, args)
    # transfer weight
    # weight_path = args.save_path + args.pretrained_model + '.pth'
    # model = transfer_weights(args.save_path+args.pretrained_model+ '.pth', model)
    ##TODO 已经修改
    # model_path ="patchtst_pretrained_datasetPower-Battery_patch500_stride500_epochs-pretrain100_mask0.1_model1"
    model = transfer_weights(args.save_path+args.pretrained_model+'.pth',model)
    loss_func = torch.nn.MSELoss(reduction='mean')
    # get loss
    # get callbacks
    if dls.valid is not None:
        monitor  = 'test_loss'
    else:
        monitor = 'train_loss'
    cbs = [
         PatchCB(patch_len=args.patch_len, stride=args.stride),
         SaveModelCB(monitor=monitor, fname=args.save_linear_probe_model, path=args.save_path),
         EarlyStoppingCB(monitor=monitor,patient=20)
        ]
    # define learner
    learn = Learner(dls, model, 
                        lr=lr, 
                        cbs=cbs, args = args,
                        metrics=[rmse,mae,REP],flag=args.task_flag, 
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
    score = [rmse,mae,REP]
    out  = learn.test(dls.test,model, scores=score) 
    print('test rmse:{} mae:{} REP:{}'.format(out[2][0],out[2][1],out[2][2]))

    #读取当前时间，存为str
    now = datetime.datetime.now()
    now = now.strftime('%Y-%m-%d-%H-%M-%S')
    #保存预测结果
    pred = out[0]
    true = out[1]
    scaler = joblib.load('./data/target_scaler.pkl')
    pred = scaler.inverse_transform(pred)
    true = scaler.inverse_transform(true)
    out_len = args.output_len
    #保存预测结果为dict文件
    np.save(now +'out_len'+str(out_len)+ '_pred.npy', pred)
    np.save(now +'out_len'+str(out_len)+ '_true.npy', true)
    #对于out[2]中的数据，如果不在cpu上，需要转换为cpu上的数据
    metric= [i.cpu().numpy() if isinstance(i, torch.Tensor) else i for i in out[2]]
    pd.DataFrame(np.array(metric).reshape(1,-1), columns=['rmse','mae','REP']).to_csv(now + '_metrics.csv', float_format='%.6f', index=False)
    
    return out

def cal_gpu(module):
    if isinstance(module, torch.nn.DataParallel):
        module = module.module
    for submodule in module.children():
        if hasattr(submodule, "_parameters"):
            parameters = submodule._parameters
            if "weight" in parameters:
                return parameters["weight"].device
def get_memory_usage():
    return memory_allocated() / 1e6  # 返回已用显存，单位为MB


def find_bs(args,dataset_dict):
    start_mem = get_memory_usage()
    while True:
        try:
            dls = get_dls(args,dataset_dict)    
            suggested_lr = find_lr(dls,find_bs=True)
            print(f"Batch size {args.batch_size} is safe. Memory usage: {get_memory_usage()-start_mem}MB")
            torch.cuda.empty_cache()  # 清空缓存，否则之前的显存不会立刻释放
            args.batch_size *= 2  # 如果当前 batch_size 安全，尝试将其翻倍
            if args.batch_size >= 512:
                break
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"Batch size {args.batch_size} caused CUDA out of memory. Reverting to safe batch size.")
                args.batch_size =int(args.batch_size/8)  # 如果出现显存溢出，那么返回上一个安全的 batch_size
                torch.cuda.empty_cache()  # 清空缓存，否则之前的显存不会立刻释放
                break
            else:
                raise e  # 如果是其它类型的错误，那么抛出
    args.batch_size =min(args.batch_size,128)
    return args.batch_size
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
        
if __name__ == '__main__':
    ##保持预训练stride_ratio=1，finetune时再恢复
    input_len = args.input_len
    finetune_strie_ratio = args.stride_ratio
    args.dset = args.dset_pretrain
    args.initialize_wandb = False
    args.head_type = 'pretrain'
    if args.is_pretrain:
        args.input_len = 1
        args.stride_ratio = 1
        args.task_flag = 'pretrain'
        # args.batch_size = 4
        args.batch_size = 16
        dataset_dict = get_dataset_dict(args)
        # args.batch_size = find_bs(args,dataset_dict)
        torch.cuda.empty_cache()
        dls = get_dls(args,dataset_dict)    
        suggested_lr = find_lr(dls)
        torch.cuda.empty_cache()

        # Pretrain
        args.initialize_wandb = True
        pretrain_func(dls,suggested_lr)
        print('pretraining completed')
        ##清空显存
        torch.cuda.empty_cache()
        del dls
    
    if args.is_finetune:
        args.batch_size = 16
        args.input_len = input_len
        
        # args.dset = args.dset_finetune
        # Finetune
        #判断wandb是否初始化
        args.initialize_wandb = True
        args.stride_ratio = finetune_strie_ratio
        args.task_flag = 'finetune'
        args.head_type = 'prior_pooler'
        dataset_dict = get_dataset_dict(args)
        # args.batch_size = find_bs(args,dataset_dict)
        torch.cuda.empty_cache()

        # get dataloader
        dls = get_dls(args,dataset_dict)
        suggested_lr = find_lr(dls)  
        torch.cuda.empty_cache()
      
        # suggested_lr = 1e-4
        model=finetune_func(dls,suggested_lr)        
        print('finetune completed')
        ##清空显存
        torch.cuda.empty_cache()
        # # Test
        # out = test_func(model=model)         
        # print('----------- Complete! -----------')
        del dls
 
    if args.is_linear_probe:
        args.batch_size = 64
        args.input_len = input_len
        print('begin linear_probe')
        args.initialize_wandb = True
        args.stride_ratio = finetune_strie_ratio
        args.task_flag = 'linear_probe'
        args.head_type = 'prior_pooler'
        dataset_dict = get_dataset_dict(args)
        # args.batch_size = find_bs(args,dataset_dict)
        # args.dset = args.dset_finetune----------------
        torch.cuda.empty_cache()
        dls = get_dls(args,dataset_dict)
        suggested_lr = find_lr(dls)        
        torch.cuda.empty_cache()
        model =  linear_probe_func(dls,suggested_lr)        
        print('linear_probe completed')
        del dls
        # # Test
        # out = test_func(model)        
        # print('----------- Complete! -----------')

    if args.test:
        args.task_flag = 'test'
        args.batch_size = 1
        dataset_dict = get_dataset_dict(args)
        dls = get_dls(args,dataset_dict)
        #如果model不存在
        if 'model' not in locals():
            args.head_type = 'prior_pooler'
            model = get_model(dls.vars, args.head_type,args)
            #根据device_id选择设备
            model.to('cuda')
            if args.is_linear_probe:
                model = transfer_weights('/data/home/ps/WYL/琶洲新能源电池比赛-正式代码/saved_models/Power-Battery/masked_patchtst/based_model/linear_probe_time2024-02-23-04-03-33_ol192_patch1000_stride1000_epochs-finetune200_model1_n_layer4_n_dec2_n_head16_d_model512_dropout0.15_head_dropout0.05.pth',model)
                print('load linear_probe model')
            if args.is_finetune:
                model = transfer_weights(args.save_path+args.save_finetuned_model+'.pth',model)
        
        test_func(model,dls)
        
    
        