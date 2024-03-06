
from typing import List
import torch
import torch.cuda
from torch.optim import Adam
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from .basics import *
from .callback.core import * 
from .callback.tracking import * 
from .callback.scheduler import *
from .callback.distributed import *
from .utils import *
from pathlib import Path
import numpy as np
from sklearn.base import BaseEstimator
import torch.nn.functional as F
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau

class Learner(GetAttr):
    """
    A high-level PyTorch module for training and fine-tuning deep learning models.

    Parameters:
    -----------
    dls : dataloader
        A dataloader object containing training and validation datasets.
    model : nn.Module
        A PyTorch neural network model.
    loss_func : nn.Module
        A loss function to optimize during training.
    lr : float
        Learning rate for the optimizer. Default is 1e-3.
    cbs : List[Callback]
        List of callback objects to use during training.
    metrics : List[nn.Module]
        List of evaluation metrics.
    opt_func : optimizer
        An optimizer for the model. Default is Adam.
    """
    def __init__(self, dls, model, 
                        lr=1e-3, 
                        cbs=None, 
                        metrics=None, 
                        opt_func=Adam,
                        flag = 'pretrain',
                        args = None,
                        loss_func = nn.MSELoss()
                        ):
                
        self.model, self.dls, self.lr = model, dls, lr
        self.opt_func = opt_func
        #self.opt = self.opt_func(self.model.parameters(), self.lr) 
        self.constrast_temp =args.kl_temperature
        self.loss_weight = args.recon_weight
        self.metrics = metrics
        self.n_inp  = 2
        self.flag = flag
        self.loss_func = loss_func
        # Indicator of running lr_finder
        self.run_finder = False
        self.args = args        
        self.set_opt()
  
        if cbs and not isinstance(cbs, List): cbs = [cbs]    
        self.initialize_callbacks(cbs)        
        # if args.dist:
        #     self.to_distributed()


    def set_opt(self):
        if self.model:
            self.opt = self.opt_func(self.model.parameters(), self.lr) 
        else: self.opt = None


    def default_callback(self):
        "get a set of default callbacks"
        default_cbs = [ SetupLearnerCB(), TrackTimerCB(), 
                        TrackTrainingCB(train_metrics=False, valid_metrics=True,args = self.args)]                  
        return default_cbs
    

    def initialize_callbacks(self, cbs):        
        default_cbs = self.default_callback()       
        self.cbs = update_callbacks(cbs, default_cbs) if cbs else default_cbs        
        # add print CB
        self.cbs += [PrintResultsCB(self.args)]        
        
        for cb in self.cbs: cb.learner = self     
        self('init_cb')       


    def add_callback(self, cb):                
        if not cb: return
        cb.learner = self
        self.cbs = update_callback(cb, self.cbs)           

    def add_callbacks(self, cbs):        
        if not isinstance(cbs, list):  cbs = [cbs]
        for cb in cbs: self.add_callback(cb)

    def remove_callback(self, cb): 
        cb.learn = None
        self.cbs, removed_cb = remove_callback(cb, self.cbs)
        return removed_cb
    
    def remove_callbacks(self, cb_list):
        for cb in cb_list: self.remove_callback(cb)


    def pretrain_loss_func(self, pred, target,z1,z2):
        """
        preds:   [bs x num_patch  x patch_len]
        targets: [bs x num_patch  x patch_len] 
        """
        loss_re = (pred - target) ** 2
        loss_re = loss_re.mean(dim=-1)
        loss_re = (loss_re * self.mask).sum() / self.mask.sum()
        loss_con = self.contrast_loss(z1, z2)/10
        loss =(1-self.loss_weight)* loss_re +self.loss_weight* loss_con
        return loss,loss_re,loss_con
   

    def contrast_loss(self, z1, z2):
        #如果z1和z2的维度不一致，需要进行维度的转换
        if z1.dim() == 1:
            z1 = z1.unsqueeze(0)
        B = z1.size(0)
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        z = torch.cat([z1, z2], dim=0)
        
        # 计算相似度矩阵
        sim = torch.matmul(z, z.T) / self.constrast_temp
        diag_mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill_(diag_mask, float('-inf'))  # 掩盖对角线元素
        
        # 计算损失
        logits = F.log_softmax(sim, dim=-1)
        contrast_loss = (logits.diagonal(B).mean() + logits.diagonal(-B).mean()) / 2
        
        return -contrast_loss


    def fit_pretrain(self, n_epochs, lr_max=None, lr=None,cbs=None, do_valid=False, pct_start=0.3):
        " fit the model based on the pretrain loss "
        self.lr_max = lr_max if lr_max else self.lr
        cb = OneCycleLR(lr_max=self.lr_max, pct_start=pct_start)
        self.n_epochs = n_epochs
        if not self.dls.valid: do_valid = False
        if cbs: self.add_callbacks(cbs)
        if lr: self.opt = self.opt_func(self.model.parameters(), lr) 
        self('before_fit')
        try:
            self.dl = self.dls.train
            for self.epoch in range(n_epochs):            
                self('before_epoch')                     
                self('before_epoch_train')
                self.model.train()                
                for num,batch in enumerate(self.dl):
                    self.iter, self.batch = num, batch
                    self('before_batch_train')
                    self.xb, self.target,self.prior,self.xb_mark,self.xb_dec,self.dec_mark = self.batch['encoder_input'],self.batch['label'],self.batch['prior'],self.batch['encoder_mark'],self.batch['decoder_input'],self.batch['decoder_mark']              # forward
                    # forward
                    self('before_forward')
                    pred,con1,con2 = self.model(self.xb,self.prior,self.xb_dec,self.xb_mark,self.dec_mark)
                    self('after_forward')
                    # compute loss
                    pretrain_loss,recon_loss,kl_loss = self.pretrain_loss_func(pred, self.target,con1,con2)
                    self.pred, self.loss,self.recon_loss,self.kl_loss = pred,pretrain_loss,recon_loss,kl_loss                                      
                    # zero the parameter gradients
                    self.opt.zero_grad()                 
                    # gradient
                    self.loss.backward()
                    # update weights
                    self.opt.step() 
                    self('after_batch_train')  
                self('after_epoch_train')               # if self.dls.valid:                    
                if do_valid: self.epoch_validate()                   
                self('after_epoch')                 
        except KeyboardInterrupt: pass 
        self('after_fit')


    def Probability_prediction_loss(self, pred, target):
        """
        preds:   [bs x time_length  x feature_dim]
        targets: [bs x time_length  x feature_dim] 
        """
        
        loss = F.binary_cross_entropy_with_logits(pred, target)
        return loss

         
    def fit_downstreams(self, n_epochs,loss_func, lr_max=None, lr=None, cbs=None, do_valid=True,pct_start=0.3):
        " fit the model to the label target "        
        self.lr_max = lr_max if lr_max else self.lr
        # self.add_callback( OneCycleLR(lr_max=self.lr_max, pct_start=pct_start))
        self.n_epochs = n_epochs
        if not self.dls.valid: do_valid = False
        if cbs: self.add_callbacks(cbs)
        if lr: self.opt = self.opt_func(self.model.parameters(), lr) 
        scheduler = ReduceLROnPlateau(self.opt, mode='min', factor=0.5, patience=2, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
        self('before_fit')
        try:
            self.dl = self.dls.train
            for self.epoch in range(n_epochs):            
                self('before_epoch')                     
                self('before_epoch_train')
                self.model.train()                
                for num,batch in enumerate(self.dl):
                    self.iter, self.batch = num, batch
                    self('before_batch_train')
                    self.xb, self.target,self.prior,self.xb_mark,self.xb_dec,self.dec_mark = self.batch['encoder_input'],self.batch['label'],self.batch['prior'],self.batch['encoder_mark'],self.batch['decoder_input'],self.batch['decoder_mark']                # forward
                    self('before_forward')
                    pred = self.model(self.xb,self.prior,self.xb_dec,self.xb_mark,self.dec_mark)
                    self('after_forward')
                    # compute loss
                    down_stream_loss = loss_func(pred, self.target)
                    self.pred, self.loss = pred,down_stream_loss                                      
                    # zero the parameter gradients
                    self.opt.zero_grad()                 
                    # gradient
                    self.loss.backward()
                    # update weights
                    self.opt.step() 
                    self('after_batch_train')  
                self('after_epoch_train')               # if self.dls.valid:                    
                if do_valid: self.epoch_validate(),self.epoch_test() 
                scheduler.step(self.recorder['train_loss'][-1])
                current_lr = self.opt.param_groups[0]['lr']
                print(f'Epoch {self.epoch+1}, Learning rate: {current_lr}')
                self('after_epoch')                 
        except KeyboardInterrupt: pass 
        self('after_fit')

    

    def fine_tune(self, n_epochs, loss_func,base_lr=None, freeze_epochs=1, pct_start=0.3):
        """
        fintune the pretrained model. First the entire model is freezed, only head is trained
        up to a freeze_epochs number. Then the model is unfreezed and the entire model is trained
        """
        assert (n_epochs>0)|(freeze_epochs>0), "Either n_epochs or freeze_epochs has to be > 0"
        if not base_lr: base_lr = self.lr
        # Finetune the head of freeze_epochs > 0:
        head_epoch_ratio = self.args.head_epochs_ratio
        n_epochs_head = int(n_epochs*head_epoch_ratio)
        n_epochs = n_epochs - n_epochs_head
        if n_epochs_head > 0:
            print('Finetune the head and decoder')
            self.freeze()
            self.fit_downstreams(n_epochs_head,loss_func=loss_func, lr_max=base_lr, pct_start=pct_start)
        
        # Finetune the entire network if n_epochs > 0
        if n_epochs > 0:
            print('Finetune the entire network')        
            self.unfreeze()
            self.fit_downstreams(n_epochs, lr_max=base_lr/8, pct_start=pct_start,loss_func=loss_func)

    def linear_probe(self, n_epochs,loss_func, base_lr=None, pct_start=0.3):
        """
        linear probing the pretrained model. The model is freeze except the head during finetuning
        """
        assert (n_epochs>0), "n_epochs has to be > 0"
        if not base_lr: base_lr = self.lr
        print('Finetune the head')
        self.freeze()
        self.fit_downstreams(n_epochs,loss_func=loss_func, lr_max=base_lr, lr=base_lr,pct_start=pct_start)
    

    def lr_finder(self,task_flag, start_lr=1e-5, end_lr=1e-2, num_iter=100, step_mode='exp', show_plot=False, suggestion='valley',loss_func=None):                
        """
        find the learning rate
        """
        n_epochs = max(num_iter//max(len(self.dls.train),1) + 5,10)
        # indicator of lr_finder method is applied
        self.run_finder = True
        # add LRFinderCB to callback list and will remove later
        cb = LRFinderCB(start_lr, end_lr, num_iter, step_mode, suggestion=suggestion)                
        # fit
        if task_flag == 'pretrain':
            self.fit_pretrain(n_epochs=n_epochs, cbs=cb, do_valid=False)        
        if task_flag == 'finetune':
            self.add_callback(cb)
            self.fine_tune(n_epochs=n_epochs,loss_func=loss_func)
        if task_flag == 'linear_probe':
            self.add_callback(cb)
            self.linear_probe(n_epochs=n_epochs,loss_func=loss_func)
        # should remove LRFinderCB callback after fitting                
        self.remove_callback(cb)        
        self.run_finder = False        
        if show_plot: cb.plot_lr_find()
        if suggestion: return cb.suggested_lr     
    
    def epoch_validate(self, dl=None):
        self('before_epoch_valid')
        # model at evaluation mode  
        self.model.eval()                
        self.dl = dl if dl else self.dls.valid
        if self.dl:        
            with torch.no_grad(): self.all_batches('valid')
        self('after_epoch_valid')

    def epoch_test(self, dl=None):
        self('before_epoch_test')
        # model at evaluation mode  
        self.model.eval()                
        self.dl = dl if dl else self.dls.test
        if self.dl:        
            with torch.no_grad(): self.all_batches('test')
        self('after_epoch_test')



    def all_batches(self, type_):
        # for self.num,self.batch in enumerate(progress_bar(dl, leave=False)):        
        for num, batch in enumerate(self.dl):            
            self.iter, self.batch = num, batch            
            if type_ == 'valid': self.batch_validate()
            elif type_ == 'predict': self.batch_predict()             
            elif type_ == 'test': self.batch_test()


    def batch_train(self):
        self('before_batch_train')
        self._do_batch_train()
        self('after_batch_train')  
 

    def batch_validate(self):
        self('before_batch_valid')
        self._do_batch_validate()
        self('after_batch_valid')  
    
    def batch_predict(self):
        self('before_batch_predict')
        self._do_batch_predict()
        self('after_batch_predict') 

    def batch_test(self):
        self('before_batch_test')
        self._do_batch_test()
        self('after_batch_test') 
        
    def model_forward(self):
        self('before_forward')
        self.pred = self.model(self.xb,self.prior,self.xb_dec,self.xb_mark,self.dec_mark)
        self('after_forward')
        return self.pred

    def _do_batch_validate(self):       
        # forward + calculate loss
        self.pred, self.loss = self.valid_step(self.batch)     

    def valid_step(self, batch):
        # get the inputs
        self.xb, self.target,self.prior,self.xb_mark,self.xb_dec,self.dec_mark = self.batch['encoder_input'],self.batch['label'],self.batch['prior'],self.batch['encoder_mark'],self.batch['decoder_input'],self.batch['decoder_mark']                    # forward

        # forward
        pred = self.model_forward()
        # compute loss
        loss = self.loss_func(pred, self.target)
        return pred, loss                                     


    def _do_batch_predict(self):   
        self.pred = self.predict_step(self.batch)     
           
    def predict_step(self, batch):
        # get the inputs
        self.xb, self.target,self.prior,self.xb_mark,self.xb_dec,self.dec_mark = self.batch['encoder_input'],self.batch['label'],self.batch['prior'],self.batch['encoder_mark'],self.batch['decoder_input'],self.batch['decoder_mark']                    # forward
        # forward
        pred = self.model_forward()
        return pred 
    
    def _do_batch_test(self):   
        self.pred, self.target,self.loss = self.test_step(self.batch)     
           
    def test_step(self, batch):
        # get the inputs
        self.xb, self.target,self.prior,self.xb_mark,self.xb_dec,self.dec_mark = self.batch['encoder_input'],self.batch['label'],self.batch['prior'],self.batch['encoder_mark'],self.batch['decoder_input'],self.batch['decoder_mark']                    # forward
        # forward
        pred = self.model_forward()
        loss = self.loss_func(pred, self.target)
        return pred, self.target,loss


    def _predict(self, dl=None):
        # self('before_validate')
        self('before_predict')
        if dl is None: return
        self.dl = dl
        self.n_inp = dl.dataset.n_inp                
        self.model.eval()        #  model at evaluation mode  
        with torch.no_grad(): self.all_batches('predict')        
        self('after_predict')


    def predict(self, test_data, weight_path=None, Dataset=None, Dataloader=None, batch_size=None):
        """_summary_
        Args:
            test_data can be a tensor, numpy array, dataset or dataloader
        Returns:
            _type_: _description_
        """                
        if weight_path is not None: self.load(weight_path)
        cb = GetPredictionsCB()
        self.add_callback(cb)                    
        test_dl = self._prepare_data(test_data, Dataset, Dataloader, batch_size)
        self._predict(test_dl)        
        self.preds = cb.preds
        return to_numpy(self.preds) 
   
    
    def test(self, dl, model, scores=None):
        """_summary_
        Args:
            test_data can be a tensor, numpy array, dataset or dataloader
        Returns:
            _type_: _description_
        """          
        if dl is None: return
        else: self.dl = dl
        self.model= model
        cb = GetTestCB()
        self.add_callback(cb)
        self('before_test')
        self.model.eval()
        torch.cuda.synchronize()
        start_time = time.time()
        memory_before = torch.cuda.memory_allocated() / 1024 / 1024  # 显存使用量（MB）

        with torch.no_grad():
            self.all_batches('test')

        torch.cuda.synchronize()
        end_time = time.time()
        memory_after = torch.cuda.memory_allocated() / 1024 / 1024  # 显存使用量（MB）
        memory_allocated = memory_after - memory_before  # 计算显存增加量（MB）
        elapsed_time = end_time - start_time  # 计算时间（秒）
        self('after_test')   
        self.preds, self.targets = to_numpy([cb.preds, cb.targets])
        # calculate scores
        if scores: 
            s_vals = [score(cb.targets, cb.preds) for score in list(scores)]
            return self.preds, self.targets, s_vals
        else: return self.preds, self.targets


    def _prepare_data(self, test_data, Dataset=None, Dataloader=None, batch_size=None):
        if test_data is None: return test_data
        if Dataset and Dataloader:
            test_dset = Dataset(test_data)
            if not batch_size: batch_size=16
            test_dl = Dataloader(test_dset, batch_size)        
        else:            
            if self.dls: 
                # add test_data to the dataloader defined in the dls.train
                test_dl = self.dls.add_dl(test_data, batch_size=batch_size)  
            else: test_dl = test_data       # assume test_data is already a form of dataloader
        return test_dl
   
    
    def get_layer_output(self, inp, layers=None, unwrap=False):
        """
        Args:
            inp: can be numpy array, torch tensor or dataloader
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        if isinstance(inp, np.ndarray): inp = torch.Tensor(inp).to(device)
        if isinstance(inp, torch.Tensor): inp = inp.to(device)
        
        return get_layer_output(inp, model=self.model, layers=layers, unwrap=unwrap)
    
    
    

    def freeze(self):
        """ 
        freeze the model head
        require the model to have head attribute
        """
        if hasattr(get_model(self.model), 'head'): 
            # print('model head is available')
            for param in get_model(self.model).parameters(): param.requires_grad = True
            for param in get_model(self.model.backbone).parameters(): param.requires_grad = False        
            print('backbone of model is frozen')
            
            
    def unfreeze(self):
        for param in get_model(self.model).parameters(): param.requires_grad = True        


    def __call__(self, name):        
        for cb in self.cbs: 
            attr = getattr(cb, name)
            if attr is not None: attr()
          

    def save(self, fname, path, **kwargs):
        """
        Save model and optimizer state (if `with_opt`) to `self.path/file`
        """
        fname = join_path_file(fname, path, ext='.pth')        
        save_model(fname, self.model, getattr(self,'opt',None), **kwargs)
        return fname


    def load(self, fname, with_opt=False, device='cuda', strict=True, **kwargs):
        """
        load the model
        """
        if not torch.cuda.is_available():
            device = "cpu"
        load_model(fname, self.model, self.opt, with_opt, device=device, strict=strict)


    def get_params(self, deep=True, **kwargs):
        params = BaseEstimator.get_params(self, deep=deep, **kwargs)
        return params

    def _get_param_names(self):
        return (k for k in self.__dict__ if not k.endswith('_'))


    def set_params(self, **kwargs):
        params = {}
        for key, val in kwargs.items():
            params[key] = val
        BaseEstimator.set_params(self, **params)

    def to_distributed(self,
                       sync_bn=True,  # Whether to replace all batch norm with `nn.SyncBatchNorm`
                       **kwargs
                       ):
        # local_rank = int(os.environ.get('LOCAL_RANK'))
        # world_size = int(os.environ.get('WORLD_SIZE'))
        # rank = int(os.environ.get('RANK'))
        # print('Process {} (out of {})'.format(
        #     rank, torch.distributed.get_world_size()))
        print('begin distributing by DeepSpeed')

        self.add_callback(DistributedTrainer(**kwargs))

        return self


def save_model(path, model, opt, with_opt=True, pickle_protocol=2):
    "Save `model` to `file` along with `opt` (if available, and if `with_opt`)"
    if opt is None: with_opt=False
    state = get_model(model).state_dict()
    if with_opt: state = {'model': state, 'opt':opt.state_dict()}
    torch.save(state, path, pickle_protocol=pickle_protocol)


def load_model(path, model, opt=None, with_opt=False, device='cpu', strict=True):
    " load the saved model "
    state = torch.load(path, map_location=device)
    if not opt: with_opt=False
    model_state = state['model'] if with_opt else state
    get_model(model).load_state_dict(model_state, strict=strict)
    if with_opt: opt.load_state_dict(state['opt'])
    model = model.to(device)
      

def join_path_file(file, path, ext=''):
    "Return `path/file` if file is a string or a `Path`, file otherwise"
    if not isinstance(file, (str, Path)): return file
    if not isinstance(path, Path): path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path/f'{file}{ext}'


def get_model(model):
    "Return the model maybe wrapped inside `model`."    
    return model.module if isinstance(model, (DistributedDataParallel, nn.DataParallel)) else model


def transfer_weights(weights_path, model, exclude_head=True, device='cpu'):
    # state_dict = model.state_dict()
    new_state_dict = torch.load(weights_path, map_location=device)
    #print('new_state_dict',new_state_dict)
    matched_layers = 0
    unmatched_layers = []
    for name, param in model.state_dict().items():        
        if exclude_head and 'head' in name: continue
        if name in new_state_dict:            
            matched_layers += 1
            input_param = new_state_dict[name]
            if input_param.shape == param.shape: param.copy_(input_param)
            else: unmatched_layers.append(name)
        else:
            unmatched_layers.append(name)
            pass # these are weights that weren't in the original model, such as a new head
    if matched_layers == 0: raise Exception("No shared weight names were found between the models")
    else:
        if len(unmatched_layers) > 0:
            print(f'check unmatched_layers: {unmatched_layers}')
        else:
            print(f"weights from {weights_path} successfully transferred!\n")
    model = model.to(device)
    return model


def update_callback(cb, list_cbs):
    for cb_ in list_cbs:
        if type(cb_) ==  type(cb): list_cbs.remove(cb_)
    list_cbs += [cb]
    return list_cbs

def update_callbacks(list_cbs, default_cbs):
    for cb in list_cbs: default_cbs = update_callback(cb, default_cbs)
    return default_cbs

def remove_callback(cb, list_cbs):
    for cb_ in list_cbs:
        if type(cb_) ==  type(cb):             
            list_cbs.remove(cb_)
            break
    return list_cbs, cb_


def get_layer_output(inp, model, layers=None, unwrap=False):
    """
    layers is a list of module names
    """
    orig_model = model
    
    if unwrap: model = unwrap_model(model)
    if not layers: layers = list(dict(model.named_children()).keys())
    if not isinstance(layers, list): layers = [layers]

    activation = {}
    def getActivation(name):
        # the hook signature
        def hook(model, input, output):
            activation[name] = output.detach().cpu().numpy()
        return hook

    # register forward hooks on the layers of choice    
    h_list = [getattr(model, layer).register_forward_hook(getActivation(layer)) for layer in layers]
    
    model.eval()
    out = orig_model(inp)    
    for h in h_list: h.remove()
    return activation
