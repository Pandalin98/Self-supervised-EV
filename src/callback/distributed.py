# import deepspeed
from .core import Callback
import torch
from typing import Optional, Dict, Any
import logging
logger = logging.getLogger(__name__)
# from power_battery import Vari_len_collate_func



class DistributedTrainer(Callback):
    "Wrap `model` in `DistributedDataParallel` and `dls` in `DistributedDL`"
    def __init__(self,
                 sync_bn=True,  # Whether to replace all batch norm with `nn.SyncBatchNorm`
                 **kwargs
                 ):
        self.kwargs = kwargs
        self.args = {
            "num_gpus": 4,
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "activation_checkpointing": {
                "partition_activations": True,
                "cpu_checkpointing": True,
                "contiguous_memory_optimization": False,
                "number_checkpoints": None,
                "synchronize_checkpoint_boundary": False,
                "profile": True
            },
            "fp16": {
                "enabled": True,
                "auto_cast": False,
                "loss_scale": 0,
                "initial_scale_power": 16,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "consecutive_hysteresis": False,
                "min_loss_scale": 1
            },
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": 0.001,  # 使用学习率
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0.01
                }
            },
            "zero_optimization": {
                "stage": 3,
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "contiguous_gradients": True,
                "overlap_comm": True
            }
        }
        super().__init__()

    def before_fit(self):

        # self.old_train_dl = self.dls.train
        # self.old_valid_dl = self.dls.valid
        lr = self.learner.opt.param_groups[0]['lr']
        self.args['optimizer']['params']['lr'] = lr
        #更改config='deepspeed_config.json'中的参数lr后传入
        
        self.learner.model, self.learner.opt,_, _ = deepspeed.initialize(args=None,
            model=self.model,
            # optimizer=self.opt,
            model_parameters=self.model.parameters(),
            config  =self.args
            # training_data=self.dls.train.dataset,
            # collate_fn=Vari_len_collate_func,
        )
        
        # self.learner.dls.train = train_dls
        # self.learner.dls.valid = self._wrap_dl(self.dls.valid)
        



    def after_fit(self): 
        self.learner.model = self.learner.model.module 
        # self.learner.dls.train = self.old_train_dl
        # self.learner.dls.valid = self.old_valid_dl


