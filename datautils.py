

import numpy as np
import pandas as pd
import torch
from torch import nn
import sys

from src.data.datamodule import DataLoaders

def get_dls(params,dataset_dict):
    
    if not hasattr(params,'use_time_features'): params.use_time_features = False
    if params.dset == 'Power-Battery':
        dls = DataLoaders(
                datasetCls=dataset_dict,
                batch_size=params.batch_size,
                workers=params.num_workers,
                distrubuted=params.dist,
                flag =params.task_flag,
                )
    # dataset is assume to have dimension len x nvars
    dls.vars = dls.train.dataset.__getitem__(0)['encoder_input'].shape[-1]
    return dls

