

import numpy as np
import pandas as pd
import torch
from torch import nn
import sys

from src.data.datamodule import DataLoaders
from src.data.CALCE_Battary import CALCE_dataset
from src.data.engine import Engine
from src.data.NASA_battery import NASA_Battery


DSETS = ['N-MAPPS', 'CALCE_Battery', 'NASA_Battery']

def get_dls(params):
    
    assert params.dset in DSETS, f"Unrecognized dset (`{params.dset}`). Options include: {DSETS}"
    if not hasattr(params,'use_time_features'): params.use_time_features = False

    if params.dset == 'N-MAPPS':
        root_path = 'datasets/N-MAPPS/'
        dls = DataLoaders(
            datasetCls=Engine,
            dataset_kwargs={
                'root_path': root_path,
                'data_path': 'N-CMAPSS_DS05.h5',
                'scale': True,
                'out_sample_rate':params.out_sample_rate,
                'inside_sample_rate':params.inside_sample_rate,
                'data_sort' : params.data_sort,
                'snr'    : params.snr,
                'target_points':params.target_points,
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                distrubuted=params.dist,
                flag =params.task_flag,
                )
        
    elif params.dset == 'CALCE_Battery':
        root_path = 'datasets/CALCE_Battery/'
        dls = DataLoaders(
            datasetCls=CALCE_dataset,
            dataset_kwargs={
                'root_path': root_path,
                'data_path': ['CS2_35.csv', 'CS2_36.csv', 'CS2_37.csv', 'CS2_38.csv'],
                'scale': True,
                'out_sample_rate':params.out_sample_rate,
                'inside_sample_rate':params.inside_sample_rate,
                'data_sort' : params.data_sort,
                'snr'    : params.snr,
                'target_points':params.target_points,

                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                distrubuted=params.dist,
                flag =params.task_flag,

                )
    elif params.dset == 'NASA_Battery':
        root_path = 'datasets/NASA_Battery/'
        dls = DataLoaders(
            datasetCls=NASA_Battery,
            dataset_kwargs={
                'root_path': root_path,
                'data_path': ['B0005', 'B0006', 'B0007', 'B0018'],
                'scale': True,
                'out_sample_rate':params.out_sample_rate,
                'inside_sample_rate':params.inside_sample_rate,
                'data_sort' : params.data_sort,
                'snr'    : params.snr,
                'target_points':params.target_points,

                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                distrubuted=params.dist,
                flag =params.task_flag,

                )

    # dataset is assume to have dimension len x nvars
    dls.vars = dls.train.dataset.__getitem__(0)['feature'].shape[-1]+dls.train.dataset.__getitem__(0)['WC'].shape[-1]
    return dls



if __name__ == "__main__":
    class Params:
        dset= 'CALCE_Battery'
        out_sample_rate= 1
        inside_sample_rate= 1
        batch_size= 64
        data_sort = True
        snr = 0
        num_workers= 32
        dist = False
        task_flag = 'pretrain'
        target_points = 10

    params = Params 
    dls = get_dls(params)
    for i, batch in enumerate(dls.train):
        print(i, len(batch), batch['prior'].shape)
    breakpoint()
