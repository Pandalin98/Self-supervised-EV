

import numpy as np
import pandas as pd
import torch
from torch import nn
import sys

from src.data.datamodule import DataLoaders
from src.data.CALCE_Battary import CALCE_dataset
from src.data.engine import Engine
from src.data.NASA_battery import NASA_Battery
from src.power_battery import PowerBatteryData
DSETS = ['N-MAPPS', 'CALCE_Battery', 'NASA_Battery','Power-Battery']

def get_dls(params):
    
    assert params.dset in DSETS, f"Unrecognized dset (`{params.dset}`). Options include: {DSETS}"
    if not hasattr(params,'use_time_features'): params.use_time_features = False
    if params.dset == 'Power-Battery':
        root_path = 'datasets/Power-Battery/'
        dls = DataLoaders(
            datasetCls=PowerBatteryData,
            dataset_kwargs={
                'data_path': params.data_path,
                'scale': params.scale,
                'size':[params.input_len,params.output_len],
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                distrubuted=params.dist,
                flag =params.task_flag,
                )

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
    dls.vars = dls.train.dataset.__getitem__(0)['encoder_input'].shape[-1]
    return dls

if __name__ == "__main__":
    class Params:
        dset= 'Power-Battery'
        data_path = './data/local_data_structure'
        scale = True
        input_len = 24
        label_len = 12
        output_len = 3
        batch_size= 64
        num_workers= 32
        dist = False
        task_flag = 'train'
        target_points = 10

    params = Params 
    dls = get_dls(params)
    for i, batch in enumerate(dls.train):
        print(i, len(batch), batch['prior'].shape)
    breakpoint()
