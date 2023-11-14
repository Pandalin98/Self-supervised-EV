import numpy as np
import pandas as pd
import os
import torch
from src.callback.tracking import *
from src.callback.patch_mask import *
from src.callback.transforms import *
from src.metrics import *
from datautils import *
from src.callback.patch_mask import create_patch,reverse_patch
from matplotlib.colors import LinearSegmentedColormap

import matplotlib.pyplot as plt
def explain_funciton(model,encoder_input,label,dict,i,args,prior,decoder_input,encoder_mark,decoder_mark):
    ### Explain the model based on the input gradient
    # model - the model to be explained
    # encoder_input - the input to the model
    # label - the label of the input
    # i - the index of the input
    # args - the arguments
    ###
    encoder_input.requires_grad = True
    ## Catch the output
    output = model(encoder_input,prior,decoder_input,encoder_mark,decoder_mark,prior,decoder_input,encoder_mark,decoder_mark)

    output_mean = output.mean()
    label = label.mean()
    # Do backpropagation to get the derivative of the output based on the image
    output_mean.backward()
    # Retireve the saliency map and also pick the maximum value from channels on each pixel.
    # In this case, we look at dim=1. Recall the shape (batch_size, channel, width, height)
    saliency = encoder_input.grad.data.abs()
    saliency = saliency.reshape(encoder_input.shape)
    
    # 将梯度和结果还原到原始空间中
    orin_input = dict['encoder_input']
    orin_input = torch.unsqueeze(torch.tensor(orin_input), axis=0)
    saliency = reverse_patch(saliency, args.patch_len, args.stride,orin_input.shape)
    orin_input = torch.squeeze(orin_input, axis=0).cpu().numpy()
    saliency = torch.squeeze(saliency, axis=0).cpu().numpy()
    #nan值处理
    saliency[np.isnan(saliency)] = 0
    #变换到0~1空间中
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
    fig, axes = plt.subplots(17, 4, figsize=(32, 16))
    cmap_red = LinearSegmentedColormap.from_list('red_alpha', [(0, 0, 0), (1, 0, 0)])

    for j, ax in enumerate(axes.flat):
        #不要绘制x 轴和 y 轴的刻度
        ax.set_xticks([])
        ax.set_yticks([])
        # 将saliency映射到[0,1]范围内的透明度
        y = orin_input[:, j]
        alpha_values = saliency[:, j] 
        colors = cmap_red(alpha_values)[:, :3]
        # 创建一个用于imshow的2D数组
        image = np.zeros((2, len(y), 4))
        image[:, :, :3] = colors
        image[:, :, 3] = alpha_values  # 设置透明度
        ax.imshow(image, aspect='auto', origin='lower', extent=[0, len(y)-1, y.min(), y.max()])
        ax.plot(np.arange(len(y)), y, color='blue')  # 绘制原始信号

    fig_name = 'The {}th explanatory sample image of the reasoning model - model predict {} - real label {}.png'.format(i,output_mean,label.cpu().numpy())
    #设置图片标题  
    fig.suptitle(fig_name, fontsize=20)
    if not os.path.exists('explain'):
        os.mkdir('explain')
    fig.savefig(os.path.join('explain',fig_name), dpi=300, bbox_inches='tight')
    print('保存'+fig_name)