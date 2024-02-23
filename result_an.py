import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_percentage_error

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    #设置上下界
    plt.ylim(90, 130)
    plt.savefig(name, bbox_inches='tight')

# 指定StandardScaler的路径
scaler_path = 'data/target_scaler.pkl'
result_path  = 'result'
# 加载StandardScaler
scaler = joblib.load(scaler_path)

# 初始化一个空的DataFrame来保存结果
results = pd.DataFrame(columns=['file', 'mse','rmse','mae','mape'])
file_dir_list = os.listdir(result_path)

# 遍历"result"目录下的所有npy文件
for file_dir in os.listdir(result_path):
    file_list = os.listdir(os.path.join(result_path, file_dir))
    #找出其中包含true和pred的文件
    true_file = [file for file in file_list if 'true' in file][0]
    pred_file = [file for file in file_list if 'pred' in file][0]
    
    # 加载npy文件
    true = np.load(os.path.join(result_path,file_dir, true_file))
    pred = np.load(os.path.join(result_path,file_dir, pred_file))
    
    predict_len = 10
    
    true_visual = true[::predict_len,:]
    pred_visual = pred[::predict_len,:]
    # 可视化结果
    for i in range(true_visual.shape[0]):
        visual(true_visual[i,:],pred_visual[i,:],name=os.path.join(result_path,file_dir, 'test_{}.pdf'.format(i)))    
    
    # # 使用StandardScaler将数据转换回原始空间
    # true = scaler.inverse_transform(true)
    # pred = scaler.inverse_transform(pred)

    # 计算各项指标
    MSE = mse(true, pred)
    RMSE = rmse(true, pred)
    MAE = mae(true, pred)
    MAPE = mape(true, pred)*100
    
    # 将结果保存在DataFrame中
    results = results.append({'file': file_dir, 'mse': MSE,
                                'rmse': RMSE, 'mae': MAE,
                                'mape': MAPE,
                              }, ignore_index=True)

# 保存DataFrame到CSV文件
results.to_excel('accuracy_results.xlsx', index=False)