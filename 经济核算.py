import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns
import pandas as pd

# 参数初始化
N = 300  # 样本数量
Q = 70   # 电池的初始容量
L = 600000 # 行驶至L公里时强制报废电池
a_min, a_max = 68, 72
b_min, b_max = -11e-8, -9e-8
c_min, c_max = -7e-1,-5.5e-1 
d_min, d_max = 4.5e-6, 5.5e-6
m_mea = 3  # 电池个体差异的标准差
p1 = 1   # 每行驶1公里带来的经济价值
p2 = 300  # 电池回收的基础价格
deta = 0.6 # 回收折扣系数
z = 5000    # 预测未来某个里程l的最小步长
e_values = [0.5, 1, 1.2, 1.5, 2]  # 电池容量预测算法的误差标准差
err_increment = 0.1 # 误差标准差的增量


# 电池容量随行驶里程的变化函数
def battery_capacity(l, a, b, c, d, er):
    return a * np.exp(b * l) + c * np.exp(d * l) + er

# 电池容量预测算法

def predict_capacity(current_mileage, future_mileage, capacities, err, z):
    # 计算未来里程相对于当前里程增加的步数
    steps = (future_mileage - current_mileage) // z
    # 更新预测误差的标准差
    updated_err = err * (1 + err_increment * steps)
    # 获取未来行驶里程时的真实剩余容量
    eQ = capacities[min(future_mileage // z, len(capacities) - 1)]
    # 计算预测剩余容量
    mer = np.random.normal(0, updated_err)
    return eQ + mer



# 生成电池容量退化过程样本
def generate_samples(N, L, m_mea):
    samples = []
    for _ in range(N):
        a = np.random.uniform(a_min, a_max)
        b = np.random.uniform(b_min, b_max)
        c = np.random.uniform(c_min, c_max)
        d = np.random.uniform(d_min, d_max)
        er = np.random.normal(0, m_mea)
        eQ = battery_capacity(L, a, b, c, d, er)
        samples.append(eQ)
    return samples

# 生成电池容量退化曲线
def generate_capacity_curves(N, L, z, m_mea):
    curves = []
    for _ in range(N):
        a = np.random.uniform(a_min, a_max)
        b = np.random.uniform(b_min, b_max)
        c = np.random.uniform(c_min, c_max)
        d = np.random.uniform(d_min, d_max)
        er = np.random.normal(0, m_mea)
        capacities = [battery_capacity(l, a, b, c, d, er) for l in range(0, L + z, z)]
        curves.append(capacities)
    return curves

# 绘制电池容量退化曲线
def plot_capacity_curves(curves, L, z):
    plt.figure(figsize=(10, 6))
    for capacities in curves:
        plt.plot(range(0, L + z, z), capacities)
    plt.title("Battery Capacity Degradation Curves")
    plt.xlabel("Mileage (km)")
    plt.ylabel("Capacity")
    plt.grid(True)
    plt.show()
    

# 固定公里数回收策略的真实总收益计算
def fixed_mileage_strategy(samples, Q, L, p1, p2, deta):
    profits = []
    recycle_values = []
    for eQ in samples:
        v = calculate_recycle_value(eQ, Q, p2, deta)
        OL = L * p1 + v
        profits.append(OL)
        recycle_values.append(v)
    return profits, recycle_values



# 灵活回收策略的真实总收益计算及记录回收里程和容量
def flexible_recycling_strategy(capacity_curves, Q, z, p1, p2, deta, e_values, L):
    profits_per_method = {e: [] for e in e_values}
    recycle_points = {e: [] for e in e_values}  # 用于记录回收点

    for capacities in capacity_curves:
        current_mileage = 0
        for e in e_values:
            max_profit = 0
            best_k = 0
            current_mileage = 0

            for k in range(z, 2 * L + z, z):                    
                predicted_capacity = predict_capacity(current_mileage, k, capacities, e, z)
                if predicted_capacity < 0.8 * Q:
                    # 预测容量小于80%，开始计算收益
                    while k <= 2 * L and predicted_capacity >= 0.6 * Q:
                        v = calculate_recycle_value(predicted_capacity, Q, p2, deta)
                        O = p1 * k + v
                        if O > max_profit:
                            max_profit = O
                            best_k = k
                        k += z
                        predicted_capacity = predict_capacity(current_mileage, k, capacities, e, z)
                    break

                current_mileage = k
            real_eQ = capacities[min(best_k // z - 1, len(capacities) - 1)]
            v = calculate_recycle_value(real_eQ, Q, p2, deta)
            OL = p1 * (best_k) + v
            profits_per_method[e].append(OL)
            recycle_points[e].append((best_k, real_eQ,v))

    return profits_per_method, recycle_points



def plot_recycle_points(recycle_points, fixed_mileage_samples, L):
    # 第一张图：散点图
    plt.figure(figsize=(10, 6))
    
    # 灵活回收策略的散点图
    for e, points in recycle_points.items():
        # 提取每个点的最佳回收里程和真实电池容量
        mileage_capacity_pairs = [(mileage, capacity) for mileage, capacity, _ in points]
        # 绘制散点图
        plt.scatter(*zip(*mileage_capacity_pairs), alpha=0.6, label=f'e={e}')
    
    # 固定公里数策略的散点图（竖线）
    plt.scatter([L] * len(fixed_mileage_samples), fixed_mileage_samples, color='black', label='Fixed Mileage')
    
    plt.title("Recycle Mileage vs. Battery Capacity")
    plt.xlabel("Mileage (km)")
    plt.ylabel("Battery Capacity")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 第二张图：5个子图，展示行驶里程的概率分布
    # 概率分布图
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'green', 'red', 'cyan', 'magenta']
    for i, e in enumerate(e_values):
        mileage = [point[0] for point in recycle_points[e]]  # 提取每种误差下的里程数据
        if mileage:
            kde = gaussian_kde(mileage)
            x = np.linspace(min(mileage) - np.std(mileage), max(mileage) + np.std(mileage), 1000)
            y = kde(x)
            mean = np.mean(mileage)
            variance = np.var(mileage)
            plt.plot(x, y, color=colors[i], label=f'e={e} (Mean: {mean:.2f}, Var: {variance:.2f})')
            plt.fill_between(x, y, color=colors[i], alpha=0.3)
    plt.title("Mileage Distribution across Prediction Errors")
    plt.xlabel("Mileage (km)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()



    
# 绘制核密度估计的概率分布图并标注均值和方差
def plot_kde_distribution(profits, title):
    kde = gaussian_kde(profits)
    mean = np.mean(profits)
    variance = np.var(profits)

    # 创建值域以绘制
    x = np.linspace(min(profits) - np.std(profits), max(profits) + np.std(profits), 1000)
    y = kde(x)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label=f'Mean: {mean:.2f}, Variance: {variance:.2f}')
    plt.title(title)
    plt.xlabel('Total Profit')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_kde_distributions(flexible_recycle_values, fixed_recycle_value):
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']
    labels = ['Fixed Mileage'] + [f'Flexible e={e}' for e in flexible_recycle_values.keys()]

    # 固定公里数策略下的回收电池的价值v的概率分布曲线
    mean_fixed = np.mean(fixed_recycle_value)
    variance_fixed = np.var(fixed_recycle_value)
    kde = gaussian_kde(fixed_recycle_value)
    x = np.linspace(min(fixed_recycle_value) - np.std(fixed_recycle_value), max(fixed_recycle_value) + np.std(fixed_recycle_value), 1000)
    y = kde(x)
    plt.plot(x, y, color=colors[0], label=f'{labels[0]} (Mean: {mean_fixed:.2f}, Var: {variance_fixed:.2f})')

    # 灵活回收策略
    for i, (e, values) in enumerate(flexible_recycle_values.items()):
        mean_flexible = np.mean(values)
        variance_flexible = np.var(values)
        kde = gaussian_kde(values)
        x = np.linspace(min(values) - np.std(values), max(values) + np.std(values), 1000)
        y = kde(x)
        plt.plot(x, y, color=colors[i+1], label=f'{labels[i+1]} (Mean: {mean_flexible:.2f}, Var: {variance_flexible:.2f})')

    plt.title("Recycle Value Distribution across Strategies")
    plt.xlabel('Recycle Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()


    
    # 绘制核密度估计的概率分布图并标注均值和方差
def plot_kde_subplots(fixed_profits, flexible_profits):
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']
    labels = ['Fixed Mileage'] + [f'Flexible e={e}' for e in flexible_profits.keys()]

    # 固定公里数策略
    kde = gaussian_kde(fixed_profits)
    mean = np.mean(fixed_profits)
    variance = np.var(fixed_profits)
    x = np.linspace(min(fixed_profits) - np.std(fixed_profits), max(fixed_profits) + np.std(fixed_profits), 1000)
    y = kde(x)
    plt.plot(x, y, color=colors[0], label=f'{labels[0]} (Mean: {mean:.2f}, Var: {variance:.2f})')

    # 灵活回收策略
    for i, (e, profits) in enumerate(flexible_profits.items()):
        kde = gaussian_kde(profits)
        mean = np.mean(profits)
        variance = np.var(profits)
        x = np.linspace(min(profits) - np.std(profits), max(profits) + np.std(profits), 1000)
        y = kde(x)
        plt.plot(x, y, color=colors[i+1], label=f'{labels[i+1]} (Mean: {mean:.2f}, Var: {variance:.2f})')

    plt.title("Profit Distribution across Strategies")
    plt.xlabel('Total Profit')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()



# 计算回收价值
def calculate_recycle_value(eQ, Q, p2, deta):
    if eQ > 0.8 * Q:
        return p2 * eQ
    elif eQ >= 0.6 * Q:
        return deta * p2 * eQ
    else:
        return 0.01 * p2 * Q

# 绘制概率分布图
def plot_distributions(profits, title):
    plt.figure(figsize=(10, 6))
    plt.hist(profits, bins=30, density=True, alpha=0.6)
    plt.title(title)
    plt.xlabel('Total Profit')
    plt.ylabel('Probability')
    plt.grid(True)
    plt.show()
    
def plot_violin_subplots(fixed_profits, flexible_profits):
    data = {'Fixed Mileage': fixed_profits}
    data.update({f'Flexible e={e}': profits for e, profits in flexible_profits.items()})
    df = pd.DataFrame(data)

    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df, cut=0, inner=None, palette='pastel')  # inner=None 移除小提琴内部的棒状图

    # 添加均值散点和标准差注释
    for i, column in enumerate(df.columns):
        mean = df[column].mean()
        std = df[column].std()
        plt.scatter(i, mean, color='red', s=50, zorder=3)  # zorder=3 确保散点在最上层
        plt.text(i, mean, f'Mean: {mean:.2f}\nSD: {std:.2f}', ha='center', va='bottom')

    plt.title("Total Revenue Distribution across Strategies")
    plt.ylabel("Total Revenue")
    plt.grid(True)
    plt.legend(df.columns)  # 添加图例
    plt.show()

def plot_violin_distributions(flexible_recycle_values, fixed_recycle_value):
    data = {'Fixed Mileage': fixed_recycle_value}
    data.update({f'Flexible e={e}': values for e, values in flexible_recycle_values.items()})
    df = pd.DataFrame(data)

    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df, cut=0, inner=None, palette='pastel')

    # 添加均值散点和标准差注释
    for i, column in enumerate(df.columns):
        mean = df[column].mean()
        std = df[column].std()
        plt.scatter(i, mean, color='red', s=50, zorder=3)
        plt.text(i, mean, f'Mean: {mean:.2f}\nSD: {std:.2f}', ha='center', va='bottom')

    plt.title("Recycle Value Distribution across Strategies")
    plt.ylabel("Recycle Value")
    plt.grid(True)
    plt.legend(df.columns)  # 添加图例
    plt.show()


def main():
    samples = generate_samples(N, L, m_mea)
    fixed_profits, fixed_recycle_values = fixed_mileage_strategy(samples, Q, L, p1, p2, deta)

    # 生成电池容量退化曲线
    capacity_curves = generate_capacity_curves(N, 2 * L, z, m_mea)


    # 绘制电池容量退化曲线
    plot_capacity_curves(capacity_curves, 2 * L, z)
    
    
    # 灵活回收策略的真实总收益计算
    flexible_profits, recycle_points = flexible_recycling_strategy(capacity_curves, Q, z, p1, p2, deta, e_values, L)

    
    flexible_recycle_values = {e: [v for _, _, v in points] for e, points in recycle_points.items()}

    # 绘制固定公里数回收策略的概率分布图
    plot_kde_distribution(fixed_profits, "Fixed Mileage Strategy Profit Distribution")

    plot_recycle_points(recycle_points, samples, L)
        

    # 绘制灵活回收策略的概率分布图
    plot_kde_subplots(fixed_profits, flexible_profits)

    plot_kde_distributions(flexible_recycle_values, fixed_recycle_values)
    
    plot_violin_subplots(fixed_profits, flexible_profits)
    
    plot_violin_distributions(flexible_recycle_values, fixed_recycle_values)
        

if __name__ == "__main__":
    main()

