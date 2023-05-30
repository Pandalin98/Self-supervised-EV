# __author__='童国炜'
# -- coding: utf-8 --**
import re
import pandas as pd
import os
import xlrd
import numpy as np

START = ""
END = ""
CL = ""


def extract(file):
    xls_file = xlrd.open_workbook(file)
    xls_table = xls_file.sheets()[0]  # 第1个工作簿
    print(file + "数据读取完毕")
    row = xls_table.nrows
    # 初始化变量
    eneryQ = 0
    result = []
    i = 0
    # 从第二行开始遍历
    while i <= row - 2:
        # 保存行驶数据
        i = i + 1
        if xls_table.row_values(i)[3] == 1.0:
            #  判断本次充电行为是否为40-90的充电行为，不是的话就跳过。
            j = i
            soclist = []
            while xls_table.row_values(j)[3] == 1.0:
                soclist.append([int(xls_table.row_values(j)[7]), float(xls_table.row_values(j)[6])])  # 保存SOC，电流。
                # 以充电结束时跳出循环
                if j >= row-1:
                    break
                j += 1
            i = j
            soc_list = np.array(soclist)
            # 使用unique函数去除重复值
            unique_soc_list = np.unique(soc_list[:, 0])
            # 使用sort函数将数组排序
            sorted_soc_list = np.sort(unique_soc_list)
            # 判断是否是连续变化的整数
            for ii in range(1, len(sorted_soc_list)):
                if sorted_soc_list[ii - 1] - sorted_soc_list[ii] != -1.0:
                    is_continuous = False
                    break
            else:
                is_continuous = True

            if 40 in soc_list[:, 0] and 90 in soc_list[:, 0] and is_continuous:
                for num in soc_list:
                    if 90 >= num[0] >= 40:
                        eneryQ = eneryQ + num[1]
                eneryQ = eneryQ/180
            else:
                eneryQ = 0
                continue

            date_1 = xls_table.row_values(i-1)[0]  # 获取该帧的时间
            date_2 = xlrd.xldate_as_datetime(date_1, 0)
            date = date_2.strftime('%Y/%m/%d %H:%M:%S')
            Q_info = {"结束时间": date, "车辆状态": xls_table.row_values(i-1)[3],
                      "里程": xls_table.row_values(i-1)[4], "电量": str(eneryQ)}
            result.append(Q_info)
    # 保存数据
    if result:
        pf = pd.DataFrame(list(result))
        global START
        global CL
        pf.to_csv('结果\\' + CL + '充电能耗分析_' + START + '.csv', index=False)


def main():
    global START
    global CL
    pat = re.compile(r'(?P<CL>.*)_(?P<START>\d{14})_(?P<END>\d{14})')  # 这里存在一个空格，是因为文件名里存在空格
    filePath = "H:\\广州琶洲算法赛数据\\用于打分测试的数据\\"
    file_list = os.listdir(filePath)
    for file in file_list:
        inf = pat.findall(file)
        START = inf[0][1]
        CL = inf[0][0]
        extract(filePath + file)


if __name__ == '__main__':
    main()



