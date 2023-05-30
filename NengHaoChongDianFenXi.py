# __author__='童国炜'
# -- coding: utf-8 --**
import pandas as pd
import re
import os
import xlrd
from datetime import datetime

filePath = "H:\\广州琶洲算法赛数据\\用于打分测试的数据\\"
file_list = os.listdir(filePath)

pat = re.compile(r'(?P<CL>.*)_(?P<START>\d{14})_(?P<END>\d{14})')  # 这里存在一个空格，是因为文件名里存在空格

for file in file_list:
    xls_file = xlrd.open_workbook(filePath + file)
    xls_table = xls_file.sheets()[0]  # 第1个工作簿
    inf = pat.findall(file)
    START = inf[0][1]
    END = inf[0][2]
    CL = inf[0][0]
    print(file + " 数据读取完毕")

    row = xls_table.nrows
    # 第一行是列名，用第二行数据初始化变量
    Vehicle_statue = xls_table.row_values(1)[3]  # 充电状态 1为停车充电  3为未充电状态
    date_1 = xls_table.row_values(1)[0]  # 获取该帧的时间
    date_2 = xlrd.xldate_as_datetime(date_1, 0)
    date = date_2.strftime('%Y/%m/%d %H:%M:%S')

    SOC = xls_table.row_values(1)[7]  # 获取SOC值
    Mileage = xls_table.row_values(1)[4]  # 获取里程
    eneryQ = 0
    result = []
    # 从第三行开始遍历
    for i in range(row - 2):
        i = i + 2
        Vehicle_statue_1 = xls_table.row_values(i)[3]  # 充电状态 1为停车充电  3为未充电状态
        if Vehicle_statue != Vehicle_statue_1:  # 由1变为3或者3变为1时进行判断

            vehicle_info = {"车辆号": CL, "充电或驾驶行为的开始时间": date, "车辆状态（1为停车充电  3为未充电状态）": Vehicle_statue, "驾驶或充电行为开始时的SOC值": SOC,
                            "驾驶或充电行为开始时的累计里程": Mileage, "驾驶或充电行为结束时的能耗/充电": str(eneryQ)}
            result.append(vehicle_info)

            Vehicle_statue = Vehicle_statue_1
            eneryQ = 0
            date_1 = xls_table.row_values(i)[0]  # 获取该帧的时间
            date_2 = xlrd.xldate_as_datetime(date_1, 0)
            date = date_2.strftime('%Y/%m/%d %H:%M:%S')
            SOC = xls_table.row_values(i)[7]  # 获取SOC值
            Mileage = xls_table.row_values(i)[4]  # 获取里程
            continue

        chargingI = float(xls_table.row_values(i)[6])
        eneryQ = eneryQ + chargingI / 180

    pf = pd.DataFrame(list(result))

    order = ['车辆号', '充电或驾驶行为的开始时间', '车辆状态（1为停车充电  3为未充电状态）', '驾驶或充电行为开始时的SOC值', '驾驶或充电行为开始时的累计里程', '驾驶或充电行为结束时的能耗/充电']

    pf = pf[order]
    # 指定生成的Excel表格名称
    pf.to_csv(CL + '充电能耗分析_' + START +'.csv', index=False)







