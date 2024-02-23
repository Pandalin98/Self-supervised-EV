import pstats
p = pstats.Stats('power_battery_profile.txt')
p.sort_stats('cumulative').print_stats(50)  # 按照累计时间排序并打印前10个最耗时的函数