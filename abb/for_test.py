# import numpy as np
# import robot_con.gofa_con as gofa_con
# import robot_sim.robots.gofa5.gofa5 as gf5
# from abb.check_grasps import base
#
# rbt_s = gf5.GOFA5()
# rbt_r = gofa_con.GoFaArmController()
# current_jnts = rbt_r.get_jnt_values()
# print(current_jnts)
# start_conf = np.array([0, 0, 0, 0, 0, 0])
# rbt_r.move_j(start_conf)
# base.run()

import pickle

# 读取 pickle 文件
with open('ag145_grasps.pickle', 'rb') as file:
    data = pickle.load(file)

print(data)  # 输出文件中的内容
