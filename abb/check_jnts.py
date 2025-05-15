"""

首先通过 go_init() 函数让机械臂回到起点.接着进入无限循环,持续获取并打印机械臂的关节扭矩值
通过这种方式,您可以确保机械臂回到起点并且继续执行后续的操作

"""

import numpy as np
from motion.probabilistic.comparison_rrt_connect import rrtc_s
import robot_sim.robots.gofa5.gofa5 as gf5
import robot_con.gofa_con.gofa_con as gofa_con


def go_init():
    init_jnts = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    rbt_s = gf5.GOFA5()
    current_jnts = rbt_s.get_jnt_values("arm")

    path = rrtc_s.plan(component_name="arm",
                       start_conf=current_jnts,
                       goal_conf=init_jnts,
                       ext_dist=0.05,
                       max_time=300)

    # 将机械臂沿着规划的路径移动到目标位置
    rbt_r.move_jntspace_path(path)


if __name__ == '__main__':
    rbt_r = gofa_con.GoFaArmController()
    go_init()
    rbt_r.get_torques()
    print(rbt_r.get_torques())
    while True:
        print(rbt_r.get_torques())  # 进入一个无限循环,不断获取并打印机械臂的关节扭矩值
        current_jnts = rbt_r.get_jnt_values()  # 获取当前关节值并打印
        print(current_jnts)
