"""
Author: Yixuan Su
Date: 2025/03/04 11:10
File: demo_dmeo.py
Description: 

"""
import os

import numpy as np

folder_path = r'E:\ABB-Project\ABB_wrs\suyixuan\ABB\data\out_rack\ob_in_cam'

files_list = sorted([f for f in os.listdir(folder_path) if f.endswith(".txt")])

select_files = [f for f in files_list if 680 <= int(f.split(".")[0]) <= 685]

for file_name in select_files:
    with open(os.path.join(folder_path, file_name), "r") as f:

        lines = f.readlines()
        print(lines)

        matrix = np.array([[float(num) for num in line.split()] for line in lines])

        if matrix.shape==(4, 4):
            print(matrix)
