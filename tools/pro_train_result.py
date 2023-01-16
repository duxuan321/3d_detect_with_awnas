import numpy as np

path = "/home/user1/ytx_workspace/mvlidarnet_pcdet/output/cfgs/kitti_models/pointpillar_bev_free/with_plane_iou/log_train_20220927-131955.txt"

with open(path,"r") as file1:
	contents=file1.readlines()	# 读取所有行，返回一个list

key = "Pedestrian AP@0.50, 0.50, 0.50:\n"

# 找出所有的有效值
count = 70
record = []
for i in range(len(contents)):
    if key==contents[i]:
        # print("car:",contents[i-20:i-16],contents[i-17][17:22])
        # print("ped:",contents[i:i+4],contents[i+3][17:22])
        # print("cyc:",contents[i+20:i+24],contents[i+23][17:22])
        print("3D medium的结果为：")
        print(count,[contents[i-17][17:22],contents[i+3][17:22],contents[i+23][17:22]])
        count+=1

# 打印指定的epoch结果：
target = 75
count = 70
for i in range(len(contents)):
    if key==contents[i]:
        if target == count:
            print("指定的结果为:")
            print(contents[i-20])
            print(contents[i-18])
            print(contents[i-17])
            print(contents[i])
            print(contents[i+2])
            print(contents[i+3])
            print(contents[i+20])
            print(contents[i+22])
            print(contents[i+23])
        count+=1