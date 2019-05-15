import os

file_dir = 'C:/Users/sn_96/Desktop/毕业设计_基于神经网络/Database/PETA dataset_19000/rescale/'
target_dir = 'C:/Users/sn_96/Desktop/毕业设计_基于神经网络/Database/PETA dataset_19000/val/'
filename = []
for _, _, _file in os.walk(file_dir):
    filename = _file
print(filename)

for _id, file in enumerate(filename):
    if _id % 5 == 0:
        os.rename(file_dir + file, target_dir + file)
