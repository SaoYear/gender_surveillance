import os

train = 'C:/Users/sn_96/Desktop/毕业设计_基于神经网络/Database/data_train/'
test = "C:/Users/sn_96/Desktop/毕业设计_基于神经网络/Database/data_test/"
val = "C:/Users/sn_96/Desktop/毕业设计_基于神经网络/Database/data_val/"
root = "C:/Users/sn_96/Desktop/毕业设计_基于神经网络/Database/rescale/"

def cutmove(root, target, file_name):
    os.rename(root + file_name, target + file_name)

img = []
for _, _, file in os.walk(root):
        img = file

img = [item.split('/')[-1] for item in img]

for id, item in enumerate(img):
    id = id % 10
    if id < 6:
        cutmove(root, train, item)
    elif id < 8:
        cutmove(root, test, item)
    else:
        cutmove(root, val, item)





