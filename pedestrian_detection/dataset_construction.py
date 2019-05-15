import os
from PIL import Image
import math
import random


correct_dir = 'C:/Users/sn_96/Desktop/FYP_Gender_Processing/Database/data_test/'
wrong_dir = 'E:/Cache/INRIAPerson/Train/neg/'
correct_file = []
wrong_file = []

for _, _, file in os.walk(correct_dir):
    correct_file = file
# os.rename(item, )
for _, _, file in os.walk(wrong_dir):
    wrong_file = file

print("Cropping the negative examples:")
for id, file in enumerate(wrong_file):
    if id % 200 == 0:
        print('.', end='')
    img = Image.open(wrong_dir + file)  # Read the img
    rand_length = math.ceil(random.uniform(0, img.size[0] - 96))    # Read the size of each img
    rand_weight = math.ceil(random.uniform(0, img.size[1] - 160))
    box = (rand_length, rand_weight, rand_length + 96, rand_weight + 160)
    img_new = img.crop(box)
    file_type = file.split('.')[-1]
    img_new.save('C:/Users/sn_96/Desktop/FYP_Gender_Processing/Database/pd_dataset/neg/neg_{}.{}'.format(id, file_type))


print("\nCropping the negative examples for twice:")
for id, file in enumerate(wrong_file):
    if id % 200 == 0:
        print('.', end='')
    img = Image.open(wrong_dir + file)  # Read the img
    rand_length = math.ceil(random.uniform(0, img.size[0] - 96))    # Read the size of each img
    rand_weight = math.ceil(random.uniform(0, img.size[1] - 160))
    box = (rand_length, rand_weight, rand_length + 96, rand_weight + 160)
    img_new = img.crop(box)
    file_type = file.split('.')[-1]
    img_new.save('C:/Users/sn_96/Desktop/FYP_Gender_Processing/Database/pd_dataset/neg/neg_{}.{}'.format(
        id + len(wrong_file), file_type))


# print("\nRemoving the positive examples:")
# for id, file in enumerate(correct_file):
#     if id % 200 == 0:
#         print('.', end='')
#     img = Image.open(correct_dir + file)
#     file_type = file.split('.')[-1]
#     img.save('C:/Users/sn_96/Desktop/FYP_Gender_Processing/Database/pd_dataset/pos/pos_{}.{}'.format(id+2000, file_type))
    # 这里的pos图像经过了镜面对称




