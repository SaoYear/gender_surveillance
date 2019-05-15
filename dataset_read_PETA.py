import os
import numpy as np
from PIL import Image


class ImgRead:
    def __init__(self, file_dir, padding='ONE'):
        self.cache = ['3DPeS', 'CAVIAR4REID', 'CUHK', 'GRID', 'i-LID', 'MIT', 'PRID', 'SARC3D', 'TownCentre',  'VIPeR']
        self.file_dir = file_dir
        self.padding = padding
        self.filenames = []
        for _, _, file in os.walk(self.file_dir):   # 读取路径下所有的文件名
            self.filenames.append(file)
        self.filenames = self.filenames[0]
        self.filename_dir = [self.file_dir + item for item in self.filenames]
        self.label_file = self.filename_dir[-1]                         # 得到label文件的地址
        self.filename_dir.pop()  # 获取所有图片的地址

        self.img_ary = []       # 图片的数值化,[{'person_id': int, 'img':array}]
        self.label = {}         # label提取的数值化,[{'person_id': int, 'label': label}]
                                # 收集好之后需要做对齐
        self.read_label()
        self.read_img()

    def read_img(self):         # 对不同数据集通用
        for _id, img_path in enumerate(self.filename_dir):
            img = Image.open(img_path)
            p_id = img_path.split('/')[-1].split('.')[0]    # cuhk
            # p_id = img_path.split('/')[-1].split('_')[0]
            img = img.resize((96, 160))
            gender = self.label.get(int(p_id))
            filename = 'C:/Users/sn_96/Desktop/毕业设计_基于神经网络/Database/rescale/{person}_{id}_{gender}_.{fm}'.format(
                                                                                     person=p_id,
                                                                                     id=_id,
                                                                                     gender=gender,
                                                                                     fm=img_path.split('.')[-1])
            img.save(filename)
            self.img_ary.append({int(p_id): np.array(img)})

    def read_label(self):       # 对不同数据集通用，输出为一个list [照片人数， 标签 ]
        with open(self.label_file, 'r') as f:
            content = f.readlines()
            male = 0
            female = 0
        for item in content:
            if item.find('personalFemale')+1:
                self.label[int(item.split(' ')[0])] = 0
                female += 1
            else:
                self.label[int(item.split(' ')[0])] = 1
                male += 1
        print('Male:', male, 'Female:', female)





# cache = ['3DPeS', 'CAVIAR4REID', 'GRID', 'i-LID', 'MIT', 'PRID', 'SARC3D', 'TownCentre',  'VIPeR']
# for item in cache:
#     print(item)
#     try1 = ImgRead('C:/Users/sn_96/Desktop/毕业设计_基于神经网络/Database/PETA dataset_19000/{}/archive/'.format(item))
#     try1.read_img()
print('CUHK')
try1 = ImgRead('C:/Users/sn_96/Desktop/毕业设计_基于神经网络/Database/PETA dataset_19000/CUHK/archive/')
try1.read_img()

