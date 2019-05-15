import os
import random
from PIL import Image
import numpy as np


class ImgInput:
    def __init__(self, img_dir, batch_size):
        self.batch_size = batch_size
        self.img_dir = img_dir
        self.img_name = []
        for _, _, _img_name in os.walk(self.img_dir):
            self.img_name = _img_name
        # print(self.img_name)
        random.shuffle(self.img_name)
        self.img, self.label = self.read_img()
        self._num_examples = len(self.img)
        self.indices = list(range(self._num_examples))  # 获取indice
        random.shuffle(self.indices)
        self.total_batch = self._num_examples
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def read_img(self):
        _data = []
        _label = []
        male = 0
        female = 0
        print('Reading images:')
        for img_path in self.img_name:
            if img_path != '__init__.py':                           # 丢掉 init 文件
                _img = Image.open(self.img_dir + str(img_path))      # array [150, 100, 3]
                _lab = img_path.split('_')[2]
                # print(img_path, end='\t\t\t')
                _data.append(np.array(_img))
                # print('GOAL!')
                if _lab == '1':
                    male += 1
                    _label.append([0, 1])
                else:
                    female += 1
                    _label.append([1, 0])
                _img.close()
        print('Total:', male + female,
              '\nMale:', male, male / (male + female + 1) * 100, '%',
              '\nFemale:', female, female / (male + female + 1) * 100, '%')
        return _data, _label    # [total batch, 150, 100, 3]

    def return_batch_indices(self):
        start = self._index_in_epoch  # 初始是0 随后会随着batch的值改变

        # Go to the next epoch
        if start + self.batch_size > self._num_examples:  # 如果初始加下一个batch_size后的例子超过了所有的样本数
            self._epochs_completed += 1  # Finished epoch, epoch++
            rest_num_examples = self._num_examples - start  # 获取剩下example的数量
            filenames_rest_part = self.indices[start:self._num_examples]  # 获得剩下example的indice
            start = 0
            # 对于下一个epoch来说，batch_size - rest = 下个epoch需要提供的example的数量来满足上个epoch的最后一个batch
            self._index_in_epoch = self.batch_size - rest_num_examples
            end = self._index_in_epoch
            filenames_new_part = self.indices[start:end]  # 把下一个epoch的indice取出来
            # 最后拼接成一个完整的batch，然后下一个start的值会更新为self._index_in_epoch
            return filenames_rest_part + filenames_new_part
        else:
            self._index_in_epoch += self.batch_size  # 如果没有进入epoch则index+batch_size
            end = self._index_in_epoch  # 更新batch的值
            return self.indices[start:end]  # start:end 就是需要处理的数据indices

    def next_batch(self):  # 根据上面函数的传入的indice来构成训练的数据
        batch_indices = self.return_batch_indices()  # 获取上面的indice
        _img = [self.img[i] for i in batch_indices]  # 获取data
        _label = [self.label[i] for i in batch_indices]
        return _img, _label


# data = ImgInput('C:/Users/sn_96/Desktop/毕业设计_基于神经网络/Database/PETA dataset_19000/rescale/', 100)
# data.next_batch()
