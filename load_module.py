import tensorflow as tf
import numpy as np
import NN_code.Image_Process.test_process as t
from PIL import Image

img = Image.open('DIR')
print(np.asarray(img).shape)
img = img.resize((96, 160))
img.show()
img_ary = np.array(img)
img_list = []
img_list.append(img_ary)


with tf.Session() as sess:
    saver = tf.train.import_meta_graph('DIR/model_final.ckpt.meta')
    saver.restore(sess, 'DIR/model_final.ckpt')
    print(sess.run('dense/Softmax:0', feed_dict={'Placeholder:0': 1,'truediv:0': img_list}))
