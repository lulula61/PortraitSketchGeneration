from skimage import io, transform, color
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import os

class DataGenerator():
  def __init__(self):
    pass

  def process(self, batch_path, is_train):
    imgs_A, imgs_B = [], []

    for img_path in batch_path:
      img_A = io.imread(img_path)
      img_B = io.imread(os.path.join('edges2portrait/trainB', os.path.basename(img_path)))  # 1024*1024S
      # img_A = io.imread(img_path, as_gray=True)
      # img_B = io.imread(os.path.join('edges2portrait/trainB', os.path.basename(img_path)), as_gray=True)  # 1024*1024S

      # 将图片从1024*1024转化为256*256
      # 缩小图片并转换为 RGB 模式（即，3通道图片）
      size = (256, 256)
      resized_img_a = transform.resize(img_A, size)
      resized_img_b = transform.resize(img_B, size)

      # 将 RGB 图像转换为灰度图（即，1通道图片）
      gray_img_a = color.rgb2gray(resized_img_a)
      gray_img_b = color.rgb2gray(resized_img_b)

      # 转换为整型数据类型
      img_A = np.uint8(gray_img_a * 255)
      img_B = np.uint8(gray_img_b * 255)

      if is_train and np.random.random() < 0.5:
        img_A = np.fliplr(img_A)
        img_B = np.fliplr(img_B)

      imgs_A.append(np.expand_dims(img_A, axis=-1))
      imgs_B.append(np.expand_dims(img_B, axis=-1))

    imgs_A = np.array(imgs_A) / 127.5 - 1.
    imgs_B = np.array(imgs_B) / 127.5 - 1.

    return imgs_A, imgs_B

  def load_data(self, batch_size=1, is_train=True):
    listA = glob('edges2portrait/trainA/*.jpg')

    batch_path = np.random.choice(listA, size=batch_size)

    imgs_A, imgs_B = self.process(batch_path, is_train)

    return imgs_A, imgs_B

  def load_test_data(self, index, is_train=False):
    listA = glob('edges2portrait/test/*.jpg')

    img_path = listA[index]

    imgs_A = []
    img_A = io.imread(img_path)
    size = (256, 256)
    resized_img_a = transform.resize(img_A, size)

    # 将 RGB 图像转换为灰度图（即，1通道图片）
    gray_img_a = color.rgb2gray(resized_img_a)
   

    # 转换为整型数据类型
    img_A = np.uint8(gray_img_a * 255)

    imgs_A.append(np.expand_dims(img_A, axis=-1))

    imgs_A = np.array(imgs_A) / 127.5 - 1.
    

    return imgs_A

  def load_batch(self, batch_size=1, is_train=True):
    listA = glob('edges2portrait/trainA/*.jpg')

    self.n_batches = int(len(listA) / batch_size)

    for i in range(self.n_batches-1):
      batch_path = listA[i*batch_size:(i+1)*batch_size]
      
      imgs_A, imgs_B = self.process(batch_path, is_train)

      yield imgs_A, imgs_B

if __name__ == '__main__':
  dg = DataGenerator()
  a = dg.load_test_data(0, is_train=False)

  print(a)