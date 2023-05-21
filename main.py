from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers import LeakyReLU
from keras.layers import UpSampling2D, Conv2D
from keras.models import Model,load_model
from keras.optimizers import Adam
from skimage import io,transform,util
from PIL import Image
import numpy as np
import datetime, sys, os

from DataGenerator import DataGenerator

class Pix2Pix():
  def __init__(self):
    self.img_rows = 256
    self.img_cols = 256
    self.channels = 1
    self.img_shape = (self.img_rows, self.img_cols, self.channels)

    self.data_loader = DataGenerator()

    # calculate output shape of D (PatchGAN)
    patch = int(self.img_rows / 2**4)
    self.disc_patch = (patch, patch, 1)

    # number of filters in the first layer of G and D
    self.gf = 64
    self.df = 64

    # build discriminator
    self.discriminator = self.build_discriminator()
    self.discriminator.compile(
      loss='mse',
      optimizer='adam',
      metrics=['accuracy']
    )

    # build generator
    self.generator = self.build_generator()

    img_A = Input(shape=self.img_shape)
    img_B = Input(shape=self.img_shape)

    fake_B = self.generator(img_A)

    # do not train discriminator while training generator
    self.discriminator.trainable = False

    valid = self.discriminator([img_A, fake_B])

    self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_B])
    self.combined.compile(
      loss=['mse', 'mae'],
      loss_weights=[1, 100],
      optimizer='adam'
    )

  def build_generator(self):
    def conv2d(layer_input, filters, kernel_size=4, bn=True):
      d = Conv2D(filters, kernel_size=kernel_size, strides=2, padding='same')(layer_input)
      d = LeakyReLU(alpha=0.2)(d)
      if bn:
        d = BatchNormalization(momentum=0.8)(d)
      return d

    def deconv2d(layer_input, skip_input, filters, kernel_size=4, dropout_rate=0):
      u = UpSampling2D(size=2)(layer_input)
      u = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same', activation='relu')(u)
      if dropout_rate:
        u = Dropout(dropout_rate)(u)
      u = BatchNormalization(momentum=0.8)(u)
      u = Concatenate()([u, skip_input])
      return u

    # image input
    d0 = Input(shape=self.img_shape)

    # downsampling
    d1 = conv2d(d0, self.gf, bn=False)
    d2 = conv2d(d1, self.gf*2)
    d3 = conv2d(d2, self.gf*4)
    d4 = conv2d(d3, self.gf*8)
    d5 = conv2d(d4, self.gf*8)
    d6 = conv2d(d5, self.gf*8)
    d7 = conv2d(d6, self.gf*8)

    # upsampling
    u1 = deconv2d(d7, d6, self.gf*8)
    u2 = deconv2d(u1, d5, self.gf*8)
    u3 = deconv2d(u2, d4, self.gf*8)
    u4 = deconv2d(u3, d3, self.gf*4)
    u5 = deconv2d(u4, d2, self.gf*2)
    u6 = deconv2d(u5, d1, self.gf)

    u7 = UpSampling2D(size=2)(u6)

    output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

    return Model(d0, output_img)

  def build_discriminator(self):

    def d_layer(layer_input, filters, kernel_size=4, bn=True):
      d = Conv2D(filters, kernel_size=kernel_size, strides=2, padding='same')(layer_input)
      d = LeakyReLU(alpha=0.2)(d)
      if bn:
        d = BatchNormalization(momentum=0.8)(d)
      return d

    img_A = Input(shape=self.img_shape)
    img_B = Input(shape=self.img_shape)

    combined_imgs = Concatenate(axis=-1)([img_A, img_B])

    d1 = d_layer(combined_imgs, self.df, bn=False)
    d2 = d_layer(d1, self.df*1)
    d3 = d_layer(d2, self.df*2)
    d4 = d_layer(d3, self.df*4)

    validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

    return Model([img_A, img_B], validity)

  def train(self, epochs, batch_size=1, sample_interval=50):
    start_time = datetime.datetime.now()
    num = 0
    # adversarial loss ground truths
    valid = np.ones((batch_size,) + self.disc_patch)
    fake = np.zeros((batch_size,) + self.disc_patch)

    for epoch in range(epochs):
      for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):

        # ---------------------
        #  Train Discriminator
        # ---------------------
        fake_B = self.generator.predict(imgs_A)

        if np.random.random() < 0.5:
          d_loss = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
        else:
          d_loss = self.discriminator.train_on_batch([imgs_A, fake_B], fake)
        # d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # -----------------
        #  Train Generator
        # -----------------
        g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_B])

        # print
        elapsed_time = datetime.datetime.now() - start_time

        print('[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s' % (epoch, epochs, batch_i, self.data_loader.n_batches, d_loss[0], 100*d_loss[1], g_loss[0], elapsed_time))

        # save sample images
        if batch_i % sample_interval == 0 and d_loss[1] < 0.3:
          self.sample_images(num)
          num += 1

        # save sample images when discriminator has low accuracy
        # if epoch > 9 and d_loss[1] < 0.3:
        #   self.sample_images(num)
        #   num += 1
      
      # 每25个epoch在测试集上运行一次
      if (epoch+1) % 25 == 0:
        gan.test(180, epoch+1)


  def sample_images(self, num):
    path = ['samples/img_A','samples/img_B','samples/fake']
    os.makedirs(path[0], exist_ok=True)
    os.makedirs(path[1], exist_ok=True)
    os.makedirs(path[2], exist_ok=True)

    imgs_A, imgs_B = self.data_loader.load_data(batch_size=1, is_train=False)
    fake_B = self.generator.predict(imgs_A)    

    gen_imgs = [imgs_A, imgs_B, fake_B]
    for i in range(3):
      savept = path[i] + '/' + str(num) + '.jpg'
      self.draw(gen_imgs[i], savept)

  
  def test(self, test_size, epoch):
    path = []
    p1 = 'result'+str(epoch)+'/'+'imgs'
    path.append(p1)
    p2 = 'result'+str(epoch)+'/'+'fake'
    path.append(p2)
    os.makedirs(path[0], exist_ok=True)
    os.makedirs(path[1], exist_ok=True)
    for index in range(test_size):
      imgs_A = self.data_loader.load_test_data(index, is_train=False)
      fake = self.generator.predict(imgs_A)
      
      num = index+1
      savept = path[0] + '/' + str(num) + '.jpg'
      self.draw(imgs_A, savept)
      savept = path[1] + '/' + str(num) + '.jpg'
      self.draw(fake, savept)
    
  def draw(self, data, path):
    data = data.squeeze()     # 1*256*256*1 -->  256*256
    data = data*0.5+0.5       # 将data的值变为0~1之间
    little_img = util.img_as_ubyte(data)      # float32 --> utf8
    little_img_rgb = np.stack((little_img,)*3, axis=-1)   # L --> RGB
    big_img = transform.resize(little_img_rgb, (1024, 1024))       # 256*256  -->  1024*1024
    big_img = util.img_as_ubyte(big_img)
    img = Image.fromarray(big_img)
    img.save(path, 'JPEG', quality=95)


if __name__ == '__main__':
  # param
  epochs=75
  batch_size=4
  sample_interval=100
  test_size=180

  gan = Pix2Pix()
  gan.train(epochs=epochs, batch_size=batch_size, sample_interval=sample_interval)  # batch = 100/1 = 100   savepic = 100/25 * 50 = 200
  # gan.test(test_size)
