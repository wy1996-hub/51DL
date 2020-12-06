import os
import zipfile
import paddle
from multiprocessing import cpu_count

# step1.数据准备

# 解压数据集
src_path="I:/workspace/python/paddlepaddle/FaceScore/data/data17941/face_data_5.zip"
target_path="I:/workspace/python/paddlepaddle/FaceScore/data/face_image"
if(not os.path.isdir(target_path)):
    z = zipfile.ZipFile(src_path, 'r')
    z.extractall(path=target_path)
    z.close()

# 1、定义数据提供器 data_reader
# 2、定义train_reader、test_reader
def data_mapper(data):
    img, label = data
    img = paddle.dataset.image.load_and_transform(img,224,224, False).astype('float32')  #img.shape是(3, 224, 224)
    img = img / 255.0
    return img, int(label)

def data_reader(data_path, buffered_size=512):
  print(data_path)
  def reader():
      for image in os.listdir(data_path):
          label = int(image.split('-')[0])-1
          img = os.path.join(data_path+ '/' + image)
          yield img, label
  return paddle.reader.xmap_readers(data_mapper, reader, cpu_count(), buffered_size)

#构造训练、测试数据提供器
BATCH_SIZE = 16
train_r = data_reader(data_path='I:/workspace/python/paddlepaddle/FaceScore/data/face_image/face_image_train')
train_reader = paddle.batch(paddle.reader.shuffle(reader=train_r,buf_size=128),batch_size=BATCH_SIZE)
test_r = data_reader(data_path='I:/workspace/python/paddlepaddle/FaceScore/data/face_image/face_image_test')
test_reader = paddle.batch(test_r, batch_size=BATCH_SIZE)