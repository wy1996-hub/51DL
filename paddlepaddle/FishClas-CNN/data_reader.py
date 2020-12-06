from fish_data import *
import paddle
from multiprocessing import cpu_count


# 根据TXT里面的路径读取相应的图片，进行相关处理，并吐出img和标签
def data_mapper(sample):
    img_path, label = sample
    # 进行图片的读取，由于数据集的像素维度各不相同，需要进一步处理对图像进行变换
    img = paddle.dataset.image.load_image(img_path)
    # 进行了简单的图像变换，这里对图像进行crop修剪操作，输出img的维度为(3, 47, 47)
    img = paddle.dataset.image.simple_transform(im=img,
                                                resize_size=47,  # 剪裁图片
                                                crop_size=47,
                                                is_color=True,  # 是否彩色图像
                                                is_train=True)  # 是否训练集
    # 将img数组进行进行归一化处理，得到0到1之间的数值
    img = img.flatten().astype('float32') / 255.0
    return img, label

def data_r(file_list, buffered_size=1024):
    def reader():
        with open(file_list, 'r') as f:
            lines = [line.strip() for line in f]
            for line in lines:
                img_path, lab = line.strip().split('\t')
                yield img_path, int(lab)
    return paddle.reader.xmap_readers(data_mapper, reader, cpu_count(), buffered_size)


BATCH_SIZE = 64
BUF_SIZE=512

# 构造训练数据提供器
train_r = data_r(file_list=TRAIN_LIST_PATH)
train_reader = paddle.batch(
    paddle.reader.shuffle(
        reader=train_r,buf_size=BUF_SIZE),
    batch_size=BATCH_SIZE)

# 构造测试数据提供器
eval_r = data_r(file_list=EVAL_LIST_PATH)
eval_reader = paddle.batch(eval_r, batch_size=BATCH_SIZE)