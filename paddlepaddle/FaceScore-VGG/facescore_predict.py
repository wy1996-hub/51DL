from facescore_train_valid import *
import numpy as np
from PIL import Image

# step5模型预测

# 模型预测-使用已保存好的模型对图像中的人脸进行打分
infer_exe = fluid.Executor(place)
# 图片预处理
def load_image(file):
    im = Image.open(file)
    im = im.resize((224, 224), Image.ANTIALIAS) # resize image with high-quality 图像大小为224*224
    if np.array(im).shape[2] == 4: # 判断图片是否为4通道，若为四通道，则转换为3通道
        im = np.array(im)[:,:,:3]
    im = np.array(im).reshape(1, 3, 224, 224).astype(np.float32) # 返回新形状的数组,把它变成一个 numpy 数组以匹配数据馈送格式。
    im = im / 255.0                            # 归一化到【-1~1】之间
    return im

infer_img='/home/aistudio/data/data19168/image1.png'
# 获取训练好的模型
# 从指定目录中加载 推理model(inference model)
[inference_program, # 预测用的program
feed_target_names, # 是一个str列表，它包含需要在推理 Program 中提供数据的变量的名称。
fetch_targets] = fluid.io.load_inference_model(model_save_dir, infer_exe) # fetch_targets：是一个 Variable 列表，从中我们可以得到推断结果。

img = Image.open(infer_img)
plt.imshow(img)   # 根据数组绘制图像
plt.show()        # 显示图像

image=load_image(infer_img)

# 开始预测
results = infer_exe.run(
                        inference_program,                      # 运行预测程序
                        feed={feed_target_names[0]: image},     # 喂入要预测的数据
                        fetch_list=fetch_targets)               # 得到推测结果
print('results',results)
label_list = ["1", "2", "3", "4", "5"]
print("infer results: %s" % label_list[np.argmax(results[0])])