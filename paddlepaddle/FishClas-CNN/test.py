import paddle.fluid as fluid
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 使用CPU进行预测
place = fluid.CPUPlace()
# 定义一个executor
infer_exe = fluid.Executor(place)
inference_scope = fluid.core.Scope()  # 要想运行一个网络，需要指明它运行所在的域，确切的说： exe.Run(&scope)
# 选择保存不同的训练模型
params_dirname = 'I:/workspace/python/paddlepaddle/FishClas/work/model'

# 图片预处理
def load_image(file):
    im = Image.open(file)
    im = im.resize((47, 47), Image.ANTIALIAS)  # resize image with high-quality 图像大小为28*28
    im = np.array(im).reshape(1, 3, 47, 47).astype(np.float32)  # 返回新形状的数组,把它变成一个 numpy 数组以匹配数据馈送格式。
    im = im / 255.0  # 归一化到【-1~1】之间
    return im

infer_img = 'I:/workspace/python/paddlepaddle/FishClas/work/test/1.png'

# fluid.scope_guard修改全局/默认作用域（scope）, 运行时中的所有变量都将分配给新的scope
with fluid.scope_guard(inference_scope):
    # 获取训练好的模型
    # 从指定目录中加载 推理model(inference model)
    [inference_program,  # 预测用的program
     feed_target_names,  # 是一个str列表，它包含需要在推理 Program 中提供数据的变量的名称。
     fetch_targets] = fluid.io.load_inference_model(params_dirname,
                                                    infer_exe)  # fetch_targets：是一个 Variable 列表，从中我们可以得到推断结果。

    img = Image.open(infer_img)
    plt.imshow(img)  # 根据数组绘制图像
    plt.show()  # 显示图像

    image = load_image(infer_img)

    # 开始预测
    results = infer_exe.run(
        inference_program,  # 运行预测程序
        feed={feed_target_names[0]: image},  # 喂入要预测的数据
        fetch_list=fetch_targets)  # 得到推测结果
    print('results', results)
    label_list = ["fish_1", "fish_2", "fish_3", "fish_4", "fish_5"]
    print("infer results: %s" % label_list[np.argmax(results[0])])