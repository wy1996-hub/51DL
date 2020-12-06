import zipfile
import os
import random
import json

# 解压原始数据集，将fish_image.zip解压至data目录下
src_path = 'I:/workspace/python/paddlepaddle/FishClas/data/fish_image23.zip'
target_path = 'I:/workspace/python/paddlepaddle/FishClas/data/fish_image'
# if(not os.path.isdir(target_path)):
#     z = zipfile.ZipFile(src_path, 'r')
#     z.extractall(path=target_path)
#     z.close()

# 存放所有类别的信息
class_detail = []
# 获取所有类别保存的文件夹名称
class_dirs = os.listdir(target_path+'/fish_image')

data_list_path = 'I:/workspace/python/paddlepaddle/FishClas/data/'

TRAIN_LIST_PATH = data_list_path + 'train.txt'
EVAL_LIST_PATH = data_list_path + 'eval.txt'

# 每次执行代码，首先清空train.txt和eval.txt
with open(TRAIN_LIST_PATH, 'w') as ft:
    ft.truncate(0)
with open(EVAL_LIST_PATH, 'w') as fe:
    fe.truncate(0)

# 总的图像数量
all_class_images = 0
# 存放类别标签
class_label = 0
# 设置要生成文件的路径
data_root_path = "I:/workspace/python/paddlepaddle/FishClas/data/fish_image/fish_image"
# 存储要写进test.txt和train.txt中的内容
trainer_list = []
eval_list = []
# 读取每个类别，['fish_1', 'fish_2', 'fish_3']
for class_dir in class_dirs:
    # 每个类别的信息
    class_detail_list = {}
    eval_sum = 0
    trainer_sum = 0
    # 统计每个类别有多少张图片
    class_sum = 0
    # 获取类别路径
    path = data_root_path + "/" + class_dir
    # 获取所有图片
    img_paths = os.listdir(path)
    for img_path in img_paths:  # 遍历文件夹下的每个图片
        name_path = path + '/' + img_path  # 每张图片的路径
        if class_sum % 10 == 0:  # 每10张图片取一个做验证数据
            eval_sum += 1  # eval_sum为验证数据的数目
            eval_list.append(name_path + "\t%d" % class_label + "\n")
        else:
            trainer_sum += 1
            trainer_list.append(name_path + "\t%d" % class_label + "\n")  # trainer_sum为训练数据的数目
        class_sum += 1  # 每类图片的数目
        all_class_images += 1  # 所有类图片的数目
    class_label += 1
    # 说明的json文件的class_detail数据
    class_detail_list['class_name'] = class_dir  # 类别名称
    class_detail_list['class_label'] = class_label  # 类别标签
    class_detail_list['class_eval_images'] = eval_sum  # 该类数据的测试集数目
    class_detail_list['class_trainer_images'] = trainer_sum  # 该类数据的训练集数目
    class_detail.append(class_detail_list)

random.shuffle(eval_list)
with open(data_list_path + "eval.txt", 'a') as f:
    for eval_image in eval_list:
        f.write(eval_image)

random.shuffle(trainer_list)
with open(data_list_path + "train.txt", 'a') as f2:
    for train_image in trainer_list:
        f2.write(train_image)

# 说明的json文件信息
readjson = {}
readjson['all_class_name'] = data_root_path  # 文件父目录
readjson['all_class_sum'] = class_sum
readjson['all_class_images'] = all_class_images
readjson['class_detail'] = class_detail
jsons = json.dumps(readjson, sort_keys=True, indent=4, separators=(',', ': '))
with open(data_list_path + "readme.json", 'w') as f:
    f.write(jsons)
print('生成数据列表完成！')