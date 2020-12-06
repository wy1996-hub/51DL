from data_reader import *
from clas_nn import *
import matplotlib.pyplot as plt

use_cuda = True
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

# 将数据喂给网络模型
feeder = fluid.DataFeeder(feed_list=[image, label], place=place)

all_train_iter = 0
all_train_iters = []
all_train_costs = []
all_train_accs = []

all_test_iter = 0
all_test_iters = []
all_test_costs = []
all_test_accs = []

def draw_train_process(title, iters, costs, accs, label_cost, lable_acc):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel("cost/acc", fontsize=20)
    plt.plot(iters, costs, color='red', label=label_cost)
    plt.plot(iters, accs, color='green', label=lable_acc)
    plt.legend()
    plt.grid()
    plt.show()

EPOCH_NUM = 20
for pass_id in range(EPOCH_NUM):
    train_cost = 0
    for batch_id, data in enumerate(train_reader()):  # 遍历train_reader的迭代器，并为数据加上索引batch_id
        train_cost, train_acc = exe.run(
            program=fluid.default_main_program(),  # 运行主程序
            feed=feeder.feed(data),  # 喂入一个batch的数据
            fetch_list=[avg_cost, accuracy])  # fetch均方误差和准确率

        all_train_iter = all_train_iter + BATCH_SIZE
        all_train_iters.append(all_train_iter)
        all_train_costs.append(train_cost[0])
        all_train_accs.append(train_acc[0])

        if batch_id % 100 == 0:  # 每100次batch打印一次训练、进行一次测试
            print("\nPass %d, Step %d, Cost %f, Acc %f" %
                  (pass_id, batch_id, train_cost[0], train_acc[0]))

    test_accs = []  # 测试的损失值
    test_costs = []  # 测试的准确率
    # 每训练一轮 进行一次测试
    for batch_id, data in enumerate(eval_reader()):  # 遍历test_reader
        test_cost, test_acc = exe.run(program=fluid.default_main_program(),  # #运行测试主程序
                                      feed=feeder.feed(data),  # 喂入一个batch的数据
                                      fetch_list=[avg_cost, accuracy])  # fetch均方误差、准确率
        test_accs.append(test_acc[0])  # 记录每个batch的误差
        test_costs.append(test_cost[0])  # 记录每个batch的准确率

        all_test_iter = all_test_iter + BATCH_SIZE
        all_test_iters.append(all_test_iter)
        all_test_costs.append(test_cost[0])
        all_test_accs.append(test_acc[0])

    test_cost = (sum(test_costs) / len(test_costs))  # 每轮的平均误差
    test_acc = (sum(test_accs) / len(test_accs))  # 每轮的平均准确率
    print('Test:%d, Cost:%0.5f, ACC:%0.5f' % (pass_id, test_cost, test_acc))

    model_save_dir = "I:/workspace/python/paddlepaddle/FishClas/work/model"
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    # 保存训练的模型，executor 把所有相关参数保存到 dirname 中
    fluid.io.save_inference_model(model_save_dir,  # 保存推理model的路径
                                  ['image'],  # 推理（inference）需要 feed 的数据
                                  [predict],  # 保存推理（inference）结果的 Variables
                                  exe)  # executor 保存 inference model

print('训练模型保存完成！')
draw_train_process("training", all_train_iters, all_train_costs, all_train_accs, "trainning cost", "trainning acc")
draw_train_process("testing", all_test_iters, all_test_costs, all_test_accs, "testing cost", "testing acc")