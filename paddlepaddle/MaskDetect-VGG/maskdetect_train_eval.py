from maskdetect_data import *
from maskdetect_nn import *
import matplotlib.pyplot as plt

# 三、模型训练 && 四、模型评估

all_train_iter=0
all_train_iters=[]
all_train_costs=[]
all_train_accs=[]

def draw_train_process(title, iters, costs, accs, label_cost, lable_acc):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel("cost/acc", fontsize=20)
    plt.plot(iters, costs,color='red',label=label_cost)
    plt.plot(iters, accs,color='green',label=lable_acc)
    plt.legend()
    plt.grid()
    plt.show()


def draw_process(title, color, iters, data, label):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel(label, fontsize=20)
    plt.plot(iters, data,color=color,label=label)
    plt.legend()
    plt.grid()
    plt.show()


'''
模型训练
'''
# with fluid.dygraph.guard(place = fluid.CUDAPlace(0)):
with fluid.dygraph.guard():
    print(train_parameters['class_dim'])
    print(train_parameters['label_dict'])
    vgg = VGGNet()
    optimizer = fluid.optimizer.AdamOptimizer(learning_rate=train_parameters['learning_strategy']['lr'],
                                              parameter_list=vgg.parameters())
    for epoch_num in range(train_parameters['num_epochs']):
        for batch_id, data in enumerate(train_reader()):
            # windows下的int数据类型和linux下的int数据类型不一致
            dy_x_data = np.array([x[0] for x in data]).astype('float32')
            y_data = np.array([x[1] for x in data]).astype('int64')
            y_data = y_data[:, np.newaxis]

            # 将Numpy转换为DyGraph接收的输入
            img = fluid.dygraph.to_variable(dy_x_data)
            label = fluid.dygraph.to_variable(y_data)

            out, acc = vgg(img, label)
            loss = fluid.layers.cross_entropy(out, label)
            avg_loss = fluid.layers.mean(loss)

            # 使用backward()方法可以执行反向网络
            avg_loss.backward()
            optimizer.minimize(avg_loss)

            # 将参数梯度清零以保证下一轮训练的正确性
            vgg.clear_gradients()

            all_train_iter = all_train_iter + train_parameters['train_batch_size']
            all_train_iters.append(all_train_iter)
            all_train_costs.append(loss.numpy()[0])
            all_train_accs.append(acc.numpy()[0])

            if batch_id % 1 == 0:
                print("Loss at epoch {} step {}: {}, acc: {}".format(epoch_num, batch_id, avg_loss.numpy(), acc.numpy()))

    draw_train_process("training", all_train_iters, all_train_costs, all_train_accs, "trainning cost", "trainning acc")
    draw_process("trainning loss", "red", all_train_iters, all_train_costs, "trainning loss")
    draw_process("trainning acc", "green", all_train_iters, all_train_accs, "trainning acc")

    # 保存模型参数
    fluid.save_dygraph(vgg.state_dict(), "vgg")
    print("Final loss: {}".format(avg_loss.numpy()))

'''
模型校验
'''
with fluid.dygraph.guard():
    model, _ = fluid.load_dygraph("vgg")
    vgg = VGGNet()
    vgg.load_dict(model)
    # 验证部分，则不进行反向求导
    vgg.eval()
    accs = []
    for batch_id, data in enumerate(eval_reader()):
        dy_x_data = np.array([x[0] for x in data]).astype('float32')
        y_data = np.array([x[1] for x in data]).astype('int64')
        y_data = y_data[:, np.newaxis]

        img = fluid.dygraph.to_variable(dy_x_data)
        label = fluid.dygraph.to_variable(y_data)

        out, acc = vgg(img, label)
        lab = np.argsort(out.numpy())
        accs.append(acc.numpy()[0])
print("mean training acc:", np.mean(accs))

print("训练与验证结束")