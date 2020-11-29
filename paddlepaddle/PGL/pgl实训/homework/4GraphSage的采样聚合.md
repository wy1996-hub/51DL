# 第四课：图神经网络算法（二）

本节教程将带着同学们理解[GraphSage模型](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf)的关键代码，以便按照自己的需求修改和实现。请参照示例代码，补充实现采样函数和不同的聚合函数。


```python
# 安装依赖
# !pip install paddlepaddle==1.8.5
!pip install pgl
```

    Looking in indexes: https://mirror.baidu.com/pypi/simple/
    Collecting pgl
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/e2/84/6aac242f80a794f1169386d73bdc03f2e3467e4fa85b1286979ddf51b1a0/pgl-1.2.1-cp37-cp37m-manylinux1_x86_64.whl (7.9MB)
    [K     |████████████████████████████████| 7.9MB 22.1MB/s eta 0:00:01
    [?25hRequirement already satisfied: cython>=0.25.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pgl) (0.29)
    Requirement already satisfied: visualdl>=2.0.0b; python_version >= "3" in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pgl) (2.0.3)
    Requirement already satisfied: numpy>=1.16.4 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pgl) (1.16.4)
    Collecting redis-py-cluster (from pgl)
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/2b/c5/3236720746fa357e214f2b9fe7e517642329f13094fc7eb339abd93d004f/redis_py_cluster-2.1.0-py2.py3-none-any.whl (41kB)
    [K     |████████████████████████████████| 51kB 34.1MB/s eta 0:00:01
    [?25hRequirement already satisfied: pre-commit in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0b; python_version >= "3"->pgl) (1.21.0)
    Requirement already satisfied: Pillow>=7.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0b; python_version >= "3"->pgl) (7.1.2)
    Requirement already satisfied: requests in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0b; python_version >= "3"->pgl) (2.22.0)
    Requirement already satisfied: six>=1.14.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0b; python_version >= "3"->pgl) (1.15.0)
    Requirement already satisfied: flask>=1.1.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0b; python_version >= "3"->pgl) (1.1.1)
    Requirement already satisfied: protobuf>=3.11.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0b; python_version >= "3"->pgl) (3.12.2)
    Requirement already satisfied: flake8>=3.7.9 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0b; python_version >= "3"->pgl) (3.8.2)
    Requirement already satisfied: Flask-Babel>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0b; python_version >= "3"->pgl) (1.0.0)
    Collecting redis<4.0.0,>=3.0.0 (from redis-py-cluster->pgl)
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/a7/7c/24fb0511df653cf1a5d938d8f5d19802a88cef255706fdda242ff97e91b7/redis-3.5.3-py2.py3-none-any.whl (72kB)
    [K     |████████████████████████████████| 81kB 17.0MB/s eta 0:00:01
    [?25hRequirement already satisfied: pyyaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0b; python_version >= "3"->pgl) (5.1.2)
    Requirement already satisfied: identify>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0b; python_version >= "3"->pgl) (1.4.10)
    Requirement already satisfied: aspy.yaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0b; python_version >= "3"->pgl) (1.3.0)
    Requirement already satisfied: importlib-metadata; python_version < "3.8" in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0b; python_version >= "3"->pgl) (0.23)
    Requirement already satisfied: toml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0b; python_version >= "3"->pgl) (0.10.0)
    Requirement already satisfied: nodeenv>=0.11.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0b; python_version >= "3"->pgl) (1.3.4)
    Requirement already satisfied: virtualenv>=15.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0b; python_version >= "3"->pgl) (16.7.9)
    Requirement already satisfied: cfgv>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0b; python_version >= "3"->pgl) (2.0.1)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0b; python_version >= "3"->pgl) (1.25.6)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0b; python_version >= "3"->pgl) (2019.9.11)
    Requirement already satisfied: idna<2.9,>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0b; python_version >= "3"->pgl) (2.8)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0b; python_version >= "3"->pgl) (3.0.4)
    Requirement already satisfied: Werkzeug>=0.15 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.0.0b; python_version >= "3"->pgl) (0.16.0)
    Requirement already satisfied: itsdangerous>=0.24 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.0.0b; python_version >= "3"->pgl) (1.1.0)
    Requirement already satisfied: Jinja2>=2.10.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.0.0b; python_version >= "3"->pgl) (2.10.1)
    Requirement already satisfied: click>=5.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.0.0b; python_version >= "3"->pgl) (7.0)
    Requirement already satisfied: setuptools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from protobuf>=3.11.0->visualdl>=2.0.0b; python_version >= "3"->pgl) (41.4.0)
    Requirement already satisfied: mccabe<0.7.0,>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0b; python_version >= "3"->pgl) (0.6.1)
    Requirement already satisfied: pyflakes<2.3.0,>=2.2.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0b; python_version >= "3"->pgl) (2.2.0)
    Requirement already satisfied: pycodestyle<2.7.0,>=2.6.0a1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0b; python_version >= "3"->pgl) (2.6.0)
    Requirement already satisfied: Babel>=2.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl>=2.0.0b; python_version >= "3"->pgl) (2.8.0)
    Requirement already satisfied: pytz in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl>=2.0.0b; python_version >= "3"->pgl) (2019.3)
    Requirement already satisfied: zipp>=0.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from importlib-metadata; python_version < "3.8"->pre-commit->visualdl>=2.0.0b; python_version >= "3"->pgl) (0.6.0)
    Requirement already satisfied: MarkupSafe>=0.23 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Jinja2>=2.10.1->flask>=1.1.1->visualdl>=2.0.0b; python_version >= "3"->pgl) (1.1.1)
    Requirement already satisfied: more-itertools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from zipp>=0.5->importlib-metadata; python_version < "3.8"->pre-commit->visualdl>=2.0.0b; python_version >= "3"->pgl) (7.2.0)
    Installing collected packages: redis, redis-py-cluster, pgl
    Successfully installed pgl-1.2.1 redis-3.5.3 redis-py-cluster-2.1.0


## 1. 代码框架梳理

GraphSage的PGL代码实现位于 [PGL/examples/graphsage/](https://github.com/PaddlePaddle/PGL/tree/main/examples/graphsage)，Notebook中提供了复制版本，主要结构如下

- **数据部分** ./data/

	我们在原 github 上使用的是Reddit数据集。Reddit是一个新闻网站，以消息为节点，如果同一用户在不同消息下都发表了评论，则二者之间有边，用于预测消息属于哪类社区。
    
    但是因为数据集比较大，我们暂时没能放入 AIStudio。所以很遗憾的，这里我们不需要使用 Reddit 数据集进行评估，作业针对的 Acc 结果只需在 Cora 数据集上完成即可，这样也方便大家快速跑出结果。感兴趣的同学，自己跑跑 Reddit 啦！这里贴出数据集链接：[reddit.npz](https://drive.google.com/open?id=19SphVl_Oe8SJ1r87Hr5a6znx3nJu1F2J)和[reddit_adj.npz](https://drive.google.com/open?id=174vb0Ws7Vxk_QTUtxqTgDHSQ4El4qDHt)
    
    由于我们将原本可以全图跑的 Cora数据集进行了分 Batch 跑，因此相对的，测试集能够达到的结果就比我们昨天所讲解的 GCN、GAT 要低了。
    
- **采样部分**  reader.py 

	提供了采样代码。以某个节点为中心，按照距该节点的距离依次采样得到子图，作为训练数据
    
- **模型部分** model.py
	
    提供了聚合函数实现，包括Mean，Maxpool，MeanPool和LSTM四种方式
    
- **训练部分** train.py

	实现了数据读取、模型构建和模型训练部分。

## 2. GraphSage采样函数实现

GraphSage的作者提出了采样算法来使得模型能够以Mini-batch的方式进行训练，算法伪代码见[论文](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf)附录A。

- 假设我们要利用中心节点的k阶邻居信息，则在聚合的时候，需要从第k阶邻居传递信息到k-1阶邻居，并依次传递到中心节点。
- 采样的过程刚好与此相反，在构造第t轮训练的Mini-batch时，我们从中心节点出发，在前序节点集合中采样$N_t$个邻居节点加入采样集合。
- 接着将邻居节点作为新的中心节点继续进行第t-1轮训练的节点采样，以此类推。
- 最后将采样到的节点和边一起构造得到子图。

下面请将GraphSage的采样函数补充完整。


```python
%%writefile userdef_sample.py

import numpy as np

def traverse(item):
    """traverse
    """
    if isinstance(item, list) or isinstance(item, np.ndarray):
        for i in iter(item):
            for j in traverse(i):
                yield j
    else:
        yield item


def flat_node_and_edge(nodes):
    """flat_node_and_edge
    """
    nodes = list(set(traverse(nodes)))
    return nodes


def my_graphsage_sample(graph, batch_train_samples, samples):
    """
    输入：graph - 图结构 Graph
         batch_train_samples - 中心节点 list (batch_size,)
         samples - 采样时的最大邻节点数列表 list 
    输出：被采样节点下标的集合 
         对当前节点进行k阶采样后得到的子图 
    """
    
    start_nodes = batch_train_samples
    nodes = start_nodes
    edges = []
    for max_deg in samples:
        #################################
        # 请在这里补充每阶邻居采样的代码：此部分课堂实践内容已详细讲解，加油~
        # 提示：graph.sample_predecessor(该 API用于获取目标节点对应的源节点，具体用法到 pgl.Graph 结构中查看)
        pred_nodes = graph.sample_predecessor(start_nodes, max_degree = max_deg)
        # 根据采样的子节点， 恢复边
        for dst_node, src_nodes in zip(start_nodes, pred_nodes):
            for node in src_nodes:
                edges.append((node, dst_node))
        #################################

        # 合并已采样节点并找出新的节点作为start_nodes
        last_nodes = nodes
        nodes = [nodes, pred_nodes]
        nodes = flat_node_and_edge(nodes)
        start_nodes = list(set(nodes) - set(last_nodes))
        if len(start_nodes) == 0:
            break

    subgraph = graph.subgraph(
         nodes=nodes,
         edges=edges,
         with_node_feat=False,
         with_edge_feat=False)
         
    return nodes, subgraph
```

    Overwriting userdef_sample.py


运行一下代码看看自己实现的采样算法与PGL相比效果如何吧~ 


```python
!python train.py --use_my_sample
```

    [INFO] 2020-11-27 14:07:46,344 [    train.py:  310]:	Namespace(batch_size=128, epoch=50, graphsage_type='graphsage_maxpool', hidden_size=128, lr=0.01, normalize=False, sample_workers=5, samples_1=25, samples_2=10, symmetry=False, use_cuda=False, use_my_lstm=False, use_my_maxpool=False, use_my_sample=True)
    [INFO] 2020-11-27 14:07:47,314 [    train.py:  176]:	preprocess finish
    [INFO] 2020-11-27 14:07:47,314 [    train.py:  177]:	Train Examples: 140
    [INFO] 2020-11-27 14:07:47,315 [    train.py:  178]:	Val Examples: 300
    [INFO] 2020-11-27 14:07:47,315 [    train.py:  179]:	Test Examples: 1000
    [INFO] 2020-11-27 14:07:47,315 [    train.py:  180]:	Num nodes 2708
    [INFO] 2020-11-27 14:07:47,315 [    train.py:  181]:	Num edges 8137
    [INFO] 2020-11-27 14:07:47,315 [    train.py:  182]:	Average Degree 3.00480059084195
    [INFO] 2020-11-27 14:07:47,624 [    train.py:  171]:	train Epoch 0 Loss 1.95629 Acc 0.10000 Speed(per batch) 0.09468 sec
    [INFO] 2020-11-27 14:07:47,754 [    train.py:  171]:	val Epoch 0 Loss 1.80798 Acc 0.35000 Speed(per batch) 0.04322 sec
    [INFO] 2020-11-27 14:07:47,931 [    train.py:  171]:	train Epoch 1 Loss 1.82604 Acc 0.29286 Speed(per batch) 0.08853 sec
    [INFO] 2020-11-27 14:07:48,068 [    train.py:  171]:	val Epoch 1 Loss 1.80123 Acc 0.35000 Speed(per batch) 0.04555 sec
    [INFO] 2020-11-27 14:07:48,245 [    train.py:  171]:	train Epoch 2 Loss 1.82217 Acc 0.29286 Speed(per batch) 0.08810 sec
    [INFO] 2020-11-27 14:07:48,376 [    train.py:  171]:	val Epoch 2 Loss 1.79983 Acc 0.35000 Speed(per batch) 0.04339 sec
    [INFO] 2020-11-27 14:07:48,553 [    train.py:  171]:	train Epoch 3 Loss 1.81503 Acc 0.29286 Speed(per batch) 0.08837 sec
    [INFO] 2020-11-27 14:07:48,683 [    train.py:  171]:	val Epoch 3 Loss 1.77965 Acc 0.35000 Speed(per batch) 0.04335 sec
    [INFO] 2020-11-27 14:07:48,861 [    train.py:  171]:	train Epoch 4 Loss 1.77288 Acc 0.30714 Speed(per batch) 0.08850 sec
    [INFO] 2020-11-27 14:07:48,999 [    train.py:  171]:	val Epoch 4 Loss 1.70808 Acc 0.37000 Speed(per batch) 0.04606 sec
    [INFO] 2020-11-27 14:07:49,189 [    train.py:  171]:	train Epoch 5 Loss 1.66278 Acc 0.44286 Speed(per batch) 0.09465 sec
    [INFO] 2020-11-27 14:07:49,322 [    train.py:  171]:	val Epoch 5 Loss 1.61045 Acc 0.45333 Speed(per batch) 0.04396 sec
    [INFO] 2020-11-27 14:07:49,499 [    train.py:  171]:	train Epoch 6 Loss 1.50166 Acc 0.53571 Speed(per batch) 0.08836 sec
    [INFO] 2020-11-27 14:07:49,630 [    train.py:  171]:	val Epoch 6 Loss 1.49076 Acc 0.49333 Speed(per batch) 0.04346 sec
    [INFO] 2020-11-27 14:07:49,806 [    train.py:  171]:	train Epoch 7 Loss 1.34940 Acc 0.65000 Speed(per batch) 0.08798 sec
    [INFO] 2020-11-27 14:07:49,939 [    train.py:  171]:	val Epoch 7 Loss 1.44040 Acc 0.53333 Speed(per batch) 0.04425 sec
    [INFO] 2020-11-27 14:07:50,121 [    train.py:  171]:	train Epoch 8 Loss 1.21672 Acc 0.67857 Speed(per batch) 0.09053 sec
    [INFO] 2020-11-27 14:07:50,254 [    train.py:  171]:	val Epoch 8 Loss 1.31202 Acc 0.57000 Speed(per batch) 0.04435 sec
    [INFO] 2020-11-27 14:07:50,440 [    train.py:  171]:	train Epoch 9 Loss 1.07870 Acc 0.71429 Speed(per batch) 0.09282 sec
    [INFO] 2020-11-27 14:07:50,572 [    train.py:  171]:	val Epoch 9 Loss 1.22092 Acc 0.61667 Speed(per batch) 0.04372 sec
    [INFO] 2020-11-27 14:07:50,751 [    train.py:  171]:	train Epoch 10 Loss 0.96651 Acc 0.77857 Speed(per batch) 0.08949 sec
    [INFO] 2020-11-27 14:07:50,885 [    train.py:  171]:	val Epoch 10 Loss 1.17209 Acc 0.70000 Speed(per batch) 0.04447 sec
    [INFO] 2020-11-27 14:07:51,070 [    train.py:  171]:	train Epoch 11 Loss 0.87253 Acc 0.92143 Speed(per batch) 0.09203 sec
    [INFO] 2020-11-27 14:07:51,201 [    train.py:  171]:	val Epoch 11 Loss 1.11805 Acc 0.70000 Speed(per batch) 0.04346 sec
    [INFO] 2020-11-27 14:07:51,380 [    train.py:  171]:	train Epoch 12 Loss 0.78332 Acc 0.92857 Speed(per batch) 0.08957 sec
    [INFO] 2020-11-27 14:07:51,510 [    train.py:  171]:	val Epoch 12 Loss 1.06687 Acc 0.74667 Speed(per batch) 0.04322 sec
    [INFO] 2020-11-27 14:07:51,699 [    train.py:  171]:	train Epoch 13 Loss 0.70337 Acc 0.93571 Speed(per batch) 0.09420 sec
    [INFO] 2020-11-27 14:07:51,831 [    train.py:  171]:	val Epoch 13 Loss 1.02686 Acc 0.75333 Speed(per batch) 0.04385 sec
    [INFO] 2020-11-27 14:07:52,010 [    train.py:  171]:	train Epoch 14 Loss 0.63609 Acc 0.93571 Speed(per batch) 0.08922 sec
    [INFO] 2020-11-27 14:07:52,142 [    train.py:  171]:	val Epoch 14 Loss 0.99060 Acc 0.75667 Speed(per batch) 0.04364 sec
    [INFO] 2020-11-27 14:07:52,320 [    train.py:  171]:	train Epoch 15 Loss 0.57406 Acc 0.93571 Speed(per batch) 0.08923 sec
    [INFO] 2020-11-27 14:07:52,451 [    train.py:  171]:	val Epoch 15 Loss 0.95512 Acc 0.75000 Speed(per batch) 0.04325 sec
    [INFO] 2020-11-27 14:07:52,629 [    train.py:  171]:	train Epoch 16 Loss 0.51924 Acc 0.93571 Speed(per batch) 0.08872 sec
    [INFO] 2020-11-27 14:07:52,759 [    train.py:  171]:	val Epoch 16 Loss 0.92708 Acc 0.75333 Speed(per batch) 0.04338 sec
    [INFO] 2020-11-27 14:07:52,943 [    train.py:  171]:	train Epoch 17 Loss 0.46879 Acc 0.93571 Speed(per batch) 0.09163 sec
    [INFO] 2020-11-27 14:07:53,076 [    train.py:  171]:	val Epoch 17 Loss 0.90296 Acc 0.76000 Speed(per batch) 0.04436 sec
    [INFO] 2020-11-27 14:07:53,260 [    train.py:  171]:	train Epoch 18 Loss 0.42165 Acc 0.93571 Speed(per batch) 0.09159 sec
    [INFO] 2020-11-27 14:07:53,394 [    train.py:  171]:	val Epoch 18 Loss 0.88118 Acc 0.75000 Speed(per batch) 0.04459 sec
    [INFO] 2020-11-27 14:07:53,575 [    train.py:  171]:	train Epoch 19 Loss 0.37898 Acc 0.93571 Speed(per batch) 0.09000 sec
    [INFO] 2020-11-27 14:07:53,709 [    train.py:  171]:	val Epoch 19 Loss 0.86691 Acc 0.75667 Speed(per batch) 0.04441 sec
    [INFO] 2020-11-27 14:07:53,894 [    train.py:  171]:	train Epoch 20 Loss 0.33778 Acc 0.99286 Speed(per batch) 0.09238 sec
    [INFO] 2020-11-27 14:07:54,030 [    train.py:  171]:	val Epoch 20 Loss 0.86163 Acc 0.75667 Speed(per batch) 0.04511 sec
    [INFO] 2020-11-27 14:07:54,217 [    train.py:  171]:	train Epoch 21 Loss 0.30016 Acc 1.00000 Speed(per batch) 0.09328 sec
    [INFO] 2020-11-27 14:07:54,346 [    train.py:  171]:	val Epoch 21 Loss 0.85962 Acc 0.74333 Speed(per batch) 0.04295 sec
    [INFO] 2020-11-27 14:07:54,523 [    train.py:  171]:	train Epoch 22 Loss 0.26987 Acc 1.00000 Speed(per batch) 0.08831 sec
    [INFO] 2020-11-27 14:07:54,653 [    train.py:  171]:	val Epoch 22 Loss 0.84581 Acc 0.74667 Speed(per batch) 0.04309 sec
    [INFO] 2020-11-27 14:07:54,831 [    train.py:  171]:	train Epoch 23 Loss 0.24316 Acc 1.00000 Speed(per batch) 0.08889 sec
    [INFO] 2020-11-27 14:07:54,960 [    train.py:  171]:	val Epoch 23 Loss 0.83537 Acc 0.75667 Speed(per batch) 0.04292 sec
    [INFO] 2020-11-27 14:07:55,139 [    train.py:  171]:	train Epoch 24 Loss 0.22020 Acc 1.00000 Speed(per batch) 0.08907 sec
    [INFO] 2020-11-27 14:07:55,269 [    train.py:  171]:	val Epoch 24 Loss 0.83474 Acc 0.76000 Speed(per batch) 0.04316 sec
    [INFO] 2020-11-27 14:07:55,456 [    train.py:  171]:	train Epoch 25 Loss 0.20001 Acc 1.00000 Speed(per batch) 0.09352 sec
    [INFO] 2020-11-27 14:07:55,587 [    train.py:  171]:	val Epoch 25 Loss 0.83880 Acc 0.76333 Speed(per batch) 0.04352 sec
    [INFO] 2020-11-27 14:07:55,766 [    train.py:  171]:	train Epoch 26 Loss 0.18210 Acc 1.00000 Speed(per batch) 0.08892 sec
    [INFO] 2020-11-27 14:07:55,897 [    train.py:  171]:	val Epoch 26 Loss 0.83633 Acc 0.76000 Speed(per batch) 0.04359 sec
    [INFO] 2020-11-27 14:07:56,076 [    train.py:  171]:	train Epoch 27 Loss 0.16630 Acc 1.00000 Speed(per batch) 0.08957 sec
    [INFO] 2020-11-27 14:07:56,209 [    train.py:  171]:	val Epoch 27 Loss 0.83240 Acc 0.75667 Speed(per batch) 0.04416 sec
    [INFO] 2020-11-27 14:07:56,388 [    train.py:  171]:	train Epoch 28 Loss 0.15237 Acc 1.00000 Speed(per batch) 0.08917 sec
    [INFO] 2020-11-27 14:07:56,532 [    train.py:  171]:	val Epoch 28 Loss 0.83215 Acc 0.75667 Speed(per batch) 0.04777 sec
    [INFO] 2020-11-27 14:07:56,710 [    train.py:  171]:	train Epoch 29 Loss 0.13993 Acc 1.00000 Speed(per batch) 0.08890 sec
    [INFO] 2020-11-27 14:07:56,840 [    train.py:  171]:	val Epoch 29 Loss 0.83427 Acc 0.75667 Speed(per batch) 0.04303 sec
    [INFO] 2020-11-27 14:07:57,021 [    train.py:  171]:	train Epoch 30 Loss 0.12887 Acc 1.00000 Speed(per batch) 0.09044 sec
    [INFO] 2020-11-27 14:07:57,152 [    train.py:  171]:	val Epoch 30 Loss 0.83515 Acc 0.75667 Speed(per batch) 0.04356 sec
    [INFO] 2020-11-27 14:07:57,418 [    train.py:  171]:	train Epoch 31 Loss 0.11902 Acc 1.00000 Speed(per batch) 0.13263 sec
    [INFO] 2020-11-27 14:07:57,552 [    train.py:  171]:	val Epoch 31 Loss 0.83520 Acc 0.75667 Speed(per batch) 0.04442 sec
    [INFO] 2020-11-27 14:07:57,735 [    train.py:  171]:	train Epoch 32 Loss 0.11020 Acc 1.00000 Speed(per batch) 0.09139 sec
    [INFO] 2020-11-27 14:07:57,877 [    train.py:  171]:	val Epoch 32 Loss 0.83731 Acc 0.75667 Speed(per batch) 0.04709 sec
    [INFO] 2020-11-27 14:07:58,054 [    train.py:  171]:	train Epoch 33 Loss 0.10227 Acc 1.00000 Speed(per batch) 0.08871 sec
    [INFO] 2020-11-27 14:07:58,185 [    train.py:  171]:	val Epoch 33 Loss 0.84129 Acc 0.75333 Speed(per batch) 0.04332 sec
    [INFO] 2020-11-27 14:07:58,362 [    train.py:  171]:	train Epoch 34 Loss 0.09516 Acc 1.00000 Speed(per batch) 0.08854 sec
    [INFO] 2020-11-27 14:07:58,494 [    train.py:  171]:	val Epoch 34 Loss 0.84417 Acc 0.75333 Speed(per batch) 0.04357 sec
    [INFO] 2020-11-27 14:07:58,671 [    train.py:  171]:	train Epoch 35 Loss 0.08874 Acc 1.00000 Speed(per batch) 0.08839 sec
    [INFO] 2020-11-27 14:07:58,801 [    train.py:  171]:	val Epoch 35 Loss 0.84600 Acc 0.75333 Speed(per batch) 0.04313 sec
    [INFO] 2020-11-27 14:07:58,978 [    train.py:  171]:	train Epoch 36 Loss 0.08295 Acc 1.00000 Speed(per batch) 0.08846 sec
    [INFO] 2020-11-27 14:07:59,118 [    train.py:  171]:	val Epoch 36 Loss 0.84769 Acc 0.75333 Speed(per batch) 0.04634 sec
    [INFO] 2020-11-27 14:07:59,295 [    train.py:  171]:	train Epoch 37 Loss 0.07773 Acc 1.00000 Speed(per batch) 0.08871 sec
    [INFO] 2020-11-27 14:07:59,425 [    train.py:  171]:	val Epoch 37 Loss 0.84985 Acc 0.76000 Speed(per batch) 0.04300 sec
    [INFO] 2020-11-27 14:07:59,604 [    train.py:  171]:	train Epoch 38 Loss 0.07297 Acc 1.00000 Speed(per batch) 0.08916 sec
    [INFO] 2020-11-27 14:07:59,733 [    train.py:  171]:	val Epoch 38 Loss 0.85286 Acc 0.76000 Speed(per batch) 0.04299 sec
    [INFO] 2020-11-27 14:07:59,908 [    train.py:  171]:	train Epoch 39 Loss 0.06867 Acc 1.00000 Speed(per batch) 0.08718 sec
    [INFO] 2020-11-27 14:08:00,037 [    train.py:  171]:	val Epoch 39 Loss 0.85544 Acc 0.76000 Speed(per batch) 0.04301 sec
    [INFO] 2020-11-27 14:08:00,229 [    train.py:  171]:	train Epoch 40 Loss 0.06475 Acc 1.00000 Speed(per batch) 0.09556 sec
    [INFO] 2020-11-27 14:08:00,368 [    train.py:  171]:	val Epoch 40 Loss 0.85798 Acc 0.76000 Speed(per batch) 0.04612 sec
    [INFO] 2020-11-27 14:08:00,545 [    train.py:  171]:	train Epoch 41 Loss 0.06117 Acc 1.00000 Speed(per batch) 0.08825 sec
    [INFO] 2020-11-27 14:08:00,674 [    train.py:  171]:	val Epoch 41 Loss 0.86092 Acc 0.76333 Speed(per batch) 0.04302 sec
    [INFO] 2020-11-27 14:08:00,850 [    train.py:  171]:	train Epoch 42 Loss 0.05791 Acc 1.00000 Speed(per batch) 0.08750 sec
    [INFO] 2020-11-27 14:08:00,980 [    train.py:  171]:	val Epoch 42 Loss 0.86457 Acc 0.76000 Speed(per batch) 0.04345 sec
    [INFO] 2020-11-27 14:08:01,165 [    train.py:  171]:	train Epoch 43 Loss 0.05491 Acc 1.00000 Speed(per batch) 0.09183 sec
    [INFO] 2020-11-27 14:08:01,295 [    train.py:  171]:	val Epoch 43 Loss 0.86830 Acc 0.76333 Speed(per batch) 0.04314 sec
    [INFO] 2020-11-27 14:08:01,479 [    train.py:  171]:	train Epoch 44 Loss 0.05216 Acc 1.00000 Speed(per batch) 0.09212 sec
    [INFO] 2020-11-27 14:08:01,680 [    train.py:  171]:	val Epoch 44 Loss 0.87198 Acc 0.76333 Speed(per batch) 0.06673 sec
    [INFO] 2020-11-27 14:08:01,857 [    train.py:  171]:	train Epoch 45 Loss 0.04961 Acc 1.00000 Speed(per batch) 0.08854 sec
    [INFO] 2020-11-27 14:08:01,987 [    train.py:  171]:	val Epoch 45 Loss 0.87568 Acc 0.76333 Speed(per batch) 0.04314 sec
    [INFO] 2020-11-27 14:08:02,163 [    train.py:  171]:	train Epoch 46 Loss 0.04727 Acc 1.00000 Speed(per batch) 0.08744 sec
    [INFO] 2020-11-27 14:08:02,291 [    train.py:  171]:	val Epoch 46 Loss 0.87893 Acc 0.76333 Speed(per batch) 0.04276 sec
    [INFO] 2020-11-27 14:08:02,468 [    train.py:  171]:	train Epoch 47 Loss 0.04510 Acc 1.00000 Speed(per batch) 0.08791 sec
    [INFO] 2020-11-27 14:08:02,597 [    train.py:  171]:	val Epoch 47 Loss 0.88248 Acc 0.76333 Speed(per batch) 0.04289 sec
    [INFO] 2020-11-27 14:08:02,771 [    train.py:  171]:	train Epoch 48 Loss 0.04309 Acc 1.00000 Speed(per batch) 0.08692 sec
    [INFO] 2020-11-27 14:08:02,918 [    train.py:  171]:	val Epoch 48 Loss 0.88639 Acc 0.76000 Speed(per batch) 0.04885 sec
    [INFO] 2020-11-27 14:08:03,096 [    train.py:  171]:	train Epoch 49 Loss 0.04123 Acc 1.00000 Speed(per batch) 0.08861 sec
    [INFO] 2020-11-27 14:08:03,225 [    train.py:  171]:	val Epoch 49 Loss 0.89057 Acc 0.76333 Speed(per batch) 0.04297 sec
    [INFO] 2020-11-27 14:08:03,540 [    train.py:  171]:	test Epoch 49 Loss 0.92137 Acc 0.72800 Speed(per batch) 0.03934 sec


## 3. GraphSage聚合函数实现

对于GraphSage中的聚合函数，首先用PGL中的Send和Receive接口实现邻居信息的聚合，然后分别学习两个全连接层，映射得到当前节点和邻居信息的表示，最后将二者拼接起来经过L2标准化，得到新的的节点表示。不同聚合函数的区别就在于信息传递机制的不同。

### 3.1 Mean Aggregator示例代码

以下代码实现了Mean Aggregator的消息传递机制，得到邻居聚合信息后的代码与其他聚合函数相同。具体实现细节可参考第三节实践教程中的消息传递机制。

``` python
def graphsage_mean(gw, feature, hidden_size, act, name):
    # 消息的传递和接收
    def copy_send(src_feat, dst_feat, edge_feat):
    	return src_feat["h"]
    def mean_recv(feat):
    	return fluid.layers.sequence_pool(feat, pool_type="average")
    msg = gw.send(copy_send, nfeat_list=[("h", feature)])
    neigh_feature = gw.recv(msg, mean_recv)
    
    # 自身表示和邻居表示的结合
    self_feature = feature
    self_feature = fluid.layers.fc(self_feature,
                                   hidden_size,
                                   act=act,
                                   name=name + '_l')
    neigh_feature = fluid.layers.fc(neigh_feature,
                                    hidden_size,
                                    act=act,
                                    name=name + '_r')
    output = fluid.layers.concat([self_feature, neigh_feature], axis=1)
    output = fluid.layers.l2_normalize(output, axis=1)
    return output
```

### 3.2 MaxPool Aggregator实现

MaxPool Aggregator在进行邻居聚合时会选取最大的值作为当前节点接收到的消息，实现API可参考[Paddle文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/layers_cn.html)。

实际实现的时候，与上述给出的例子 Mean Aggregator 非常类似。大家可以自行填空完成。


```python
%%writefile userdef_maxpool.py
import paddle.fluid as fluid

def my_graphsage_maxpool(gw,
                      feature,
                      hidden_size,
                      act,
                      name,
                      inner_hidden_size=512):
    """
    输入：gw - GraphWrapper对象
         feature - 当前节点表示 (num_nodes, embed_dim)
         hidden_size - 新的节点表示维数 int
         act - 激活函数名 str
         name - 聚合函数名 str
         inner_hidden_size - 消息传递过程中邻居信息的维数 int
    输出：新的节点表示
    """
    
    ####################################
    # 请在这里实现MaxPool Aggregator

    def copy_send(src_feat, dst_feat, edge_feat):
          return src_feat["h"]
    def maxpool_recv(feat):
          return fluid.layers.sequence_pool(feat, pool_type="max")

    # 求和聚合函数
    def sumpool_recv(feat):
          return fluid.layers.sequence_pool(feat, pool_type="sum")

    # 补充消息传递机制触发代码
    neigh_feature = fluid.layers.fc(feature, inner_hidden_size, act="relu")
    msg = gw.send(copy_send, nfeat_list=[("h", neigh_feature)])
    neigh_feature = gw.recv(msg, maxpool_recv)
    ####################################
    
    # 自身表示和邻居表示的结合
    self_feature = feature
    self_feature = fluid.layers.fc(self_feature,
                                   hidden_size,
                                   act=act,
                                   name=name + '_l')
    neigh_feature = fluid.layers.fc(neigh_feature,
                                    hidden_size,
                                    act=act,
                                    name=name + '_r')
    output = fluid.layers.concat([self_feature, neigh_feature], axis=1)
    output = fluid.layers.l2_normalize(output, axis=1)
    return output
```


```python
!python train.py --use_my_maxpool
```

    [INFO] 2020-11-27 14:02:42,444 [    train.py:  310]:	Namespace(batch_size=128, epoch=50, graphsage_type='graphsage_maxpool', hidden_size=128, lr=0.01, normalize=False, sample_workers=5, samples_1=25, samples_2=10, symmetry=False, use_cuda=False, use_my_lstm=False, use_my_maxpool=True, use_my_sample=False)
    [INFO] 2020-11-27 14:02:43,420 [    train.py:  176]:	preprocess finish
    [INFO] 2020-11-27 14:02:43,420 [    train.py:  177]:	Train Examples: 140
    [INFO] 2020-11-27 14:02:43,420 [    train.py:  178]:	Val Examples: 300
    [INFO] 2020-11-27 14:02:43,420 [    train.py:  179]:	Test Examples: 1000
    [INFO] 2020-11-27 14:02:43,420 [    train.py:  180]:	Num nodes 2708
    [INFO] 2020-11-27 14:02:43,420 [    train.py:  181]:	Num edges 8137
    [INFO] 2020-11-27 14:02:43,420 [    train.py:  182]:	Average Degree 3.00480059084195
    [INFO] 2020-11-27 14:02:43,726 [    train.py:  171]:	train Epoch 0 Loss 1.90015 Acc 0.25714 Speed(per batch) 0.09209 sec
    [INFO] 2020-11-27 14:02:43,857 [    train.py:  171]:	val Epoch 0 Loss 1.79026 Acc 0.35000 Speed(per batch) 0.04362 sec
    [INFO] 2020-11-27 14:02:44,030 [    train.py:  171]:	train Epoch 1 Loss 1.82240 Acc 0.29286 Speed(per batch) 0.08581 sec
    [INFO] 2020-11-27 14:02:44,172 [    train.py:  171]:	val Epoch 1 Loss 1.78642 Acc 0.35000 Speed(per batch) 0.04726 sec
    [INFO] 2020-11-27 14:02:44,348 [    train.py:  171]:	train Epoch 2 Loss 1.82115 Acc 0.29286 Speed(per batch) 0.08716 sec
    [INFO] 2020-11-27 14:02:44,478 [    train.py:  171]:	val Epoch 2 Loss 1.78108 Acc 0.35000 Speed(per batch) 0.04326 sec
    [INFO] 2020-11-27 14:02:44,653 [    train.py:  171]:	train Epoch 3 Loss 1.81323 Acc 0.29286 Speed(per batch) 0.08681 sec
    [INFO] 2020-11-27 14:02:44,789 [    train.py:  171]:	val Epoch 3 Loss 1.75395 Acc 0.35000 Speed(per batch) 0.04527 sec
    [INFO] 2020-11-27 14:02:44,967 [    train.py:  171]:	train Epoch 4 Loss 1.77110 Acc 0.29286 Speed(per batch) 0.08833 sec
    [INFO] 2020-11-27 14:02:45,102 [    train.py:  171]:	val Epoch 4 Loss 1.68843 Acc 0.35000 Speed(per batch) 0.04498 sec
    [INFO] 2020-11-27 14:02:45,281 [    train.py:  171]:	train Epoch 5 Loss 1.67405 Acc 0.30714 Speed(per batch) 0.08876 sec
    [INFO] 2020-11-27 14:02:45,413 [    train.py:  171]:	val Epoch 5 Loss 1.58481 Acc 0.44667 Speed(per batch) 0.04368 sec
    [INFO] 2020-11-27 14:02:45,583 [    train.py:  171]:	train Epoch 6 Loss 1.52583 Acc 0.42857 Speed(per batch) 0.08471 sec
    [INFO] 2020-11-27 14:02:45,714 [    train.py:  171]:	val Epoch 6 Loss 1.47736 Acc 0.46667 Speed(per batch) 0.04345 sec
    [INFO] 2020-11-27 14:02:45,885 [    train.py:  171]:	train Epoch 7 Loss 1.36049 Acc 0.55714 Speed(per batch) 0.08504 sec
    [INFO] 2020-11-27 14:02:46,015 [    train.py:  171]:	val Epoch 7 Loss 1.38589 Acc 0.51333 Speed(per batch) 0.04307 sec
    [INFO] 2020-11-27 14:02:46,189 [    train.py:  171]:	train Epoch 8 Loss 1.21250 Acc 0.64286 Speed(per batch) 0.08691 sec
    [INFO] 2020-11-27 14:02:46,319 [    train.py:  171]:	val Epoch 8 Loss 1.30452 Acc 0.55333 Speed(per batch) 0.04310 sec
    [INFO] 2020-11-27 14:02:46,514 [    train.py:  171]:	train Epoch 9 Loss 1.09263 Acc 0.76429 Speed(per batch) 0.09714 sec
    [INFO] 2020-11-27 14:02:46,652 [    train.py:  171]:	val Epoch 9 Loss 1.21955 Acc 0.67333 Speed(per batch) 0.04584 sec
    [INFO] 2020-11-27 14:02:46,827 [    train.py:  171]:	train Epoch 10 Loss 0.97625 Acc 0.90714 Speed(per batch) 0.08751 sec
    [INFO] 2020-11-27 14:02:46,958 [    train.py:  171]:	val Epoch 10 Loss 1.13879 Acc 0.70333 Speed(per batch) 0.04355 sec
    [INFO] 2020-11-27 14:02:47,131 [    train.py:  171]:	train Epoch 11 Loss 0.86728 Acc 0.92857 Speed(per batch) 0.08628 sec
    [INFO] 2020-11-27 14:02:47,264 [    train.py:  171]:	val Epoch 11 Loss 1.07350 Acc 0.73333 Speed(per batch) 0.04399 sec
    [INFO] 2020-11-27 14:02:47,439 [    train.py:  171]:	train Epoch 12 Loss 0.76964 Acc 0.92857 Speed(per batch) 0.08716 sec
    [INFO] 2020-11-27 14:02:47,569 [    train.py:  171]:	val Epoch 12 Loss 1.01665 Acc 0.73667 Speed(per batch) 0.04326 sec
    [INFO] 2020-11-27 14:02:47,750 [    train.py:  171]:	train Epoch 13 Loss 0.67788 Acc 0.94286 Speed(per batch) 0.09051 sec
    [INFO] 2020-11-27 14:02:47,881 [    train.py:  171]:	val Epoch 13 Loss 0.97333 Acc 0.73333 Speed(per batch) 0.04334 sec
    [INFO] 2020-11-27 14:02:48,054 [    train.py:  171]:	train Epoch 14 Loss 0.59430 Acc 0.94286 Speed(per batch) 0.08634 sec
    [INFO] 2020-11-27 14:02:48,184 [    train.py:  171]:	val Epoch 14 Loss 0.96632 Acc 0.74000 Speed(per batch) 0.04324 sec
    [INFO] 2020-11-27 14:02:48,357 [    train.py:  171]:	train Epoch 15 Loss 0.52394 Acc 0.94286 Speed(per batch) 0.08633 sec
    [INFO] 2020-11-27 14:02:48,487 [    train.py:  171]:	val Epoch 15 Loss 0.92447 Acc 0.74000 Speed(per batch) 0.04303 sec
    [INFO] 2020-11-27 14:02:48,661 [    train.py:  171]:	train Epoch 16 Loss 0.46451 Acc 0.94286 Speed(per batch) 0.08664 sec
    [INFO] 2020-11-27 14:02:48,793 [    train.py:  171]:	val Epoch 16 Loss 0.92030 Acc 0.71667 Speed(per batch) 0.04411 sec
    [INFO] 2020-11-27 14:02:48,973 [    train.py:  171]:	train Epoch 17 Loss 0.40760 Acc 0.94286 Speed(per batch) 0.08951 sec
    [INFO] 2020-11-27 14:02:49,104 [    train.py:  171]:	val Epoch 17 Loss 0.88607 Acc 0.73667 Speed(per batch) 0.04345 sec
    [INFO] 2020-11-27 14:02:49,275 [    train.py:  171]:	train Epoch 18 Loss 0.36033 Acc 0.94286 Speed(per batch) 0.08495 sec
    [INFO] 2020-11-27 14:02:49,410 [    train.py:  171]:	val Epoch 18 Loss 0.85521 Acc 0.75000 Speed(per batch) 0.04493 sec
    [INFO] 2020-11-27 14:02:49,583 [    train.py:  171]:	train Epoch 19 Loss 0.31944 Acc 0.94286 Speed(per batch) 0.08575 sec
    [INFO] 2020-11-27 14:02:49,713 [    train.py:  171]:	val Epoch 19 Loss 0.83468 Acc 0.76000 Speed(per batch) 0.04303 sec
    [INFO] 2020-11-27 14:02:49,886 [    train.py:  171]:	train Epoch 20 Loss 0.28460 Acc 0.94286 Speed(per batch) 0.08624 sec
    [INFO] 2020-11-27 14:02:50,016 [    train.py:  171]:	val Epoch 20 Loss 0.82019 Acc 0.76667 Speed(per batch) 0.04296 sec
    [INFO] 2020-11-27 14:02:50,200 [    train.py:  171]:	train Epoch 21 Loss 0.25462 Acc 0.98571 Speed(per batch) 0.09140 sec
    [INFO] 2020-11-27 14:02:50,330 [    train.py:  171]:	val Epoch 21 Loss 0.81067 Acc 0.76333 Speed(per batch) 0.04326 sec
    [INFO] 2020-11-27 14:02:50,504 [    train.py:  171]:	train Epoch 22 Loss 0.22805 Acc 1.00000 Speed(per batch) 0.08630 sec
    [INFO] 2020-11-27 14:02:50,634 [    train.py:  171]:	val Epoch 22 Loss 0.80362 Acc 0.75667 Speed(per batch) 0.04313 sec
    [INFO] 2020-11-27 14:02:50,807 [    train.py:  171]:	train Epoch 23 Loss 0.20506 Acc 1.00000 Speed(per batch) 0.08632 sec
    [INFO] 2020-11-27 14:02:50,937 [    train.py:  171]:	val Epoch 23 Loss 0.79992 Acc 0.76333 Speed(per batch) 0.04297 sec
    [INFO] 2020-11-27 14:02:51,109 [    train.py:  171]:	train Epoch 24 Loss 0.18536 Acc 1.00000 Speed(per batch) 0.08580 sec
    [INFO] 2020-11-27 14:02:51,238 [    train.py:  171]:	val Epoch 24 Loss 0.79616 Acc 0.77333 Speed(per batch) 0.04287 sec
    [INFO] 2020-11-27 14:02:51,418 [    train.py:  171]:	train Epoch 25 Loss 0.16796 Acc 1.00000 Speed(per batch) 0.08957 sec
    [INFO] 2020-11-27 14:02:51,548 [    train.py:  171]:	val Epoch 25 Loss 0.79176 Acc 0.78000 Speed(per batch) 0.04325 sec
    [INFO] 2020-11-27 14:02:51,720 [    train.py:  171]:	train Epoch 26 Loss 0.15259 Acc 1.00000 Speed(per batch) 0.08549 sec
    [INFO] 2020-11-27 14:02:51,849 [    train.py:  171]:	val Epoch 26 Loss 0.78799 Acc 0.77667 Speed(per batch) 0.04298 sec
    [INFO] 2020-11-27 14:02:52,021 [    train.py:  171]:	train Epoch 27 Loss 0.13910 Acc 1.00000 Speed(per batch) 0.08534 sec
    [INFO] 2020-11-27 14:02:52,151 [    train.py:  171]:	val Epoch 27 Loss 0.78542 Acc 0.77667 Speed(per batch) 0.04315 sec
    [INFO] 2020-11-27 14:02:52,325 [    train.py:  171]:	train Epoch 28 Loss 0.12714 Acc 1.00000 Speed(per batch) 0.08652 sec
    [INFO] 2020-11-27 14:02:52,467 [    train.py:  171]:	val Epoch 28 Loss 0.78433 Acc 0.78000 Speed(per batch) 0.04732 sec
    [INFO] 2020-11-27 14:02:52,640 [    train.py:  171]:	train Epoch 29 Loss 0.11647 Acc 1.00000 Speed(per batch) 0.08608 sec
    [INFO] 2020-11-27 14:02:52,770 [    train.py:  171]:	val Epoch 29 Loss 0.78531 Acc 0.78000 Speed(per batch) 0.04312 sec
    [INFO] 2020-11-27 14:02:52,942 [    train.py:  171]:	train Epoch 30 Loss 0.10688 Acc 1.00000 Speed(per batch) 0.08542 sec
    [INFO] 2020-11-27 14:02:53,073 [    train.py:  171]:	val Epoch 30 Loss 0.78827 Acc 0.78333 Speed(per batch) 0.04374 sec
    [INFO] 2020-11-27 14:02:53,252 [    train.py:  171]:	train Epoch 31 Loss 0.09830 Acc 1.00000 Speed(per batch) 0.08844 sec
    [INFO] 2020-11-27 14:02:53,386 [    train.py:  171]:	val Epoch 31 Loss 0.79213 Acc 0.78333 Speed(per batch) 0.04479 sec
    [INFO] 2020-11-27 14:02:53,564 [    train.py:  171]:	train Epoch 32 Loss 0.09062 Acc 1.00000 Speed(per batch) 0.08793 sec
    [INFO] 2020-11-27 14:02:53,710 [    train.py:  171]:	val Epoch 32 Loss 0.79547 Acc 0.78667 Speed(per batch) 0.04866 sec
    [INFO] 2020-11-27 14:02:53,891 [    train.py:  171]:	train Epoch 33 Loss 0.08375 Acc 1.00000 Speed(per batch) 0.08976 sec
    [INFO] 2020-11-27 14:02:54,026 [    train.py:  171]:	val Epoch 33 Loss 0.79792 Acc 0.78333 Speed(per batch) 0.04472 sec
    [INFO] 2020-11-27 14:02:54,206 [    train.py:  171]:	train Epoch 34 Loss 0.07757 Acc 1.00000 Speed(per batch) 0.08971 sec
    [INFO] 2020-11-27 14:02:54,335 [    train.py:  171]:	val Epoch 34 Loss 0.79972 Acc 0.78333 Speed(per batch) 0.04313 sec
    [INFO] 2020-11-27 14:02:54,509 [    train.py:  171]:	train Epoch 35 Loss 0.07201 Acc 1.00000 Speed(per batch) 0.08645 sec
    [INFO] 2020-11-27 14:02:54,641 [    train.py:  171]:	val Epoch 35 Loss 0.80166 Acc 0.78333 Speed(per batch) 0.04389 sec
    [INFO] 2020-11-27 14:02:54,814 [    train.py:  171]:	train Epoch 36 Loss 0.06701 Acc 1.00000 Speed(per batch) 0.08621 sec
    [INFO] 2020-11-27 14:02:54,953 [    train.py:  171]:	val Epoch 36 Loss 0.80419 Acc 0.78333 Speed(per batch) 0.04632 sec
    [INFO] 2020-11-27 14:02:55,128 [    train.py:  171]:	train Epoch 37 Loss 0.06251 Acc 1.00000 Speed(per batch) 0.08684 sec
    [INFO] 2020-11-27 14:02:55,258 [    train.py:  171]:	val Epoch 37 Loss 0.80757 Acc 0.78000 Speed(per batch) 0.04314 sec
    [INFO] 2020-11-27 14:02:55,432 [    train.py:  171]:	train Epoch 38 Loss 0.05846 Acc 1.00000 Speed(per batch) 0.08613 sec
    [INFO] 2020-11-27 14:02:55,562 [    train.py:  171]:	val Epoch 38 Loss 0.81070 Acc 0.78333 Speed(per batch) 0.04324 sec
    [INFO] 2020-11-27 14:02:55,736 [    train.py:  171]:	train Epoch 39 Loss 0.05481 Acc 1.00000 Speed(per batch) 0.08657 sec
    [INFO] 2020-11-27 14:02:55,871 [    train.py:  171]:	val Epoch 39 Loss 0.81459 Acc 0.78333 Speed(per batch) 0.04484 sec
    [INFO] 2020-11-27 14:02:56,053 [    train.py:  171]:	train Epoch 40 Loss 0.05151 Acc 1.00000 Speed(per batch) 0.09087 sec
    [INFO] 2020-11-27 14:02:56,187 [    train.py:  171]:	val Epoch 40 Loss 0.81879 Acc 0.78000 Speed(per batch) 0.04427 sec
    [INFO] 2020-11-27 14:02:56,361 [    train.py:  171]:	train Epoch 41 Loss 0.04852 Acc 1.00000 Speed(per batch) 0.08679 sec
    [INFO] 2020-11-27 14:02:56,493 [    train.py:  171]:	val Epoch 41 Loss 0.82274 Acc 0.78000 Speed(per batch) 0.04371 sec
    [INFO] 2020-11-27 14:02:56,667 [    train.py:  171]:	train Epoch 42 Loss 0.04582 Acc 1.00000 Speed(per batch) 0.08656 sec
    [INFO] 2020-11-27 14:02:56,804 [    train.py:  171]:	val Epoch 42 Loss 0.82607 Acc 0.78000 Speed(per batch) 0.04531 sec
    [INFO] 2020-11-27 14:02:56,980 [    train.py:  171]:	train Epoch 43 Loss 0.04336 Acc 1.00000 Speed(per batch) 0.08720 sec
    [INFO] 2020-11-27 14:02:57,111 [    train.py:  171]:	val Epoch 43 Loss 0.82895 Acc 0.77667 Speed(per batch) 0.04349 sec
    [INFO] 2020-11-27 14:02:57,293 [    train.py:  171]:	train Epoch 44 Loss 0.04113 Acc 1.00000 Speed(per batch) 0.09082 sec
    [INFO] 2020-11-27 14:02:57,503 [    train.py:  171]:	val Epoch 44 Loss 0.83156 Acc 0.77667 Speed(per batch) 0.06996 sec
    [INFO] 2020-11-27 14:02:57,680 [    train.py:  171]:	train Epoch 45 Loss 0.03909 Acc 1.00000 Speed(per batch) 0.08809 sec
    [INFO] 2020-11-27 14:02:57,810 [    train.py:  171]:	val Epoch 45 Loss 0.83427 Acc 0.77667 Speed(per batch) 0.04335 sec
    [INFO] 2020-11-27 14:02:57,986 [    train.py:  171]:	train Epoch 46 Loss 0.03722 Acc 1.00000 Speed(per batch) 0.08754 sec
    [INFO] 2020-11-27 14:02:58,116 [    train.py:  171]:	val Epoch 46 Loss 0.83730 Acc 0.77667 Speed(per batch) 0.04329 sec
    [INFO] 2020-11-27 14:02:58,288 [    train.py:  171]:	train Epoch 47 Loss 0.03551 Acc 1.00000 Speed(per batch) 0.08528 sec
    [INFO] 2020-11-27 14:02:58,417 [    train.py:  171]:	val Epoch 47 Loss 0.84062 Acc 0.77667 Speed(per batch) 0.04307 sec
    [INFO] 2020-11-27 14:02:58,590 [    train.py:  171]:	train Epoch 48 Loss 0.03394 Acc 1.00000 Speed(per batch) 0.08591 sec
    [INFO] 2020-11-27 14:02:58,740 [    train.py:  171]:	val Epoch 48 Loss 0.84414 Acc 0.77667 Speed(per batch) 0.04974 sec
    [INFO] 2020-11-27 14:02:58,914 [    train.py:  171]:	train Epoch 49 Loss 0.03248 Acc 1.00000 Speed(per batch) 0.08649 sec
    [INFO] 2020-11-27 14:02:59,049 [    train.py:  171]:	val Epoch 49 Loss 0.84771 Acc 0.77667 Speed(per batch) 0.04481 sec
    [INFO] 2020-11-27 14:02:59,360 [    train.py:  171]:	test Epoch 49 Loss 0.95433 Acc 0.71800 Speed(per batch) 0.03869 sec


***此截图测试sum聚合函数***
![](https://ai-studio-static-online.cdn.bcebos.com/311c749f9a664de08f752be49e12157084675bb611a940408891663dd5f79aaa)


***lstm聚合函数测试***


```python
%%writefile userdef_lstm.py
import paddle.fluid as fluid

def my_graphsage_lstm(gw,
                      feature,
                      hidden_size,
                      act,
                      name,
                      inner_hidden_size=128):
    """
    输入：gw - GraphWrapper对象
         feature - 当前节点表示 (num_nodes, embed_dim)
         hidden_size - 新的节点表示维数 int
         act - 激活函数名 str
         name - 聚合函数名 str
         inner_hidden_size - 消息传递过程中邻居信息的维数 int
    输出：新的节点表示
    """
    def copy_send(src_feat, dst_feat, edge_feat):
          return src_feat["h"]
    def lstmpool_recv(feat):
        hidden_dim = 128
        forward, _ = fluid.layers.dynamic_lstm(
            input=feat, size=hidden_dim * 4, use_peepholes=False)
        output = fluid.layers.sequence_last_step(forward)
        return output

    hidden_dim=128
    # 补充消息传递机制触发代码
    neigh_feature = fluid.layers.fc(feature, inner_hidden_size, act="relu")
    forward_proj = fluid.layers.fc(input=neigh_feature,size=hidden_dim * 4, bias_attr=False,name="lstm_proj")
    msg = gw.send(copy_send, nfeat_list=[("h", forward_proj)])
    neigh_feature = gw.recv(msg, lstmpool_recv)
    ####################################
    
    # 自身表示和邻居表示的结合
    self_feature = feature
    self_feature = fluid.layers.fc(self_feature,
                                   hidden_size,
                                   act=act,
                                   name=name + '_l')
    neigh_feature = fluid.layers.fc(neigh_feature,
                                    hidden_size,
                                    act=act,
                                    name=name + '_r')
    output = fluid.layers.concat([self_feature, neigh_feature], axis=1)
    output = fluid.layers.l2_normalize(output, axis=1)
    return output
```

    Overwriting userdef_lstm.py



```python
!python train.py --use_my_lstm
```

    [INFO] 2020-11-27 14:46:30,832 [    train.py:  310]:	Namespace(batch_size=128, epoch=50, graphsage_type='graphsage_maxpool', hidden_size=128, lr=0.01, normalize=False, sample_workers=5, samples_1=25, samples_2=10, symmetry=False, use_cuda=False, use_my_lstm=True, use_my_maxpool=False, use_my_sample=False)
    [INFO] 2020-11-27 14:46:31,787 [    train.py:  176]:	preprocess finish
    [INFO] 2020-11-27 14:46:31,787 [    train.py:  177]:	Train Examples: 140
    [INFO] 2020-11-27 14:46:31,787 [    train.py:  178]:	Val Examples: 300
    [INFO] 2020-11-27 14:46:31,787 [    train.py:  179]:	Test Examples: 1000
    [INFO] 2020-11-27 14:46:31,787 [    train.py:  180]:	Num nodes 2708
    [INFO] 2020-11-27 14:46:31,787 [    train.py:  181]:	Num edges 8137
    [INFO] 2020-11-27 14:46:31,787 [    train.py:  182]:	Average Degree 3.00480059084195
    [INFO] 2020-11-27 14:46:32,111 [    train.py:  171]:	train Epoch 0 Loss 1.95309 Acc 0.15714 Speed(per batch) 0.09531 sec
    [INFO] 2020-11-27 14:46:32,241 [    train.py:  171]:	val Epoch 0 Loss 1.80898 Acc 0.35000 Speed(per batch) 0.04334 sec
    [INFO] 2020-11-27 14:46:32,417 [    train.py:  171]:	train Epoch 1 Loss 1.80531 Acc 0.29286 Speed(per batch) 0.08781 sec
    [INFO] 2020-11-27 14:46:32,552 [    train.py:  171]:	val Epoch 1 Loss 1.79475 Acc 0.35000 Speed(per batch) 0.04463 sec
    [INFO] 2020-11-27 14:46:32,728 [    train.py:  171]:	train Epoch 2 Loss 1.77183 Acc 0.29286 Speed(per batch) 0.08772 sec
    [INFO] 2020-11-27 14:46:32,861 [    train.py:  171]:	val Epoch 2 Loss 1.77096 Acc 0.35000 Speed(per batch) 0.04428 sec
    [INFO] 2020-11-27 14:46:33,036 [    train.py:  171]:	train Epoch 3 Loss 1.68965 Acc 0.37143 Speed(per batch) 0.08725 sec
    [INFO] 2020-11-27 14:46:33,169 [    train.py:  171]:	val Epoch 3 Loss 1.69542 Acc 0.41000 Speed(per batch) 0.04428 sec
    [INFO] 2020-11-27 14:46:33,351 [    train.py:  171]:	train Epoch 4 Loss 1.51051 Acc 0.50714 Speed(per batch) 0.09080 sec
    [INFO] 2020-11-27 14:46:33,483 [    train.py:  171]:	val Epoch 4 Loss 1.57789 Acc 0.42000 Speed(per batch) 0.04381 sec
    [INFO] 2020-11-27 14:46:33,657 [    train.py:  171]:	train Epoch 5 Loss 1.29998 Acc 0.62857 Speed(per batch) 0.08691 sec
    [INFO] 2020-11-27 14:46:33,793 [    train.py:  171]:	val Epoch 5 Loss 1.51135 Acc 0.46667 Speed(per batch) 0.04500 sec
    [INFO] 2020-11-27 14:46:33,967 [    train.py:  171]:	train Epoch 6 Loss 1.17892 Acc 0.70714 Speed(per batch) 0.08708 sec
    [INFO] 2020-11-27 14:46:34,099 [    train.py:  171]:	val Epoch 6 Loss 1.44757 Acc 0.49333 Speed(per batch) 0.04385 sec
    [INFO] 2020-11-27 14:46:34,273 [    train.py:  171]:	train Epoch 7 Loss 1.05609 Acc 0.70714 Speed(per batch) 0.08649 sec
    [INFO] 2020-11-27 14:46:34,414 [    train.py:  171]:	val Epoch 7 Loss 1.39659 Acc 0.50000 Speed(per batch) 0.04696 sec
    [INFO] 2020-11-27 14:46:34,589 [    train.py:  171]:	train Epoch 8 Loss 0.94496 Acc 0.80714 Speed(per batch) 0.08737 sec
    [INFO] 2020-11-27 14:46:34,722 [    train.py:  171]:	val Epoch 8 Loss 1.34523 Acc 0.53000 Speed(per batch) 0.04409 sec
    [INFO] 2020-11-27 14:46:34,899 [    train.py:  171]:	train Epoch 9 Loss 0.84671 Acc 0.91429 Speed(per batch) 0.08811 sec
    [INFO] 2020-11-27 14:46:35,034 [    train.py:  171]:	val Epoch 9 Loss 1.29491 Acc 0.55333 Speed(per batch) 0.04510 sec
    [INFO] 2020-11-27 14:46:35,210 [    train.py:  171]:	train Epoch 10 Loss 0.76164 Acc 0.95714 Speed(per batch) 0.08760 sec
    [INFO] 2020-11-27 14:46:35,342 [    train.py:  171]:	val Epoch 10 Loss 1.27133 Acc 0.55333 Speed(per batch) 0.04392 sec
    [INFO] 2020-11-27 14:46:35,524 [    train.py:  171]:	train Epoch 11 Loss 0.68256 Acc 0.97143 Speed(per batch) 0.09069 sec
    [INFO] 2020-11-27 14:46:35,658 [    train.py:  171]:	val Epoch 11 Loss 1.25935 Acc 0.56000 Speed(per batch) 0.04432 sec
    [INFO] 2020-11-27 14:46:35,832 [    train.py:  171]:	train Epoch 12 Loss 0.61430 Acc 0.99286 Speed(per batch) 0.08706 sec
    [INFO] 2020-11-27 14:46:35,966 [    train.py:  171]:	val Epoch 12 Loss 1.24047 Acc 0.57667 Speed(per batch) 0.04440 sec
    [INFO] 2020-11-27 14:46:36,140 [    train.py:  171]:	train Epoch 13 Loss 0.55268 Acc 0.99286 Speed(per batch) 0.08693 sec
    [INFO] 2020-11-27 14:46:36,279 [    train.py:  171]:	val Epoch 13 Loss 1.21481 Acc 0.58333 Speed(per batch) 0.04610 sec
    [INFO] 2020-11-27 14:46:36,460 [    train.py:  171]:	train Epoch 14 Loss 0.49471 Acc 0.99286 Speed(per batch) 0.09044 sec
    [INFO] 2020-11-27 14:46:36,593 [    train.py:  171]:	val Epoch 14 Loss 1.20541 Acc 0.58333 Speed(per batch) 0.04421 sec
    [INFO] 2020-11-27 14:46:36,767 [    train.py:  171]:	train Epoch 15 Loss 0.44436 Acc 0.99286 Speed(per batch) 0.08661 sec
    [INFO] 2020-11-27 14:46:36,900 [    train.py:  171]:	val Epoch 15 Loss 1.19875 Acc 0.58333 Speed(per batch) 0.04424 sec
    [INFO] 2020-11-27 14:46:37,075 [    train.py:  171]:	train Epoch 16 Loss 0.39846 Acc 0.99286 Speed(per batch) 0.08715 sec
    [INFO] 2020-11-27 14:46:37,207 [    train.py:  171]:	val Epoch 16 Loss 1.18778 Acc 0.59000 Speed(per batch) 0.04393 sec
    [INFO] 2020-11-27 14:46:37,380 [    train.py:  171]:	train Epoch 17 Loss 0.35765 Acc 1.00000 Speed(per batch) 0.08645 sec
    [INFO] 2020-11-27 14:46:37,521 [    train.py:  171]:	val Epoch 17 Loss 1.17348 Acc 0.59333 Speed(per batch) 0.04688 sec
    [INFO] 2020-11-27 14:46:37,696 [    train.py:  171]:	train Epoch 18 Loss 0.32175 Acc 1.00000 Speed(per batch) 0.08694 sec
    [INFO] 2020-11-27 14:46:37,830 [    train.py:  171]:	val Epoch 18 Loss 1.16043 Acc 0.60667 Speed(per batch) 0.04473 sec
    [INFO] 2020-11-27 14:46:38,006 [    train.py:  171]:	train Epoch 19 Loss 0.28957 Acc 1.00000 Speed(per batch) 0.08731 sec
    [INFO] 2020-11-27 14:46:38,140 [    train.py:  171]:	val Epoch 19 Loss 1.15043 Acc 0.60667 Speed(per batch) 0.04453 sec
    [INFO] 2020-11-27 14:46:38,314 [    train.py:  171]:	train Epoch 20 Loss 0.26108 Acc 1.00000 Speed(per batch) 0.08702 sec
    [INFO] 2020-11-27 14:46:38,457 [    train.py:  171]:	val Epoch 20 Loss 1.14452 Acc 0.61000 Speed(per batch) 0.04742 sec
    [INFO] 2020-11-27 14:46:38,631 [    train.py:  171]:	train Epoch 21 Loss 0.23532 Acc 1.00000 Speed(per batch) 0.08712 sec
    [INFO] 2020-11-27 14:46:38,765 [    train.py:  171]:	val Epoch 21 Loss 1.13882 Acc 0.60667 Speed(per batch) 0.04439 sec
    [INFO] 2020-11-27 14:46:38,941 [    train.py:  171]:	train Epoch 22 Loss 0.21217 Acc 1.00000 Speed(per batch) 0.08796 sec
    [INFO] 2020-11-27 14:46:39,075 [    train.py:  171]:	val Epoch 22 Loss 1.13712 Acc 0.61333 Speed(per batch) 0.04452 sec
    [INFO] 2020-11-27 14:46:39,256 [    train.py:  171]:	train Epoch 23 Loss 0.19180 Acc 1.00000 Speed(per batch) 0.09029 sec
    [INFO] 2020-11-27 14:46:39,392 [    train.py:  171]:	val Epoch 23 Loss 1.14086 Acc 0.61000 Speed(per batch) 0.04511 sec
    [INFO] 2020-11-27 14:46:39,581 [    train.py:  171]:	train Epoch 24 Loss 0.17381 Acc 1.00000 Speed(per batch) 0.09409 sec
    [INFO] 2020-11-27 14:46:39,717 [    train.py:  171]:	val Epoch 24 Loss 1.14533 Acc 0.61333 Speed(per batch) 0.04520 sec
    [INFO] 2020-11-27 14:46:39,895 [    train.py:  171]:	train Epoch 25 Loss 0.15780 Acc 1.00000 Speed(per batch) 0.08864 sec
    [INFO] 2020-11-27 14:46:40,033 [    train.py:  171]:	val Epoch 25 Loss 1.14792 Acc 0.61667 Speed(per batch) 0.04605 sec
    [INFO] 2020-11-27 14:46:40,220 [    train.py:  171]:	train Epoch 26 Loss 0.14356 Acc 1.00000 Speed(per batch) 0.09295 sec
    [INFO] 2020-11-27 14:46:40,359 [    train.py:  171]:	val Epoch 26 Loss 1.14682 Acc 0.62000 Speed(per batch) 0.04617 sec
    [INFO] 2020-11-27 14:46:40,543 [    train.py:  171]:	train Epoch 27 Loss 0.13091 Acc 1.00000 Speed(per batch) 0.09208 sec
    [INFO] 2020-11-27 14:46:40,676 [    train.py:  171]:	val Epoch 27 Loss 1.14303 Acc 0.61667 Speed(per batch) 0.04420 sec
    [INFO] 2020-11-27 14:46:40,850 [    train.py:  171]:	train Epoch 28 Loss 0.11964 Acc 1.00000 Speed(per batch) 0.08660 sec
    [INFO] 2020-11-27 14:46:40,983 [    train.py:  171]:	val Epoch 28 Loss 1.14131 Acc 0.61667 Speed(per batch) 0.04434 sec
    [INFO] 2020-11-27 14:46:41,157 [    train.py:  171]:	train Epoch 29 Loss 0.10963 Acc 1.00000 Speed(per batch) 0.08665 sec
    [INFO] 2020-11-27 14:46:41,290 [    train.py:  171]:	val Epoch 29 Loss 1.14258 Acc 0.62333 Speed(per batch) 0.04416 sec
    [INFO] 2020-11-27 14:46:41,462 [    train.py:  171]:	train Epoch 30 Loss 0.10073 Acc 1.00000 Speed(per batch) 0.08597 sec
    [INFO] 2020-11-27 14:46:41,675 [    train.py:  171]:	val Epoch 30 Loss 1.14302 Acc 0.62333 Speed(per batch) 0.07073 sec
    [INFO] 2020-11-27 14:46:41,853 [    train.py:  171]:	train Epoch 31 Loss 0.09274 Acc 1.00000 Speed(per batch) 0.08863 sec
    [INFO] 2020-11-27 14:46:41,986 [    train.py:  171]:	val Epoch 31 Loss 1.14327 Acc 0.62667 Speed(per batch) 0.04420 sec
    [INFO] 2020-11-27 14:46:42,161 [    train.py:  171]:	train Epoch 32 Loss 0.08558 Acc 1.00000 Speed(per batch) 0.08760 sec
    [INFO] 2020-11-27 14:46:42,296 [    train.py:  171]:	val Epoch 32 Loss 1.14555 Acc 0.63333 Speed(per batch) 0.04463 sec
    [INFO] 2020-11-27 14:46:42,470 [    train.py:  171]:	train Epoch 33 Loss 0.07918 Acc 1.00000 Speed(per batch) 0.08713 sec
    [INFO] 2020-11-27 14:46:42,604 [    train.py:  171]:	val Epoch 33 Loss 1.15028 Acc 0.63667 Speed(per batch) 0.04439 sec
    [INFO] 2020-11-27 14:46:42,791 [    train.py:  171]:	train Epoch 34 Loss 0.07344 Acc 1.00000 Speed(per batch) 0.09326 sec
    [INFO] 2020-11-27 14:46:42,929 [    train.py:  171]:	val Epoch 34 Loss 1.15637 Acc 0.63333 Speed(per batch) 0.04572 sec
    [INFO] 2020-11-27 14:46:43,103 [    train.py:  171]:	train Epoch 35 Loss 0.06829 Acc 1.00000 Speed(per batch) 0.08702 sec
    [INFO] 2020-11-27 14:46:43,237 [    train.py:  171]:	val Epoch 35 Loss 1.16082 Acc 0.63333 Speed(per batch) 0.04434 sec
    [INFO] 2020-11-27 14:46:43,413 [    train.py:  171]:	train Epoch 36 Loss 0.06361 Acc 1.00000 Speed(per batch) 0.08798 sec
    [INFO] 2020-11-27 14:46:43,548 [    train.py:  171]:	val Epoch 36 Loss 1.16234 Acc 0.64000 Speed(per batch) 0.04489 sec
    [INFO] 2020-11-27 14:46:43,734 [    train.py:  171]:	train Epoch 37 Loss 0.05937 Acc 1.00000 Speed(per batch) 0.09293 sec
    [INFO] 2020-11-27 14:46:43,868 [    train.py:  171]:	val Epoch 37 Loss 1.16232 Acc 0.64000 Speed(per batch) 0.04418 sec
    [INFO] 2020-11-27 14:46:44,043 [    train.py:  171]:	train Epoch 38 Loss 0.05552 Acc 1.00000 Speed(per batch) 0.08742 sec
    [INFO] 2020-11-27 14:46:44,175 [    train.py:  171]:	val Epoch 38 Loss 1.16369 Acc 0.64000 Speed(per batch) 0.04400 sec
    [INFO] 2020-11-27 14:46:44,351 [    train.py:  171]:	train Epoch 39 Loss 0.05205 Acc 1.00000 Speed(per batch) 0.08745 sec
    [INFO] 2020-11-27 14:46:44,484 [    train.py:  171]:	val Epoch 39 Loss 1.16588 Acc 0.64000 Speed(per batch) 0.04425 sec
    [INFO] 2020-11-27 14:46:44,659 [    train.py:  171]:	train Epoch 40 Loss 0.04889 Acc 1.00000 Speed(per batch) 0.08760 sec
    [INFO] 2020-11-27 14:46:44,810 [    train.py:  171]:	val Epoch 40 Loss 1.16803 Acc 0.64333 Speed(per batch) 0.05003 sec
    [INFO] 2020-11-27 14:46:44,986 [    train.py:  171]:	train Epoch 41 Loss 0.04599 Acc 1.00000 Speed(per batch) 0.08792 sec
    [INFO] 2020-11-27 14:46:45,125 [    train.py:  171]:	val Epoch 41 Loss 1.17049 Acc 0.64333 Speed(per batch) 0.04602 sec
    [INFO] 2020-11-27 14:46:45,305 [    train.py:  171]:	train Epoch 42 Loss 0.04333 Acc 1.00000 Speed(per batch) 0.09009 sec
    [INFO] 2020-11-27 14:46:45,439 [    train.py:  171]:	val Epoch 42 Loss 1.17398 Acc 0.65000 Speed(per batch) 0.04442 sec
    [INFO] 2020-11-27 14:46:45,621 [    train.py:  171]:	train Epoch 43 Loss 0.04089 Acc 1.00000 Speed(per batch) 0.09094 sec
    [INFO] 2020-11-27 14:46:45,756 [    train.py:  171]:	val Epoch 43 Loss 1.17823 Acc 0.64333 Speed(per batch) 0.04461 sec
    [INFO] 2020-11-27 14:46:45,944 [    train.py:  171]:	train Epoch 44 Loss 0.03866 Acc 1.00000 Speed(per batch) 0.09372 sec
    [INFO] 2020-11-27 14:46:46,079 [    train.py:  171]:	val Epoch 44 Loss 1.18076 Acc 0.64000 Speed(per batch) 0.04489 sec
    [INFO] 2020-11-27 14:46:46,257 [    train.py:  171]:	train Epoch 45 Loss 0.03662 Acc 1.00000 Speed(per batch) 0.08906 sec
    [INFO] 2020-11-27 14:46:46,391 [    train.py:  171]:	val Epoch 45 Loss 1.18096 Acc 0.64000 Speed(per batch) 0.04454 sec
    [INFO] 2020-11-27 14:46:46,565 [    train.py:  171]:	train Epoch 46 Loss 0.03473 Acc 1.00000 Speed(per batch) 0.08672 sec
    [INFO] 2020-11-27 14:46:46,699 [    train.py:  171]:	val Epoch 46 Loss 1.18044 Acc 0.64333 Speed(per batch) 0.04448 sec
    [INFO] 2020-11-27 14:46:46,889 [    train.py:  171]:	train Epoch 47 Loss 0.03297 Acc 1.00000 Speed(per batch) 0.09478 sec
    [INFO] 2020-11-27 14:46:47,025 [    train.py:  171]:	val Epoch 47 Loss 1.18130 Acc 0.64667 Speed(per batch) 0.04531 sec
    [INFO] 2020-11-27 14:46:47,206 [    train.py:  171]:	train Epoch 48 Loss 0.03136 Acc 1.00000 Speed(per batch) 0.09012 sec
    [INFO] 2020-11-27 14:46:47,341 [    train.py:  171]:	val Epoch 48 Loss 1.18454 Acc 0.64667 Speed(per batch) 0.04488 sec
    [INFO] 2020-11-27 14:46:47,516 [    train.py:  171]:	train Epoch 49 Loss 0.02986 Acc 1.00000 Speed(per batch) 0.08742 sec
    [INFO] 2020-11-27 14:46:47,651 [    train.py:  171]:	val Epoch 49 Loss 1.18955 Acc 0.64333 Speed(per batch) 0.04454 sec
    [INFO] 2020-11-27 14:46:47,988 [    train.py:  171]:	test Epoch 49 Loss 1.24549 Acc 0.62000 Speed(per batch) 0.04207 sec


***这个我是按照examples里面的代码添加的，不知道lstm聚合为什么效果不是很好........***

其实在 GraphSage 原文中，还提出可以使用 LSTM 进行聚合。由于LSTM的输入是有序的而节点的邻居是无序的，论文将邻居节点随机排列作为LSTM的输入。这里我们就不做作业要求了，感兴趣的同学可以查看对应的[代码](https://github.com/PaddlePaddle/PGL/blob/main/examples/graphsage/model.py)。
