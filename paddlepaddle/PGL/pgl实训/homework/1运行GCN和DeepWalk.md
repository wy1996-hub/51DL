# 第一课：图学习初印象习题

搭建环境，运行[GCN](https://arxiv.org/abs/1609.02907)和[DeepWalk](https://dl.acm.org/doi/10.1145/2623330.2623732)。

## 1. 环境搭建


```python
# !pip install paddlepaddle==1.8.5 # 安装PaddlePaddle
!pip install pgl # 安装PGL
```

    Looking in indexes: https://mirror.baidu.com/pypi/simple/
    Collecting pgl
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/e2/84/6aac242f80a794f1169386d73bdc03f2e3467e4fa85b1286979ddf51b1a0/pgl-1.2.1-cp37-cp37m-manylinux1_x86_64.whl (7.9MB)
    [K     |████████████████████████████████| 7.9MB 11.6MB/s eta 0:00:01
    [?25hCollecting redis-py-cluster (from pgl)
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/2b/c5/3236720746fa357e214f2b9fe7e517642329f13094fc7eb339abd93d004f/redis_py_cluster-2.1.0-py2.py3-none-any.whl (41kB)
    [K     |████████████████████████████████| 51kB 19.9MB/s eta 0:00:01
    [?25hRequirement already satisfied: numpy>=1.16.4 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pgl) (1.16.4)
    Requirement already satisfied: cython>=0.25.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pgl) (0.29)
    Requirement already satisfied: visualdl>=2.0.0b; python_version >= "3" in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pgl) (2.0.3)
    Collecting redis<4.0.0,>=3.0.0 (from redis-py-cluster->pgl)
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/a7/7c/24fb0511df653cf1a5d938d8f5d19802a88cef255706fdda242ff97e91b7/redis-3.5.3-py2.py3-none-any.whl (72kB)
    [K     |████████████████████████████████| 81kB 20.4MB/s eta 0:00:01
    [?25hRequirement already satisfied: flake8>=3.7.9 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0b; python_version >= "3"->pgl) (3.8.2)
    Requirement already satisfied: six>=1.14.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0b; python_version >= "3"->pgl) (1.15.0)
    Requirement already satisfied: flask>=1.1.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0b; python_version >= "3"->pgl) (1.1.1)
    Requirement already satisfied: pre-commit in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0b; python_version >= "3"->pgl) (1.21.0)
    Requirement already satisfied: protobuf>=3.11.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0b; python_version >= "3"->pgl) (3.12.2)
    Requirement already satisfied: Flask-Babel>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0b; python_version >= "3"->pgl) (1.0.0)
    Requirement already satisfied: requests in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0b; python_version >= "3"->pgl) (2.22.0)
    Requirement already satisfied: Pillow>=7.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0b; python_version >= "3"->pgl) (7.1.2)
    Requirement already satisfied: mccabe<0.7.0,>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0b; python_version >= "3"->pgl) (0.6.1)
    Requirement already satisfied: pyflakes<2.3.0,>=2.2.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0b; python_version >= "3"->pgl) (2.2.0)
    Requirement already satisfied: pycodestyle<2.7.0,>=2.6.0a1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0b; python_version >= "3"->pgl) (2.6.0)
    Requirement already satisfied: importlib-metadata; python_version < "3.8" in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0b; python_version >= "3"->pgl) (0.23)
    Requirement already satisfied: click>=5.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.0.0b; python_version >= "3"->pgl) (7.0)
    Requirement already satisfied: itsdangerous>=0.24 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.0.0b; python_version >= "3"->pgl) (1.1.0)
    Requirement already satisfied: Jinja2>=2.10.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.0.0b; python_version >= "3"->pgl) (2.10.3)
    Requirement already satisfied: Werkzeug>=0.15 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.0.0b; python_version >= "3"->pgl) (0.16.0)
    Requirement already satisfied: identify>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0b; python_version >= "3"->pgl) (1.4.10)
    Requirement already satisfied: pyyaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0b; python_version >= "3"->pgl) (5.1.2)
    Requirement already satisfied: cfgv>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0b; python_version >= "3"->pgl) (2.0.1)
    Requirement already satisfied: toml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0b; python_version >= "3"->pgl) (0.10.0)
    Requirement already satisfied: nodeenv>=0.11.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0b; python_version >= "3"->pgl) (1.3.4)
    Requirement already satisfied: aspy.yaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0b; python_version >= "3"->pgl) (1.3.0)
    Requirement already satisfied: virtualenv>=15.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0b; python_version >= "3"->pgl) (16.7.9)
    Requirement already satisfied: setuptools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from protobuf>=3.11.0->visualdl>=2.0.0b; python_version >= "3"->pgl) (41.4.0)
    Requirement already satisfied: Babel>=2.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl>=2.0.0b; python_version >= "3"->pgl) (2.8.0)
    Requirement already satisfied: pytz in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl>=2.0.0b; python_version >= "3"->pgl) (2019.3)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0b; python_version >= "3"->pgl) (3.0.4)
    Requirement already satisfied: idna<2.9,>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0b; python_version >= "3"->pgl) (2.8)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0b; python_version >= "3"->pgl) (2019.9.11)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0b; python_version >= "3"->pgl) (1.25.6)
    Requirement already satisfied: zipp>=0.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from importlib-metadata; python_version < "3.8"->flake8>=3.7.9->visualdl>=2.0.0b; python_version >= "3"->pgl) (0.6.0)
    Requirement already satisfied: MarkupSafe>=0.23 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Jinja2>=2.10.1->flask>=1.1.1->visualdl>=2.0.0b; python_version >= "3"->pgl) (1.1.1)
    Requirement already satisfied: more-itertools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from zipp>=0.5->importlib-metadata; python_version < "3.8"->flake8>=3.7.9->visualdl>=2.0.0b; python_version >= "3"->pgl) (7.2.0)
    Installing collected packages: redis, redis-py-cluster, pgl
    Successfully installed pgl-1.2.1 redis-3.5.3 redis-py-cluster-2.1.0


## 2. 下载PGL代码库


```python
# 由于 AIStudio 上访问 github速度比较慢，因此我们提供已经下载好了的 PGL 代码库
# !git clone --depth=1 https://github.com/PaddlePaddle/PGL
!ls PGL # 查看PGL库根目录
```

    docs	  LICENSE	pgl	   README.zh.md      setup.py  tutorials
    examples  ogb_examples	README.md  requirements.txt  tests


## 3. 运行示例

### 3.1 GCN

GCN层的具体实现见 PGL/pgl/layers/conv.py

NOTE：

1. 在GCN模型中，对于图中的某个节点N，相邻节点会将学到的信息发送给它。节点N根据节点的度数给收到的信息加上权重，组合起来作为它新的表示向量。

2. GCN模型会在第三节课进行详细介绍。
    
     


```python
!cd PGL/examples/gcn; python train.py --epochs 100 # 切换到gcn的目录，运行train.py在cora数据集上训练  
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/sklearn/externals/joblib/externals/cloudpickle/cloudpickle.py:47: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
      import imp
    [INFO] 2020-11-24 14:35:23,012 [    train.py:  153]:	Namespace(dataset='cora', epochs=100, use_cuda=False)
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/numpy/core/fromnumeric.py:3118: RuntimeWarning: Mean of empty slice.
      out=out, **kwargs)
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/numpy/core/_methods.py:85: RuntimeWarning: invalid value encountered in double_scalars
      ret = ret.dtype.type(ret / rcount)
    [INFO] 2020-11-24 14:35:24,072 [    train.py:  135]:	Epoch 0 (nan sec) Train Loss: 1.946185 Train Acc: 0.142857 Val Loss: 1.937398 Val Acc: 0.350000 
    [INFO] 2020-11-24 14:35:24,115 [    train.py:  135]:	Epoch 1 (nan sec) Train Loss: 1.935671 Train Acc: 0.342857 Val Loss: 1.927820 Val Acc: 0.523333 
    [INFO] 2020-11-24 14:35:24,170 [    train.py:  135]:	Epoch 2 (nan sec) Train Loss: 1.924336 Train Acc: 0.471429 Val Loss: 1.918572 Val Acc: 0.413333 
    [INFO] 2020-11-24 14:35:24,213 [    train.py:  135]:	Epoch 3 (0.02768 sec) Train Loss: 1.911200 Train Acc: 0.435714 Val Loss: 1.908998 Val Acc: 0.410000 
    [INFO] 2020-11-24 14:35:24,255 [    train.py:  135]:	Epoch 4 (0.02735 sec) Train Loss: 1.898517 Train Acc: 0.464286 Val Loss: 1.898026 Val Acc: 0.426667 
    [INFO] 2020-11-24 14:35:24,298 [    train.py:  135]:	Epoch 5 (0.02719 sec) Train Loss: 1.883320 Train Acc: 0.478571 Val Loss: 1.886568 Val Acc: 0.453333 
    [INFO] 2020-11-24 14:35:24,341 [    train.py:  135]:	Epoch 6 (0.02736 sec) Train Loss: 1.870368 Train Acc: 0.457143 Val Loss: 1.875734 Val Acc: 0.460000 
    [INFO] 2020-11-24 14:35:24,384 [    train.py:  135]:	Epoch 7 (0.02734 sec) Train Loss: 1.855206 Train Acc: 0.471429 Val Loss: 1.864528 Val Acc: 0.460000 
    [INFO] 2020-11-24 14:35:24,429 [    train.py:  135]:	Epoch 8 (0.02764 sec) Train Loss: 1.839174 Train Acc: 0.478571 Val Loss: 1.853004 Val Acc: 0.450000 
    [INFO] 2020-11-24 14:35:24,472 [    train.py:  135]:	Epoch 9 (0.02761 sec) Train Loss: 1.825923 Train Acc: 0.471429 Val Loss: 1.841074 Val Acc: 0.456667 
    [INFO] 2020-11-24 14:35:24,514 [    train.py:  135]:	Epoch 10 (0.02751 sec) Train Loss: 1.813848 Train Acc: 0.435714 Val Loss: 1.829241 Val Acc: 0.463333 
    [INFO] 2020-11-24 14:35:24,558 [    train.py:  135]:	Epoch 11 (0.02757 sec) Train Loss: 1.795466 Train Acc: 0.421429 Val Loss: 1.817481 Val Acc: 0.460000 
    [INFO] 2020-11-24 14:35:24,600 [    train.py:  135]:	Epoch 12 (0.02752 sec) Train Loss: 1.787655 Train Acc: 0.421429 Val Loss: 1.806191 Val Acc: 0.460000 
    [INFO] 2020-11-24 14:35:24,647 [    train.py:  135]:	Epoch 13 (0.02789 sec) Train Loss: 1.772201 Train Acc: 0.407143 Val Loss: 1.795176 Val Acc: 0.460000 
    [INFO] 2020-11-24 14:35:24,690 [    train.py:  135]:	Epoch 14 (0.02779 sec) Train Loss: 1.758189 Train Acc: 0.471429 Val Loss: 1.784524 Val Acc: 0.460000 
    [INFO] 2020-11-24 14:35:24,732 [    train.py:  135]:	Epoch 15 (0.02772 sec) Train Loss: 1.740702 Train Acc: 0.428571 Val Loss: 1.774283 Val Acc: 0.456667 
    [INFO] 2020-11-24 14:35:24,775 [    train.py:  135]:	Epoch 16 (0.02770 sec) Train Loss: 1.735026 Train Acc: 0.435714 Val Loss: 1.764541 Val Acc: 0.456667 
    [INFO] 2020-11-24 14:35:24,817 [    train.py:  135]:	Epoch 17 (0.02764 sec) Train Loss: 1.703781 Train Acc: 0.464286 Val Loss: 1.754949 Val Acc: 0.463333 
    [INFO] 2020-11-24 14:35:24,860 [    train.py:  135]:	Epoch 18 (0.02760 sec) Train Loss: 1.702633 Train Acc: 0.421429 Val Loss: 1.745419 Val Acc: 0.463333 
    [INFO] 2020-11-24 14:35:24,901 [    train.py:  135]:	Epoch 19 (0.02754 sec) Train Loss: 1.687885 Train Acc: 0.457143 Val Loss: 1.735724 Val Acc: 0.466667 
    [INFO] 2020-11-24 14:35:24,945 [    train.py:  135]:	Epoch 20 (0.02756 sec) Train Loss: 1.679132 Train Acc: 0.428571 Val Loss: 1.726115 Val Acc: 0.470000 
    [INFO] 2020-11-24 14:35:24,987 [    train.py:  135]:	Epoch 21 (0.02754 sec) Train Loss: 1.659574 Train Acc: 0.421429 Val Loss: 1.716568 Val Acc: 0.466667 
    [INFO] 2020-11-24 14:35:25,030 [    train.py:  135]:	Epoch 22 (0.02753 sec) Train Loss: 1.634195 Train Acc: 0.492857 Val Loss: 1.707134 Val Acc: 0.466667 
    [INFO] 2020-11-24 14:35:25,074 [    train.py:  135]:	Epoch 23 (0.02752 sec) Train Loss: 1.630441 Train Acc: 0.471429 Val Loss: 1.697799 Val Acc: 0.466667 
    [INFO] 2020-11-24 14:35:25,116 [    train.py:  135]:	Epoch 24 (0.02749 sec) Train Loss: 1.628998 Train Acc: 0.485714 Val Loss: 1.688270 Val Acc: 0.473333 
    [INFO] 2020-11-24 14:35:25,160 [    train.py:  135]:	Epoch 25 (0.02754 sec) Train Loss: 1.597275 Train Acc: 0.492857 Val Loss: 1.678585 Val Acc: 0.476667 
    [INFO] 2020-11-24 14:35:25,203 [    train.py:  135]:	Epoch 26 (0.02753 sec) Train Loss: 1.594020 Train Acc: 0.478571 Val Loss: 1.668684 Val Acc: 0.476667 
    [INFO] 2020-11-24 14:35:25,246 [    train.py:  135]:	Epoch 27 (0.02751 sec) Train Loss: 1.579134 Train Acc: 0.471429 Val Loss: 1.658567 Val Acc: 0.476667 
    [INFO] 2020-11-24 14:35:25,288 [    train.py:  135]:	Epoch 28 (0.02748 sec) Train Loss: 1.552680 Train Acc: 0.464286 Val Loss: 1.648050 Val Acc: 0.480000 
    [INFO] 2020-11-24 14:35:25,335 [    train.py:  135]:	Epoch 29 (0.02761 sec) Train Loss: 1.559275 Train Acc: 0.500000 Val Loss: 1.637107 Val Acc: 0.486667 
    [INFO] 2020-11-24 14:35:25,377 [    train.py:  135]:	Epoch 30 (0.02759 sec) Train Loss: 1.526868 Train Acc: 0.500000 Val Loss: 1.626147 Val Acc: 0.496667 
    [INFO] 2020-11-24 14:35:25,421 [    train.py:  135]:	Epoch 31 (0.02761 sec) Train Loss: 1.524132 Train Acc: 0.492857 Val Loss: 1.615052 Val Acc: 0.510000 
    [INFO] 2020-11-24 14:35:25,465 [    train.py:  135]:	Epoch 32 (0.02760 sec) Train Loss: 1.482377 Train Acc: 0.528571 Val Loss: 1.603748 Val Acc: 0.516667 
    [INFO] 2020-11-24 14:35:25,507 [    train.py:  135]:	Epoch 33 (0.02758 sec) Train Loss: 1.475376 Train Acc: 0.542857 Val Loss: 1.592267 Val Acc: 0.523333 
    [INFO] 2020-11-24 14:35:25,550 [    train.py:  135]:	Epoch 34 (0.02758 sec) Train Loss: 1.481455 Train Acc: 0.528571 Val Loss: 1.580759 Val Acc: 0.530000 
    [INFO] 2020-11-24 14:35:25,593 [    train.py:  135]:	Epoch 35 (0.02758 sec) Train Loss: 1.471441 Train Acc: 0.485714 Val Loss: 1.569347 Val Acc: 0.530000 
    [INFO] 2020-11-24 14:35:25,638 [    train.py:  135]:	Epoch 36 (0.02761 sec) Train Loss: 1.423521 Train Acc: 0.614286 Val Loss: 1.557598 Val Acc: 0.550000 
    [INFO] 2020-11-24 14:35:25,682 [    train.py:  135]:	Epoch 37 (0.02763 sec) Train Loss: 1.427988 Train Acc: 0.585714 Val Loss: 1.545247 Val Acc: 0.556667 
    [INFO] 2020-11-24 14:35:25,726 [    train.py:  135]:	Epoch 38 (0.02763 sec) Train Loss: 1.361888 Train Acc: 0.642857 Val Loss: 1.532764 Val Acc: 0.566667 
    [INFO] 2020-11-24 14:35:25,769 [    train.py:  135]:	Epoch 39 (0.02762 sec) Train Loss: 1.404664 Train Acc: 0.600000 Val Loss: 1.520443 Val Acc: 0.573333 
    [INFO] 2020-11-24 14:35:25,813 [    train.py:  135]:	Epoch 40 (0.02763 sec) Train Loss: 1.350403 Train Acc: 0.585714 Val Loss: 1.507765 Val Acc: 0.580000 
    [INFO] 2020-11-24 14:35:25,857 [    train.py:  135]:	Epoch 41 (0.02764 sec) Train Loss: 1.386726 Train Acc: 0.585714 Val Loss: 1.495053 Val Acc: 0.596667 
    [INFO] 2020-11-24 14:35:25,899 [    train.py:  135]:	Epoch 42 (0.02762 sec) Train Loss: 1.342807 Train Acc: 0.628571 Val Loss: 1.482347 Val Acc: 0.593333 
    [INFO] 2020-11-24 14:35:25,943 [    train.py:  135]:	Epoch 43 (0.02764 sec) Train Loss: 1.311738 Train Acc: 0.678571 Val Loss: 1.469172 Val Acc: 0.606667 
    [INFO] 2020-11-24 14:35:25,991 [    train.py:  135]:	Epoch 44 (0.02775 sec) Train Loss: 1.324387 Train Acc: 0.642857 Val Loss: 1.455853 Val Acc: 0.610000 
    [INFO] 2020-11-24 14:35:26,035 [    train.py:  135]:	Epoch 45 (0.02774 sec) Train Loss: 1.272827 Train Acc: 0.721429 Val Loss: 1.442416 Val Acc: 0.616667 
    [INFO] 2020-11-24 14:35:26,079 [    train.py:  135]:	Epoch 46 (0.02774 sec) Train Loss: 1.282629 Train Acc: 0.692857 Val Loss: 1.429290 Val Acc: 0.626667 
    [INFO] 2020-11-24 14:35:26,125 [    train.py:  135]:	Epoch 47 (0.02779 sec) Train Loss: 1.240881 Train Acc: 0.692857 Val Loss: 1.416146 Val Acc: 0.636667 
    [INFO] 2020-11-24 14:35:26,171 [    train.py:  135]:	Epoch 48 (0.02783 sec) Train Loss: 1.223143 Train Acc: 0.678571 Val Loss: 1.402819 Val Acc: 0.643333 
    [INFO] 2020-11-24 14:35:26,215 [    train.py:  135]:	Epoch 49 (0.02784 sec) Train Loss: 1.211636 Train Acc: 0.657143 Val Loss: 1.389506 Val Acc: 0.646667 
    [INFO] 2020-11-24 14:35:26,258 [    train.py:  135]:	Epoch 50 (0.02783 sec) Train Loss: 1.195890 Train Acc: 0.721429 Val Loss: 1.376252 Val Acc: 0.646667 
    [INFO] 2020-11-24 14:35:26,301 [    train.py:  135]:	Epoch 51 (0.02782 sec) Train Loss: 1.155226 Train Acc: 0.714286 Val Loss: 1.363149 Val Acc: 0.646667 
    [INFO] 2020-11-24 14:35:26,344 [    train.py:  135]:	Epoch 52 (0.02782 sec) Train Loss: 1.152210 Train Acc: 0.714286 Val Loss: 1.350167 Val Acc: 0.656667 
    [INFO] 2020-11-24 14:35:26,387 [    train.py:  135]:	Epoch 53 (0.02781 sec) Train Loss: 1.166166 Train Acc: 0.728571 Val Loss: 1.337320 Val Acc: 0.660000 
    [INFO] 2020-11-24 14:35:26,430 [    train.py:  135]:	Epoch 54 (0.02779 sec) Train Loss: 1.096565 Train Acc: 0.771429 Val Loss: 1.324485 Val Acc: 0.670000 
    [INFO] 2020-11-24 14:35:26,473 [    train.py:  135]:	Epoch 55 (0.02778 sec) Train Loss: 1.118437 Train Acc: 0.728571 Val Loss: 1.312128 Val Acc: 0.673333 
    [INFO] 2020-11-24 14:35:26,516 [    train.py:  135]:	Epoch 56 (0.02777 sec) Train Loss: 1.086052 Train Acc: 0.757143 Val Loss: 1.299736 Val Acc: 0.676667 
    [INFO] 2020-11-24 14:35:26,559 [    train.py:  135]:	Epoch 57 (0.02776 sec) Train Loss: 1.109819 Train Acc: 0.750000 Val Loss: 1.287504 Val Acc: 0.680000 
    [INFO] 2020-11-24 14:35:26,602 [    train.py:  135]:	Epoch 58 (0.02776 sec) Train Loss: 1.066681 Train Acc: 0.771429 Val Loss: 1.275070 Val Acc: 0.693333 
    [INFO] 2020-11-24 14:35:26,648 [    train.py:  135]:	Epoch 59 (0.02776 sec) Train Loss: 1.030894 Train Acc: 0.757143 Val Loss: 1.262529 Val Acc: 0.703333 
    [INFO] 2020-11-24 14:35:26,691 [    train.py:  135]:	Epoch 60 (0.02773 sec) Train Loss: 1.046237 Train Acc: 0.764286 Val Loss: 1.250013 Val Acc: 0.706667 
    [INFO] 2020-11-24 14:35:26,734 [    train.py:  135]:	Epoch 61 (0.02773 sec) Train Loss: 1.057186 Train Acc: 0.764286 Val Loss: 1.237465 Val Acc: 0.716667 
    [INFO] 2020-11-24 14:35:26,777 [    train.py:  135]:	Epoch 62 (0.02772 sec) Train Loss: 1.027045 Train Acc: 0.728571 Val Loss: 1.225595 Val Acc: 0.720000 
    [INFO] 2020-11-24 14:35:26,820 [    train.py:  135]:	Epoch 63 (0.02771 sec) Train Loss: 1.012417 Train Acc: 0.800000 Val Loss: 1.214479 Val Acc: 0.720000 
    [INFO] 2020-11-24 14:35:26,863 [    train.py:  135]:	Epoch 64 (0.02771 sec) Train Loss: 0.944846 Train Acc: 0.771429 Val Loss: 1.203937 Val Acc: 0.720000 
    [INFO] 2020-11-24 14:35:26,906 [    train.py:  135]:	Epoch 65 (0.02770 sec) Train Loss: 0.972979 Train Acc: 0.771429 Val Loss: 1.193881 Val Acc: 0.730000 
    [INFO] 2020-11-24 14:35:26,949 [    train.py:  135]:	Epoch 66 (0.02770 sec) Train Loss: 0.948729 Train Acc: 0.792857 Val Loss: 1.183838 Val Acc: 0.730000 
    [INFO] 2020-11-24 14:35:26,992 [    train.py:  135]:	Epoch 67 (0.02769 sec) Train Loss: 0.922943 Train Acc: 0.750000 Val Loss: 1.173660 Val Acc: 0.730000 
    [INFO] 2020-11-24 14:35:27,035 [    train.py:  135]:	Epoch 68 (0.02768 sec) Train Loss: 0.958292 Train Acc: 0.778571 Val Loss: 1.163172 Val Acc: 0.736667 
    [INFO] 2020-11-24 14:35:27,077 [    train.py:  135]:	Epoch 69 (0.02767 sec) Train Loss: 0.925448 Train Acc: 0.814286 Val Loss: 1.153209 Val Acc: 0.736667 
    [INFO] 2020-11-24 14:35:27,120 [    train.py:  135]:	Epoch 70 (0.02767 sec) Train Loss: 0.878701 Train Acc: 0.842857 Val Loss: 1.143276 Val Acc: 0.736667 
    [INFO] 2020-11-24 14:35:27,164 [    train.py:  135]:	Epoch 71 (0.02768 sec) Train Loss: 0.905662 Train Acc: 0.814286 Val Loss: 1.133338 Val Acc: 0.740000 
    [INFO] 2020-11-24 14:35:27,207 [    train.py:  135]:	Epoch 72 (0.02767 sec) Train Loss: 0.879022 Train Acc: 0.785714 Val Loss: 1.123667 Val Acc: 0.740000 
    [INFO] 2020-11-24 14:35:27,250 [    train.py:  135]:	Epoch 73 (0.02766 sec) Train Loss: 0.880817 Train Acc: 0.828571 Val Loss: 1.113799 Val Acc: 0.743333 
    [INFO] 2020-11-24 14:35:27,293 [    train.py:  135]:	Epoch 74 (0.02766 sec) Train Loss: 0.871175 Train Acc: 0.835714 Val Loss: 1.104007 Val Acc: 0.746667 
    [INFO] 2020-11-24 14:35:27,341 [    train.py:  135]:	Epoch 75 (0.02772 sec) Train Loss: 0.870027 Train Acc: 0.814286 Val Loss: 1.094272 Val Acc: 0.750000 
    [INFO] 2020-11-24 14:35:27,386 [    train.py:  135]:	Epoch 76 (0.02773 sec) Train Loss: 0.835869 Train Acc: 0.814286 Val Loss: 1.085243 Val Acc: 0.756667 
    [INFO] 2020-11-24 14:35:27,430 [    train.py:  135]:	Epoch 77 (0.02774 sec) Train Loss: 0.883487 Train Acc: 0.800000 Val Loss: 1.076818 Val Acc: 0.763333 
    [INFO] 2020-11-24 14:35:27,474 [    train.py:  135]:	Epoch 78 (0.02774 sec) Train Loss: 0.839400 Train Acc: 0.800000 Val Loss: 1.068500 Val Acc: 0.766667 
    [INFO] 2020-11-24 14:35:27,518 [    train.py:  135]:	Epoch 79 (0.02774 sec) Train Loss: 0.841634 Train Acc: 0.850000 Val Loss: 1.060051 Val Acc: 0.770000 
    [INFO] 2020-11-24 14:35:27,561 [    train.py:  135]:	Epoch 80 (0.02774 sec) Train Loss: 0.812359 Train Acc: 0.821429 Val Loss: 1.052244 Val Acc: 0.776667 
    [INFO] 2020-11-24 14:35:27,605 [    train.py:  135]:	Epoch 81 (0.02774 sec) Train Loss: 0.799842 Train Acc: 0.828571 Val Loss: 1.044836 Val Acc: 0.770000 
    [INFO] 2020-11-24 14:35:27,648 [    train.py:  135]:	Epoch 82 (0.02774 sec) Train Loss: 0.720036 Train Acc: 0.878571 Val Loss: 1.037211 Val Acc: 0.770000 
    [INFO] 2020-11-24 14:35:27,692 [    train.py:  135]:	Epoch 83 (0.02774 sec) Train Loss: 0.737968 Train Acc: 0.864286 Val Loss: 1.030172 Val Acc: 0.766667 
    [INFO] 2020-11-24 14:35:27,735 [    train.py:  135]:	Epoch 84 (0.02774 sec) Train Loss: 0.763071 Train Acc: 0.878571 Val Loss: 1.023598 Val Acc: 0.770000 
    [INFO] 2020-11-24 14:35:27,778 [    train.py:  135]:	Epoch 85 (0.02774 sec) Train Loss: 0.775600 Train Acc: 0.857143 Val Loss: 1.017404 Val Acc: 0.773333 
    [INFO] 2020-11-24 14:35:27,821 [    train.py:  135]:	Epoch 86 (0.02773 sec) Train Loss: 0.754057 Train Acc: 0.892857 Val Loss: 1.011005 Val Acc: 0.776667 
    [INFO] 2020-11-24 14:35:27,864 [    train.py:  135]:	Epoch 87 (0.02772 sec) Train Loss: 0.756406 Train Acc: 0.878571 Val Loss: 1.004001 Val Acc: 0.783333 
    [INFO] 2020-11-24 14:35:27,907 [    train.py:  135]:	Epoch 88 (0.02772 sec) Train Loss: 0.743661 Train Acc: 0.885714 Val Loss: 0.996710 Val Acc: 0.783333 
    [INFO] 2020-11-24 14:35:27,950 [    train.py:  135]:	Epoch 89 (0.02772 sec) Train Loss: 0.755894 Train Acc: 0.864286 Val Loss: 0.989898 Val Acc: 0.783333 
    [INFO] 2020-11-24 14:35:27,999 [    train.py:  135]:	Epoch 90 (0.02772 sec) Train Loss: 0.728832 Train Acc: 0.864286 Val Loss: 0.983047 Val Acc: 0.786667 
    [INFO] 2020-11-24 14:35:28,043 [    train.py:  135]:	Epoch 91 (0.02773 sec) Train Loss: 0.696529 Train Acc: 0.900000 Val Loss: 0.976196 Val Acc: 0.793333 
    [INFO] 2020-11-24 14:35:28,086 [    train.py:  135]:	Epoch 92 (0.02773 sec) Train Loss: 0.741594 Train Acc: 0.871429 Val Loss: 0.969892 Val Acc: 0.793333 
    [INFO] 2020-11-24 14:35:28,130 [    train.py:  135]:	Epoch 93 (0.02773 sec) Train Loss: 0.681129 Train Acc: 0.871429 Val Loss: 0.963948 Val Acc: 0.793333 
    [INFO] 2020-11-24 14:35:28,173 [    train.py:  135]:	Epoch 94 (0.02773 sec) Train Loss: 0.670672 Train Acc: 0.892857 Val Loss: 0.958294 Val Acc: 0.793333 
    [INFO] 2020-11-24 14:35:28,216 [    train.py:  135]:	Epoch 95 (0.02772 sec) Train Loss: 0.700566 Train Acc: 0.871429 Val Loss: 0.952930 Val Acc: 0.793333 
    [INFO] 2020-11-24 14:35:28,259 [    train.py:  135]:	Epoch 96 (0.02771 sec) Train Loss: 0.687270 Train Acc: 0.871429 Val Loss: 0.948623 Val Acc: 0.796667 
    [INFO] 2020-11-24 14:35:28,302 [    train.py:  135]:	Epoch 97 (0.02771 sec) Train Loss: 0.637869 Train Acc: 0.878571 Val Loss: 0.944386 Val Acc: 0.793333 
    [INFO] 2020-11-24 14:35:28,347 [    train.py:  135]:	Epoch 98 (0.02772 sec) Train Loss: 0.679080 Train Acc: 0.900000 Val Loss: 0.939481 Val Acc: 0.793333 
    [INFO] 2020-11-24 14:35:28,392 [    train.py:  135]:	Epoch 99 (0.02773 sec) Train Loss: 0.696660 Train Acc: 0.864286 Val Loss: 0.935254 Val Acc: 0.793333 
    [INFO] 2020-11-24 14:35:28,408 [    train.py:  143]:	Accuracy: 0.759000


<br>

### 3.2 DeepWalk

模型代码详见 PGL/examples/deepwalk/deepwalk.py

NOTE: 

1. DeepWalk的主要原理是通过随机游走生成节点路径，然后将其作为词向量模型SkipGram的输入来学习节点表示。

2. DeepWalk 模型会在第二节课详细介绍。
    
<br>

**Step1 学习节点表示**

查看deepwalk.py中的parser(239行起)，修改不同参数的值，观察其对训练结果的影响，比如游走路径长度walk_len，SkipGram窗口大小win_size等

**Tips** 

1. 如果出现内存不足的问题，可以调小batch_size参数
2. 以下设置的参数为了让同学们可以快速跑出结果，设置的 epoch、walk_len、hidden_size 均比较小，可以自行尝试调大这些值。



```python
!cd PGL/examples/deepwalk/; python deepwalk.py --dataset ArXiv --save_path ./tmp/deepwalk_ArXiv --offline_learning --epoch 2 --batch_size 256 --processes 1 --walk_len 10 --hidden_size 10 
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/sklearn/externals/joblib/externals/cloudpickle/cloudpickle.py:47: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
      import imp
    [INFO] 2020-11-24 14:36:01,632 [ deepwalk.py:  258]:	Namespace(batch_size=256, dataset='ArXiv', epoch=2, hidden_size=10, neg_num=20, offline_learning=True, processes=1, save_path='./tmp/deepwalk_ArXiv', use_cuda=False, walk_len=10, win_size=10)
    [INFO] 2020-11-24 14:36:02,430 [ deepwalk.py:  172]:	Start random walk on disk...
    [INFO] 2020-11-24 14:36:03,184 [ deepwalk.py:  182]:	Random walk on disk Done.
    2020-11-24 14:36:03,185-WARNING: paddle.fluid.layers.py_reader() may be deprecated in the near future. Please use paddle.fluid.io.DataLoader.from_generator() instead.
    [INFO] 2020-11-24 14:36:03,601 [ deepwalk.py:  228]:	Step 0 Deepwalk Loss: 0.724576  0.372420 s/step.
    [INFO] 2020-11-24 14:36:14,487 [ deepwalk.py:  228]:	Step 50 Deepwalk Loss: 0.644198  0.215872 s/step.
    [INFO] 2020-11-24 14:36:21,202 [ deepwalk.py:  228]:	Step 100 Deepwalk Loss: 0.568986  0.075070 s/step.
    [INFO] 2020-11-24 14:36:25,024 [ deepwalk.py:  228]:	Step 150 Deepwalk Loss: 0.546118  0.074450 s/step.
    [INFO] 2020-11-24 14:36:28,813 [ deepwalk.py:  228]:	Step 200 Deepwalk Loss: 0.547902  0.076921 s/step.
    [INFO] 2020-11-24 14:36:32,637 [ deepwalk.py:  228]:	Step 250 Deepwalk Loss: 0.545251  0.075461 s/step.
    [INFO] 2020-11-24 14:36:36,471 [ deepwalk.py:  228]:	Step 300 Deepwalk Loss: 0.544639  0.077662 s/step.
    [INFO] 2020-11-24 14:36:40,274 [ deepwalk.py:  228]:	Step 350 Deepwalk Loss: 0.536977  0.074419 s/step.
    [INFO] 2020-11-24 14:36:48,757 [ deepwalk.py:  228]:	Step 400 Deepwalk Loss: 0.529961  0.213211 s/step.
    [INFO] 2020-11-24 14:36:58,960 [ deepwalk.py:  228]:	Step 450 Deepwalk Loss: 0.524064  0.207546 s/step.
    [INFO] 2020-11-24 14:37:09,422 [ deepwalk.py:  228]:	Step 500 Deepwalk Loss: 0.527823  0.211965 s/step.
    [INFO] 2020-11-24 14:37:19,968 [ deepwalk.py:  228]:	Step 550 Deepwalk Loss: 0.525310  0.214373 s/step.
    [INFO] 2020-11-24 14:37:28,677 [ deepwalk.py:  228]:	Step 600 Deepwalk Loss: 0.539174  0.078960 s/step.
    [INFO] 2020-11-24 14:37:32,627 [ deepwalk.py:  228]:	Step 650 Deepwalk Loss: 0.540447  0.077189 s/step.
    [INFO] 2020-11-24 14:37:42,022 [ deepwalk.py:  228]:	Step 700 Deepwalk Loss: 0.537241  0.211411 s/step.
    [INFO] 2020-11-24 14:37:50,225 [ deepwalk.py:  228]:	Step 750 Deepwalk Loss: 0.541432  0.079251 s/step.
    [INFO] 2020-11-24 14:37:54,213 [ deepwalk.py:  228]:	Step 800 Deepwalk Loss: 0.540765  0.077490 s/step.
    [INFO] 2020-11-24 14:37:58,127 [ deepwalk.py:  228]:	Step 850 Deepwalk Loss: 0.539415  0.080149 s/step.
    [INFO] 2020-11-24 14:38:02,101 [ deepwalk.py:  228]:	Step 900 Deepwalk Loss: 0.532488  0.079324 s/step.
    [INFO] 2020-11-24 14:38:05,991 [ deepwalk.py:  228]:	Step 950 Deepwalk Loss: 0.534465  0.077003 s/step.
    [INFO] 2020-11-24 14:38:16,322 [ deepwalk.py:  228]:	Step 1000 Deepwalk Loss: 0.534607  0.216003 s/step.
    [INFO] 2020-11-24 14:38:23,716 [ deepwalk.py:  228]:	Step 1050 Deepwalk Loss: 0.541579  0.076334 s/step.
    [INFO] 2020-11-24 14:38:27,767 [ deepwalk.py:  228]:	Step 1100 Deepwalk Loss: 0.530346  0.215230 s/step.
    [INFO] 2020-11-24 14:38:38,293 [ deepwalk.py:  228]:	Step 1150 Deepwalk Loss: 0.523066  0.215668 s/step.
    [INFO] 2020-11-24 14:38:48,817 [ deepwalk.py:  228]:	Step 1200 Deepwalk Loss: 0.524576  0.206781 s/step.
    [INFO] 2020-11-24 14:38:59,457 [ deepwalk.py:  228]:	Step 1250 Deepwalk Loss: 0.531868  0.214398 s/step.
    [INFO] 2020-11-24 14:39:09,990 [ deepwalk.py:  228]:	Step 1300 Deepwalk Loss: 0.523190  0.209884 s/step.
    [INFO] 2020-11-24 14:39:20,550 [ deepwalk.py:  228]:	Step 1350 Deepwalk Loss: 0.532175  0.212429 s/step.
    [INFO] 2020-11-24 14:39:30,194 [ deepwalk.py:  228]:	Step 1400 Deepwalk Loss: 0.534883  0.085832 s/step.
    [INFO] 2020-11-24 14:39:34,124 [ deepwalk.py:  228]:	Step 1450 Deepwalk Loss: 0.538972  0.078368 s/step.


<br>

**Step2 链接预测任务上的测试**

这里选用的数据集是ArXiv，它包含了天体物理学类的论文作者间的合作关系图，得到的节点表示存储在 ./tmp/deepwalk_Arxiv


```python
!cd ./PGL/examples/deepwalk/; python link_predict.py --ckpt_path ./tmp/deepwalk_Arxiv/paddle_model --epoch 50
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/sklearn/externals/joblib/externals/cloudpickle/cloudpickle.py:47: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
      import imp
    [INFO] 2020-11-24 14:40:44,192 [link_predict.py:  233]:	Namespace(batch_size=None, ckpt_path='./tmp/deepwalk_Arxiv/paddle_model', dataset='ArXiv', epoch=50, hidden_size=128, use_cuda=False)
    2020-11-24 14:40:44,980-WARNING: paddle.fluid.layers.py_reader() may be deprecated in the near future. Please use paddle.fluid.io.DataLoader.from_generator() instead.
    2020-11-24 14:40:44,999-WARNING: paddle.fluid.layers.py_reader() may be deprecated in the near future. Please use paddle.fluid.io.DataLoader.from_generator() instead.
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/executor.py:1093: UserWarning: There are no operators in the program to be executed. If you pass Program manually, please use fluid.program_guard to ensure the current Program is being used.
      warnings.warn(error_info)
    [INFO] 2020-11-24 14:40:45,836 [link_predict.py:  215]:		Step 1 Test Loss: 0.693220 Test AUC: 0.503291 
    [INFO] 2020-11-24 14:40:46,125 [link_predict.py:  215]:		Step 2 Test Loss: 0.693154 Test AUC: 0.504987 
    [INFO] 2020-11-24 14:40:46,407 [link_predict.py:  215]:		Step 3 Test Loss: 0.693156 Test AUC: 0.506682 
    [INFO] 2020-11-24 14:40:46,692 [link_predict.py:  215]:		Step 4 Test Loss: 0.693183 Test AUC: 0.508276 
    [INFO] 2020-11-24 14:40:46,979 [link_predict.py:  215]:		Step 5 Test Loss: 0.693178 Test AUC: 0.509675 
    [INFO] 2020-11-24 14:40:47,265 [link_predict.py:  215]:		Step 6 Test Loss: 0.693150 Test AUC: 0.510915 
    [INFO] 2020-11-24 14:40:47,546 [link_predict.py:  215]:		Step 7 Test Loss: 0.693138 Test AUC: 0.512049 
    [INFO] 2020-11-24 14:40:47,831 [link_predict.py:  215]:		Step 8 Test Loss: 0.693144 Test AUC: 0.513095 
    [INFO] 2020-11-24 14:40:48,122 [link_predict.py:  215]:		Step 9 Test Loss: 0.693156 Test AUC: 0.514046 
    [INFO] 2020-11-24 14:40:48,290 [link_predict.py:  192]:	Step 10 Train Loss: 0.693155 Train AUC: 0.518355 
    [INFO] 2020-11-24 14:40:48,418 [link_predict.py:  215]:		Step 10 Test Loss: 0.693159 Test AUC: 0.514895 
    [INFO] 2020-11-24 14:40:48,710 [link_predict.py:  215]:		Step 11 Test Loss: 0.693156 Test AUC: 0.515638 
    [INFO] 2020-11-24 14:40:49,004 [link_predict.py:  215]:		Step 12 Test Loss: 0.693147 Test AUC: 0.516283 
    [INFO] 2020-11-24 14:40:49,357 [link_predict.py:  215]:		Step 13 Test Loss: 0.693137 Test AUC: 0.516842 
    [INFO] 2020-11-24 14:40:49,651 [link_predict.py:  215]:		Step 14 Test Loss: 0.693139 Test AUC: 0.517322 
    [INFO] 2020-11-24 14:40:49,945 [link_predict.py:  215]:		Step 15 Test Loss: 0.693141 Test AUC: 0.517741 
    [INFO] 2020-11-24 14:40:50,238 [link_predict.py:  215]:		Step 16 Test Loss: 0.693148 Test AUC: 0.518102 
    [INFO] 2020-11-24 14:40:50,535 [link_predict.py:  215]:		Step 17 Test Loss: 0.693147 Test AUC: 0.518418 
    [INFO] 2020-11-24 14:40:50,830 [link_predict.py:  215]:		Step 18 Test Loss: 0.693143 Test AUC: 0.518698 
    [INFO] 2020-11-24 14:40:51,124 [link_predict.py:  215]:		Step 19 Test Loss: 0.693138 Test AUC: 0.518942 
    [INFO] 2020-11-24 14:40:51,289 [link_predict.py:  192]:	Step 20 Train Loss: 0.693135 Train AUC: 0.525268 
    [INFO] 2020-11-24 14:40:51,415 [link_predict.py:  215]:		Step 20 Test Loss: 0.693135 Test AUC: 0.519153 
    [INFO] 2020-11-24 14:40:51,706 [link_predict.py:  215]:		Step 21 Test Loss: 0.693137 Test AUC: 0.519336 
    [INFO] 2020-11-24 14:40:51,998 [link_predict.py:  215]:		Step 22 Test Loss: 0.693137 Test AUC: 0.519498 
    [INFO] 2020-11-24 14:40:52,289 [link_predict.py:  215]:		Step 23 Test Loss: 0.693137 Test AUC: 0.519640 
    [INFO] 2020-11-24 14:40:52,581 [link_predict.py:  215]:		Step 24 Test Loss: 0.693137 Test AUC: 0.519764 
    [INFO] 2020-11-24 14:40:52,871 [link_predict.py:  215]:		Step 25 Test Loss: 0.693136 Test AUC: 0.519876 
    [INFO] 2020-11-24 14:40:53,161 [link_predict.py:  215]:		Step 26 Test Loss: 0.693134 Test AUC: 0.519972 
    [INFO] 2020-11-24 14:40:53,455 [link_predict.py:  215]:		Step 27 Test Loss: 0.693133 Test AUC: 0.520059 
    [INFO] 2020-11-24 14:40:53,741 [link_predict.py:  215]:		Step 28 Test Loss: 0.693133 Test AUC: 0.520135 
    [INFO] 2020-11-24 14:40:54,026 [link_predict.py:  215]:		Step 29 Test Loss: 0.693132 Test AUC: 0.520203 
    [INFO] 2020-11-24 14:40:54,187 [link_predict.py:  192]:	Step 30 Train Loss: 0.693127 Train AUC: 0.527323 
    [INFO] 2020-11-24 14:40:54,312 [link_predict.py:  215]:		Step 30 Test Loss: 0.693133 Test AUC: 0.520262 
    [INFO] 2020-11-24 14:40:54,596 [link_predict.py:  215]:		Step 31 Test Loss: 0.693133 Test AUC: 0.520315 
    [INFO] 2020-11-24 14:40:54,882 [link_predict.py:  215]:		Step 32 Test Loss: 0.693133 Test AUC: 0.520361 
    [INFO] 2020-11-24 14:40:55,164 [link_predict.py:  215]:		Step 33 Test Loss: 0.693133 Test AUC: 0.520402 
    [INFO] 2020-11-24 14:40:55,449 [link_predict.py:  215]:		Step 34 Test Loss: 0.693131 Test AUC: 0.520441 
    [INFO] 2020-11-24 14:40:55,733 [link_predict.py:  215]:		Step 35 Test Loss: 0.693130 Test AUC: 0.520473 
    [INFO] 2020-11-24 14:40:56,015 [link_predict.py:  215]:		Step 36 Test Loss: 0.693130 Test AUC: 0.520503 
    [INFO] 2020-11-24 14:40:56,299 [link_predict.py:  215]:		Step 37 Test Loss: 0.693130 Test AUC: 0.520528 
    [INFO] 2020-11-24 14:40:56,579 [link_predict.py:  215]:		Step 38 Test Loss: 0.693129 Test AUC: 0.520549 
    [INFO] 2020-11-24 14:40:56,865 [link_predict.py:  215]:		Step 39 Test Loss: 0.693130 Test AUC: 0.520569 
    [INFO] 2020-11-24 14:40:57,025 [link_predict.py:  192]:	Step 40 Train Loss: 0.693122 Train AUC: 0.528016 
    [INFO] 2020-11-24 14:40:57,149 [link_predict.py:  215]:		Step 40 Test Loss: 0.693129 Test AUC: 0.520585 
    [INFO] 2020-11-24 14:40:57,433 [link_predict.py:  215]:		Step 41 Test Loss: 0.693129 Test AUC: 0.520598 
    [INFO] 2020-11-24 14:40:57,714 [link_predict.py:  215]:		Step 42 Test Loss: 0.693129 Test AUC: 0.520611 
    [INFO] 2020-11-24 14:40:57,999 [link_predict.py:  215]:		Step 43 Test Loss: 0.693130 Test AUC: 0.520623 
    [INFO] 2020-11-24 14:40:58,288 [link_predict.py:  215]:		Step 44 Test Loss: 0.693129 Test AUC: 0.520633 
    [INFO] 2020-11-24 14:40:58,581 [link_predict.py:  215]:		Step 45 Test Loss: 0.693128 Test AUC: 0.520641 
    [INFO] 2020-11-24 14:40:58,869 [link_predict.py:  215]:		Step 46 Test Loss: 0.693128 Test AUC: 0.520647 
    [INFO] 2020-11-24 14:40:59,152 [link_predict.py:  215]:		Step 47 Test Loss: 0.693129 Test AUC: 0.520652 
    [INFO] 2020-11-24 14:40:59,434 [link_predict.py:  215]:		Step 48 Test Loss: 0.693129 Test AUC: 0.520656 
    [INFO] 2020-11-24 14:40:59,717 [link_predict.py:  215]:		Step 49 Test Loss: 0.693128 Test AUC: 0.520659 
    [INFO] 2020-11-24 14:40:59,882 [link_predict.py:  192]:	Step 50 Train Loss: 0.693121 Train AUC: 0.528213 
    [INFO] 2020-11-24 14:41:00,010 [link_predict.py:  215]:		Step 50 Test Loss: 0.693128 Test AUC: 0.520660 


## 4.本地作业
**本地作业：飞桨本地测试代码运行成功截图和GCN例子运行成功截图**

**####请在下面cell中上传飞桨安装成功的截图和GCN例子运行成功截图####**

### 4.1 飞桨相关信息(上传paddle.fluid.install_check.run_check()之后的截图)：

飞桨安装文档：[https://paddlepaddle.org.cn/install/quick](https://paddlepaddle.org.cn/install/quick)

提示：使用 python 进入python解释器，输入import paddle.fluid ，再输入 paddle.fluid.install_check.run_check()。
如果出现 Your Paddle Fluid is installed successfully!，说明您已成功安装。

本地安装PaddlePaddle的常见错误：[https://aistudio.baidu.com/aistudio/projectdetail/697227](https://aistudio.baidu.com/aistudio/projectdetail/697227)

手把手教你 win10 安装Paddlepaddle-GPU：[https://aistudio.baidu.com/aistudio/projectdetail/696822](https://aistudio.baidu.com/aistudio/projectdetail/696822)

### 4.2 PGL相关信息(上传运行示例-GCN的截图)：

    pip install pgl # 安装PGL

    git clone --depth=1 https://github.com/PaddlePaddle/PGL #下载PGL代码库(或者直接把左边文件中的下载到本地)
		
    # 运行示例-GCN
    cd PGL/examples/gcn; python train.py --epochs 100 # 切换到gcn的目录，运行train.py在cora数据集上训练  
    

		

![](https://ai-studio-static-online.cdn.bcebos.com/f5d157d35ae2429d8f62d3bbb8949dc0b2a941d31ff04e26a79db1d0b121ea8e)
![](https://ai-studio-static-online.cdn.bcebos.com/824e77d482e14fa19b2360876a3d9341f391f0dace16475d99e20d6123bc07d1)



```python

```

## 5. 代码框架梳理（可选）

本小节以GCN的 PGL/examples/gcn/train.py 为例，简单介绍一下图模型的训练框架。

### 5.1 参数设置

可修改parser的参数来使用不同的数据集进行训练

```python
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    # 设置数据集，默认选择cora数据集
    parser.add_argument(
        "--dataset", type=str, default="cora", help="dataset (cora, pubmed)")
    # 设置是否使用GPU
    parser.add_argument("--use_cuda", action='store_true', help="use_cuda")
    args = parser.parse_args()
    log.info(args)
    main(args)
```


### 5.2 数据预处理

读取数据后，需要进行一些预处理，例如GCN中对图中节点度数进行了标准化

```python
dataset = load(args.dataset)

indegree = dataset.graph.indegree()
norm = np.zeros_like(indegree, dtype="float32")
norm[indegree > 0] = np.power(indegree[indegree > 0], -0.5)
dataset.graph.node_feat["norm"] = np.expand_dims(norm, -1)
```


### 5.3 模型构建


**Step1 实例化[GraphWrapper](https://github.com/PaddlePaddle/PGL/blob/main/pgl/graph_wrapper.py)和[Program](https://www.paddlepaddle.org.cn/documentation/docs/zh/beginners_guide/basic_concept/program.html)**

- 定义train_program、startup_program和test_program等程序

```python
  place = fluid.CUDAPlace(0) if args.use_cuda else fluid.CPUPlace()
  train_program = fluid.Program()
  startup_program = fluid.Program()
  test_program = fluid.Program()
```

- 实例化GraphWrapper，它提供了图的基本信息，以及GNN算法message passing机制中的send和receive两个接口。

```python
with fluid.program_guard(train_program, startup_program): 
    gw = pgl.graph_wrapper.GraphWrapper(
        name="graph",
        place=place,
        node_feat=dataset.graph.node_feat_info())
```

**Step2 模型定义**

在train_program中定义要使用的模型结构，这里是双层的GCN模型

```python
    output = pgl.layers.gcn(gw,
                            gw.node_feat["words"],
                            hidden_size,
                            activation="relu",
                            norm=gw.node_feat['norm'],
                            name="gcn_layer_1")
    output = fluid.layers.dropout(
        output, 0.5, dropout_implementation='upscale_in_train')
    output = pgl.layers.gcn(gw,
                            output,
                            dataset.num_classes,
                            activation=None,
                            norm=gw.node_feat['norm'],
                            name="gcn_layer_2")
```

**Step3 损失函数计算**

- node_index和node_label定义了有标签样本的数据下标和标签数据

```python
    node_index = fluid.layers.data(
        "node_index",
        shape=[None, 1],
        dtype="int64",
        append_batch_size=False)
    node_label = fluid.layers.data(
        "node_label",
        shape=[None, 1],
        dtype="int64",
        append_batch_size=False)
```        

- 使用gather函数找出output中有标签样本的预测结果后，计算得到交叉熵损失函数值以及准确度

```python
    pred = fluid.layers.gather(output, node_index)
    loss, pred = fluid.layers.softmax_with_cross_entropy(
        logits=pred, label=node_label, return_softmax=True)
    acc = fluid.layers.accuracy(input=pred, label=node_label, k=1)
    loss = fluid.layers.mean(loss)
```

**Step4 构造测试程序**

复制构造test_program的静态图。到此为止，train_program和test_program的静态图结构完全相同，区别在于test_program不需要梯度计算和反向传播过程。

```python
test_program = train_program.clone(for_test=True)
```

**Step5 定义优化器**

为了实现train_program上的参数更新，需要定义优化器和优化目标，这里是用Adam最小化loss

```python
with fluid.program_guard(train_program, startup_program):
    adam = fluid.optimizer.Adam(
        learning_rate=1e-2,
        regularization=fluid.regularizer.L2DecayRegularizer(
            regularization_coeff=0.0005))
    adam.minimize(loss)
```


### 5.4 模型训练和测试

模型构建完成后，就可以定义一个Executor来执行program了

```python
exe = fluid.Executor(place)
```

**Step1 初始化**

执行startup_program进行初始化

```python
exe.run(startup_program)
```

**Step2 数据准备**

将预处理阶段读取到的数据集填充到GraphWrapper中，同时准备好训练、验证和测试阶段用到的样本下标和标签数据

```python
feed_dict = gw.to_feed(dataset.graph)

train_index = dataset.train_index
train_label = np.expand_dims(dataset.y[train_index], -1)
train_index = np.expand_dims(train_index, -1)

val_index = dataset.val_index
val_label = np.expand_dims(dataset.y[val_index], -1)
val_index = np.expand_dims(val_index, -1)

test_index = dataset.test_index
test_label = np.expand_dims(dataset.y[test_index], -1)
test_index = np.expand_dims(test_index, -1)
```

**Step3 训练和测试**

给Executor分别传入不同的program来执行训练和测试过程

- feed以字典形式给定了输入数据 {变量名：numpy数据}
- fetch_list给定了模型中需要取出结果的变量名，可以根据需要自行修改

```python
dur = []
for epoch in range(200):
    if epoch >= 3:
        t0 = time.time()
    feed_dict["node_index"] = np.array(train_index, dtype="int64")
    feed_dict["node_label"] = np.array(train_label, dtype="int64")
    train_loss, train_acc = exe.run(train_program,
                                        feed=feed_dict,
                                        fetch_list=[loss, acc],
                                        return_numpy=True)

	# 3个epoch后，统计每轮训练执行的时间然后求均值。
    if epoch >= 3:
        time_per_epoch = 1.0 * (time.time() - t0)
        dur.append(time_per_epoch)
    feed_dict["node_index"] = np.array(val_index, dtype="int64")
    feed_dict["node_label"] = np.array(val_label, dtype="int64")
    val_loss, val_acc = exe.run(test_program,
                                    feed=feed_dict,
                                    fetch_list=[loss, acc],
                                    return_numpy=True)

    log.info("Epoch %d " % epoch + "(%.5lf sec) " % np.mean(dur) +
                 "Train Loss: %f " % train_loss + "Train Acc: %f " % train_acc
                 + "Val Loss: %f " % val_loss + "Val Acc: %f " % val_acc)

feed_dict["node_index"] = np.array(test_index, dtype="int64")
feed_dict["node_label"] = np.array(test_label, dtype="int64")
test_loss, test_acc = exe.run(test_program,
                                  feed=feed_dict,
                                  fetch_list=[loss, acc],
                                  return_numpy=True)
log.info("Accuracy: %f" % test_acc)
```

图模型训练的基本框架大概就是这样啦，下次再见咯~


```python

```
