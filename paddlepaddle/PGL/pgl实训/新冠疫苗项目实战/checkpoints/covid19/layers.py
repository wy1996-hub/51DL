# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Doc String
"""
import paddle
import paddle.fluid as fluid


def copy_send(src_feat, dst_feat, edge_feat):
    """doc"""
    return src_feat["h"]


def mean_recv(feat):
    """doc"""
    return fluid.layers.sequence_pool(feat, pool_type="average")


def sum_recv(feat):
    """doc"""
    return fluid.layers.sequence_pool(feat, pool_type="sum")


def max_recv(feat):
    """doc"""
    return fluid.layers.sequence_pool(feat, pool_type="max")


def simple_gnn(gw, feature, hidden_size, act, name):
    """doc"""
    msg = gw.send(copy_send, nfeat_list=[("h", feature)])
    neigh_feature = gw.recv(msg, sum_recv)
    self_feature = feature
    output = self_feature + neigh_feature

    output = fluid.layers.fc(output,
                            hidden_size,
                            act=act,
                            param_attr=fluid.ParamAttr(name=name + '_w'),
                            bias_attr=fluid.ParamAttr(name=name + '_b'))

    return output

def my_gnn(gw, node_feat, edge_feat, hidden_size, act, name):

    def my_gnn_send(src_feat, dst_feat, edge_feat):
        return src_feat['h'] + dst_feat['h']
    
    def my_gnn_recv(feat):
        return fluid.layers.sequence_pool(feat, pool_type="sum")

    msg = gw.send(my_gnn_send, nfeat_list=[('h', node_feat)], efeat_list=[('h', edge_feat)])
    neigh_feature = gw.recv(msg, my_gnn_recv)
    self_feature = node_feat
    output = self_feature + neigh_feature

    output = fluid.layers.fc(output,
                            hidden_size,
                            act=act,
                            param_attr=fluid.ParamAttr(name=name + '_w'),
                            bias_attr=fluid.ParamAttr(name=name + '_b'))

    return output























































