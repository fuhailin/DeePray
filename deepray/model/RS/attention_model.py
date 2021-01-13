#  Copyright © 2020-2020 Hailin Fu All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================

"""
Author:
    Hailin Fu, hailinfufu@outlook.com
"""

import tensorflow as tf
from absl import flags

from deepray.base.layers.interactions import SelfAttentionNet
from deepray.model.classify_model import BaseClassifyModel

FLAGS = flags.FLAGS

flags.DEFINE_string("history_item", "hist_iid",
                    "feature name of user historical sequence")
flags.DEFINE_string("candidate_item", "hist_cate_id",
                    "feature name of candidate item sequence")

flags.DEFINE_integer("heads", 2, "number of heads")
flags.DEFINE_integer("field_size", 23, "number of fields")
flags.DEFINE_integer("blocks", 2, "number of blocks")
flags.DEFINE_string("block_shape", "16,16",
                    "output shape of each block")
flags.DEFINE_bool("has_residual", False, "add has_residual")


# has_wide 是否有wide层

# isScale 是否需要对连续型变量进行scala操作，即对大于1的数据加上自然数e然后去自然对数的操作
# cut_off 计算分类词频时，低于该值的类别归为一类


class AutoIntModel(BaseClassifyModel):

    def __init__(self, flags):
        super().__init__(flags)

    def build(self, input_shape):
        hidden = [int(h) for h in self.flags.deep_layers.split(',')]
        self.deep_block = self.build_deep(hidden=hidden)
        self.attention_block = self.build_attention(concat_last_deep=True)

    def build_network(self, features, is_training=None):
        """

        :param features:
        :param is_training:
        :return:
        """
        deep_part = self.deep_block(features, is_training=is_training)
        attention_part = self.attention_block(features, is_training=is_training)
        logit = tf.concat([attention_part, deep_part], -1)
        return logit

    def build_attention(self, concat_last_deep):
        hidden = [16, 16]  # [int(h) for h in self.flags.deep_layers.split(',')]
        return SelfAttentionNet(hidden=hidden, concat_last_deep=concat_last_deep)


class DeepInterestNetwork(BaseClassifyModel):

    def __init__(self, flags):
        super().__init__(flags)
        self.flags = flags

    def build(self, input_shape):
        # 1. mlp part
        hidden = [int(h) for h in self.flags.deep_layers.split(',')]
        self.mlp_block = self.build_deep(hidden=hidden)

        # 2. DIN
        self.din_block = self.build_din()

    def build_network(self, features, is_training=None):
        """
        TODO

        :param features:
        :param is_training:
        :return:
        """
        ev_list, sparse_ev_list, fv_list = features

        din_part = self.din_block(sparse_ev_list[self.flags.candidate_item],
                                  sparse_ev_list[self.flags.history_item],
                                  is_training)
        deep_part = self.mlp_block(self.concat(ev_list + fv_list))
        logit = tf.concat(values=[din_part, deep_part], axis=1)
        return logit

    def build_features(self, features, embedding_suffix=''):
        """
        categorical feature id starts from -1 (as missing)
        """
        ev_list = [self.EmbeddingDict[key](features[key])
                   for key in self.CATEGORY_FEATURES]
        sparse_ev_list = {key: self.EmbeddingDict[key](features[key],
                                                       combiner=self.flags.sparse_embedding_combiner) for key in
                          self.VARLEN_FEATURES}
        fv_list = [self.build_dense_layer(features[key]) for key in self.NUMERICAL_FEATURES]
        return ev_list, sparse_ev_list, fv_list

    def build_din(self):
        """
        TODO

        Returns:

        """
        raise NotImplementedError("build_cin: not implemented!")

    def din_attention(self, query, facts, attention_size, mask, stag='null', mode='SUM', softmax_stag=1,
                      time_major=False):
        if isinstance(facts, tuple):
            # In case of Bi-RNN, concatenate the forward and the backward RNN
            # outputs.
            facts = tf.concat(facts, 2)
            print("querry_size mismatch")
            query = tf.concat(values=[
                query,
                query,
            ], axis=1)

        if time_major:
            # (T,B,D) => (B,T,D)
            facts = tf.transpose(facts, [1, 0, 2])
        mask = tf.equal(mask, tf.ones_like(mask))
        # D value - hidden size of the RNN layer
        facts_size = facts.get_shape().as_list()[-1]
        querry_size = query.get_shape().as_list()[-1]
        queries = tf.tile(query, [1, tf.shape(facts)[1]])
        queries = tf.reshape(queries, tf.shape(facts))
        din_all = tf.concat(
            [queries, facts, queries - facts, queries * facts], axis=-1)
        d_layer_1_all = tf.keras.layers.Dense(
            din_all, 80, activation=tf.nn.sigmoid, kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=1.0, mode="fan_avg", distribution="normal"), name='f1_att' + stag)
        d_layer_2_all = tf.keras.layers.Dense(
            d_layer_1_all, 40, activation=tf.nn.sigmoid, kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=1.0, mode="fan_avg", distribution="normal"), name='f2_att' + stag)
        d_layer_3_all = tf.keras.layers.Dense(
            d_layer_2_all, 1, activation=None, kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=1.0, mode="fan_avg", distribution="normal"), name='f3_att' + stag)
        d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])
        scores = d_layer_3_all
        # Mask
        # key_masks = tf.sequence_mask(facts_length, tf.shape(facts)[1])   # [B, T]
        key_masks = tf.expand_dims(mask, 1)  # [B, 1, T]
        paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
        scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]

        # Scale
        # scores = scores / (facts.get_shape().as_list()[-1] ** 0.5)

        # Activation
        if softmax_stag:
            scores = tf.nn.softmax(scores)  # [B, 1, T]

        # Weighted sum
        if mode == 'SUM':
            output = tf.matmul(scores, facts)  # [B, 1, H]
            # output = tf.reshape(output, [-1, tf.shape(facts)[-1]])
        else:
            scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])
            output = facts * tf.expand_dims(scores, -1)
            output = tf.reshape(output, tf.shape(facts))
        return output


class DeepInterestEvolutionNetwork(BaseClassifyModel):

    def __init__(self, flags):
        super(DeepInterestEvolutionNetwork, self).__init__(flags)

    def build_network(self, features, is_training=None):
        """
        TODO

        :param features:
        :param is_training:
        :return:
        """


class DeepSessionInterestNetwork(BaseClassifyModel):
    def __init__(self, flags):
        super(DeepSessionInterestNetwork, self).__init__(flags)

    def build_network(self, features, is_training=None):
        """
        TODO

        :param features:
        :param is_training:
        :return:
        """
