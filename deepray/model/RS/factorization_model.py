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

from deepray.base.layers.core import Linear
from deepray.base.layers.interactions import FMNet
from deepray.model.classify_model import BaseClassifyModel

FLAGS = flags.FLAGS
flags.DEFINE_integer("fm_order", 2, "FM net polynomial order")
flags.DEFINE_integer("fm_rank", 2,
                     "Number of factors in low-rank appoximation.")
flags.DEFINE_integer("latent_factors", 10,
                     "Size of factors in low-rank appoximation.")


class FactorizationMachine(BaseClassifyModel):

    def __init__(self, flags):
        super().__init__(flags)
        self.k = tf.constant(self.flags.latent_factors)

    def build(self, input_shape):
        self.fm_block = self.build_fm()

    def build_network(self, features, is_training=None):
        """

        :param features:
        :param is_training:
        :return:
        """
        logit = self.fm_block(features)
        return logit

    def build_fm(self):
        return FMNet(k=self.k)

    def get_config(self):
        config = super(FactorizationMachine, self).get_config()
        config.update({"k": self.self.flags.latent_factors})
        return config


class FieldawareFactorizationMachine(FactorizationMachine):

    def build(self, input_shape):
        self.linear_part = Linear()
        self.field_nums = len(input_shape)

        index = 0
        self.field_dict = {}
        for idx, val in enumerate(input_shape):
            if val in self.NUMERICAL_FEATURES:
                self.field_dict[index] = idx
                index += 1
            if val in self.CATEGORY_FEATURES:
                for i in range(self.voc_emb_size[val][1]):
                    self.field_dict[index] = idx
                    index += 1
        self.total_dims = len(self.field_dict)

        self.kernel = self.add_weight('v', shape=[self.total_dims, self.field_nums, self.k],
                                      initializer=tf.keras.initializers.TruncatedNormal(mean=0, stddev=0.01))

    def call(self, inputs, is_training=None, mask=None):
        linear_terms = self.linear_part(inputs)
        interaction_terms = tf.constant(0, dtype='float32')
        for i in range(self.total_dims):
            for j in range(i + 1, self.total_dims):
                interaction_terms += tf.multiply(
                    tf.reduce_sum(tf.multiply(self.kernel[i, self.field_dict[j]], self.kernel[j, self.field_dict[i]])),
                    tf.multiply(inputs[:, i], inputs[:, j]))
        interaction_terms = tf.reshape(interaction_terms, [-1, 1])
        logit = tf.math.add(linear_terms, interaction_terms)
        return logit


class NeuralFactorizationMachine(FactorizationMachine):
    def build(self, input_shape):
        hidden = [int(h) for h in self.flags.deep_layers.split(',')]
        self.B_Interaction_Layer = self.build_fm()
        self.Hidden_Layers = self.build_deep(hidden)

    def build_network(self, features, is_training=None):
        """

        :param features:
        :param is_training:
        :return:
        """
        logit = self.B_Interaction_Layer(features)
        logit = self.Hidden_Layers(logit)
        return logit


class AttentionalFactorizationMachine(FactorizationMachine):
    def build(self, input_shape):
        self.fm_block = self.build_fm()

    def build_network(self, features, is_training=None):
        """
        TODO

        :param features:
        :param is_training:
        :return:
        """


class DeepFM(FactorizationMachine):

    def build(self, input_shape):
        hidden = [int(h) for h in self.flags.deep_layers.split(',')]
        self.deep_block = self.build_deep(hidden=hidden)
        self.fm_block = self.build_fm()

    def build_network(self, features, is_training=None):
        """

        :param features:
        :param is_training:
        :return:
        """
        fm_out = self.fm_block(features)
        deep_out = self.deep_block(features)
        logit = tf.concat([deep_out, fm_out], -1)
        return logit


class FMNeuralNetwork(BaseClassifyModel):
    def __init__(self):
        super(FMNeuralNetwork, self).__init__()

    def build_network(self, features, is_training=None):
        """
        TODO

        :param features:
        :param is_training:
        :return:
        """