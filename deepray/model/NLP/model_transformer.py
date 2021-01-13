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

from absl import flags
from tensorflow.keras import layers

from deepray.base.layers.transformer import TransformerBlock
from deepray.base.layers.embedding import TokenAndPositionEmbedding
from deepray.model.NLP.model_lstm import CustomModel

flags.DEFINE_integer("maxlen",
                     200,
                     "Whether use residual structure to fuse the results from each layer of CIN.")
flags.DEFINE_integer("vocab_size", None, "")
flags.DEFINE_integer("num_heads", 2, "")
flags.DEFINE_integer("ff_dim", 32, "")


class Transformer(CustomModel):
    def __init__(self, flags):
        super(Transformer, self).__init__(flags)
        self.maxlen = flags.maxlen
        self.vocab_size = flags.vocab_size
        self.embed_dim = flags.embed_dim
        self.num_heads = flags.num_heads
        self.ff_dim = flags.ff_dim

    def build(self, input_shape):
        self.embedding_layer = TokenAndPositionEmbedding(self.maxlen, self.vocab_size, self.embed_dim)
        self.transformer_block = TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim)
        self.pool_layer = layers.GlobalAveragePooling1D()
        self.deep_layer = layers.Dense(20, activation="relu")

    def build_network(self, features, is_training=None):
        x = self.embedding_layer(features)
        x = self.transformer_block(x)
        x = self.pool_layer(x)
        x = layers.Dropout(0.1)(x)
        x = self.deep_layer(x)
        logit = layers.Dropout(0.1)(x)
        return logit
