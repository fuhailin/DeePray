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
import os

import pandas as pd
import tensorflow as tf
from absl import flags
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers

from deepray.model.model_classify import BaseClassifyModel
from deepray.utils.converter import CSV2TFRecord
from deepray.utils.encoder import MultiColumnLabelEncoder

FLAGS = flags.FLAGS

flags.DEFINE_integer("embed_dim", 128, "Embedding dim")
flags.DEFINE_integer("MAX_SEQUENCE_LENGTH", 50, "")

VECTOR_WORD_PATH = '/examples/tencent2020/data/vector_word.npz'
output_path = '/examples/tencent2020/data/'


class CustomModel(BaseClassifyModel):
    def build(self, input_shape):
        self.lstm_block = tf.keras.layers.LSTM(50)

    def build_network(self, features, is_training=None):
        logit = self.lstm_block(features)
        return logit

    def build_features(self, features, embedding_suffix=''):
        sparse_ev_list = []
        for key in self.VARIABLE_FEATURES:
            value = tf.sparse.to_dense(features[key])
            sparse_ev_list.append(
                self.EmbeddingDict[key](value))
        # inputs = self.concat(sparse_ev_list)
        return sparse_ev_list

    def build_EmbeddingDict(self):
        _dict = {}
        for feat in self.VARIABLE_FEATURES:
            if 1:
                import gensim
                if not os.path.exists(VECTOR_WORD_PATH):
                    raise ValueError("")
                w2v_model = gensim.models.KeyedVectors.load_word2vec_format(VECTOR_WORD_PATH, binary=False)
                w2v_weights = w2v_model.wv.vectors
                vocab_size, embedding_size = w2v_weights.shape
                print("Vocabulary Size: {} - Embedding Dim: {}".format(vocab_size, embedding_size))
            emb = layers.Embedding(input_dim=vocab_size,
                                   output_dim=embedding_size,
                                   weights=[w2v_weights],
                                   input_length=self.flags.MAX_SEQUENCE_LENGTH,
                                   mask_zero=True,
                                   trainable=False)
            _dict[feat] = emb
        return _dict

    @classmethod
    def tfrecord_pipeline(cls, tfrecord_file, batch_size,
                          epochs, shuffle=True):
        flags = FLAGS
        dataset = tf.data.TFRecordDataset(tfrecord_file, compression_type='GZIP' if flags.gzip else None,
                                          num_parallel_reads=tf.data.experimental.AUTOTUNE
                                          if flags.parallel_reads_per_file is None else flags.parallel_reads_per_file).map(
            cls.parser,
            num_parallel_calls=tf.data.experimental.AUTOTUNE if flags.parallel_parse is None else flags.parallel_parse)
        dataset = dataset.batch(batch_size)
        return dataset

    def create_train_data_iterator(self):
        if not os.path.exists(output_path + 'train_seq.tfrecord'):
            train_path = '/Users/vincent/Dataset/tencent2020/train_preliminary/'
            test_path = '/Users/vincent/Dataset/tencent2020/test/'
            user = pd.read_csv(train_path + 'user.csv')
            ad = pd.read_csv(train_path + 'ad.csv')
            click = pd.read_csv(train_path + 'click_log.csv').head(1000)
            print(click.shape)
            train_df = click.merge(user, on=['user_id'], how='left')
            train_df = train_df.merge(ad, on=['creative_id'], how='left')
            train_df['gender'] = train_df['gender'] - 1
            train_df['age'] = train_df['age'] - 1
            train_df['part'] = 'train'

            ad_test = pd.read_csv(test_path + 'ad.csv')
            click_test = pd.read_csv(test_path + 'click_log.csv').head(1000)
            test_df = click_test.merge(ad_test, on=['creative_id'], how='left')
            test_df['part'] = 'test'

            total_df = pd.concat([train_df, test_df], axis=0)
            ######### LABEL ENCODER

            total_df = MultiColumnLabelEncoder(columns=['ad_id']).fit_transform(total_df)
            total_df['ad_id'] = total_df['ad_id'] % 65536
            print('ad_id categorical size:', len(total_df['ad_id'].unique()))

            train_df = total_df[total_df['part'] == 'train']
            train_df.sort_values(by=['time'], inplace=True)
            aggregated_train = train_df.groupby('user_id').agg({'ad_id': lambda x: list(x)}).reset_index()
            aggregated_train['age'] = train_df['age']
            aggregated_train['gender'] = train_df['gender']
            aggregated_train['age'] = train_df['age'].astype(int)
            aggregated_train['gender'] = train_df['gender'].astype(int)

            train, valid = train_test_split(aggregated_train, test_size=0.2)
            converter = CSV2TFRecord(LABEL=['gender'],
                                     CATEGORY_FEATURES=[],
                                     NUMERICAL_FEATURES=[],
                                     VARIABLE_FEATURES=['ad_id'])
            converter(train, out_file=output_path + 'train_seq.tfrecord')
            converter(valid, out_file=output_path + 'train_seq.tfrecord')

        self.train_iterator = self.tfrecord_pipeline(
            [output_path + 'train_seq.tfrecord'], self.flags.batch_size, epochs=1
        )
        self.valid_iterator = self.tfrecord_pipeline(
            [output_path + 'train_seq.tfrecord'], self.flags.batch_size, epochs=1, shuffle=False
        )

    def load_voc_summary(self):
        """
        model doesn't support categorical feature custom embedding size
        """
        voc_emb_size = dict()
        for key, voc_size in self.VOC_SIZE.items():
            voc_size = 1736
            if key in self.VARIABLE_FEATURES:
                emb_size = self.flags.embed_dim
                voc_emb_size[key] = [voc_size, emb_size]
        for k, v in voc_emb_size.items():
            if k == self.LABEL:
                continue
            self.print_emb_info(k, v[0], v[1])
        return voc_emb_size
