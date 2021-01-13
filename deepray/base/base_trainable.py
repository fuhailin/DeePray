#  -*- coding: utf-8 -*-
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

"""
Base training model

Author:
    Hailin Fu, hailinfufu@outlook.com
"""

import os
import time
from datetime import datetime

import tensorflow as tf
import tensorflow_addons as tfa
from absl import flags, logging

from deepray.base.callbacks import LearningRateScheduler, CSVLogger, LossAndErrorPrintingCallback

FLAGS = flags.FLAGS
TIME_STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
flags.DEFINE_bool("gzip", False, 'tfrecord file is gzip or not')
flags.DEFINE_bool("lr_schedule", False, 'lr_schedule')
flags.DEFINE_enum("optimizer", "lazyadam",
                  ["adam", "adagrad", "adadelta", "lazyadam", "sgd", "RMSprop", "ftrl", "AdamW"],
                  "optimizer type")
flags.DEFINE_integer("patient_valid_passes", None,
                     "number of valid passes before early stopping")
flags.DEFINE_string("profile_batch", None, "batch range to profile")
flags.DEFINE_string("checkpoint_path", "summaries/{0}/cpk/".format(TIME_STAMP),
                    "path to save checkpoint")
flags.DEFINE_string("summaries_dir", "summaries/" + TIME_STAMP, "summary dir")
flags.DEFINE_string("train_data", None, "training data")
flags.DEFINE_string("valid_data", None, "validating data")

flags.DEFINE_integer("batch_size", 1, "batch size")
flags.DEFINE_integer("epochs", 1, "number of training epochs")
flags.DEFINE_integer("parallel_parse", 8, "Number of parallel parsing")
flags.DEFINE_integer("shuffle_buffer", 512, "Size of shuffle buffer")
flags.DEFINE_integer("prefetch_buffer", 4096, "Size of prefetch buffer")
flags.DEFINE_integer("random_seed", 42, "random seed")

LR_SCHEDULE = [
    # (epoch to start, learning rate) tuples
    (3, 0.05), (6, 0.01), (9, 0.005), (12, 0.001)
]


class BaseTrainable(object):
    def __init__(self, flags):
        super().__init__(flags)
        self.seed_everything(self.flags.random_seed)
        if not os.path.exists(flags.summaries_dir):
            os.makedirs(flags.summaries_dir)
        logging.get_absl_handler().use_absl_log_file(
            program_name='DeePray',
            log_dir=flags.summaries_dir
        )
        logging.info(' {} Initialize training'.format(
            time.strftime("%Y%m%d %H:%M:%S")))

        self.flags = FLAGS
        logging.info('\ttf.app.flags.FLAGS:')
        for key, value in sorted(self.flags.flag_values_dict().items()):
            logging.info('\t{:25}= {}'.format(key, value))

        self.LABEL, self.CATEGORY_FEATURES, self.NUMERICAL_FEATURES, \
        self.VOC_SIZE, self.VARIABLE_FEATURES, self.COL_NAMES = self.get_summary()

    def seed_everything(self, seed=10):
        tf.random.set_seed(seed)
        # random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        # np.random.seed(seed)

    def build_loss(self):
        if self.VOC_SIZE[self.LABEL] == 2:
            return tf.keras.losses.BinaryCrossentropy()
        else:
            return tf.keras.losses.SparseCategoricalCrossentropy()

    @classmethod
    def get_summary(cls):
        raise NotImplementedError(
            "parser(called by tfrecord_pipeline): not implemented!")

    def build_metrics(self):
        metrics = []
        if self.VOC_SIZE[self.LABEL] == 2:
            metrics.append(tf.keras.metrics.AUC(curve='ROC', name='roc'))
            metrics.append(tf.keras.metrics.AUC(curve='PR', name='pr'))
            metrics.append(tf.keras.metrics.BinaryAccuracy())
        else:
            metrics.append(tf.keras.metrics.AUC(curve='ROC', name='roc', multi_label=True))
            metrics.append(tf.keras.metrics.AUC(curve='PR', name='pr', multi_label=True))
            metrics.append(tf.keras.metrics.SparseCategoricalAccuracy())
        return metrics

    @classmethod
    def build_data_iterator(cls, files, batch_size, epochs, shuffle=True):
        raise NotImplementedError(
            "data_iterator(called by create_train_data_iterator): not implemented!")

    # @classmethod
    # def read_list_from_file(cls, filename):
    #     if not os.path.isfile(filename):
    #         raise ValueError('{} should be a text file'.format(filename))
    #     with open(filename) as f:
    #         record_files = [path.strip() for path in f]
    #         return record_files

    def create_train_data_iterator(self):
        # train_data = self.read_list_from_file(self.flags.train_data)
        # valid_data = self.read_list_from_file(self.flags.valid_data)
        self.train_iterator = self.build_data_iterator(self.flags.train_data,
                                                       self.flags.batch_size,
                                                       epochs=self.flags.epochs, shuffle=False)
        self.valid_iterator = self.build_data_iterator(self.flags.valid_data,
                                                       self.flags.batch_size,
                                                       epochs=1,
                                                       shuffle=False)

    def build_callbacks(self):
        callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=self.flags.summaries_dir),
            CSVLogger(self.flags.summaries_dir + '/log.csv', append=True, separator=','),
            LossAndErrorPrintingCallback()
        ]
        if self.flags.profile_batch:
            tb_callback = tf.keras.callbacks.TensorBoard(log_dir=self.flags.summaries_dir,
                                                         profile_batch=self.flags.profile_batch)
            callbacks.append(tb_callback)
        if self.flags.patient_valid_passes:
            EarlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                             patience=self.flags.patient_valid_passes,
                                                             mode='min',
                                                             restore_best_weights=True)
            callbacks.append(EarlyStopping)
        if self.flags.checkpoint_path:
            # Create a callback that saves the model's weights
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.flags.checkpoint_path,
                                                             save_weights_only=True,
                                                             monitor='val_loss',
                                                             mode='auto',
                                                             save_best_only=True)
            callbacks.append(cp_callback)
        if self.flags.lr_schedule:
            callbacks.append(LearningRateScheduler(self.lr_schedule))
        return callbacks

    def run(self, model):
        self.create_train_data_iterator()
        metrics_object = self.build_metrics()
        loss_object = self.build_loss()
        optimizer_object = self.build_optimizer()
        # for features, labels in self.train_iterator:
        #     for key, value in features.items():
        #         if isinstance(value, tf.sparse.SparseTensor):
        #             test = tf.sparse.to_dense(value)
        #             test1 = test.numpy()
        #             print(key, test)
        #         else:
        #             print(key, value)
        #     print('\n')
        #     with tf.GradientTape() as tape:
        #         predictions = model(features, training=True)
        #         loss = loss_object(labels, predictions)
        #     gradients = tape.gradient(loss, model.trainable_variables)
        #     optimizer_object.apply_gradients(zip(gradients, model.trainable_variables))
        model.compile(optimizer=optimizer_object,
                      loss=loss_object,
                      metrics=metrics_object)


        callbacks_object = None#self.build_callbacks()
        # import tempfile
        # model_dir = tempfile.mkdtemp()
        # keras_estimator = tf.keras.estimator.model_to_estimator(
        #     keras_model=model, model_dir=model_dir)
        # keras_estimator.train(input_fn=self.train_iterator, steps=500)
        # eval_result = keras_estimator.evaluate(input_fn=self.valid_iterator, steps=10)
        # print('Eval result: {}'.format(eval_result))
        history = model.fit(self.train_iterator, validation_data=self.valid_iterator,
                            epochs=self.flags.epochs, callbacks=callbacks_object)
        return history

    def build_optimizer(self):
        if self.flags.optimizer == "adam":
            optimizer = tf.keras.optimizers.Adam
        elif self.flags.optimizer == "adadelta":
            optimizer = tf.keras.optimizers.Adadelta
        elif self.flags.optimizer == "adagrad":
            optimizer = tf.keras.optimizers.Adagrad
        elif self.flags.optimizer == "lazyadam":
            optimizer = tfa.optimizers.LazyAdam
        elif self.flags.optimizer == "ftrl":
            optimizer = tf.keras.optimizers.Ftrl
        elif self.flags.optimizer == "sgd":
            optimizer = tf.keras.optimizers.SGD
        elif self.flags.optimizer == "RMSprop":
            optimizer = tf.keras.optimizers.RMSprop
        elif self.flags.optimizer == "AdamW":
            optimizer = tfa.optimizers.AdamW
        else:
            raise ValueError('--optimizer {} was not found.'.format(self.flags.optimizer))
        return optimizer(learning_rate=self.flags.learning_rate)

    def lr_schedule(self, epoch, lr):
        """Helper function to retrieve the scheduled learning rate based on epoch."""
        if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
            return lr
        for i in range(len(LR_SCHEDULE)):
            if epoch == LR_SCHEDULE[i][0]:
                return LR_SCHEDULE[i][1]
        return lr
