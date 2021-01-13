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
import tensorflow as tf
from absl import flags

from deepray.model.model_classify import BaseClassifyModel

flags.DEFINE_enum("ResNetCore", "ResNet50",
                  ["ResNet50", "ResNet101", "ResNet152", "ResNet50V2", "ResNet101V2", "ResNet152V2"],
                  "ResNet type")


class ResNet(BaseClassifyModel):
    def __init__(self, flags):
        super().__init__(flags)

    def build(self, input_shape):
        resNetCore = self.build_resnet(self.flags.ResNetCore)
        self.resNet = resNetCore(input_shape=input_shape,
                                 include_top=False,
                                 weights='imagenet')

    def build_network(self, features, is_training=None):
        logit = self.resNet(features)
        return logit

    def build_resnet(self, core):
        if core == 'ResNet50':
            return tf.keras.applications.ResNet50
        elif core == 'ResNet101':
            return tf.keras.applications.ResNet101
        elif core == 'ResNet152':
            return tf.keras.applications.ResNet152
        elif core == 'ResNet50V2':
            return tf.keras.applications.ResNet50V2
        elif core == 'ResNet101V2':
            return tf.keras.applications.ResNet101V2
        elif core == 'ResNet152V2':
            return tf.keras.applications.ResNet152V2
        else:
            raise ValueError('--ResNetCore {} was not found.'.format(self.flags.ResNetCore))
