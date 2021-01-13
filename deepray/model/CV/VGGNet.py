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

flags.DEFINE_enum("VGGNetCore", "VGG16",
                  ["VGG16", "VGG19"],
                  "VGGNet type")

from deepray.model.model_classify import BaseClassifyModel


class VGGNet(BaseClassifyModel):
    def __init__(self, flags):
        super().__init__(flags)

    def build(self, input_shape):
        vgg = self.build_vggnet(self.flags.VGGNetCore)
        self.vgg = vgg(input_shape=input_shape,
                       include_top=False,
                       weights='imagenet')

    def build_network(self, features, is_training=None):
        logit = self.vgg(features)
        return logit

    def build_vggnet(self, core):
        if core == 'ResNet50':
            return tf.keras.applications.ResNet50
        elif core == 'ResNet101':
            return tf.keras.applications.ResNet101
        else:
            raise ValueError('--ResNetCore {} was not found.'.format(self.flags.ResNetCore))
