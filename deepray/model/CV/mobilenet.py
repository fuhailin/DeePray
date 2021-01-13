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

from deepray.model.model_classify import BaseClassifyModel


class MobileNet(BaseClassifyModel):
    def build(self, input_shape):
        self.mobile = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                        include_top=False,
                                                        weights='imagenet')

    def build_network(self, features, is_training=None):
        logit = self.vgg(features)
        return logit
