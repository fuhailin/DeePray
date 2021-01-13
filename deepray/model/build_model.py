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
Build model

Author:
    Hailin Fu, hailinfufu@outlook.com
"""
from absl import flags

from deepray.model import linear_model

flags.DEFINE_string("model", "lr", "model")


def BuildModel(flags):
    if flags.model == 'lr':
        model = linear_model.LogisitcRegression(flags)
    elif flags.model == 'fm':
        from deepray.model.RS import factorization_model
        model = factorization_model.FactorizationMachine(flags)
    elif flags.model == 'ffm':
        from deepray.model.RS import factorization_model
        model = factorization_model.FieldawareFactorizationMachine(flags)
    elif flags.model == 'nfm':
        from deepray.model.RS import factorization_model
        model = factorization_model.NeuralFactorizationMachine(flags)
    elif flags.model == 'afm':
        from deepray.model.RS import factorization_model
        model = factorization_model.AttentionalFactorizationMachine(flags)
    elif flags.model == 'deepfm':
        from deepray.model.RS import factorization_model
        model = factorization_model.DeepFM(flags)
    elif flags.model == 'xdeepfm':
        from deepray.model.RS import xDeepFM
        model = xDeepFM.ExtremeDeepFMModel(flags)
    elif flags.model == 'wdl':
        from deepray.model.RS import WDL
        model = WDL.WideDeepModel(flags)
    elif flags.model == 'dcn':
        from deepray.model.RS import DCN
        model = DCN.DeepCrossModel(flags)
    elif flags.model == 'autoint':
        from deepray.model.RS import attention_model
        model = attention_model.AutoIntModel(flags)
    elif flags.model == 'din':
        from deepray.model.RS import attention_model
        model = attention_model.DeepInterestNetwork(flags)
    elif flags.model == 'dien':
        from deepray.model.RS import attention_model
        model = attention_model.DeepInterestEvolutionNetwork
    elif flags.model == 'dsin':
        from deepray.model.RS import attention_model
        model = attention_model.DeepSessionInterestNetwork(flags)
    elif flags.model == 'flen':
        from deepray.model.RS import FLEN
        model = FLEN.FLENModel(flags)

    elif flags.model == 'lstm':
        from deepray.model.NLP import model_lstm
        model = model_lstm.CustomModel(flags)
    else:
        raise ValueError('--model {} was not found.'.format(flags.model))
    return model
