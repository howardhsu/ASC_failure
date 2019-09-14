# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team and authors from University of Illinois at Chicago.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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


import os
import logging

import random
import json

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, PreTrainedBertModel, BertModel
from pytorch_pretrained_bert.optimization import BertAdam

from . import absa_data_utils as data_utils

from . import modelconfig
from . import models
from .trainer import Trainer


logger = logging.getLogger(__name__)

class Manual_bal(Trainer):
    """
    Manually balance the weights: contra has weights #noncontra/total; noncontra has weights: #contra/total
    """
    
    def initial_train_sample_weights(self, train_features):
        total_count = len(train_features)
        contra_count = sum([f.contra for f in train_features])
        noncontra_count = total_count - contra_count
        contra_weight = float(noncontra_count) / total_count
        noncontra_weight = float(contra_count) / total_count
        return torch.tensor([contra_weight if f.contra else noncontra_weight for f in train_features], dtype=torch.float)
    
    def initial_valid_sample_weights(self, valid_features):
        total_count = len(valid_features)
        contra_count = sum([f.contra for f in valid_features])
        noncontra_count = total_count - contra_count
        contra_weight = float(noncontra_count) / total_count
        noncontra_weight = float(contra_count) / total_count
        return torch.tensor([contra_weight if f.contra else noncontra_weight for f in valid_features], dtype=torch.float)


