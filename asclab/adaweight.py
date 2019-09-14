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
import sklearn.metrics
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, PreTrainedBertModel, BertModel
from pytorch_pretrained_bert.optimization import BertAdam

from . import absa_data_utils as data_utils

from . import modelconfig
from . import models

from .trainer import Trainer

logger = logging.getLogger(__name__)

class AdaWeight(Trainer):
    """Adaweight use a adaboost-style example weighting function.
    """

    def initial_train_sample_weights(self, train_features):
        return torch.ones(len(train_features) )/len(train_features)
    
    def epoch_weight_update(self, args, model, eval_dataloader, all_label_ids, all_sample_weights, train_features):
        """in-place change to all_sample_weights.
        """
        epsilon = 1e-07
        
        #>>>> perform weight adjustment the end of each epoch.            
        all_y_preds=self._evalutate_on_train(model, eval_dataloader)
        incorrect = (all_y_preds != all_label_ids.numpy() )
        estimator_error = np.average(incorrect, weights=all_sample_weights.numpy(), axis=0)
        estimator_weight = np.log(max(epsilon, (1. - estimator_error) + args.factor) / max(epsilon, estimator_error - args.factor) )
        scale = np.exp(estimator_weight * incorrect)
        all_sample_weights.mul_(torch.from_numpy(scale).float() )
        logger.info("sample_weights %s", str(all_sample_weights[:20]) )
        logger.info("****************************************************************")
        logger.info("estimator_error %f", estimator_error)
        logger.info("estimator_weight (should be >0) %f", estimator_weight)
        
        logger.info("# hard examples %i / %i", sum(incorrect), len(incorrect))
        
        all_contra = np.array([f.contra for f in train_features])
        
        p, r, _, _=sklearn.metrics.precision_recall_fscore_support(all_contra, incorrect, average = 'binary')
        logger.info("precision and recall of contra %f %f", p, r)
        
        pos_ratio = float(sum(np.logical_and(all_label_ids.numpy() == 0, incorrect))) / sum(incorrect)
        neg_ratio = float(sum(np.logical_and(all_label_ids.numpy() == 1, incorrect))) / sum(incorrect)
        neu_ratio = float(sum(np.logical_and(all_label_ids.numpy() == 2, incorrect))) / sum(incorrect)
        logger.info("pos neg neu ratio %f %f %f", pos_ratio, neg_ratio, neu_ratio)
        