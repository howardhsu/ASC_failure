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


logger = logging.getLogger(__name__)

class Trainer:
    """
    A base ASC classifier trainer.
    """
    
    def _warmup_linear(self, x, warmup=0.002):
        if x < warmup:
            return x/warmup
        return 1.0 - x
    
    def _evalutate_on_train(self, model, eval_dataloader):
        model.eval()
        all_y_preds=[]
        for step, batch in enumerate(eval_dataloader):
            batch = tuple(t.cuda() for t in batch)
            input_ids, segment_ids, input_mask, _ = batch
            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask)
            y_preds = logits.detach().cpu().numpy().argmax(axis=-1)

            all_y_preds.append(y_preds)
        all_y_preds=np.concatenate(all_y_preds) 
        model.train()
        return all_y_preds
    
    def initial_train_sample_weights(self, train_features):
        return torch.ones(len(train_features), dtype=torch.float)
    
    def initial_valid_sample_weights(self, valid_features):
        return torch.ones(len(valid_features), dtype=torch.float)
    
    def epoch_weight_update(self, args, model, eval_dataloader, all_label_ids, all_sample_weights, train_features):
        # only print basic info without weight adjustment here.
        all_y_preds=self._evalutate_on_train(model, eval_dataloader)
        incorrect = (all_y_preds != all_label_ids.numpy() )
        logger.info("****************************************************************")        
        logger.info("# hard examples %i / %i", sum(incorrect), len(incorrect))
        all_contra = np.array([f.contra for f in train_features])
        p, r, _, _=sklearn.metrics.precision_recall_fscore_support(all_contra, incorrect, average = 'binary')
        logger.info("precision and recall of contra %f %f", p, r)
        pos_ratio = float(sum(np.logical_and(all_label_ids.numpy() == 0, incorrect))) / sum(incorrect)
        neg_ratio = float(sum(np.logical_and(all_label_ids.numpy() == 1, incorrect))) / sum(incorrect)
        neu_ratio = float(sum(np.logical_and(all_label_ids.numpy() == 2, incorrect))) / sum(incorrect)
        logger.info("pos neg neu ratio %f %f %f", pos_ratio, neg_ratio, neu_ratio)

    
    def set_model_param(self, model, args):
        pass
    
    def train(self, args):
        
        processor = data_utils.AscProcessor()
        label_list = processor.get_labels()
        tokenizer = BertTokenizer.from_pretrained(modelconfig.MODEL_ARCHIVE_MAP[args.bert_model])
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_steps = int(len(train_examples) / args.train_batch_size) * args.num_train_epochs

        train_features = data_utils.convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, "asc")
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        
        all_sample_weights = self.initial_train_sample_weights(train_features)
        
        train_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids, all_sample_weights)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        eval_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        #>>>>> validation        
        valid_examples = processor.get_dev_examples(args.data_dir)
        valid_features=data_utils.convert_examples_to_features(
            valid_examples, label_list, args.max_seq_length, tokenizer, "asc")
        valid_all_input_ids = torch.tensor([f.input_ids for f in valid_features], dtype=torch.long)
        valid_all_segment_ids = torch.tensor([f.segment_ids for f in valid_features], dtype=torch.long)
        valid_all_input_mask = torch.tensor([f.input_mask for f in valid_features], dtype=torch.long)
        valid_all_label_ids = torch.tensor([f.label_id for f in valid_features], dtype=torch.long)

        valid_all_weights = self.initial_valid_sample_weights(valid_features)
        
        valid_data = TensorDataset(valid_all_input_ids, valid_all_segment_ids, valid_all_input_mask, valid_all_label_ids, valid_all_weights)

        logger.info("***** Running validations *****")
        logger.info("  Num orig examples = %d", len(valid_examples))
        logger.info("  Num split examples = %d", len(valid_features))
        logger.info("  Batch size = %d", args.train_batch_size)

        valid_sampler = SequentialSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.train_batch_size)    

        best_valid_loss=float('inf')
        valid_losses=[]
        #<<<<< end of validation declaration


        model = getattr(models, args.model).from_pretrained(modelconfig.MODEL_ARCHIVE_MAP[args.bert_model], num_labels = len(label_list) )
        self.set_model_param(model, args)
        model.cuda()
        # Prepare optimizer
        param_optimizer = [(k, v) for k, v in model.named_parameters() if v.requires_grad==True]
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        t_total = num_train_steps
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)

        global_step = 0
        model.train()
        for epoch in range(args.num_train_epochs):
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.cuda() for t in batch)

                input_ids, segment_ids, input_mask, label_ids, sample_weights = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids, sample_weights)
                loss.backward()

                lr_this_step = args.learning_rate * self._warmup_linear(global_step/t_total, args.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            self.epoch_weight_update(args, model, eval_dataloader, all_label_ids, all_sample_weights, train_features)
                
            model.eval()
            with torch.no_grad():
                losses=0.
                for step, batch in enumerate(valid_dataloader):
                    batch = tuple(t.cuda() for t in batch)
                    input_ids, segment_ids, input_mask, label_ids, weights = batch
                    loss = model(input_ids, segment_ids, input_mask, label_ids, weights)
                    loss *= len(input_ids)
                    losses += loss.item()
                    valid_loss = losses / len(valid_all_input_ids)

                logger.info("validation loss: %f", valid_loss)
                valid_losses.append(valid_loss)
            if valid_loss < best_valid_loss:
                torch.save(model, os.path.join(args.output_dir, "model.pt") )
                best_valid_loss=valid_loss
            model.train()

        with open(os.path.join(args.output_dir, "valid.json"), "w") as fw:
            json.dump({"valid_losses": valid_losses}, fw)
