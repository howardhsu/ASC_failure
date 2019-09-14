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


import torch
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, PreTrainedBertModel, BertModel

class BaseClassifier(BertForSequenceClassification):
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, sample_weights=None):
        return super().forward(input_ids, token_type_ids, attention_mask, labels)
        

class ClassifierWithSampleWeight(BertForSequenceClassification):
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, sample_weights=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            if sample_weights is not None:
                if isinstance(sample_weights, str):
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1) )
                else:
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1) )
                    loss = (loss * sample_weights / sample_weights.sum() ).sum()
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1) )
            return loss
        else:
            return logits


class ClassifierWithFocalLoss(BertForSequenceClassification):
    def set_gamma(self, gamma):
        self.gamma = gamma
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, sample_weights=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            softmax = torch.nn.Softmax(dim = -1)
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1) )
            probs = softmax(logits)
            probs = probs.gather(1, labels.view(-1, 1) ).squeeze(-1)
            loss = ((1. - probs) ** self.gamma * loss).mean()
            return loss
        else:
            return logits
        

class ClassifierWithLogP(BertForSequenceClassification):
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, sample_weights=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            softmax = torch.nn.Softmax(dim = -1)
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1) )
            loss = (loss * loss / loss.sum() ).sum()
            return loss
        else:
            return logits