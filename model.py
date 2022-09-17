"""
Copyright (c) 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch.nn as nn
import torch

from modules.feature_extraction import VGG_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        
        self.batch_max_length = 25
        self.imgH = 32
        self.imgW = 100 
        self.Transformation = 'None'
        self.FeatureExtraction = 'VGG' 
        self.SequenceModeling = 'None'
        self.Prediction = 'CTC' 
        self.num_fiducial = 20 
        self.input_channel = 3
        self.output_channel = 512
        self.hidden_size = 256
        self.num_class = 67
        self.stages = {'Trans': self.Transformation, 'Feat': self.FeatureExtraction,
                       'Seq': self.SequenceModeling, 'Pred': self.Prediction}

        """ Transformation """

        """ FeatureExtraction """
        self.FeatureExtraction = VGG_FeatureExtractor(self.input_channel, self.output_channel)
        self.FeatureExtraction_output = self.output_channel  # int(imgH/16-1) * 512
        # self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, self.hidden_size, self.hidden_size),
            BidirectionalLSTM(self.hidden_size, self.hidden_size, self.hidden_size))
        self.SequenceModeling_output = self.hidden_size

        """ Prediction """
        self.Prediction = nn.Linear(self.SequenceModeling_output, self.num_class)
        
    def forward(self, input):
        """ Transformation stage """
        # None
        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)     # [1, 512, 1, 24]
        b, c, h, w = visual_feature.size()
        if torch.is_tensor(c):
        	c = c.item() 
        avgpool2d = nn.AdaptiveAvgPool2d((c, 1))
        visual_feature = avgpool2d(visual_feature.permute(0, 3, 1, 2))  
        # visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  
        # [b, c, h, w] -> [b, w, c, h]  [1, 24, 512, 1]
        
        visual_feature = visual_feature.squeeze(3)     # [1, 24, 512]

        """ Sequence modeling stage """
        contextual_feature = self.SequenceModeling(visual_feature)     # [1, 24, 256]   
        
        """ Prediction stage """
        prediction = self.Prediction(contextual_feature.contiguous())   # [1, 24, 67]
        return prediction
