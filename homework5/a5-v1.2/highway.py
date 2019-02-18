#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

### YOUR CODE HERE for part 1h
class Highway(nn.Module):
    """
    Highway module to generate word embedding
    """

    def __init__(self, word_embedding_size):
        """
        
        :param word_embedding_size: 
        :param max_sentense_length: 
        """
        super(Highway, self).__init__()
        self.word_embedding_size = word_embedding_size

        #Define Layers
        self.proj_layer = nn.Linear(in_features=self.word_embedding_size, out_features=self.word_embedding_size, bias=True)
        self.gate_layer = nn.Linear(in_features=self.word_embedding_size, out_features=self.word_embedding_size, bias=True)

    def forward(self, input):
        """

        :param input: shape of (max_sentence_length,embedding size)
        :return: shape of (max_sentence_length,embedding size)
        """
        proj_val = self.proj_layer(input)
        proj = F.relu(proj_val)

        gate_val = self.gate_layer(input)
        gate = F.sigmoid(gate_val)

        high_way = gate * proj + (1-gate) * input

        return high_way


### END YOUR CODE 

