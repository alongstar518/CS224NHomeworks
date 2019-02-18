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

    def __init__(self, word_embedding_size, max_sentense_length):
        """
        
        :param word_embedding_size: 
        :param max_sentense_length: 
        """
        super(Highway, self).__init__()
        self.word_embedding_size = word_embedding_size
        self.max_sentense_length = max_sentense_length

        #Define Layers
        self.proj_layer = nn.Linear(in_features=self.word_embedding_size, out_features=self.word_embedding_size, bias=True)
        self.gate_layer = nn.Linear(in_features=self.word_embedding_size, out_features=self.word_embedding_size, bias=True)


    def projection(self, input):
        """

        :param input: shape of (batch, max_sentence_length,embedding size)
        :return: projection , shape of (batch, max_sentence_length,embedding_size)
        """
        proj_val = self.proj_layer(input)
        x_proj = F.relu(proj_val)
        #print("Projection is %s, shape= %s" % (x_proj, x_proj.size()))
        return x_proj

    def gate(self, input):
        """

        :param input:shape of (batch, max_sentence_length,embedding size)
        :return: shape of (batch, max_sentence_length,embedding size)
        """
        gate_val = self.gate_layer(input)
        x_gate = F.sigmoid(gate_val)
        #print("Gate is %s, shape=%s" % (x_gate,x_gate.size()))
        return x_gate

    def highway(self, input, projection, gate):
        """

        :param input: shape of (batch, max_sentence_length,embedding size)
        :param projection: shape of (batch, max_sentence_length,embedding size)
        :param gate: shape of (batch, max_sentence_length,embedding size)
        :return: shape of (batch, max_sentence_length,embedding size)
        """
        gate_proj = torch.mul(gate , projection)
        #print("gate_proj is %s, shape = %s" % (gate_proj, gate_proj.size()))
        gate_input = torch.mul((1-gate) , input)
        #print("gate_proj is %s , \n shape = %s" % (gate_input, gate_input.size()))

        high_way = gate_proj + gate_input

        #print("gate_proj is %s, shape= %s" % (high_way,high_way.size()))

        return high_way

    def forward(self, input):
        """

        :param input: shape of (batch, max_sentence_length,embedding size)
        :return: shape of (batch, max_sentence_length,embedding size)
        """
        proj = self.projection(input)
        gate = self.gate(input)
        highway = self.highway(input, proj,gate)

        return highway


### END YOUR CODE 

