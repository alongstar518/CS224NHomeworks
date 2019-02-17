#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

### YOUR CODE HERE for part 1i

class CNN(nn.Module):

    """
    CNN module
    """

    def __init__(self, word_embedding_size, filter_count, kernel_size=5):
        super(CNN, self).__init__()

        self.word_embedding_size = word_embedding_size
        self.filter_count = filter_count
        self.kernel_size = kernel_size

        self.cov1d_layer = nn.Conv1d(input_channel_size=self.word_embedding_size, out_channels=self.filter_count, bias=True)
        self.maxpool_layer = nn.MaxPool1d(kernel_size)

    def conv1d(self, input):
        """

        :param input:  size (batch_size, word_embedding_size, max_sentense_length)
        :return: size: (batch_size, word_embedding_size, max_word_lenth-kernel_size+1)
        """

        conv = self.conv1d(input)
        print("conv1d result: %s, shape=%s" % (conv, conv.size()))

        return conv

    def maxpool(self, input):
        """

        :param input:  size (batch_size, word_embedding_size, max_word_lenth-kernel_size+1)
        :return: word_embedding vector
        """

        max_pool = self.maxpool_layer(input)
        print("maxpool result: %s" % (max_pool, max_pool.size()))

        return max_pool

    def forward(self, input):

        return self.maxpool(F.relu(self.conv1d(input)))


### END YOUR CODE

