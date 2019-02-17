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

    def __init__(self, char_embedding_size, word_embedding_size, kernel_size=5):
        super(CNN, self).__init__()

        self.char_embedding_size = char_embedding_size
        self.word_embedding_size = word_embedding_size
        self.kernel_size = kernel_size

        self.cov1d_layer = nn.Conv1d(in_channels=self.char_embedding_size, out_channels=self.word_embedding_size, kernel_size=self.kernel_size, bias=True)
        self.maxpool_layer = nn.MaxPool1d(kernel_size)

    def conv1d(self, input):
        """

        :param input:  size (batch_size, char_embedding_size, max_word_length)
        :return: size: (batch_size, word_embedding_size, max_word_lenth-kernel_size+1)
        """

        conv = self.cov1d_layer(input)
        print("conv1d result: %s, shape=%s" % (conv, conv.size()))

        return conv

    def maxpool(self, input):
        """

        :param input:  size (batch_size, word_embedding_size, max_word_lenth-kernel_size+1)
        :return: word_embedding vector
        """

        max_pool = torch.max(input, dim=2)


        return max_pool[0]

    def forward(self, input):
        """

        :param input: (max_sentense_lenth, batch_size, max_word_length, char_embedding_size)
        :return: word embedding vectors for sentense. (batch_size, word_embedding_size, max_word_length)
        """
        input = input.transpose(2,3).transpose(0,1)
        out = []
        for i in range(input.size(0)):
            conv = self.conv1d(input[i])
            relu = F.relu(conv)
            out_sent = self.maxpool(relu)
            out.append(out_sent)

        forward_out = torch.stack(out)

        return forward_out


### END YOUR CODE

