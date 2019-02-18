#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn
import torch
# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        self.embed_size = embed_size
        self.dropout_rate = 0.3
        self.char_embedding_size = 50
        pad_token_idx = vocab.char2id['<pad>']
        self.charembedding_layer = nn.Embedding(num_embeddings=len(vocab.char2id), embedding_dim=self.char_embedding_size,padding_idx=pad_token_idx)
        self.dropout_layer = nn.Dropout(self.dropout_rate)
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        device = input.device
        char_embeddings = self.charembedding_layer(input) # (max_sentense_lenth, batch_size, max_word_length, char_embedding_size)
        char_embeddings = char_embeddings.permute(1,0,3,2)
        word_embedding = []
        for emb in char_embeddings:
            cnn = CNN(self.char_embedding_size, self.embed_size)
            cnn.to(device)
            conv_out_for_high_way = cnn(emb)
            highway = Highway(self.embed_size)
            highway.to(device)
            highway_out = highway(conv_out_for_high_way)
            embedding = self.dropout_layer(highway_out)
            word_embedding.append(embedding)
        word_embedding = torch.stack(word_embedding)
        return word_embedding.permute(1,0,2)

        ### END YOUR CODE

