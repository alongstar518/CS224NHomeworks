#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        super(CharDecoder, self).__init__()
        self.target_vocab = target_vocab
        self.hidden_size = hidden_size
        self.vocab_size = len(target_vocab.char2id)
        self.charDecoder = nn.LSTM(input_size=char_embedding_size, hidden_size=hidden_size, num_layers=1, bidirectional=False)
        self.char_output_projection = nn.Linear(in_features=hidden_size, out_features= self.vocab_size, bias=True)
        self.decoderCharEmb = nn.Embedding(len(target_vocab.char2id), char_embedding_size)
        self.softmax = nn.Softmax()
        ### END YOUR CODE


    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        self.to(input.device)
        Y = self.decoderCharEmb(input)

        _scores = []
        last_hidden = dec_hidden
        for Y_t in torch.split(Y, 1, dim=0):
            outputs, last_hidden = self.charDecoder(Y_t, last_hidden)
            h_t = last_hidden[0].permute(1,0,2)
            score = self.char_output_projection(h_t)
            score = torch.squeeze(score, dim=1)
            _scores.append(score)
        scores = torch.stack(_scores)
        return scores, last_hidden


        ### END YOUR CODE


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).

        scores, _ = self.forward(char_sequence, dec_hidden)
        masks = []
        for char_seq in torch.split(char_sequence, 1, dim=0):
            _y = torch.zeros((char_sequence.size(1), self.vocab_size), dtype = torch.long, device = char_sequence.device)
            _y.scatter_(1, char_seq.transpose(0,1), 1)
            _mask = (_y == self.target_vocab.char2id['<pad>']).float()
            _mask[torch.arange(0, _mask.size(0)).long(), 0] = 1
            masks.append(_mask)
        masks_tensor = torch.stack(masks)

        scores.data.masked_fill_(masks_tensor.byte(), 0)
        scores = scores.permute(1,2,0)
        char_sequence = char_sequence.t()
        ce = nn.CrossEntropyLoss()
        loss = ce(scores, char_sequence)
        return loss

        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.

        decodedWords = []
        batch_size = initialStates[0].size(1)
        current_chars = torch.tensor([[self.target_vocab.start_of_word]]*batch_size, device=device).t()
        states = initialStates
        for t in range(max_length):
            s_t1 , states = self.forward(current_chars, states)
            p_t1 = self.softmax(s_t1)
            _, max_indxs = torch.max(p_t1, dim=2)
            decodedWords.append(max_indxs)
            current_chars = max_indxs

        decodedWords = torch.stack(decodedWords)
        decodedWords = decodedWords.view(batch_size, -1)
        decodedWords = decodedWords.tolist()
        output = []
        for dw in decodedWords:
            dw = map(lambda x: self.target_vocab.id2char[x],dw)
            output.append(''.join(dw).replace('<pad>', '').replace('{', '').replace('}', ''))

        return output





        ### END YOUR CODE

