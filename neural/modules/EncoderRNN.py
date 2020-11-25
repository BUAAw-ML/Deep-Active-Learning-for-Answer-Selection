import torch
import torch.nn as nn
from torch.autograd import Variable

from .baseRNN import baseRNN


class EncoderRNN(baseRNN):

    def __init__(self, vocab_size, embedding_size, hidden_size=200, input_dropout_p=0,
                 output_dropout_p=0, n_layers=1, bidirectional=True, rnn_cell='lstm', cuda_device=0):

        super(EncoderRNN, self).__init__(vocab_size, hidden_size, input_dropout_p,
                                             output_dropout_p, n_layers, rnn_cell)

        self.cuda_device = cuda_device
        self.rnn = self.rnn_cell(embedding_size, hidden_size, n_layers,
                                 bidirectional=bidirectional, dropout=output_dropout_p,
                                 batch_first=True)

    def forward(self, words, input_lengths):

        batch_size = words.size()[0]
        input_lengths = torch.LongTensor(input_lengths)

        _, sorted_indices = torch.sort(input_lengths, dim=0, descending=True)

        _, unsort_indices = torch.sort(sorted_indices, dim=0)

        sorted_lengths = list(input_lengths[sorted_indices])
        sorted_indices = Variable(sorted_indices).cuda(self.cuda_device)
        unsort_indices = Variable(unsort_indices).cuda(self.cuda_device)

        words = words.index_select(dim=0, index=sorted_indices)

        embedded = words
        embedded = self.input_dropout(embedded)
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_lengths, batch_first=True)
        _, output = self.rnn(embedded)

        output = output[0].transpose(0,1).contiguous().view(batch_size, -1)

        output = output.index_select(0, unsort_indices)

        return output