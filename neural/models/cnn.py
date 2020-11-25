import torch
import torch.nn as nn

from neural.util import Initializer
from neural.util import Loader
from neural.modules import EncoderCNN

class CNN(nn.Module):
    
    def __init__(self, word_vocab_size, word_embedding_dim, word_out_channels, output_size, 
                 dropout_p=0.5,
                 pretrained=None,
                 cuda_device=0):
        
        super(CNN, self).__init__()
        self.cuda_device = cuda_device
        self.word_vocab_size = word_vocab_size
        self.word_embedding_dim = word_embedding_dim
        self.word_out_channels = word_out_channels
        
        self.initializer = Initializer()
        self.loader = Loader()

        self.embedding = nn.Embedding(word_vocab_size, word_embedding_dim)

        if pretrained is not None:
            self.embedding.weight = nn.Parameter(torch.FloatTensor(pretrained))

        #Q_CNN
        self.question_encoder = EncoderCNN(word_vocab_size, word_embedding_dim, word_out_channels)
        #A_CNN
        self.answer_encoder = EncoderCNN(word_vocab_size, word_embedding_dim, word_out_channels)

        self.interaction = nn.Parameter(torch.FloatTensor(word_out_channels, word_out_channels).uniform_(0, .1))
        self.dropout = nn.Dropout(p=dropout_p)
        
        hidden_size = word_out_channels * 2 + 1
        self.linear = nn.Linear(hidden_size, hidden_size//2)
        self.linear2 = nn.Linear(hidden_size//2, output_size)
        self.Tanh = nn.Tanh()

    def forward(self, questions, answers, encoder_only=False):

        questions_embedded = self.embedding(questions)
        answers_embedded = self.embedding(answers)

        question_features = self.question_encoder(questions_embedded)
        answer_features = self.answer_encoder(answers_embedded)

        i_question_features = torch.matmul(question_features, self.interaction)
        i_feature = torch.sum(i_question_features * answer_features, dim=1, keepdim=True)

        join_features = torch.cat((question_features, i_feature, answer_features), dim=1)

        if encoder_only:
            return question_features.data.cpu().numpy(), answer_features.data.cpu().numpy(),

        join_features = self.dropout(join_features)
        output = self.linear(join_features)
        output = self.Tanh(output)
        output = self.dropout(output)
        output = self.linear2(output)

        return output
