import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Implement mini-batch
class GruRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size=1, layers=1, bi=False):
        """
        IMPORTANT: Use batch_first convention for ease of use.
                   However, the hidden layer still use batch middle convension.
        """
        super(GruRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.layers = layers
        self.bi_mul = 2 if bi else 1

        self.encoder = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(input_size, hidden_size, self.layers, bidirectional=bi, batch_first=True)
        self.decoder = nn.Linear(hidden_size * self.bi_mul, output_size)
        self.softmax = F.softmax

    def forward(self, x, hidden):
        embeded = x
        gru_output, hidden = self.gru(embeded, hidden.view(self.layers * self.bi_mul, -1, self.hidden_size))
        gru_output.contiguous()
        output = self.decoder(gru_output.view(-1, self.hidden_size * self.bi_mul))
        return output.view(self.batch_size, -1, self.output_size), hidden

    def init_hidden(self, random=False):
        if random:
            return Variable(torch.randn(self.layers * self.bi_mul, self.batch_size, self.hidden_size))
        else:
            return Variable(torch.zeros(self.layers * self.bi_mul, self.batch_size, self.hidden_size))


class Engadget():
    def __init__(self, model, char2vec=None, output_char2vec=None):
        print('****** Engadget Model Initialize ******')
        self.model = model
        if char2vec is None:
            self.char2vec = Char2Vec()
        else:
            self.char2vec = char2vec

        if output_char2vec is None:
            self.output_char2vec = self.char2vec
        else:
            self.output_char2vec = output_char2vec

        self.loss = 0
        self.losses = []

    def init_hidden_(self, random=False):
        self.hidden = self.model.init_hidden(random)
        return self

    def save(self, fn="GRU_Engadget.tar"):
        torch.save({
            "hidden": self.hidden,
            "state_dict": self.model.state_dict(),
            "losses": self.losses
        }, fn)

    def load(self, fn):
        checkpoint = torch.load(fn)
        self.hidden = checkpoint['hidden']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.losses = checkpoint['losses']

    def setup_training(self, learning_rate):
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.optimizer.zero_grad()
        self.loss_fn = nn.CrossEntropyLoss()
        self.init_hidden_()

    def reset_loss(self):
        self.loss = 0

    def forward(self, input_batch, target_batch):

        self.hidden = self.hidden.detach()

        self.optimizer.zero_grad()
        self.next_(input_batch)
        self.target_vec = Variable(self.output_char2vec.char_code_batch(target_batch))
        new_loss = self.loss_fn(self.output.view(-1, self.model.output_size), self.target_vec.view(-1))
        self.loss += new_loss

    def descent(self):
        if self.loss is 0:
            print(self.loss)
            print('Warning: loss is zero.')
            return

        self.loss.backward()
        self.optimizer.step()
        self.losses.append(self.loss.cpu().data.numpy()[0])
        self.reset_loss()

    def embed(self, input_data):
        self.embeded = Variable(self.char2vec.one_hot_batch(input_data))
        return self.embeded

    def next_(self, input_text):
        self.output, self.hidden = self.model(self.embed(input_text), self.hidden)
        return self

    def output_chars(self, temperature = 1):
        self.softmax = self.model.softmax(self.output.view(-1, self.model.output_size) / temperature
                                          ).view(self.model.batch_size, -1, self.model.output_size)
        indexes = torch.multinomial(self.softmax.view(-1, self.model.output_size)
                                    ).view(self.model.batch_size, -1)
        return self.output_char2vec.vec2list_batch(indexes)
