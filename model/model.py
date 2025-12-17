import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, vocab_size=None):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size

        self.embedding = nn.Embedding(vocab_size, input_size)
        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        
    def forward(self, input_seq, input_lengths, hidden=None):
        embedded = self.embedding(input_seq)
        packed_seq = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed_seq, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)

        return outputs, hidden

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

    def dot_score(self, hidden, encoder_outputs):
        return torch.sum(hidden * encoder_outputs, dim=2)

    def forward(self, hidden, encoder_outputs):
        attn_energies = self.dot_score(hidden, encoder_outputs)
        attn_energies = attn_energies.t()
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        rnn_output, hidden = self.gru(embedded, last_hidden)
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))

        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)

        concat_input = torch.cat((rnn_output, context), dim=1)
        concat_output = torch.tanh(self.concat(concat_input))
        output = self.out(concat_output)

        return output, hidden

class EncoderDecoder(nn.Module):
    def __init__(self, attn_model, embedding, input_size, hidden_size, output_size, vocab_size, n_layers=1, dropout=0.1):
        self.embedding = embedding
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.encoder = EncoderRNN(input_size=input_size, hidden_size=hidden_size, num_layers=n_layers, vocab_size=vocab_size)
        self.decoder = LuongAttnDecoderRNN(attn_model=attn_model, embedding=embedding, hidden_size=hidden_size, output_size=output_size, n_layers=n_layers, dropout=dropout)

    def forward(self, input_seq, input_lengths, target_seq, target_lengths):
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lengths, hidden=None)
        decoder_outputs, decoder_hidden = self.decoder(target_seq, encoder_hidden, encoder_outputs)

        return decoder_outputs, decoder_hidden