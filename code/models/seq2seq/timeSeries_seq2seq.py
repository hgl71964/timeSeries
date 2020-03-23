import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim as optim
import random
import os


class seq2seq_utility():

    @staticmethod
    def show_parameter():

        print(
            '''
            This is a seq2seq model, embedding should be done before input into this model

            RNN used is GRU

            default loss function is MSELoss()
            
            #####
            Instantiate:
            seq2seq_utility.instan_things(**kwargs),
                    in which you should define the following dictionary parameters
            e.g.
            param = {'max_epochs':64,
                    'learning_rate':1e-3,
                    'clip':1,                  # clip grad norm
                    'teacher_forcing_ratio':1, # during training
                    'OUTPUT_DIM':1,            # intented output dimension
                    'ENC_EMB_DIM':21,          # embedding space of your input
                    'ENC_HID_DIM':32,
                    'DEC_HID_DIM':32,          # hidden dimension should be the same
                    'ENC_DROPOUT':0,
                    'DEC_DROPOUT':0,
                    'encode_len': 10,
                    'pred_len': 1,
                    'batch_size':1,
                    'teacher_forcing_ratio': 1,  # teacher forcing during training
                    'device':device}
            #####
            Training:
            seq2seq_utility.run_epoch(self, X_train,
                            y_train, X_test, y_test, teacher_forcing_ratio)
            #####
            Evaluation:
            seq2seq_utility.seq2seq_evaluate(test_seg)

            #####
            Prediction:
            self.model(seq2seq_input, target, teacher_forcing_ratio = 0)

            Where:
                seq2seq_input = [seq_len, batch size,Enc_emb_dim]
                target = [trg_len, batch size,output_dim], trg_len is prediction len

            ''')

    def __init__(self, **kwargs):
        try:
            self.grid = {'max_epochs': kwargs['max_epochs'],
                         'learning_rate': kwargs['learning_rate'],
                         # during training
                         'clip': kwargs['clip'],
                         'teacher_forcing_ratio': kwargs['teacher_forcing_ratio'],
                         'encode_len': kwargs['encode_len'],
                         'pred_len': kwargs['pred_len'],
                         'batch_size': kwargs['batch_size'],
                         'teacher_forcing_ratio': kwargs['teacher_forcing_ratio'],
                         }
            OUTPUT_DIM = kwargs['OUTPUT_DIM']
            ENC_EMB_DIM = kwargs['ENC_EMB_DIM']
            # DEC_EMB_DIM = 1
            ENC_HID_DIM = kwargs['ENC_HID_DIM']
            DEC_HID_DIM = kwargs['DEC_HID_DIM']
            ENC_DROPOUT = kwargs['ENC_DROPOUT']
            DEC_DROPOUT = kwargs['DEC_DROPOUT']
            self.device = kwargs['device']
        except:
            print('''hyper-parameter setting fails, 
            model uses default settings''')
            self.default_model_setting

        '''
        Instatiate Model
        '''
        attn = _Attention(ENC_HID_DIM, DEC_HID_DIM)
        enc = _Encoder(ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
        dec = _Decoder(output_dim=OUTPUT_DIM,  enc_hid_dim=ENC_HID_DIM,
                       dec_hid_dim=DEC_HID_DIM, dropout=DEC_DROPOUT, attention=attn)
        self.model = _Seq2Seq(enc, dec, self.device).to(self.device)

        '''
        Loss function & optimiser 
        '''
        self.optimiser = optim.Adam(
            self.model.parameters(), lr=self.grid['learning_rate'])
        self.lossfunction = nn.MSELoss().to(self.device)

    @property
    def default_model_setting(self):
        self.grid = {'max_epochs': 1024,
                     'learning_rate': 0.9,
                     'clip': 1,
                     'OUTPUT_DIM': 1,
                     'ENC_EMB_DIM': 23,           # dim that input to encoder  == number of your feature!
                     'ENC_HID_DIM': 24,
                     'DEC_HID_DIM': 24,
                     'ENC_DROPOUT': 0,
                     'DEC_DROPOUT': 0,
                     'encode_len': 1,           # how many days we want the ecoder encode
                     'pred_len': 1,              # how many days we hope to predict
                     'batch_size': 10,         # your batch size is constrained by the chunk of seq length
                     'teacher_forcing_ratio': 1,
                     'device': 'cuda'}

    def seq2seq_training(self, training_Loader, clip, teacher_forcing_ratio, batch_size):
        self.model.train()
        epoch_loss = 0

        # print('load')
        # print(X_local.size())
        # print(y_local.size())
        for local_batch, local_labels in training_Loader.batcher(batch_size):

            local_batch, local_labels = local_batch.transpose(0, 1).to(
                self.device), local_labels.transpose(0, 1).to(self.device)
            local_labels = local_labels.unsqueeze(2)
            # print('input')
            # print(local_batch.size())
            # print(local_labels.size())

            # forward pass
            self.optimiser.zero_grad()

            local_output = self.model(seq2seq_input=local_batch, target=local_labels,
                                      teacher_forcing_ratio=teacher_forcing_ratio)
            local_output = local_output.squeeze(2)[1:]
            local_labels = local_labels.squeeze(2)[1:]

            # print('output:')
            # print(local_output.size())
            # print(local_labels.size())
            loss = self.lossfunction(local_output, local_labels)

            # backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.optimiser.step()
            epoch_loss += loss.item()
        return epoch_loss

    def seq2seq_evaluate(self, test_Loader, batch_size):
        self.model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for local_batch, local_labels in test_Loader.batcher(batch_size):

                local_batch, local_labels = local_batch.transpose(0, 1).to(
                    self.device), local_labels.transpose(0, 1).to(self.device)
                local_labels = local_labels.unsqueeze(2)

                local_output = self.model(seq2seq_input=local_batch,
                                          target=local_labels, teacher_forcing_ratio=0)
                local_output = local_output.squeeze(2)[1:]
                local_labels = local_labels.squeeze(2)[1:]
                loss = self.lossfunction(local_output, local_labels)
                epoch_loss += loss.item()
        return epoch_loss

    def run_epoch(self, X_train, y_train, X_test, y_test):
        '''
        Args:
            X_train: [N_samples,N-features] -> Tensor
            X_test: [N_samples,N-features] -> Tensor
            y_train: [N_samples]  labels -> Tensor
            y_test: [N_samples]  labels -> Tensor

        '''

        best_valid_loss = float('inf')

        '''
        TODO: make dataloader compatible
        '''
        training_Loader = seq2seq_format_input(X_train, y_train)
        training_Loader.walk_forward_split(
            self.grid['encode_len'], self.grid['pred_len'])

        test_Loader = seq2seq_format_input(X_test, y_test)
        test_Loader.walk_forward_split(
            self.grid['encode_len'], self.grid['pred_len'])

        for epoch in range(self.grid['max_epochs']):

            train_loss = self.seq2seq_training(training_Loader,
                                               self.grid['clip'], self.grid['teacher_forcing_ratio'], self.grid['batch_size'])
            valid_loss = self.seq2seq_evaluate(test_Loader,
                                               self.grid['batch_size'])

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                self.m = copy.deepcopy(self.model)
                print(f'Epoch: {epoch+1}:')
                print(f'Train Loss: {train_loss:.3f}')
                print(f'Validation Loss: {valid_loss:.3f}')

        return best_valid_loss

    def save_model(self, path):
        '''
        Args:
            path: root path
        '''
        try:
            checkpoint = {'model_state_dict': self.m.state_dict(),
                          'optimizer_state_dict': self.optimiser.state_dict(),
                          }
            torch.save(checkpoint, os.path.join(path, 'seq2seq.pt'))

        except NameError:
            checkpoint = {'model_state_dict': self.model.state_dict(),
                          'optimizer_state_dict': self.optimiser.state_dict(),
                          }
            torch.save(checkpoint, os.path.join(path, 'seq2seq.pt'))

            print('have not trained model yet')

    def load_model(self, path):
        '''
        Args:
            path: root path
        '''
        try:
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
            self.model.eval()

        except:
            print('cannot load')


class _Encoder(nn.Module):
    def __init__(self, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.rnn = nn.GRU(input_size=emb_dim,
                          hidden_size=enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_input):

        # enc_input = [enc_input_len, batch size,emb_dim]

        # embedded = [enc_input_len, batch size, emb_dim]
        embedded = self.dropout(enc_input)

        outputs, hidden = self.rnn(embedded)

        # outputs = [enc_input len, batch size, hid dim * num directions]
        # hidden = [n layers * num directions, batch size, hid dim]

        # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer

        # hidden [-2, :, : ] is the last of the forwards RNN
        # hidden [-1, :, : ] is the last of the backwards RNN

        # initial decoder hidden is final hidden state of the forwards and backwards
        #  encoder RNNs fed through a linear layer
        hidden = torch.tanh(
            self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        # outputs = [src len, batch size, enc hid dim * 2]
        # hidden = [batch size, dec hid dim]
        return outputs, hidden


class _Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):

        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [enc_seq_len, batch size, enc hid dim * 2]

        enc_seq_len = encoder_outputs.shape[0]

        # repeat decoder hidden state enc_seq_len times
        hidden = hidden.unsqueeze(1).repeat(1, enc_seq_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # hidden = [batch size, enc_seq_len, dec hid dim]
        # encoder_outputs = [batch size, enc_seq_len, enc hid dim * 2]

        energy = torch.tanh(
            self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # energy = [batch size, enc_seq_len, dec hid dim]

        # attention= [batch size, enc_seq_len]
        attention = self.v(energy).squeeze(2)

        return F.softmax(attention, dim=1)


class _Decoder(nn.Module):
    def __init__(self, output_dim, enc_hid_dim,  dec_hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention

        self.rnn = nn.GRU((enc_hid_dim * 2) + output_dim, dec_hid_dim)

        self.fc_out = nn.Linear(
            (enc_hid_dim * 2) + dec_hid_dim + output_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, dec_input, hidden, encoder_outputs):

        # dec_input = [1,batch size,dec_emb dim]
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [enc_seq_len, batch size, enc hid dim * 2]

        # embedded = [1, batch size, dec_emb dim]
        embedded = self.dropout(dec_input)

        # attention = [batch size, enc_seq_len]
        attention = self.attention(hidden, encoder_outputs)

        # attention = [batch size, 1, enc_seq_len]
        attention = attention.unsqueeze(1)

        # encoder_outputs = [batch size, enc_seq_len, enc hid dim * 2]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # weighted = [batch size, 1, enc hid dim * 2]
        weighted = torch.bmm(attention, encoder_outputs)

        # weighted = [1, batch size, enc hid dim * 2]
        weighted = weighted.permute(1, 0, 2)

        # print('embedded',embedded.size())
        # rnn_input = [1, batch size, (enc hid dim * 2) + dec_emb dim]
        rnn_input = torch.cat((embedded, weighted), dim=2)

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        # output = [seq len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]

        # seq len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden
        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(
            torch.cat((output, weighted, embedded), dim=1))

        # prediction = [batch size, output dim]

        return prediction, hidden.squeeze(0)


class _Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, seq2seq_input, target, teacher_forcing_ratio: int = 0.5):
        """
        Args:
            seq2seq_input: '[seq_len, batch size,Enc_emb_dim]' -> torch.Tensor
            target: ' [trg_len, batch size,output_dim]'   -> torch.Tensor
            teacher_forcing_ratio:  1 -> full teacher
        Returns:
              [trg_len, batch_size, dec_output_size] -> torch.Tensor
        """
        # if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

        batch_size = seq2seq_input.shape[1]
        trg_len = target.shape[0]
        dec_output_size = self.decoder.output_dim
        dec_output = torch.zeros(
            trg_len, batch_size, dec_output_size).to(self.device)

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(seq2seq_input)

        # check: make dimension consistent
        dec_input = target[0]
        dec_input = dec_input.unsqueeze(0)
        # print('dec_input dim:',dec_input.size())

        for t in range(1, trg_len):
            # insert dec_input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            # print('in dec')
            # print(dec_input.size())

            output, hidden = self.decoder(dec_input, hidden, encoder_outputs)

            # place predictions in a tensor holding predictions for each token
            dec_output[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)
            top1 = top1.unsqueeze(1)

            # teacher forcing: MUST check consistency
            dec_input = target[t] if teacher_force else top1
            dec_input = dec_input.unsqueeze(0).float()
            # print(dec_input.size())
        return dec_output
