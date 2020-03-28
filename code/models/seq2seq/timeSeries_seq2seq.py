import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import pandas as pd
import torch.optim as optim
import random
import os
from sklearn.model_selection import ParameterGrid


class seq2seq_utility:

    def __init__(self, input_dim, **kwargs):
        self.input_dim = input_dim
        try:
            self.grid = {'max_epochs': kwargs['max_epochs'],
                         'learning_rate': kwargs['learning_rate'],
                         # during training
                         'clip': kwargs['clip'],
                         'batch_size': kwargs['batch_size'],
                         'teacher_forcing_ratio': kwargs['teacher_forcing_ratio'],

                         'OUTPUT_DIM': kwargs['OUTPUT_DIM'],
                         'ENC_EMB_DIM': input_dim,
                         # DEC_EMB_DIM: 1,
                         'ENC_HID_DIM': kwargs['ENC_HID_DIM'],
                         'DEC_HID_DIM': kwargs['DEC_HID_DIM'],
                         'ENC_DROPOUT': kwargs['ENC_DROPOUT'],
                         'DEC_DROPOUT': kwargs['DEC_DROPOUT'],
                         'device': kwargs['device']},
        except:
            print('''hyper-parameter setting fails, 
            model uses default settings''')
            self.default_model_setting

        '''
        Instatiate Model
        '''
        attn = _Attention(self.grid['ENC_HID_DIM'], self.grid['DEC_HID_DIM'])
        enc = _Encoder(
            self.grid['ENC_EMB_DIM'], self.grid['ENC_HID_DIM'], self.grid['DEC_HID_DIM'], self.grid['ENC_DROPOUT'])
        dec = _Decoder(output_dim=self.grid['OUTPUT_DIM'],  enc_hid_dim=self.grid['ENC_HID_DIM'],
                       dec_hid_dim=self.grid['DEC_HID_DIM'], dropout=self.grid['DEC_DROPOUT'], attention=attn)
        self.model = _Seq2Seq(enc, dec, self.grid['device']).to(
            self.grid['device'])

        '''
        Loss function & optimiser 
        '''
        self.optimiser = optim.Adam(
            self.model.parameters(), lr=self.grid['learning_rate'])
        self.lossfunction = nn.MSELoss().to(self.grid['device'])

    @property
    def default_model_setting(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.grid = {'max_epochs': 256,
                     'learning_rate': 1e-3,
                     'clip': 1,
                     'OUTPUT_DIM': 1,

                     # dim that input to encoder  == number of your feature!
                     'ENC_EMB_DIM': self.input_dim,
                     'ENC_HID_DIM': 16,
                     'DEC_HID_DIM': 16,
                     'ENC_DROPOUT': 0,
                     'DEC_DROPOUT': 0,
                     'batch_size': 10,         # your batch size is constrained by the chunk of seq length
                     'teacher_forcing_ratio': 1,
                     'device': device}

    def grid_search(self, X_train, y_train, X_test, y_test, search):
        '''
        Args:
            X_train: [N_samples,input_dim];  -> Tensor 
            y_train: [N_samples,];  -> Tensor 

            X_test: [N_samples,input_dim];  -> Tensor 
            y_test: [N_samples,];  -> Tensor 

            search -> boolean
        '''
        if search:
            param_grid = {'batch_size': [8, 32],
                          'max_epochs': [128, 1024],
                          'learning_rate': [1e-3, 1e-1, 0.3, 0.7],
                          'ENC_HID_DIM': [16, 32],
                          'DEC_HID_DIM': [8, 16],
                          'DEC_DROPOUT': [0, 0.5],
                          }
            best_loss = float('inf')
            best_grid = {}

            for one_search in list(ParameterGrid(param_grid)):
                for key in one_search:
                    if key in self.grid:
                        self.grid[key] = one_search[key]

                self.optimiser = optim.Adam(
                    self.model.parameters(), lr=self.grid['learning_rate'])

                loss = self.run_epoch(
                    X_train, y_train, X_test, y_test, verbo=False)

                if loss < best_loss:
                    best_loss = loss
                    best_grid = one_search

            for key in best_grid:
                if key in self.grid:
                    self.grid[key] = best_grid[key]

    def seq2seq_training(self, X_train, y_train):
        '''
        Args:
            X_train: [N_sample, encode_len, N_feature] -> Tensor; N_feature excludes price
            y_train: [N_sample, pred_len] -> Tensor;
        '''
        self.model.train()
        epoch_loss = 0

        for local_batch, local_labels in seq2seq_utility.batcher(X_train, y_train, self.grid['batch_size']):
            '''
            local_batch: [batch_size, encode_len, N_feature] -> Tensor; N_feature excludes price
            local_labels: [batch_size, pred_len] -> Tensor;
            '''

            local_batch, local_labels = local_batch.transpose(0, 1).to(
                self.grid['device']), local_labels.transpose(0, 1).to(self.grid['device'])
            local_labels = local_labels.unsqueeze(2)

            '''
            After to device:
                local_batch:  [encode_len, batch_size, N_feature] -> Tensor; N_feature excludes price
                local_labels:  [pred_len, batch_size,1] -> Tensor;
            '''

            # print('input')
            # print(local_batch.size())
            # print(local_labels.size())

            # forward pass
            self.optimiser.zero_grad()

            local_output = self.model(seq2seq_input=local_batch, target=local_labels,
                                      teacher_forcing_ratio=self.grid['teacher_forcing_ratio'])
            '''
            resolve the first dec_input / last enc_input issues
            '''

            local_output = local_output.squeeze(2)[1:]
            local_labels = local_labels.squeeze(2)[1:]

            # print('output:')
            # print(local_output.size())
            # print(local_labels.size())
            loss = self.lossfunction(local_output, local_labels)

            # backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grid['clip'])
            self.optimiser.step()
            epoch_loss += loss.item()
        return epoch_loss

    def seq2seq_evaluate(self, X_test, y_test):
        '''
        see function: seq2seq_training
        '''
        self.model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for local_batch, local_labels in seq2seq_utility.batcher(X_test, y_test, self.grid['batch_size']):

                local_batch, local_labels = local_batch.transpose(0, 1).to(
                    self.grid['device']), local_labels.transpose(0, 1).to(self.grid['device'])
                local_labels = local_labels.unsqueeze(2)

                local_output = self.model(seq2seq_input=local_batch,
                                          target=local_labels, teacher_forcing_ratio=0)
                local_output = local_output.squeeze(2)[1:]
                local_labels = local_labels.squeeze(2)[1:]
                loss = self.lossfunction(local_output, local_labels)
                epoch_loss += loss.item()
        return epoch_loss

    def run_epoch(self, X_train, y_train, X_test, y_test, verbo=True):
        '''
        Args:
            X_train, X_test: [N_sample, encode_len, N_feature] -> Tensor
            y_train, y_test: [N_sample, pred_len] -> Tensor;

        '''

        best_valid_loss = float('inf')

        for epoch in range(self.grid['max_epochs']):

            train_loss = self.seq2seq_training(X_train, y_train)
            valid_loss = self.seq2seq_evaluate(X_test, y_test)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                self.best_model = copy.deepcopy(self.model).cpu()
                if verbo:
                    print(f'Epoch: {epoch+1}:')
                    print(f'Train Loss: {train_loss:.3f}')
                    print(f'Validation Loss: {valid_loss:.3f}')

        return best_valid_loss

    def seq2seq_prediction(self, x, y):
        '''
        this function uses trained model to make prediction!

        for larger sequence of time series, breaks down to bag of chunks!

        Args:
            x: [N_sample, encode_len, N_feature] -> Tensor; N_feature excludes price
            y: [N_sample, pred_len] -> Tensor;

            enc_seq_len and pred_len should be the same as the data for training

        Returns:
              [pred_len, dec_output_size] -> torch.Tensor
        '''
        for i, (local_batch, local_labels) in enumerate(seq2seq_utility.batcher(x, y, batch_size=1)):

            '''
            pred -> Tensor [pred_len, dec_output_size]; first dec_input/ last enc_input problem
            '''

            local_batch, local_labels = local_batch.transpose(
                0, 1), local_labels.transpose(0, 1).unsqueeze(2)

            local_output = self.best_model(seq2seq_input=local_batch,
                                           target=local_labels, teacher_forcing_ratio=0)
            pred = local_output[1:].squeeze(1)

            if i == 0:
                predictions = pred
            else:
                predictions = torch.cat([predictions, pred], dim=0)

        return predictions.detach().cpu().numpy().ravel()

    @staticmethod
    def batcher(x, y, batch_size: int):
        '''
        make batch along first dimension

        Args:
            x: iterable
            y: iterable

        Return:
            x  [batch_size, encode_len, N_feature]
            y  [batch_size, encode_len+pred_len]
        '''

        l = len(x)
        for batch in range(0, l, batch_size):
            yield (x[batch:min(batch + batch_size, l)], y[batch:min(batch + batch_size, l)])

    def save_model(self, path):
        '''
        Args:
            path: root path
        '''
        try:
            checkpoint = {'model_state_dict': self.best_model.state_dict(),
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
                    'batch_size':1,
                    'teacher_forcing_ratio': 1,  # teacher forcing during training
                    'device':device}

            #####
            Training:
            run_epoch(X_train,y_train, X_test, y_test)

            #####
            Evaluation:
            seq2seq_evaluate(X_test, y_test)

            #####
            Prediction:
            seq2seq_prediction(x,y)
            ''')


class _Encoder(nn.Module):
    def __init__(self, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.rnn = nn.GRU(input_size=emb_dim,
                          hidden_size=enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_input):
        '''
        Args: 
            enc_input -> Tensor: [enc_input_len, batch size,emb_dim]

        Returns:
            outputs -> Tensor: [enc_seq_len, batch size, enc hid dim * 2]
            hidden -> Tensor: [batch size, dec hid dim]

        '''

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

        # outputs = [enc_seq_len, batch size, enc hid dim * 2]
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
        '''
        Args:
            dec_input -> Tensor: [1,batch size,dec_emb dim]
            hidden -> Tensor: [batch size, dec hid dim]
            encoder_outputs -> Tensor: [enc_seq_len, batch size, enc hid dim * 2]
        '''

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

        # embedded = [1, batch size, dec_emb dim]

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
        '''
        this function for time series forecasting

        Args:
            seq2seq_input: '[enc_seq_len, batch size,Enc_emb_dim]' -> torch.Tensor
            target: ' [1 + pred_len, batch size,output_dim]'  -> torch.Tensor
            teacher_forcing_ratio:  1 -> full teacher
        Returns:
              [pred_len, batch_size, dec_output_size] -> torch.Tensor
        '''
        # if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

        batch_size = seq2seq_input.shape[1]

        dec_output_size = self.decoder.output_dim

        '''
        the first input to decoder = last y input to encoder

        decoutputs[0] = all zeros, this will be resolved in loss function
        '''

        pred_len = target.shape[0] - 1
        dec_outputs = torch.zeros(
            pred_len + 1, batch_size, dec_output_size).to(self.device)

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(seq2seq_input)

        # dec_input -> Tensor [1, batch size, output_dim]
        dec_input = target[0]
        dec_input = dec_input.unsqueeze(0)

        for t in range(1, pred_len + 1):

            output, hidden = self.decoder(dec_input, hidden, encoder_outputs)

            # place predictions in a tensor holding predictions for each token
            dec_outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)
            top1 = top1.unsqueeze(1)

            # teacher forcing: MUST check consistency
            dec_input = target[t] if teacher_force else top1
            dec_input = dec_input.unsqueeze(0).float()
            # print(dec_input.size())
        return dec_outputs
