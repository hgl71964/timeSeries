import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import os
import pandas as pd
import numpy as np


class DNN_utility:
    def __init__(self, input_dim, **kwargs):
        '''
        Args:
            X_train, y_train, X_test, y_test -> np.darray

            kwargs -> dict of all parameters
        '''

        # set all parameters

        self.dim = input_dim

        try:
            self.other_param = {'max_epochs': kwargs['max_epochs'],
                                'learning_rate': kwargs['learning_rate'],
                                'batch_size': kwargs['batch_size'],
                                'device': kwargs['device']
                                }

            self.model_param = {'input_dim': input_dim,
                                'first_hidden': kwargs['first_hidden'],
                                'second_hidden': kwargs['second_hidden'],
                                }
        except:
            print('''hyper-parameter setting fails,
            model uses default settings''')
            self.default_model_setting

        # instantiate
        self.model = DNN(self.model_param['input_dim'], self.model_param['first_hidden'],
                         self.model_param['second_hidden']).to(self.other_param['device'])

        self.optimiser = optim.Adam(
            self.model.parameters(), lr=self.other_param['learning_rate'])

        self.lossfunction = nn.MSELoss().to(self.other_param['device'])

    @property
    def default_model_setting(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.other_param = {'max_epochs': 256,
                            'learning_rate': 1e-3,
                            'batch_size': 8,
                            'device': device,
                            }
        self.model_param = {'input_dim': self.dim,
                            'first_hidden': 128,
                            'second_hidden': 32,
                            }

    def run_epoch(self, X_train, y_train, X_test, y_test):
        '''
        Args:
            X_train: [N_samples,input_dim];  -> Tensor or np
            y_train: [N_samples,];  -> Tensor or np

            X_test: [N_samples,input_dim];  -> Tensor or np
            y_test: [N_samples,];  -> Tensor or np
        '''
        best_valid_loss = float('inf')

        for epoch in range(self.other_param['max_epochs']):

            train_loss = self.training(X_train, y_train)
            valid_loss = self.evaluation(X_test, y_test)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                self.best_model = copy.deepcopy(self.model.cpu())
                print(f'Epoch: {epoch+1}:')
                print(f'Train Loss: {train_loss:.3f}')
                print(f'Validation Loss: {valid_loss:.3f}')

        return best_valid_loss

    def prediction(self, x):

        if type(x) is np.ndarray:
            x = torch.from_numpy(x).float()

        elif type(x) is pd.DataFrame:
            x = torch.from_numpy(x.values).float()

        try:
            pred = self.best_model(x).detach()
        except:
            raise NameError('have not trained model')

        return pred

    def training(self, X_train, y_train):
        '''
        Args:
            X_train: [N_samples,input_dim];  -> Tensor or np or pd
            y_train: [N_samples,];  -> Tensor or np or pd

            output from _generate_batches:
                local_batch:  [batch_size, input_dim] -> Tensor
                local_labels: [batch_size,];  -> Tensor
        '''
        self.model.train()
        epoch_loss = 0

        if type(X_train) is np.ndarray:
            X_train = torch.from_numpy(X_train).float()
            y_train = torch.from_numpy(y_train).float()

        elif type(X_train) is pd.DataFrame:
            X_train = torch.from_numpy(X_train.values).float()
            y_train = torch.from_numpy(y_train.values).float()

        for local_batch, local_labels in self.batcher(X_train, y_train, self.other_param['batch_size']):

            local_batch, local_labels = local_batch.to(
                self.other_param['device']), local_labels.flatten().to(self.other_param['device'])

            # print('input are:')
            # print(local_batch.size())
            # print(local_labels.size())

            self.optimiser.zero_grad()
            local_output = self.model(local_batch)

            # print('output are:')
            # print(local_output.size())
            # print(local_labels.size())

            loss = self.lossfunction(local_output, local_labels)
            loss.backward()
            self.optimiser.step()
            epoch_loss += loss.item()
        return epoch_loss

    def evaluation(self, X_test, y_test):
        '''
        Args:
            X_test: [estmples,input_dim];  -> Tensor or np
            y_test: [N_samples,];  -> Tensor or np

            output from _generate_batches:
                local_batch:  [batch_size, input_dim] -> Tensor
                local_labels: [batch_size,];  -> Tensor
        '''
        self.model.eval()
        epoch_loss = 0

        if type(X_test) is np.ndarray:
            X_test = torch.from_numpy(X_test).float()
            y_test = torch.from_numpy(y_test).float()

        elif type(X_test) is pd.DataFrame:
            X_test = torch.from_numpy(X_test.values).float()
            y_test = torch.from_numpy(y_test.values).float()

        for local_batch, local_labels in self.batcher(X_test, y_test, self.other_param['batch_size']):

            local_batch, local_labels = local_batch.to(
                self.other_param['device']), local_labels.flatten().to(self.other_param['device'])

            local_output = self.model(local_batch)

            loss = self.lossfunction(local_output, local_labels)
            epoch_loss += loss.item()
        return epoch_loss

    def batcher(self, x, y, batch_size):
        l = len(y)
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
            torch.save(checkpoint, os.path.join(path, 'timeSeries_DNN.pt'))

        except NameError:
            checkpoint = {'model_state_dict': self.model.state_dict(),
                          'optimizer_state_dict': self.optimiser.state_dict(),
                          }
            torch.save(checkpoint, os.path.join(path, 'timeSeries_DNN.pt'))

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


class DNN(nn.Module):

    def __init__(self, input_dim, first_hidden, second_hidden):
        super(DNN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, first_hidden),
            nn.Sigmoid(),
            nn.Linear(first_hidden, second_hidden),
            nn.Sigmoid(),
            nn.Linear(second_hidden, 1),
        )

        # self.fc1 = nn.Linear(input_dim, first_hidden)
        # self.a1 = nn.Sigmoid()
        # self.fc2 = nn.Linear(first_hidden, second_hidden)
        # self.a2 = nn.Sigmoid()
        # self.fc3 = nn.Linear(second_hidden, 1)

    def forward(self, x):
        '''
        Args:
            x -> [N_samples,input_dim];

        Outputs:
            x -> [N_samples,]
        '''
        return self.layers(x).view(-1)


# if __name__ == "__main__":
#     x = torch.rand(50, 10)
#     y = torch.ones(50,)
#     input_dim = 10
#     dnn = DNN_utility(input_dim)
#     dnn.run_epoch(x, y, x, y)
#     dnn.prediction(x)
