import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import os


class DNN_utility:
    def __init__(self, X_train, y_train, X_test, y_test, **kwargs):
        '''
        Args:
            X_train, y_train, X_test, y_test -> np.darray

            kwargs -> dict of all parameters
        '''

        # set all parameters
        try:
            self.other_param = {'max_epochs': kwargs['max_epochs'],
                                'learning_rate': kwargs['learning_rate'],
                                'batch_size': kwargs['batch_size'],
                                'device': kwargs['device']
                                }

            self.model_param = {'input_dim': kwargs['input_dim'],
                                'first_hidden': kwargs['first_hidden'],
                                'second_hidden': kwargs['second_hidden'],
                                }
        except:
            print('''hyper-parameter setting fails, 
            model uses default settings''')
            self.default_model_setting

        # instantiate
        self.model = DNN(self.model_param['input_dim'],
                         self.model_param['first_hidden'], self.model_param['second_hidden'])

        self.X_train = torch.from_numpy(X_train).float()
        self.y_train = torch.from_numpy(y_train).float()
        self.X_test = torch.from_numpy(X_test).float()
        self.y_test = torch.from_numpy(y_test).float()

        self.optimiser = optim.Adam(
            self.model.parameters(), lr=self.other_param['learning_rate'])

        self.lossfunction = nn.MSELoss().to(self.other_param['device'])

    @property
    def default_model_setting(self):
        self.other_param = {'max_epochs': 256,
                            'learning_rate': 1e-3,
                            'batch_size': 8,
                            'device': 'cuda',
                            }
        self.model_param = {'input_dim': 10,
                            'first_hidden': 128,
                            'second_hidden': 32,
                            }

    def run_epoch(self, X_train, y_train, X_test, y_test):
        '''
        Args:
            X_train: [N_samples,input_dim];  -> Tensor
            y_train: [N_samples,];  -> Tensor

            X_test: [N_samples,input_dim];  -> Tensor
            y_test: [N_samples,];  -> Tensor
        '''
        best_valid_loss = float('inf')

        for epoch in range(self.other_param['max_epochs']):

            train_loss = self.training(X_train, y_train)
            valid_loss = self.evaluation(X_test, y_test)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                self.best_model = copy.deepcopy(self.model)
            print(f'Epoch: {epoch+1}:')
            print(f'Train Loss: {train_loss:.3f}')
            print(f'Validation Loss: {valid_loss:.3f}')

        return best_valid_loss

    def prediction(self, x):
        try:
            pred = self.best_model(x)
        except:
            raise NameError('have not trained model')

        return pred

    def training(self, X_train, y_train):
        '''
        Args:
            X_train: [N_samples,input_dim];  -> Tensor
            y_train: [N_samples,];  -> Tensor

            output from _generate_batches:
                local_batch:  [batch_size, input_dim] -> Tensor
                local_labels: [batch_size,];  -> Tensor
        '''
        self.model.train()
        epoch_loss = 0

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
            X_test: [estmples,input_dim];  -> Tensor
            y_test: [N_samples,];  -> Tensor

            output from _generate_batches:
                local_batch:  [batch_size, input_dim] -> Tensor
                local_labels: [batch_size,];  -> Tensor
        '''
        self.model.eval()
        epoch_loss = 0

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
        self.layers = nn.Sequential(
            nn.Linear(input_dim, first_hidden),
            nn.Sigmoid(),
            nn.Linear(first_hidden, second_hidden),
            nn.Sigmoid(),
            nn.Linear(second_hidden, 1),
        )

    def forward(self, x):
        '''
        Args:
            x -> [N_samples,input_dim]; input_dim = meta feature

        Outputs:
            x -> [N_samples,]
        '''
        return self.layers(x).view(-1)