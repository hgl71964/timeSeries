
import torch
import copy

'''
to format the dataset such that seq2seq can run
'''


class seq2seq_dataset():
    def __init__(self, X_train, y_train, X_test, y_test):
        """
        Args:
            X_train: [N_samples,N-features] -> Tensor
            X_test: [N_samples,N-features] -> Tensor
            y_train: [N_samples]  labels -> Tensor
            y_test: [N_samples]  labels -> Tensor
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def walk_forward_split(self,
                           encode_len: int,
                           pred_len: int):
        '''
        See desirable split: https://github.com/guol1nag/datagrasp/blob/master/README.md

        For seq2seq this should be able to encode arbitrary sequence and prediction arbitrary sequence

        Args:
            X: [N_samples,N-features] -> Tensor N_samples have sequential properties
            y: [N_samples,]  -> labels Tensor    N_samples have sequential properties

        Returns:
              self.X  [N_sample, encode_len, N_feature] -> Tensor; N_feature excludes price
              self.y  [N_sample, pred_len] -> Tensor;
        '''

        # [N_samples,N-features + return]
        self.full_data = torch.cat(
            [self.X_train, self.y_train.unsqueeze(1)], dim=1)

        N_samples = self.X_train.shape[0]
        for i in range(0, N_samples, pred_len):
            encode = self.full_data[i:min(
                i+encode_len, N_samples)]

            pred = self.y_train[i+encode_len:min(
                i+encode_len+pred_len, N_samples)]
            if pred.shape[0] != pred_len:
                break

            if i == 0:
                self.X = encode.unsqueeze(0)
                self.y = pred.unsqueeze(0)
            else:
                try:
                    self.X = torch.cat([self.X, encode.unsqueeze(0)], dim=0)
                    self.y = torch.cat([self.y, pred.unsqueeze(0)], dim=0)
                    # print(encode.unsqueeze(0).size())
                    # print(pred.unsqueeze(0).size())
                except:  # residual
                    pass
        self.X_train = copy.deepcopy(self.X)
        self.y_train = copy.deepcopy(self.y)
        del self.X, self.y, self.full_data

        # [N_samples,N-features + return]
        self.full_data = torch.cat(
            [self.X_test, self.y_test.unsqueeze(1)], dim=1)

        N_samples = self.X_test.shape[0]
        for i in range(0, N_samples, pred_len):
            encode = self.full_data[i:min(
                i+encode_len, N_samples)]

            pred = self.y_test[i+encode_len:min(
                i+encode_len+pred_len, N_samples)]
            if pred.shape[0] != pred_len:
                break

            if i == 0:
                self.X = encode.unsqueeze(0)
                self.y = pred.unsqueeze(0)
            else:
                try:
                    self.X = torch.cat([self.X, encode.unsqueeze(0)], dim=0)
                    self.y = torch.cat([self.y, pred.unsqueeze(0)], dim=0)
                    # print(encode.unsqueeze(0).size())
                    # print(pred.unsqueeze(0).size())
                except:  # residual
                    pass

        self.X_test = copy.deepcopy(self.X)
        self.y_test = copy.deepcopy(self.y)
        del self.X, self.y, self.full_data

    def train_loader(self, batch_size):
        '''
        Args:
          -> output results attributes:
              x  [batch_size, encode_len, N_feature]
              y  [batch_size, encode_len+pred_len]
        '''
        l = len(self.X_train)
        # l_res = len(self.X_res)
        # RNN has restriction for batch size
        for batch in range(0, l, batch_size):
            yield (self.X_train[batch:min(batch + batch_size, l)], self.y_train[batch:min(batch + batch_size, l)])

    def test_loader(self, batch_size):
        '''
        Args:
          -> output results attributes:
              x  [batch_size, encode_len, N_feature]
              y  [batch_size, encode_len+pred_len]
        '''
        l = len(self.X_test)
        # l_res = len(self.X_res)
        # RNN has restriction for batch size
        for batch in range(0, l, batch_size):
            yield (self.X_test[batch:min(batch + batch_size, l)], self.y_test[batch:min(batch + batch_size, l)])

    def shuffler(self, tensor):
        n = tensor.size(0)
        rand = torch.randperm(n)
        tensor = tensor[rand]
