
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
        Args:
            see desirable split: https://github.com/guol1nag/datagrasp/blob/master/README.md

          input:
              X: [N_samples,N-features] Tensor N_samples have sequential properties
              y: [N_samples,]  labels Tensor    N_samples have sequential properties

          output -> class attributes:
              self.X  [N_sample, encode_len, N_feature] N_feature excludes price
              self.y  [N_sample, pred_len]
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

    def bag_of_timeSeries_chunk_for_prediction(self,
                                               encode_len: int,
                                               pred_len: int,
                                               shuffle=False):  # for sequential data we should not shuffle!
        '''
        Seq2seq prediction mode: given [t,t + encode_len] indicator data + price,
                                            predict [t ,t + encode_len + pred_len] price
        Args:
          input:
              X: [N_samples,N-features] Tensor N_samples have sequential properties
              y: [N_samples,]  labels Tensor    N_samples have sequential properties

        Returns
              self.X  [N_sample, encode_len, N_feature] N_feature excludes price
              self.y  [N_sample, pred_len]
        '''
        total_len = encode_len + pred_len
        N_samples = self._X.shape[0]

        for i in range(0, N_samples, total_len):
            X_segment = self._X[i:min(
                i+encode_len+pred_len, N_samples)]
            y_segment = self._y[i:min(
                i+encode_len+pred_len, N_samples)].unsqueeze(1)

            segment = torch.cat((X_segment, y_segment), dim=1)

            if i == 0:
                self.X = segment[:encode_len, :].unsqueeze(0)  # get everything
                self.y = segment[:, -1].unsqueeze(0)           # only get price
            else:
                try:
                    self.X = torch.cat(
                        (self.X, segment[:encode_len, :].unsqueeze(0)))
                    self.y = torch.cat(
                        (self.y, segment[:, -1].unsqueeze(0)))
                except:
                    pass
                    # self._residual = segment.unsqueeze(0)

    def shuffler(self, tensor):
        n = tensor.size(0)
        rand = torch.randperm(n)
        tensor = tensor[rand]

    def batcher(self, batch_size):
        '''
        Args:
          -> output results attributes:
              x  [batch_size, encode_len, N_feature]
              y  [batch_size, encode_len+pred_len]
        '''
        l = len(self.X)
        # l_res = len(self.X_res)
        # RNN has restriction for batch size
        for batch in range(0, l, batch_size):
            yield (self.X[batch:min(batch + batch_size, l)], self.y[batch:min(batch + batch_size, l)])
        # for batch in range(0, l_res, batch_size):
        #     yield (self.X_res[batch:min(batch + batch_size, l)], self.y_res[batch:min(batch + batch_size, l)])
