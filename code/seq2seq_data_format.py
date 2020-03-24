
import torch
import copy

'''
to format the dataset such that seq2seq can run
'''


class seq2seq_dataset():
    def __init__(self, X_train, y_train, X_test, y_test):
        """
        Args:
            X_train: [N_samples,N-features] -> np.ndarray
            X_test: [N_samples,N-features] -> np.ndarray

            y_train: [N_samples]  labels -> pd.DataFrame
            y_test: [N_samples]  labels -> pd.DataFrame
        """

        N_sample = X_train.shape[1]
        self.X_train = torch.from_numpy(X_train).float()
        self.X_test = torch.from_numpy(X_test).float()

        self.y_train = torch.from_numpy(
            y_train.values.ravel()[:N_sample]).float()
        self.y_test = torch.from_numpy(
            y_test.values.ravel()[:N_sample]).float()

    def split_dataset(self,
                      encode_len: int,
                      pred_len: int,):
        '''
        Returns -> pd.DataFrame
            X_train: features, including return, [N_sample,N_feature] 
            y_train: return, [N_sample,] 
        '''

        self.X_train, self.y_train = self._walk_forward_split(
            self.X_train, self.y_train, encode_len, pred_len)

        self.X_test, self.y_test = self._walk_forward_split(
            self.X_test, self.y_test, encode_len, pred_len)

    def _walk_forward_split(self,
                            x,
                            y,
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
        full_data = torch.cat(
            [x, y.unsqueeze(1)], dim=1)

        N_samples = x.shape[0]
        for i in range(0, N_samples, pred_len):
            encode = full_data[i:min(
                i+encode_len, N_samples)]

            pred = y[i+encode_len:min(
                i+encode_len+pred_len, N_samples)]
            if pred.shape[0] != pred_len:
                break

            if i == 0:
                new_X = encode.unsqueeze(0)
                new_y = pred.unsqueeze(0)
            else:
                try:
                    new_X = torch.cat([new_X, encode.unsqueeze(0)], dim=0)
                    new_y = torch.cat([new_y, pred.unsqueeze(0)], dim=0)
                    # print(encode.unsqueeze(0).size())
                    # print(pred.unsqueeze(0).size())
                except:  # residual
                    pass
        return new_X, new_y

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
