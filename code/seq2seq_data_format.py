
import torch
import copy
import numpy as np
import pandas as pd

'''
to format the dataset such that seq2seq can run
'''


class seq2seq_dataset:

    def __init__(self, X_train, y_train, X_test, y_test):
        """
        Args:
            X_train: [N_samples,N-features] -> np.ndarray
            X_test: [N_samples,N-features] -> np.ndarray

            y_train: [N_samples]  labels -> pd.DataFrame
            y_test: [N_samples]  labels -> pd.DataFrame
        """

        N_sample = X_train.shape[0]
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

        e.g. encode_len = 3, pred_len = 2 --->
                        x = (x1,y1),...,(x3,y3); y = (y3,y4,y5) <- y[0] = X_train[-1][1]...

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
        print(N_samples)
        print(pred_len)
        for i in range(0, N_samples, pred_len):

            encode = full_data[i:min(
                i+encode_len, N_samples)]

            pred = y[i+encode_len-1:min(
                i+encode_len+pred_len, N_samples)]

            # prevent from empty list
            if pred.shape[0] != pred_len+1:
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

    def shuffler(self, tensor):
        n = tensor.size(0)
        rand = torch.randperm(n)
        tensor = tensor[rand]


# if __name__ == "__main__":
#     x = np.random.rand(100, 5)
#     y = pd.DataFrame(np.random.rand(100,))

#     a = seq2seq_dataset(x, y, x, y)
#     a.split_dataset(encode_len=3, pred_len=2)
#     print(a.X_train.size())
#     print(a.y_train.size())
