
import torch

'''
to format the dataset such that seq2seq can run
'''


class seq2seq_format_input():
    def __init__(self, X, y):
        """
        Args:
            torch (Tensor): Dataset stored in Tensor
            X: [N_samples,N-features] -> Tensor
            y: [N_samples]  labels -> Tensor
        """
        self._X = X  # feature
        self._y = y  # price

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
        self.full_data = torch.cat([self._X, self._y.unsqueeze(1)], dim=1)

        N_samples = self._X.shape[0]
        for i in range(0, N_samples, pred_len):
            encode = self.full_data[i:min(
                i+encode_len, N_samples)]

            pred = self._y[i+encode_len:min(
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
        print('maximum batch_size:', self.X.shape[0])

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

          output -> class attributes:
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
        if shuffle:
            self.shuffler()

    def shuffler(self):
        n = self._segment.size(0)
        rand = torch.randperm(n)
        self._segment = self._segment[rand]

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
