import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from tslearn.metrics import dtw
from numpy.linalg import norm
from pandas import Series
from tslearn.metrics import soft_dtw

class cluster_evaluator:
    """
    evaluation of clustering results 
    """
    def __init__(self, data, labels):
        
        self.groups = {}

        for r in np.unique(labels.astype(np.int16)):
            key = str(r)
            self.groups[key] = data[labels==r]

        self.keys = list(self.groups.keys())
        self.n_cluster = len(self.keys)

    def pred_dist(self, preds):
        _=plt.hist(preds)
        plt.show()

    def intra_inter_group(self, 
                        n: int,  # number of samples to test
                        labels: tuple,  # (label1, label2); notice label1 is the main comparison group
                        metric: str = "dtw",
                        ):
        l1, l2 = labels[0], labels[1]

        if not isinstance(l1, str):
            l1, l2 = str(l1), str(l2)

        d1, d2 = self.groups[l1], self.groups[l2]

        if 2 * n > d1.shape[0] or n > d2.shape[0]:
            raise ValueError("not enough data points")

        rng = default_rng()
        idx1, idx2 = rng.choice(d1.shape[0], size=int(2*n), replace=False), rng.choice(d2.shape[0], size=n, replace=False)

        sample, sample1, sample2 = d1[idx1[:n]], d1[idx1[n:]], d2[idx2]

        intra, inter = np.empty((n, ), dtype=np.float64), np.empty((n, ), dtype=np.float64)

        if metric == "dtw":
            for i in range(n):
                intra[i] = dtw(sample[i], sample1[i])
                inter[i] = dtw(sample[i], sample2[i])

        elif metric == "l2":
            for i in range(n):
                intra[i] = norm(sample[i]- sample1[i])
                inter[i] = norm(sample[i]- sample2[i])

        # return two array
        return intra, inter

    def visual(self,
                n: int,  # number of samples to test
                *args: list,  
                ):

        _, ax = plt.subplots()

        for l1 in args:
            if not isinstance(l1, str):
                l1 = str(l1)

            d1 = self.groups[l1]
            rng = default_rng()
            idx1 = rng.choice(d1.shape[0], size=n, replace=False)

            sample1 = d1[idx1]  # a set of time series 

            for s1 in sample1:
                ax.plot([i for i in range(len(s1))], s1) # label=f"cluster_{l1}")

        ax.set_xlabel("time series"); ax.set_ylabel("vals"); ax.set_title(f"time series plot")
        ax.legend()
        plt.show()
        return None

class forecast_metric:

    @staticmethod
    def softdtw(x, y):
        """softDTW the smaller the better"""
        if isinstance(x, Series):
            x = x.to_numpy()
        if isinstance(y, Series):
            y = y.to_numpy()

        assert type(x) is type(y)  # check typing

        #  softDTW must preserve sequential property 
        assert (x[0] > x[-1] and y[0] > y[-1]) or (x[0]< x[-1] and y[0] < y[-1]), \
                f"\n x: {x} \n y: {y}"

        return soft_dtw(x, y)

    @staticmethod
    def mse(x, y):
        """always positive, the smaller the better"""
        if isinstance(x, Series):
            x = x.to_numpy()
        if isinstance(y, Series):
            y = y.to_numpy()

        assert type(x) is type(y)
        return ((x-y)**2).mean()
    
    @staticmethod
    def rmse(x, y):
        return np.sqrt(forecast_metric.mse(x, y))