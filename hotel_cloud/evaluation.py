import numpy as np
from numpy.random import default_rng
from tslearn.metrics import dtw, dtw_path
from tslearn.barycenters import dtw_barycenter_averaging
from numpy.linalg import norm

class evaluator:

    def __init__(self, data, labels):
        
        self.groups = {}

        for i, r in enumerate(np.unique(labels)):
            key = str(r)
            self.groups[key] = data[labels==r]

        self.keys = list(self.groups.keys())
        self.n_cluster = len(self.keys)

    def intra_inter_group(self, 
                        n: int,  # number of samples to test
                        labels: tuple,  # (label1, label2); notice label1 is the main comparison group
                        metric: str = "dtw",
                        ):
        l1, l2 = labels[0], labels[1]

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


    def within_distance(self,
                        n: int,
                        label:str,
                        metric: str = "dtw",
                      ):

        if isinstance(label, tuple):
            d = self.groups[label[0]]
        else:
            d = self.groups[label]

        if 2*n > d.shape[0]:
            raise ValueError("not enough data points")
        
        idx = np.random.randint(low=0, high=d.shape[0], size = (int(2*n),))

        pool1 = d[idx[:n]]; pool2 = d[idx[n:]]

        if metric == "dtw":
            distance = dtw_barycenter_averaging(...)
        

        return None

    def inter_distance(self,
                n: int,
                label: tuple,  # (label1, label2)
                metric: str="dtw", 
                ):
        
        return None




    

        