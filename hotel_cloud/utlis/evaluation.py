import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from tslearn.metrics import dtw, dtw_path
from tslearn.barycenters import dtw_barycenter_averaging
from numpy.linalg import norm

class evaluator:
    """
    evaluation of clustering results 
    """
    def __init__(self, data, labels):
        
        self.groups = {}

        for i, r in enumerate(np.unique(labels.astype(np.int16))):
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
    
    def acc(self, pred, labels):
        """provide accuracy of the clustering if labels are available"""
        
        pred, labels = pred.flatten().astype(np.int16), labels.flatten().astype(np.int16)

        diff = labels.min() - pred.min()

        labels -= diff

        # for i in range(len(np.unique(labels))):

        #     label = labels[i]

        #     group = (pred == label)  # boolean index  

        #     majority_group = labels[group]

        #     # compute the most frequent element
        #     (values,counts) = np.unique(majority_group,return_counts=True)
        #     ind=np.argmax(counts)
        #     most_freq = values[ind]

        #     rec[str(most_freq)] = group

        return (pred==labels).mean()

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

            for j, s1 in enumerate(sample1):
                ax.plot([i for i in range(len(s1))], s1, label=f"cluster_{l1}")

        ax.set_xlabel("time series"); ax.set_ylabel("vals"); ax.set_title(f"time series plot")
        ax.legend()
        plt.show()
        return None


    def barycenter(self):

        return None




    

        