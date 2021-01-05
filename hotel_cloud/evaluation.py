import numpy as np
from tslearn.metrics import dtw, dtw_path
from tslearn.barycenters import dtw_barycenter_averaging

class evaluator:

    def __init__(self, data, labels):
        
        self.groups = {}

        for i, r in enumerate(np.unique(labels)):
            key = str(r)
            self.groups[key] = data[labels==r]

        self.keys = list(self.groups.keys())
        self.n_cluster = len(self.keys)

    def inter_distance(self,
                n: int,
                label: str,  # correspond to a key
                metric: str="dtw", 
                ):

        
        
    def within_distance(self,
                        n: int,
                        label:str,
                        metric: str = "dtw",
                      ):

        d = self.groups[label]

        if 2*n > d.shape[0]:
            raise ValueError("not enough data points")
        
        idx = np.random.randint(low = 0, high=d.shape[0], size = (int(2*n),))

        pool1 = d[idx[:n]]; pool2 = d[idx[n:]]

        if metric == "dtw":
            distance = dtw_barycenter_averaging(...)
            

        return 





    

        