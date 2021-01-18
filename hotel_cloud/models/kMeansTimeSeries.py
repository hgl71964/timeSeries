from tslearn.clustering import TimeSeriesKMeans
from tslearn.barycenters import softdtw_barycenter
from tslearn.metrics import soft_dtw
import matplotlib.pyplot as plt

def ts_k_means(data, n_cluster, **kwargs):
    epochs, metric = kwargs.get("epochs", 128), \
                            kwargs.get("metric", "softdtw")

    km = TimeSeriesKMeans(n_clusters=n_cluster,max_iter = epochs, metric=metric)
    return km.fit_predict(data)  

def k_mean_model_selection(data, n_possible_cluster: int = 7, **kwargs):

    for num in range(2, n_possible_cluster):
        labels = ts_k_means(data, num, **kwargs)

    return None
