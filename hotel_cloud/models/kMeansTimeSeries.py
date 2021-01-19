from tslearn.clustering import TimeSeriesKMeans
from tslearn.barycenters import softdtw_barycenter
from tslearn.metrics import soft_dtw
import matplotlib.pyplot as plt

def Kmeans_predict(data, n_cluster, **kwargs):
    epochs, metric = kwargs.get("epochs", 128), \
                            kwargs.get("metric", "softdtw")

    km = TimeSeriesKMeans(n_clusters=n_cluster,max_iter = epochs, metric=metric)
    return km, km.fit_predict(data)  # model, labels

def k_mean_model_selection(data, n_possible_cluster: int = 7, **kwargs):

    losses = []
    for num in range(2, n_possible_cluster+1):
        km, _ = Kmeans_predict(data, num, **kwargs)
        loss = km.inertia_
        losses.append(loss)

    indice = [i for i in range(2, n_possible_cluster+1)]

    _, ax = plt.subplots()

    ax.scatter(indice, losses, label="loss function")
    ax.set_xlabel("Num of clusters"); 
    ax.set_ylabel("loss function"); 
    # ax.set_title(f"time series plot")
    plt.show()
    return None
