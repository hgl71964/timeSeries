from tslearn.clustering import TimeSeriesKMeans

def ts_k_means(data, **kwargs):
    n_cluster, epochs, metric = kwargs.get("n_cluster", 7), kwargs.get("epochs", 128), \
                            kwargs.get("metric", "softdtw")

    km = TimeSeriesKMeans(n_clusters=n_cluster,max_iter = epochs, metric=metric)
    return km.fit_predict(data)  