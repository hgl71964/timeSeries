from tslearn.clustering import TimeSeriesKMeans

def ts_k_means(data, data_dict,**kwargs):
    n_cluster, epochs, metric = kwargs.get("n_cluster", 7), kwargs.get("epochs", 128), \
                            kwargs.get("metric", "softdtw")

    km = TimeSeriesKMeans(n_clusters=n_cluster,max_iter = epochs, metric=metric)

    preds = km.fit_predict(data)  
    for i, pred in enumerate(preds):  # assign label to each staydate
        data_dict[i][1] = pred
    return preds