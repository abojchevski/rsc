import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import kneighbors_graph


def __local_purity(H, y, nn):
    """

    :param H: embedding to evaluate
    :param y: ground-truth classes
    :param nn: number of neighbours to consider
    """
    A = kneighbors_graph(H, nn + 1, include_self=True)
    neigbourhoods = A.dot(np.eye(y.max() + 1)[y])
    frequencies = neigbourhoods / neigbourhoods.sum(1)[:, None]
    purity = frequencies.max(axis=1)
    return purity.mean()


def local_purity(H, y, nn=None, num_samples=10):
    """

    :param H: embedding to evaluate
    :param y: ground-truth classes
    :param nn: number of neighbours to consider, if nn=None evaluate for nn=[1...size of max cluster]
    :param num_samples: number of samples in the range (1, size of max cluster)
    """
    if nn is None:
        max_size_cluster = np.unique(y, return_counts=True)[1].max()
        return np.fromiter((__local_purity(H, y, nn)
                            for nn in np.linspace(0, max_size_cluster, num_samples).astype(np.int32)), np.float32)
    else:
        return __local_purity(H, y, nn)


def __P_c_cp(distances, y, c, cp, tr):
    pcc = distances[y == c][:, y == cp].flatten()

    if tr < 1.0:
        k_smallest = int(len(pcc) * tr)
        idx = np.argpartition(pcc, k_smallest)
        return pcc[idx[:k_smallest]].mean()
    else:
        pcc.sort()
        return pcc[:int(len(pcc) * tr)].mean()


def __GS_c(distances, y, c, tr):
    all_c = np.arange(y.max() + 1)
    other_c = np.setdiff1d(all_c, c)

    pcc = np.fromiter((__P_c_cp(distances, y, c, cp, tr) for cp in all_c), np.float32)
    pcc_min = pcc[other_c].min()

    return (pcc_min - pcc[c]) / np.maximum(pcc_min, pcc[c])


def global_separation(H, y, k=None, num_samples=10):
    """

    :param H: embedding to evaluate
    :param y: ground-truth classes
    :param k: if None evaluate all classes else only class k
    :param num_samples: number of samples in the range (0, 100)%
    :return:
    """
    distances = squareform(pdist(H))
    if k is None:
        ranged = range(y.max() + 1)
    else:
        ranged = [k]

    return np.fromiter((__GS_c(distances, y, c, tr)
                        for c in ranged
                        for tr in np.linspace(0.2, 1, num_samples)), np.float32).reshape((-1, num_samples))