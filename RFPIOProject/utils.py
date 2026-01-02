from sklearn.neighbors import kneighbors_graph
import numpy as np
import random

def exists(val):
    return val is not None


def generateMvG(X, k=5):
    print("data numbers:", len(X[0]))
    MvG = []
    for x in X:
        subG = kneighbors_graph(x, n_neighbors=k, n_jobs=-1, include_self=False, mode='distance')
        subG = subG.toarray()

        L = []
        for i, g in enumerate(subG):
            b = g
            sort = sorted(enumerate(b), key=lambda x: x[1])
            ind = [x[0] for x in sort]
            row = np.concatenate(([i], ind[-k:]))

            L.append(row)
        MvG.append(L)
    return np.array(MvG)

def getMvKNNGraph(X, k=5, mode='connectivity'):
    MvG = []
    for x in X:
        subG = kneighbors_graph(x, n_neighbors=k, n_jobs=-1, include_self=False, mode=mode)
        subG = subG.toarray()

        MvG.append(subG)
    return np.array(MvG)

def splitDigitData(N, V, inCP, seed):

    assert 0 <= inCP <= 1, "inCP must be within the range [0,1]"
    assert V >= 1, "The number of views must be ≥ 1"
    assert N >= V, "The number of samples must be ≥ the number of views"

    splitInd = np.ones((N, V), dtype=int)
    delNum = int(np.floor(N * inCP))

    np.random.seed(seed)
    random.seed(seed)

    view_masks = []
    for _ in range(V):
        perm = np.random.permutation(N)
        view_masks.append(perm[:delNum])


    for v in range(V):
        splitInd[view_masks[v], v] = 0


    zero_rows = np.where(splitInd.sum(axis=1) == 0)[0]

    for row in zero_rows:

        v = random.randint(0, V - 1)
        splitInd[row, v] = 1

        available = np.setdiff1d(np.arange(N), view_masks[v])
        if len(available) > 0:
            new_mask = random.choice(available)
            splitInd[new_mask, v] = 0
            view_masks[v] = np.append(view_masks[v], new_mask)

    return splitInd