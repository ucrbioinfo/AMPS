import numpy as np
import random as random

def input_maker(methylations, datasize, window_size, context, half_w):
    methylations = methylations.sort_values(["chr", "position"], ascending=(True, True))
    chrs_counts = methylations['chr'].value_counts()
    last_chr_pos = {}
    chrnums = list(chrs_counts.index)
    sum = 0
    for i in range(len(chrnums)):
        if i in chrs_counts.keys():
            last_chr_pos[i] = sum+chrs_counts[i]-1
            sum += chrs_counts[i]
    # last_chr_pos ==> {0: 5524, 1: 1042784, 2: 1713034, 3: 2550983, 4: 3205486, 5: 4145381, 6: 4153872}
    # methylations.iloc[2550983] => chr 3.0 position    23459763.0
    # methylations.iloc[2550984] => chr 4.0 position    1007
    methylations.insert(0, 'idx', range(0, len(methylations)))
    sub_methylations = methylations[methylations['context'] == context]
    idxs = sub_methylations['idx']
    mlevels = methylations['mlevel']
    mlevels = np.asarray(mlevels)
    X = np.zeros((datasize, window_size))
    Y = np.zeros(datasize)
    avlbls = np.asarray(idxs)
    for lcp in list(last_chr_pos.values()):
        if lcp > 0 and lcp < len(mlevels) - window_size:
            avlbls = np.setdiff1d(avlbls, range(lcp-half_w, lcp+half_w))
    smple = random.sample(list(avlbls), min(len(datasize), len(avlbls)))
    count_errored = 0
    print('border conditions: ', np.count_nonzero(np.asarray(smple) < half_w))
    for index, p in enumerate(smple):
        try:
            X[index] = np.concatenate((mlevels[p-half_w: p], mlevels[p+1: p+half_w+1]), axis=0)
            Y[index] = 0 if mlevels[p] < 0.5 else 1
        except ValueError:
            print(index, p)
            count_errored += 1
    X = X.reshape(list(X.shape) + [1])
    print(count_errored, ' profiles faced error')
    return X, Y

def profiler(methylations, context, datasize, window_size=20 , threshold=0.5):
    #methylations = methylations[(methylations['mlevel'] > 0.8) | (methylations['mlevel'] < 0.2)]
    half_w = int(window_size/2)
    methylated = methylations[methylations['mlevel'] > threshold]
    unmethylated = methylations[methylations['mlevel'] <= threshold]
    X_methylated, Y_methylated = input_maker(methylated, int(datasize/2), window_size, context, half_w)
    X_unmethylated, Y_unmethylated = input_maker(unmethylated, int(datasize/2), window_size, context, half_w)
    return np.concatenate((X_methylated, X_unmethylated), axis=0), np.concatenate((Y_methylated, Y_unmethylated), axis=0)
