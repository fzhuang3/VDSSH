import numpy as np

def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

def calc_metrics(qB, rB, queryL, retrievalL, k4map, k4ndcg, k4acg):
    # Calculate top mAP, NDCG, ACG
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    def dcg_at_k(r, k):
        r = np.asfarray(r)[:k]
        return np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))

    num_query = queryL.shape[0]
    old_topkmap = 0
    topkmap = 0
    topkndcg = 0
    topkacg = 0
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    for iter in range(num_query):
        gnd = np.dot(queryL[iter, :], retrievalL.transpose())
        dcg_max = dcg_at_k(sorted(gnd, reverse=True), k4ndcg)
        hamm = calc_hammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        topkndcg += dcg_at_k(gnd, k4ndcg) / dcg_max
        topkacg += np.sum(gnd[:k4acg]) / k4acg

        gnd2 = (gnd > 0).astype(np.float32)
        ttopk = np.min([k4map, np.sum(gnd2)])
        tgnd = gnd2[0:k4map]
        tsum = np.sum(tgnd)
        if (tsum == 0):
            continue
        count = np.linspace(1, tsum, tsum)
        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        old_topkmap += np.mean(count / (tindex))
        topkmap += np.sum(count / (tindex))/ttopk
    old_topkmap = old_topkmap/num_query
    topkmap = topkmap / num_query
    topkndcg = topkndcg/ num_query
    topkacg = topkacg / num_query
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    return old_topkmap, topkmap, topkndcg, topkacg

if __name__=='__main__':
    pass
