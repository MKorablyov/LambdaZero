import numpy as np
import random,time

def parse_sdhit_cluster(cluster_file):
    # parse the output of the clustering from cd-hit
    pdbids = []
    chains = []
    clusters = []
    with open(cluster_file) as f:
        for line in f:
            if line.startswith(">Cluster"):
                cluster = int(line.strip(">Cluster "))
            else:
                pdbid_chain = line.split(",")[1].split("|")[0].strip(" >")
                assert len(pdbid_chain) in [6,7] and pdbid_chain[4]==":", "parser broke"
                pdbid = pdbid_chain[:4]
                chain = pdbid_chain[5:]
                pdbids.append(pdbid.upper())
                chains.append(chain)
                clusters.append(cluster)
    return np.array(pdbids,dtype=np.str),np.array(chains,dtype=np.str),np.array(clusters,dtype=np.int32)


def splitby_clusters(pdbids,clusters,test_prob):
    """
    :param pdbids:
    :param clusters:
    :param test_prob:
    :return:
    """
    pdbids = np.asarray(pdbids)
    clusters = np.asarray(clusters)
    _test_pdbids = []
    _test_clusters = []
    _train_pdbids = []
    _train_clusters = []
    while len(pdbids) > 0:
        if len(pdbids) %500==1: print("records to split:", len(pdbids))
        # start with an empty group and one pdb
        split_pdbids = [pdbids[0]]
        split_clusters = [clusters[0]]
        last_len = 0
        # until nothing to add repeat
        while not len(split_pdbids)==last_len:
            last_len = len(split_pdbids)
            # expand pdbids into complete clusters
            pdb2clust_mask = np.in1d(pdbids,split_pdbids)
            split_pdbids = np.concatenate([split_pdbids,pdbids[pdb2clust_mask]],axis=0)
            pdbids = pdbids[np.logical_not(pdb2clust_mask)]
            split_clusters = np.concatenate([split_clusters,clusters[pdb2clust_mask]],axis=0)
            clusters = clusters[np.logical_not(pdb2clust_mask)]
            # expand clusters into pdbids
            clust2pdb_mask = np.in1d(clusters,split_clusters)
            split_clusters = np.concatenate([split_clusters,clusters[clust2pdb_mask]],axis=0)
            clusters = clusters[np.logical_not(clust2pdb_mask)]
            split_pdbids = np.concatenate([split_pdbids,pdbids[clust2pdb_mask]],axis=0)
            pdbids = pdbids[np.logical_not(clust2pdb_mask)]
        # print "cluster of size:", last_len
        # randomly add either to test or to the train subset of the pdbs
        if random.random() < test_prob:
            [_test_pdbids.append(pdbid) for pdbid in split_pdbids]
            [_test_clusters.append(cluster) for cluster in split_clusters]
        else:
            [_train_pdbids.append(pdbid) for pdbid in split_pdbids]
            [_train_clusters.append(cluster) for cluster in split_clusters]
    # every pdb might have multiple chains; remove redundancy in counting
    _, unq_train = np.unique(np.array(_train_pdbids), return_index=True)
    train_pdbids = np.array(_train_pdbids)[unq_train]
    train_clusters = np.array(_train_clusters)[unq_train]
    _, unq_test = np.unique(np.array(_test_pdbids),return_index=True)
    test_pdbids = np.array(_test_pdbids)[unq_test]
    test_clusters = np.array(_test_clusters)[unq_test]
    return train_pdbids,train_clusters,test_pdbids,test_clusters

if __name__ == "__main__":
    # cluster the pdbs at desired identity using cd-hit
    # cd-hit -i XRay2_5.txt -o XRay2_5_clust05 -c 0.5 -n 3
    cluster_file = "/home/maksym/Projects/datasets/sequence_cluster/XRay2_5_clust07.clstr"
    pdbids,chains,clusters =  parse_sdhit_cluster(cluster_file)
    print("number of records:", len(pdbids), "number of clusters:",max(clusters))
    test_pdbids, train_pdbids, test_clusters, train_clusters = splitby_clusters(pdbids, chains, clusters, 0.1)
    print("test records:", len(test_pdbids), "train_records:", len(train_pdbids))
    print ("test unique clusters", len(set(test_clusters)), "train unique clusters", len(set(train_clusters)))
