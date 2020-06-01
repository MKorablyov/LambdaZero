import os
import sys

import matplotlib.cm as cm
import numpy as np
import pandas as pd
import scipy.stats as stats
from matplotlib import pyplot as plt

sys.path.append("../..")


# sample size / top 100 affinity

class cfg:
    db_path = "/home/maksym/Projects/datasets/brutal_dock"
    #docked_csv_file = "micro.csv"
    docked_csv_file = "ampc_screen_table.csv" # 99,459,561 records
    docked_index = "ampc.parquet"
    csize = 50000
    figname = "ampc_draws.png"
    # smaller uniform 10K
    docked_index_10k = "ampc_10k.parquet"
    docked_index_100k = "ampc_100k.parquet"
    docked_index_250k = "ampc_250k.parquet"

# class cfg:
#     db_path = "/home/maksym/Projects/datasets/brutal_dock"
#     #docked_csv_file = "micro.csv"
#     docked_csv_file = "d4_screen_table.csv"
#     docked_index = "d4.parquet"
#     csize = 50000
#     figname = "d4_draws.png"
#     # smaller uniform 10K
#     docked_index_10k = "d4_10k.parquet"
#     docked_index_100k = "d4_100k.parquet"
#     docked_index_250k = "d4_250k.parquet"


# convert the slow csv format to msgpack; remove docking with no score
# docked_data_ = pd.read_csv(os.path.join(cfg.db_path,cfg.docked_csv_file),
#                          dtype={"zincid": str, "smiles":str, "dockscore":str, "hac": str}, chunksize=cfg.csize)
# for i,docked_data in enumerate(docked_data_):
#     start = time.time()
#     docked_data = docked_data[pd.to_numeric(docked_data['dockscore'], errors='coerce').notnull()]
#     print(docked_data["dockscore"].to_numpy().min())
#
#     docked_data = docked_data[["zincid", "smiles", "dockscore"]]
#     docked_data = docked_data.astype({"zincid":str, "smiles":str, "dockscore":float})
#    docked_file = os.path.join(cfg.db_path,cfg.docked_index)
#     if i ==0:
#         docked_data.to_parquet(docked_file, engine="fastparquet", compression=None)
#     else:
#         docked_data.to_parquet(docked_file, engine="fastparquet", compression=None, append=True)
#     print("raw:",i*cfg.csize, "exps:" "%.3f" % (cfg.csize / (time.time() - start)))


class Sample_size_plot:
    def __init__(self,data):
        self.data = np.sort(data)

    def plot(self,top_k=100,num_curves=7,num_draws=10,out_path=None,xlabel=None):
        n_data = self.data.shape[0]
        ns_samples = np.logspace(np.log(top_k), np.log(n_data), base=np.e, num=num_curves).astype(np.int)

        fig = plt.figure(dpi=300)
        ax = fig.gca()
        colors = cm.rainbow(np.linspace(0, 1, num_curves))

        for i,n_samples in enumerate(ns_samples):
            sample = []
            for j in range(num_draws):
                idxs = np.random.choice(n_data,n_samples,replace=False)
                idxs = idxs[np.argpartition(idxs,top_k-1)[:top_k-1]]
                sample_ = self.data[idxs]
                sample.append(sample_)
                print("iter i,j:", i,j)
            sample = np.sort(np.concatenate(sample,axis=0))

            # plot points and histogram
            fit = stats.norm.pdf(sample, np.mean(sample), np.std(sample))
            ax.plot(sample, fit, '-o', label=str(n_samples), color=colors[i])

            # annotate median
            median = (sample[(num_draws * top_k)//2], fit[(num_draws * top_k)//2])
            ax.plot(median[0],median[1], marker='o', markersize=10, color=colors[i])
            ax.annotate(np.around(median[0], decimals=3), xy=median, textcoords='data')
        plt.legend()
        ax.yaxis.set_major_locator(plt.NullLocator()) # turn off x-axis
        if plt.xlabel is not None:
            plt.xlabel(xlabel)
        if out_path is None:
            plt.show()
        else:
            plt.savefig(out_path)


# dockscore = np.asarray(pd.read_parquet(os.path.join(cfg.db_path,cfg.docked_index), columns=["dockscore"]))[:,0]
# print(dockscore.min(),dockscore.max(), dockscore.std(),dockscore.mean(),np.median(dockscore))
# np.random.shuffle(dockscore)
# sample = dockscore[:100000] # todo: -65 is the dockscore I am getting with 100K samples
# print(sample.min(),np.median(sample))


# sample_size_plot = Sample_size_plot(dockscore)
# print("loaded/sorted")
# sample_size_plot.plot(out_path=os.path.join(cfg.db_path,cfg.figname),xlabel="D4 dock kCal/mol")


# make a partitioned dataset for ML
docked_index = pd.read_parquet(os.path.join(cfg.db_path,cfg.docked_index))
# remove very bad docks
dockscore = docked_index["dockscore"].to_numpy()
idxs = np.arange(len(dockscore))[dockscore < 0]
docked_index = docked_index.iloc[idxs]
dockscore = dockscore[idxs]
del idxs # free memory
# #print(dockscore.min())
# #print(len(docked_index))

# -10k
# bin_high = 0.0 ##np.percentile(dockscore, 90)
# print("bin_high:", bin_high)
# idxs = utils.uniform_sample(dockscore,nsamples=10000,bin_high=bin_high)
# docked_index_10k = docked_index.iloc[idxs]
# docked_index_10k.to_parquet(os.path.join(cfg.db_path,cfg.docked_index_10k))
# print("done 10K")
#
# idxs = utils.uniform_sample(dockscore,nsamples=100000,bin_high=bin_high)
# docked_index_100k = docked_index.iloc[idxs]
# docked_index_100k.to_parquet(os.path.join(cfg.db_path,cfg.docked_index_100k))
#
# idxs = utils.uniform_sample(dockscore,nsamples=250000,bin_high=bin_high,nbins=200)
# docked_index_250k = docked_index.iloc[idxs]
# docked_index_250k.to_parquet(os.path.join(cfg.db_path,cfg.docked_index_250k))
#
# embed_index = pd.read_parquet(os.path.join(cfg.db_path,cfg.docked_index_10k))
# chem.ScaffoldSplit(embed_index["smiles"].tolist())

#dockscore = np.asarray(pd.read_parquet(os.path.join(cfg.db_path,cfg.docked_index_250k), columns=["dockscore"]))[:,0]
#sns.distplot(dockscore)
#plt.show()


#print("minimum dockscore:", min(dockscore), np.mean(dockscore), max(dockscore))
#print("dockscore pred MAE" ,np.abs(dockscore - np.mean(dockscore)).mean())

# I want to # max(ave(top 100)) of 100,000,000
# std = 14.0 ; top = 42 ~ 3 stds
# high labels are important; high preds are important
# I * (E1 - E2)**2

# top_100 docking