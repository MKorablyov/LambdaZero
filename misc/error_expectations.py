import numpy as np
from sklearn import metrics
from matplotlib import pyplot as plt
import seaborn as sns


class cfg:
    mean = - 9
    std = 1.31
    num_samples = 10000000
    #
    dock_err_mean = 0.0
    dock_err_std = 2.1
    #
    fep_err_mean = 0.0
    fep_err_std = 0.927

    n_mol_tests = 100 # means how many molecules tested in experiment

def data_model_v1(num_samples=cfg.num_samples):
    "assume gaussian distribution of energies, docking energies, fep_errors"
    true_data = np.random.normal(cfg.mean,cfg.std,cfg.num_samples )
    dock_errors = np.random.normal(cfg.dock_err_mean,cfg.dock_err_std,num_samples)
    dock_data = true_data + dock_errors
    fep_error = np.random.normal(cfg.fep_err_mean, cfg.fep_err_std, num_samples)
    fep_data = true_data + fep_error
    return true_data, dock_data, fep_data


def data_model_v2(num_samples=cfg.num_samples, corr = 0.15):
    "docking error strongly correlates with the actual energy of the molecule"
    true_data = np.random.normal(cfg.mean,cfg.std,cfg.num_samples)
    corr_error = - true_data
    rand_error = np.random.normal(0,1.0,num_samples)


    dock_error = (corr_error * corr) + (rand_error * (1-corr))
    dock_error = (dock_error - dock_error.mean()) / dock_error.std() * cfg.dock_err_std
    dock_data = true_data + dock_error

    fep_error = np.random.normal(cfg.fep_err_mean, cfg.fep_err_std, num_samples)
    fep_data = true_data + fep_error
    return true_data, dock_data, fep_data


def compute_stat(data, predictions, top_k):
    idxs = np.argpartition(predictions, top_k-1)[:top_k-1]
    median = np.median(data[idxs])
    return median



def plot_search_space_size(true_data,dock_data,fep_data, top_k=cfg.n_mol_tests):

    n_data = true_data.shape[0]
    ns_samples = 10**np.arange(1+int(np.log10(n_data)))[2:]

    medians = []

    for n_samples in ns_samples:

        idxs_samples = np.random.choice(n_data, n_samples, replace=False)
        true_data_sample = true_data[idxs_samples]
        dock_data_sample = dock_data[idxs_samples]
        fep_data_sample = fep_data[idxs_samples]

        data_median = compute_stat(true_data_sample, true_data_sample, top_k)
        dock_median = compute_stat(true_data_sample, dock_data_sample, top_k)
        fep_median = compute_stat(true_data_sample, fep_data_sample, top_k)

        medians.append([data_median,dock_median,fep_median])

    medians = np.asarray(medians).T

    fig = plt.figure(dpi=300)
    ax = fig.gca()
    ax.scatter(ns_samples,medians[0,:],label="true")
    ax.scatter(ns_samples, medians[1, :], label="dock")
    ax.scatter(ns_samples, medians[2, :], label="FEP")
    plt.xscale("log")
    plt.legend()
    plt.xlabel("size of the search space")
    plt.ylabel("binding affinity")

    fig.suptitle("gaussian hypothesis")

    #plt.show()
    plt.savefig("/home/maksym/Desktop/gauss_prior_docking_fep.png")




if __name__ == "__main__":
    #fep_data = np.genfromtxt('fep_data.csv', delimiter=',', skip_header=1)
    #plt.scatter(np.abs(fep_data[:, 0] - fep_data[:, 1]))
    #sns.distplot(np.abs(fep_data[:, 0] - fep_data[:, 1]))


    #plt.show()
    #print("binding energy mean", fep_data[:,0].mean(), "binding energy std", fep_data[:,0].std())
    #print("eror_std", (fep_data[:,0] - fep_data[:,1]).std())
    #plt.scatter(fep_data[:,0],fep_data[:,1])
    #plt.show()
    true_data,docking_data,fep_data = data_model_v1()
    plot_search_space_size(true_data,docking_data,fep_data)


    # todo: data is probably gaussian
    # todo: we should not assume errors are gaussian (as we assumed here)
    # todo: model error in a better way

