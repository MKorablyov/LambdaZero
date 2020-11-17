import time, os, sys, os.path as osp
import numpy as np
import pandas as pd
import LambdaZero.utils
from matplotlib import pyplot as plt
import seaborn as sns

datasets_dir, _, _ = LambdaZero.utils.get_external_dirs()

class cfg:
    dataset_dir = osp.join(datasets_dir, 'NewDrugComb')
    raw_dir = osp.join(dataset_dir,  'raw/')
    raw_file_names = ['drugcomb_data_v1.4.csv','summary_table_v1.4.csv']



#np.save(osp.join(cfg.raw_dir,"cell_line_to_idx.npy"), {"a":1, "b":99})

# cell_lines = ["A-673","T98G","L-1236","KBM-7","TMD8", "DIPG25", "HT29", "MCF7", "A375" , "A549", "VCAP", "LNCAP"]
# cell_line_to_idx = np.load(osp.join(cfg.raw_dir,"cell_line_to_idx.npy"),allow_pickle=True).flatten()[0]
# cell_line_idx = [cell_line_to_idx[line] for line in cell_lines]
# print(cell_line_idx)
#print(cell_line_to_idx[cell_lines[0]])


drugcomb_data = pd.read_csv(os.path.join(cfg.raw_dir, cfg.raw_file_names[1]), low_memory=False)
#print(drugcomb_data.columns)


# look at drugs
# drugs = drugcomb_data["drug_row"].to_numpy()
# studies_unq, raw_counts = np.unique(drugs,return_counts=True)
# plt.plot(-np.sort(-raw_counts))
# #plt.yscale("log")
# plt.savefig(osp.join(cfg.dataset_dir,"drugs.png"))
#
# plt.show()


# look at cell lines
#cell_names = drugcomb_data["cell_line_name"].to_numpy()
#cell_names_unq, cell_name_counts = np.unique(cell_names,return_counts=True)
#ord = np.argsort(-cell_name_counts)
#for c in cell_names_unq[ord]: print(c)


#plt.plot(-np.sort(-raw_counts))
# #plt.yscale("log")
# plt.savefig(osp.join(cfg.dataset_dir,"drugs.png"))
#


# # look at studies
# studies = drugcomb_data["study_name"].to_numpy()
# studies_unq, raw_counts = np.unique(studies,return_counts=True)
# for i in range(len(studies_unq)):
#     study_data = drugcomb_data.mask(studies==studies_unq[i])
#
#     print("study", studies_unq[i], "raw_count", raw_counts[i])

# look at fit data
#cell_line_summar = pd.read_csv(osp.join(datasets_dir,"NewDrugComb/cell_line_summary.csv"))

#cell_line_summar.sort_values("synergy_zip_001_found",inplace=True,ascending=False)
#cell_line_summar.reset_index(drop=True,inplace=True)
#pd.set_option('display.max_columns', 10)
#print(cell_line_summar)



#print(np.log10(10))
# almanac data: https://wiki.nci.nih.gov/display/NCIDTPdata/NCI-ALMANAC
almanac = pd.read_csv("/home/maksym/Datasets/ALMANAC/ComboDrugGrowth_Nov2017.csv")
print(almanac.columns)
print(set(almanac["CONCUNIT1"].to_list()))
print(set(almanac["CONCUNIT2"].to_list()))

#mask = almanac["CONC1"].to_numpy() > 0.005
#almanac = almanac[mask]
#print(set(almanac["CONCINDEX1"]))
#print(set(almanac["CONCINDEX2"]))
#print(min(almanac["CONC1"].to_list()))
#print(max(almanac["CONC2"].to_list()))

sns.distplot(np.log10(almanac["CONC1"].to_list()))
plt.xlabel("log concentration M")
plt.ylabel("count")

plt.savefig("/home/maksym/Datasets/ALMANAC/concentrations.png")
