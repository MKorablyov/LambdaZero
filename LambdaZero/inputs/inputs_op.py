import os,time
import numpy as np
import pandas as pd
import torch
from torch_sparse import coalesce

import rdkit
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem.rdchem import HybridizationType
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.rdchem import BondType as BT

import ray
rdBase.DisableLog('rdApp.error')


from torch.utils.data import Dataset as th_Dataset
#torch.multiprocessing.set_sharing_strategy('file_system') # bug https://github.com/pytorch/pytorch/issues/973
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#try:
from torch_geometric.data import (InMemoryDataset, download_url, extract_zip, Data)
from sklearn.preprocessing import StandardScaler as sk_StandardScaler
from sklearn.decomposition import PCA as sk_PCA
#except Exception as e:
#    print("could not import pytorch geometric")

#from ..py_tools.datasets import load_qm7
#from ..py_tools import geom
#from ..py_tools import chem
#from ..py_tools import multithread as mtr

import LambdaZero.chem

# def mtable_featurizer(elem, charge=None,organic=False):
#     """
#
#     :param elem:
#     :param charge:
#     :return:
#     """
#     elem_feat = []
#     if not organic: elem_feat.append(np.asarray(np.floor_divide(elem + 2, 8), dtype=np.float32) / 2.)
#     elem_feat.append(np.asarray(np.remainder(elem + 2, 8), dtype=np.float32) / 4.)
#     elem_feat.append(np.ones(elem.shape, dtype=np.float32))
#     if charge is not None:
#         elem_feat.append(charge)
#     return np.stack(elem_feat,axis=1)
#
# class Sampler(torch.utils.data.Sampler):
#     def __init__(self, idxs, probs=None, num_samples=None):
#         self.idxs = idxs
#         self.probs = probs
#         if num_samples is None:
#             num_samples = len(self.idxs)
#         self.num_samples = num_samples
#     def __iter__(self):
#         if self.probs is None:
#             samples = np.remainder(np.arange(self.num_samples),len(self.idxs))
#         else:
#             samples = np.random.choice(self.idxs,self.num_samples,p=(self.probs/np.sum(self.probs)))
#         return iter(samples)
#     def __len__(self):
#         return self.num_samples
#
# class ImgGen(torch.nn.Module):
#     def __init__(self, grid_pix, pix_size, sigmas, input_d, pairwise=False):
#         """
#
#         :param grid_dim:
#         :param pix_size:
#         :param sigmas:
#         :param input_d:
#         :param pairwise:
#         """
#         # todo support kernels other than gauss
#         # todo; try normalizing by the median of the energy
#         super(ImgGen, self).__init__()
#         self._pix_size = pix_size
#         self._sigmas = sigmas
#         self._pairwise = pairwise
#         self._grid_pix = grid_pix
#         # generate all possible pairwise combinations
#         if self._pairwise:
#             img_depth = input_d * len(self._sigmas)
#             pairwise_idx = np.tile(np.arange(img_depth)[:, None], [1, img_depth]).reshape(-1)[:, None]
#             pairwise_idx = np.concatenate([pairwise_idx, np.tile(np.arange(img_depth), [img_depth])[:, None]], 1)
#             self._pairwise_idx = torch.tensor(pairwise_idx)
#         grid_dim = np.asarray([grid_pix, grid_pix, grid_pix])
#         grid = np.stack([-grid_dim // 2, grid_dim - grid_dim // 2], 1)
#         grid = np.mgrid[grid[0, 0]:grid[0, 1], grid[1, 0]:grid[1, 1], grid[2, 0]:grid[2, 1]]
#         grid = torch.tensor(np.asarray(grid, dtype=np.float32))
#         self._grid = torch.nn.Parameter(grid, requires_grad=False)
#
#     def forward(self, coord, feat):
#         """
#
#         :param coord:
#         :param feat:
#         :return:
#         """
#         # crop
#         dist_mask = (coord ** 2).sum(1) < (self._pix_size * self._grid_pix) ** 2
#         coord = coord[dist_mask]
#         feat = feat[dist_mask]
#         #print("cood input", dist_mask.size(0), "coord out", coord.size(0))
#
#         # explicitly compute distances
#         scaled_coord = coord / self._pix_size
#         dist_sq = torch.sum(((self._grid[None, :, :, :] - scaled_coord[:, :, None, None, None]) ** 2), 1)
#         # generate images for each possible sigma
#         gauss_imgs = []
#         for sigma in self._sigmas:
#             gauss_dist = torch.exp(-dist_sq / (2 * (float(sigma) ** 2)))
#             gauss_img = (gauss_dist[:, None, :, :, :] * feat[:, :, None, None, None]).sum([0])
#             gauss_imgs.append(gauss_img)
#         gauss_img = torch.cat(gauss_imgs, dim=0)
#
#         # generate pairsies combinations when needed
#         if self._pairwise:
#             gauss_img = gauss_img[self._pairwise_idx]
#             gauss_img = gauss_img.sum(1)
#         #af.utils.volume_plot(sample["rec_img"].detach().cpu().numpy()[0, 1, :, :, :], cutoff=0.01)
#         # todo: verbose
#         #print("gauss img", gauss_img.mean(), gauss_img.std())
#         #print("sparsity:",
#         #     np.asarray(gauss_img[0].detach().cpu().numpy().reshape(-1) > 0.002, dtype=np.float32).mean())
#         #sns.distplot(gauss_img[0].detach().cpu().numpy().reshape(-1))
#         #plt.show()
#         return gauss_img

#
# class DatasetFrag(th_Dataset):
#     # todo: rename ufrag num into jun_bond idx
#     def __init__(self, db_path, db_index, rotate, crop_radii=9., verbose=False):
#         # initialization parameters
#         self.ufrag_path = os.path.join(db_path, "lig_frag")
#         self.rec_elem_path = os.path.join(db_path, "rec_elem")
#         self.rec_coord_path = os.path.join(db_path, "rec_coord")
#         self.rotate = rotate
#         self.crop_radii = crop_radii
#         self.verbose = verbose
#
#         # load the dataset index file
#         db_index = np.load(os.path.join(db_path, db_index))
#         self.names, self.ufrag_nums = db_index["names"], db_index["ufrag_nums"]
#
#         assert len(self.names) == len(self.ufrag_nums), "broken database index file"
#         print("loaded database with ", len(self.names))
#
#     def __len__(self):
#         return len(self.names)
#
#     def __getitem__(self, idx):
#         # find ligand name and number of the bond to break
#         name = self.names[idx]
#         ufrag_num = self.ufrag_nums[idx] # junction bond to break
#
#         # load ligand
#         sam = np.load(os.path.join(self.ufrag_path, name + ".npz"), allow_pickle=True)
#         frag_elems = sam["frag_elems"]
#         frag_coords = sam["frag_coords"]
#         jun_bonds = sam["jun_bonds"]
#         ufrag_idxs = sam["ufrag_idxs"]
#
#         # load receptor
#         rec_elem = np.load(os.path.join(self.rec_elem_path, name + ".npy"))
#         rec_coord = np.load(os.path.join(self.rec_coord_path, name + ".npy"))
#
#         # randomly rotate; data augmentation
#         if self.rotate:
#             rec_coord, tfm = geom.affine_tform(rec_coord)
#             frag_coords = [geom.affine_tform(frag_coord, tfm)[0] for frag_coord in frag_coords]
#
#         # chop the corresponding fragment
#         ufrag_idx = ufrag_idxs[ufrag_num]
#         root_idx = ufrag_idxs[(ufrag_num + jun_bonds.shape[0]) % (2 * jun_bonds.shape[0])]
#
#         if ufrag_num < len(jun_bonds):
#             jun_bond = jun_bonds[ufrag_num]
#         else:
#             ufrag_num = ufrag_num % jun_bonds.shape[0]
#             jun_bond = np.take(jun_bonds[ufrag_num], [1, 0, 3, 2])
#         frag_elem = frag_elems[jun_bond[0]]
#         frag_coord = frag_coords[jun_bond[0]]
#         root_elem = frag_elems[jun_bond[1]]
#         root_coord = frag_coords[jun_bond[1]]
#         bond_coord = np.stack([root_coord[jun_bond[3]], frag_coord[jun_bond[2]]], axis=0)
#
#         # align fragment and receptor
#         bond_len = ((bond_coord[0] - bond_coord[1]) ** 2).sum() ** 0.5
#         align_tfm = geom.align(bond_coord, np.array([[0., 0., 0.], [bond_len, 0., 0.]]))
#         root_coord, _ = geom.affine_tform(root_coord, align_tfm)
#         frag_coord, _ = geom.affine_tform(frag_coord, align_tfm)
#         rec_coord, _ = geom.affine_tform(rec_coord, align_tfm)
#
#         # crop atoms too far
#         rec_mask = (rec_coord ** 2).sum(1) < np.float32(self.crop_radii ** 2)
#         rec_elem = rec_elem[rec_mask]
#         rec_coord = rec_coord[rec_mask]
#
#         #for i in range(len(frag_elems)-1):
#         #    print(frag_elems.shape, ufrag_idxs.shape)
#         #    print("frag elem, ufrag idxs", frag_elems[i], ufrag_idxs[i])
#
#         if self.verbose:
#             # check alignment
#             print("root should be aligned at 0,0,0:", root_coord[jun_bond[3]])
#             # plot coordinates
#             fig = plt.figure(dpi=300)
#             ax = Axes3D(fig)
#             print("ufrag_idx", ufrag_idx, "ufrag_elem", frag_elem)
#             ax.scatter3D(frag_coord[:, 0], frag_coord[:, 1], frag_coord[:, 2], color="red")
#             ax.scatter3D(root_coord[:, 0], root_coord[:, 1], root_coord[:, 2], color="blue")
#             ax.scatter3D(rec_coord[:, 0], rec_coord[:, 1], rec_coord[:, 2], color="green")
#             plt.show()
#
#         out = {"name":name,
#                "ufrag_num":ufrag_num, "ufrag_idx": ufrag_idx, "frag_elem": frag_elem, "frag_coord": frag_coord,
#                "root_idx": root_idx, "root_elem": root_elem, "root_coord": root_coord,
#                "rec_elem":rec_elem, "rec_coord": rec_coord}
#         return out
#
#
# class DatasetFragImg(th_Dataset):
#     def __init__(self, datasetFrag, atm_feat="mtable", fragImgGen=None, rootImgGen=None, recImgGen=None, verbose=False,
#                  cuda=True):
#         # initial parameters
#         self.atm_feat = atm_feat
#         self.verbose = verbose
#         self.oncuda = cuda
#         # dataset
#         self.datasetFrag = DatasetFrag(**datasetFrag)
#         # image generators root
#         if rootImgGen is not None:
#             self.rootImgGen = ImgGen(**rootImgGen)
#             if cuda: self.rootImgGen.cuda()
#         else:
#             self.rootImgGen = None
#         # frag
#         if fragImgGen is not None:
#             self.fragImgGen = ImgGen(**fragImgGen)
#             if cuda: self.fragImgGen.cuda()
#         else:
#             self.fragImgGen = None
#         # rec
#         if recImgGen is not None:
#             self.recImgGen = ImgGen(**recImgGen)
#             if cuda: self.recImgGen.cuda()
#         else:
#             self.recImgGen = None
#
#     def _gen_img(self, imgGen, elem, coord, organic):
#         if self.atm_feat == None:
#             feat = elem[:, None]
#         elif self.atm_feat == "mtable":
#             feat = mtable_featurizer(elem, organic=organic)
#         else:
#             raise NotImplementedError("can't featurize")
#         coord, feat = torch.from_numpy(coord), torch.from_numpy(feat)
#         if self.oncuda: coord, feat = coord.cuda(), feat.cuda()
#         return imgGen(coord, feat)
#
#     def __len__(self):
#         return len(self.datasetFrag)
#
#     def __getitem__(self, idx):
#         #name, ufrag_num, ufrag_idx, frag_elem, frag_coord, root_elem, root_coord, rec_elem, rec_coord\
#         sam = self.datasetFrag[idx]
#         ufrag_idx = torch.from_numpy(np.asarray(sam["ufrag_idx"], dtype=np.int64)) # convert to torch tensor
#         out = {"idx": idx, "ufrag_idx":ufrag_idx}
#         if self.rootImgGen is not None:
#             out["root_img"] = self._gen_img(self.rootImgGen, sam["root_elem"], sam["root_coord"], organic=False)
#         if self.fragImgGen is not None:
#             out["frag_img"] = self._gen_img(self.fragImgGen, sam["frag_elem"], sam["frag_coord"], organic=False)
#         if self.recImgGen is not None:
#             out["rec_img"] = self._gen_img(self.recImgGen, sam["rec_elem"], sam["rec_coord"], organic=True)
#         return out

# class Dataset(th_Dataset):
#     def __init__(self, db_root, db_index, pr, pr_arg, pr_input, oncuda):
#         """ This is a generic dataset loader
#         :param db_root:
#         :param db_index:
#         """
#         # load database index
#         self.db_root = db_root
#         db_indexes = [np.load(os.path.join(db_root, index),allow_pickle=False,fix_imports=False) for index in db_index]
#         assert type(db_index) == list, "expected a list of index.npy for file locations"
#         assert all([np.array_equal(db_indexes[0][0, :], db_index[0, :]) for db_index in db_indexes]), "header mismatch"
#         db_header = np.array(db_indexes[0][0, :], np.str)
#         db_index = np.concatenate([db_index[1:, :] for db_index in db_indexes], axis=0)
#         self.db_index = np.array(db_index, np.unicode)
#         self.num_records = db_index.shape[0]
#
#         # initialize the preprocessor
#         self.pr = pr(**pr_arg)
#         if oncuda:
#             self.pr.cuda()
#         self.pr_input_key, pr_input_val = zip(*pr_input.items())
#         assert set(pr_input_val).issubset(db_header), "unknown in the db_index value requested"
#         pr_cols = np.where(np.in1d(db_header, pr_input_val))[0]
#         self.pr_inputs = np.take(self.db_index, pr_cols, axis=1)
#
#     def __len__(self):
#         return self.num_records
#
#     def __getitem__(self, idx):
#         # load data from the disk
#         np_arrays = [np.load(os.path.join(self.db_root, ref)) for ref in self.pr_inputs[idx]]
#         # feed to the preprocessor
#         pr_input = dict(zip(self.pr_input_key, np_arrays))
#         #print("parameters in instance:", list(self.pr.parameters()))
#         return (idx,) + self.pr(**pr_input)
#
# class PreprocVS(torch.nn.Module):
#     def __init__(self, rot_complex=True, rot_lig=False, rot_rec=False, rot_binder=True, shrange=3.0, featurize=None):
#         """
#
#         :param rot_complex:
#         :param rot_lig:
#         :param rot_rec:
#         :param rot_binder:
#         :param shrange:
#         :param featurize:
#         """
#         if rot_complex: assert not (rot_lig or rot_rec), \
#             "can't rotate complex when either ligand or receptor is rotated"
#         super(PreprocVS, self).__init__()
#         self.rot_complex = rot_complex
#         self.rot_lig = rot_lig
#         self.rot_rec = rot_rec
#         self.rot_binder = rot_binder
#         self.shiftrange = [shrange, shrange, shrange]
#         assert featurize in [None, "mtable"]
#         self.featurize = featurize
#
#     def forward(self, lig_elem, lig_coord, rec_elem, rec_coord, binder_elem, binder_coord):
#         """
#
#         :param lig_elem:
#         :param lig_coord:
#         :param rec_elem:
#         :param rec_coord:
#         :param binder_elem:
#         :param binder_coord:
#         :return:
#         """
#         # select frame
#         lig_coord = lig_coord[0]
#         rec_coord = rec_coord[0]
#         binder_nframes = binder_coord.shape[0]
#         binder_idx = np.random.randint(low=0, high=binder_nframes)
#         binder_coord = binder_coord[binder_idx]
#
#         # data augmentation by rotation and shift
#         if self.rot_complex:
#             #print(lig_coord.shape)
#             #old_coord = lig_coord
#             lig_center = np.mean(lig_coord, axis=0)[None, :]
#             lig_coord, tfm = geom.affine_tform(lig_coord - lig_center, shift=self.shiftrange, rot=None)
#             #print("lig coord RMSD:", ((old_coord-lig_coord)**2).mean(0).sum()**0.5)
#             rec_coord, _ = geom.affine_tform(rec_coord - lig_center, tfm)
#         if self.rot_lig:
#             lig_center = np.mean(lig_coord, axis=0)[None, :]
#             lig_coord, _ = geom.affine_tform(lig_coord - lig_center, shift=self.shiftrange, rot=None)
#         if self.rot_rec:
#             rec_center = np.mean(rec_coord, axis=0)[None, :]
#             rec_coord, _ = geom.affine_tform(rec_coord - rec_center, shift=self.shiftrange, rot=None)
#         if self.rot_binder:
#             binder_center = np.mean(binder_coord, axis=0)[None, :]
#             binder_coord, _ = geom.affine_tform(binder_coord - binder_center, shift=None, rot=None)
#
#         # featurize atoms if needed
#         if self.featurize == "mtable":
#             lig_elem = mtable_featurizer(lig_elem, organic=False)
#             rec_elem = mtable_featurizer(rec_elem, organic=True)
#             binder_elem = mtable_featurizer(binder_elem, organic=False)
#         return lig_elem, lig_coord, rec_elem, rec_coord, binder_elem, binder_coord
#
#
# class PreprocVSimg(torch.nn.Module):
#     def __init__(self,preprocVS_par,ligImgGen_par=None,recImgGen_par=None,binderImgGen_par=None):
#         super(PreprocVSimg, self).__init__()
#         # read coordinates and elements from the disk
#         self.preprocVS = PreprocVS(**preprocVS_par)
#         # generate images
#         if ligImgGen_par is None:
#             self.ligImgGen = None
#         else:
#             self.ligImgGen = ImgGen(**ligImgGen_par)
#         if recImgGen_par is None:
#             self.recImgGen = None
#         else:
#             self.recImgGen = ImgGen(**recImgGen_par)
#         if binderImgGen_par is None:
#             self.binderImgGen = None
#         else:
#             self.binderImgGen = ImgGen(**binderImgGen_par)
#
#     def forward(self, lig_elem, lig_coord, rec_elem, rec_coord, binder_elem, binder_coord):
#         # get stuff from the pipeline
#         lig_feat, lig_coord, rec_feat, rec_coord, binder_feat, binder_coord = \
#             self.preprocVS(lig_elem, lig_coord, rec_elem, rec_coord, binder_elem, binder_coord)
#         # generate images
#         imgs = []
#         if self.ligImgGen is not None:
#             lig_coord, lig_feat = torch.from_numpy(lig_coord), torch.from_numpy(lig_feat)
#             if any([p.is_cuda for p in self.parameters()]):
#                 lig_coord, lig_feat = lig_coord.cuda(), lig_feat.cuda()
#             imgs.append(self.ligImgGen(lig_coord,lig_feat))
#         if self.recImgGen is not None:
#             rec_coord, rec_feat = torch.from_numpy(rec_coord), torch.from_numpy(rec_feat)
#             if any([p.is_cuda for p in self.parameters()]):
#                 rec_coord, rec_feat = rec_coord.cuda(), rec_feat.cuda()
#             imgs.append(self.recImgGen(rec_coord,rec_feat))
#         if self.binderImgGen is not None:
#             binder_coord, binder_feat = torch.from_numpy(binder_coord), torch.from_numpy(binder_feat)
#             if any([p.is_cuda for p in self.parameters()]):
#                 binder_coord, binder_feat = binder_coord.cuda(), binder_feat.cuda()
#             imgs.append(self.binderImgGen(binder_coord,binder_feat))
#         return tuple(imgs)


#
# class DatasetQM7(th_Dataset):
#     def __init__(self, db_path, rotate, shiftrange):
#         """
#
#         :param db_path:
#         :param rotate:
#         :param shiftrange:
#         """
#         self._rotate = rotate
#         self._shiftrange = shiftrange
#         # load and split qm7
#         self.energies, self.elems, self.coords = load_qm7(db_path)
#
#         assert len(self.energies) == len(self.elems) and len(self.energies) == len(self.coords), "wrong num records"
#
#     def __len__(self):
#         return len(self.energies)
#
#     def __getitem__(self, idx):
#         coord = self.coords[idx]
#         if self._rotate:
#             shift = [self._shiftrange, self._shiftrange, self._shiftrange]
#             coord, _ = geom.affine_tform(coord, shift=shift, rot=None)
#         atm_feat = mtable_featurizer(elem=self.elems[idx])
#         return self.energies[idx], self.elems[idx], coord, atm_feat
#
#
# def qm7_input(data_dir, test_prob=0.2, shuffle=True, rotate=True, shiftrange=1.5):
#     """
#     :param data_dir:
#     :return:
#     """
#     dataset = DatasetQM7(data_dir,rotate,shiftrange)
#     ntest = int(len(dataset) * test_prob)
#     ntrain = int(len(dataset) - (len(dataset) * test_prob))
#     train_set, test_set = torch.utils.data.random_split(dataset, [ntrain, ntest])
#     train_set = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=shuffle)
#     test_set = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=shuffle)
#     return train_set, test_set
#
#
# class Embedding_qm_dataset(th_Dataset):
#     def __init__(self, data_dir, PCA_comp=False):
#         """
#
#         :param data_dir:
#         :param PCA_comp:
#         """
#         self.energies = np.load(os.path.join(data_dir, "energies.npy"))
#         self.nums_atm = np.load(os.path.join(data_dir, "nums_atm.npy"))
#         embeds = np.load(os.path.join(data_dir, "embeds.npy"))#[:,:80,...]
#         embeds = sk_StandardScaler().fit_transform(embeds)
#
#         if not PCA_comp:
#             self.embeds = embeds
#         else:
#             self.embeds = embeds
#             pca = sk_PCA(n_components=PCA_comp)
#             pca.fit(embeds)
#             self.embeds = pca.fit_transform(embeds)
#
#     def __len__(self):
#         return len(self.energies)
#
#     def __getitem__(self, idx):
#         return self.energies[idx], self.nums_atm[idx], self.embeds[idx]
#
#
# def embedding_qm_input(data_dir, batch_size, test_prob=0.2):
#     """
#     :param data_dir:
#     :return:
#     """
#     dataset = Embedding_qm_dataset(data_dir)
#     ntest = int(len(dataset) * test_prob)
#     ntrain = int(len(dataset) - (len(dataset) * test_prob))
#     train_set, test_set = torch.utils.data.random_split(dataset, [ntrain, ntest])
#     train_set = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
#     test_set = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
#     return train_set, test_set


# class LoaderQM9:
#     def __init__(self, db_root, db_index, pix_size, grid_dim, sigma,rotate=True):
#         self._db_root = db_root
#         self._db_index = np.load(os.path.join(db_root,db_index))
#         self._pix_size = pix_size
#         self._sigma = sigma
#         self._rotate = rotate
#         self.num_records = self._db_index.shape[0]
#         # generate grid
#         grid_dim = np.asarray(grid_dim)
#         grid = np.stack([-grid_dim // 2, grid_dim - grid_dim // 2], 1)
#         grid = np.mgrid[grid[0, 0]:grid[0, 1], grid[1, 0]:grid[1, 1], grid[2, 0]:grid[2, 1]]
#         grid = torch.tensor(np.asarray(grid, dtype=np.float32))
#         self._grid = grid.cuda()
#         # initialize cursor to count records
#         self.cursor = 1
#
#     def get(self):
#         while True:
#             # load record from the disk
#             name, energy, elem, coord, charge = self._db_index[self.cursor]
#             energy = np.load(os.path.join(self._db_root, energy))
#             elem = np.load(os.path.join(self._db_root, elem))
#             coord = np.load(os.path.join(self._db_root, coord))
#             charge = np.load(os.path.join(self._db_root, charge))
#             # randomly rotate inputs
#             if self._rotate:
#                 coord, _ = geom.affine_tform(coord, shift=[1.5, 1.5, 1.5], rot=None)
#             # convert to torch tensors
#             energy = torch.autograd.Variable(torch.tensor(energy), requires_grad=False).cuda()
#             elem = torch.autograd.Variable(torch.tensor(elem), requires_grad=False).cuda()
#             coord = torch.autograd.Variable(torch.tensor(coord), requires_grad=False).cuda()
#             charge = torch.autograd.Variable(torch.tensor(charge), requires_grad=False).cuda()
#             # build element features
#             elem_feat1 = torch.fmod(elem + 2, 8.).float()
#             elem_feat2 = torch.div(elem + 2, 8.).float()
#             elem_feat3 = torch.ones(elem.size(), dtype=torch.float32).cuda()
#             atm_feat = torch.stack([elem_feat1, elem_feat2, elem_feat3, charge], dim=1)
#             # compute gaussians over a grid
#             scaled_coord = coord / self._pix_size
#             dist = torch.sum(((self._grid[None, :, :, :] - scaled_coord[:, :, None, None, None]) ** 2), 1)
#             gauss_dist = torch.exp(-dist / (2 * (self._sigma ** 2)))
#             gauss_img = torch.sum(gauss_dist[:, None, :, :, :] * atm_feat[:, :, None, None, None], 0)
#             self.cursor = (self.cursor + 1) % self.num_records
#             yield name, energy, elem, coord, charge, gauss_img
#


# class DatasetMolImg(th_Dataset):
#     def __init__(self,db_index,cols, nframes, grid_pix, pix_size, sigmas, input_d, pairwise=False,oncuda=True):
#         self.nframes = nframes
#         self.oncuda = oncuda
#         db_index = pd.read_parquet(db_index,columns=cols)
#         self.elems = [np.asarray(elem) for elem in db_index[cols[0]].to_list()]
#         coords = db_index[cols[1]].to_list()
#         self.coords = [np.stack([np.stack(frame,axis=0) for frame in coord],axis=0) for coord in coords]
#         self.props = db_index[cols[2]].to_list()
#         self.imgGen = ImgGen(grid_pix,pix_size,sigmas,input_d,pairwise)
#
#         if oncuda: self.imgGen.cuda()
#
#     def __len__(self):
#         return len(self.elems)
#
#     def __getitem__(self, idx):
#         feat = mtable_featurizer(self.elems[idx])
#
#         gauss_imgs = []
#         for i in range(self.nframes):
#             coord = torch.tensor(self.coords[idx][i], dtype=torch.float32)
#             feat = torch.tensor(feat,dtype=torch.float32)
#             if self.oncuda:
#                 coord = coord.cuda()
#                 feat = feat.cuda()
#             gauss_img = self.imgGen(coord,feat)
#             gauss_imgs.append(gauss_img)
#         gauss_imgs = torch.stack(gauss_imgs,dim=0)
#
#         return gauss_imgs,self.props[idx]


def onehot(arr,num_classes,dtype=np.int):
    arr = np.asarray(arr,dtype=np.int)
    assert len(arr.shape) ==1, "dims other than 1 not implemented"
    onehot_arr = np.zeros(arr.shape + (num_classes,),dtype=dtype)
    onehot_arr[np.arange(arr.shape[0]), arr] = 1
    return onehot_arr

def mpnn_feat(mol, ifcoord=True, panda_fmt=False):
    atomtypes = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
    bondtypes = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

    natm = len(mol.GetAtoms())
    # featurize elements
    atmfeat = pd.DataFrame(index=range(natm),columns=["type_idx", "atomic_number", "acceptor", "donor", "aromatic",
                                                      "sp", "sp2", "sp3", "num_hs"])

    # featurize
    fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
    for i,atom in enumerate(mol.GetAtoms()):
        type_idx = atomtypes.get(atom.GetSymbol(),5)
        atmfeat["type_idx"][i] = onehot([type_idx], num_classes=len(atomtypes) + 1)[0]
        atmfeat["atomic_number"][i] = atom.GetAtomicNum()
        atmfeat["aromatic"][i] = 1 if atom.GetIsAromatic() else 0
        hybridization = atom.GetHybridization()
        atmfeat["sp"][i] = 1 if hybridization == HybridizationType.SP else 0
        atmfeat["sp2"][i] = 1 if hybridization == HybridizationType.SP2 else 0
        atmfeat["sp3"][i] = 1 if hybridization == HybridizationType.SP3 else 0
        atmfeat["num_hs"][i] = atom.GetTotalNumHs(includeNeighbors=True)

    # get donors and acceptors
    atmfeat["acceptor"].values[:] = 0
    atmfeat["donor"].values[:] = 0
    feats = factory.GetFeaturesForMol(mol)
    for j in range(0, len(feats)):
         if feats[j].GetFamily() == 'Donor':
             node_list = feats[j].GetAtomIds()
             for k in node_list:
                 atmfeat["donor"][k] = 1
         elif feats[j].GetFamily() == 'Acceptor':
             node_list = feats[j].GetAtomIds()
             for k in node_list:
                 atmfeat["acceptor"][k] = 1
    # get coord
    if ifcoord:
        coord = np.asarray([mol.GetConformer(0).GetAtomPosition(j) for j in range(natm)])
    else:
        coord = None
    # get bonds and bond features
    bond = np.asarray([[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in mol.GetBonds()])
    bondfeat = [bondtypes[bond.GetBondType()] for bond in mol.GetBonds()]
    bondfeat = onehot(bondfeat,num_classes=len(bondtypes))

    # convert atmfeat to numpy matrix
    if not panda_fmt:
        type_idx = np.stack(atmfeat["type_idx"].values,axis=0)
        atmfeat = atmfeat[["atomic_number", "acceptor", "donor", "aromatic", "sp", "sp2", "sp3","num_hs"]]
        atmfeat = np.concatenate([type_idx, atmfeat.to_numpy(dtype=np.int)],axis=1)
    return atmfeat, coord, bond, bondfeat

def _mol_to_graph(atmfeat, coord, bond, bondfeat, props={}):
    "convert to PyTorch geometric module"
    natm = atmfeat.shape[0]
    # transform to torch_geometric bond format; send edges both ways; sort bonds
    atmfeat = torch.tensor(atmfeat, dtype=torch.float32)
    edge_index = torch.tensor(np.concatenate([bond.T, np.flipud(bond.T)],axis=1),dtype=torch.int64)
    edge_attr = torch.tensor(np.concatenate([bondfeat,bondfeat], axis=0),dtype=torch.float32)
    edge_index, edge_attr = coalesce(edge_index, edge_attr, natm, natm)
    # make torch data
    if coord is not None:
        coord = torch.tensor(coord,dtype=torch.float32)
        data = Data(x=atmfeat, pos=coord, edge_index=edge_index, edge_attr=edge_attr, **props)
    else:
        data = Data(x=atmfeat, edge_index=edge_index, edge_attr=edge_attr, **props)
    return data

def mol_to_graph(smiles, props={}, num_conf=1, noh=True, feat="mpnn"):
    "mol to graph convertor"
    mol,_,_ = LambdaZero.chem.build_mol(smiles, num_conf=num_conf, noh=noh)
    if feat == "mpnn":
        atmfeat, coord, bond, bondfeat = mpnn_feat(mol)
    else:
        raise NotImplementedError(feat)
    graph = _mol_to_graph(atmfeat, coord, bond, bondfeat, props)
    return graph


@ray.remote
def _brutal_dock_proc(smi, props, pre_filter, pre_transform):
    try:
        graph = mol_to_graph(smi, props)
    except Exception as e:
        return None
    if pre_filter is not None and not pre_filter(graph):
        return None
    if pre_transform is not None:
        graph = pre_transform(graph)
    return graph

class BrutalDock(InMemoryDataset):
    # own internal dataset
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None,
                 props=["gridscore"], file_names=['ampc_100k'], proc_func=_brutal_dock_proc):
        self._props = props
        self.file_names = file_names
        self.proc_func = proc_func
        super(BrutalDock, self).__init__(root, transform, pre_transform, pre_filter)

        #  todo: store a list, but have a custom collate function on batch making
        graphs = []
        for processed_path in self.processed_paths:
            self.data, self.slices = torch.load(processed_path)
            graphs += [self.get(i) for i in range(len(self))]
        if len(graphs) > 0:
            self.data, self.slices = self.collate(graphs)

    @property
    def raw_file_names(self):
        return [file_name + ".feather" for file_name in self.file_names]

    @property
    def processed_file_names(self):
        return [file_name + ".pt" for file_name in self.file_names]

    def download(self):
        pass

    def process(self):       
        print("processing", self.raw_paths)
        for raw_path, processed_path in zip(self.raw_paths, self.processed_paths):
            docked_index = pd.read_feather(raw_path)
            smis = docked_index["smiles"].tolist()
            props = {pr:docked_index[pr].tolist() for pr in self._props}
            tasks = [self.proc_func.remote(smis[j], {pr: props[pr][j] for pr in props},
                                           self.pre_filter, self.pre_transform) for j in range(len(smis))]
            graphs = ray.get(tasks)
            graphs = [g for g in graphs if g is not None]
            # save to the disk
            torch.save(self.collate(graphs), processed_path)




#if __name__ == "__main__":
#    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..', 'data', 'QM9')
