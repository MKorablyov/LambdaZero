
# LambdaZero/datasets/drug_combo
# merge drug comb db

from LambdaZero.utils import get_external_dirs
datasets_dir, programs_dir, summaries_dir = get_external_dirs()

# def downloadDrugCombDB(path)
# pass

# _load_drugs(drug_combdb_path)

# _load_proteins():
    # return bunch of lists

# _load_ppi():
    # return np.array([num_interactions, 2])

# _load_ddi():

# @ray.remote
# def _get_fp(smiles)
#   return fp


# _chop_into_small_graphs(giant_graph):
    # slice big graph into g
    # [torch_geometric.Data(g) for g in list_of_small_graphs]
    # data, slices= torh.geometric.collate(graphs)


# DrugCombDB(pre_transform=None, pre_transform=chop_into_small_graphs)

    # init ()

    #   data, slices = torch.load(processed_path)


    # def download()
    #   download(dataset_dir)

    # def process()
        # load everything
        # use ray remote
        # LambdaZero.chem.getFP(mol)

        # build_giant_graph

        # self.data, self.slices = self.pre_transform(giant_graph) # todo: check how pytorch geometric does slices



# dataset = DrugCombDB()
# fixme -- there are 120 cell types (each cell type is a side effect)
# Zi Dr R Dr Zj

