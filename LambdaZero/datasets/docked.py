import torch
from torch_geometric.data import InMemoryDataset, Data

import os
import pandas as pd
from tqdm import tqdm

from LambdaZero.chem import DockVina_smi


class DockedDataset(InMemoryDataset):
    def __init__(self, root, file_name, alias, transform=None, pre_transform=None, pre_filter=None, docker_kwargs=None):
        self.file_name_no_ext = file_name.split('.')[0]
        self.alias = alias
        if pre_transform is None:
            pre_transform = DockedDataset.get_best_conformation
        self.docker_kwargs = docker_kwargs if docker_kwargs is not None else {'mode': "all_conf"}

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return f"{self.file_name_no_ext}.pth"

    @property
    def processed_file_names(self):
        return f"{self.file_name_no_ext}_{self.alias}.pth"

    def construct_graphs(self):
        graphs_list = []
        docker = DockVina_smi(outpath=self.root, **self.docker_kwargs)
        input_csv_file = os.path.join(self.root, f"{self.file_name_no_ext}.csv")
        data = pd.read_csv(input_csv_file)  # csv file is expected to be comma-separated
        for idx, entry in tqdm(data.iterrows(), total=len(data)):
            try:
                # pass over extra info from input file
                graph = Data(**entry)
                graph['idx'] = idx
                # dock molecule and append obtained info
                mol_name, dockscores, original_pos, docked_pos = docker.dock(entry['smiles'])
                graph['mol_name'] = mol_name
                graph['dockscores'] = torch.from_numpy(dockscores)
                # flatten dimensions, so that custom collate is not needed
                # those dimensions can be restored on get, since we can infer dimensions from dockscores and original_pos shapes
                graph['original_pos'] = torch.from_numpy(original_pos)
                graph['docked_pos'] = torch.from_numpy(docked_pos)
                graphs_list.append(graph)
            except Exception as e:
                # Docking can fail for some smiles, whether it does is mostly inconsequential though.
                print(entry['smiles'], e)
        torch.save(graphs_list, self.raw_paths[0])

    def process(self):
        raw_path = self.raw_paths[0]
        if not os.path.isfile(raw_path):
            print("Constructing graphs from csv file with smiles.")
            self.construct_graphs()
            print("Finished graphs construction!")

        print("Processing", self.raw_paths[0])
        graphs_list = torch.load(self.raw_paths[0])

        if self.pre_filter is not None:
            graphs_list = [graph for graph in graphs_list if self.pre_filter(graph)]

        if self.pre_transform is not None:
            graphs_list = [self.pre_transform(graph) for graph in graphs_list]

        # pre_transform can split each graph into a collection of graphs - flatten
        if isinstance(graphs_list[0], list):
            graphs_list = [graph for graphs_sublist in graphs_list for graph in graphs_sublist]

        torch.save(self.collate(graphs_list), self.processed_paths[0])

    @staticmethod
    def get_best_score_only(graph):
        graph.dockscore = graph.dockscores[0]
        graph.docked_pos = None
        graph.original_pos = None
        graph.dockscores = None
        return graph

    @staticmethod
    def get_best_conformation(graph):
        graph.dockscore = graph.dockscores[0]
        graph.docked_pos = graph.docked_pos[0]
        graph.pos = graph.original_pos
        graph.dockscores = None
        return graph

    @staticmethod
    def get_all_conformations(graph):
        graphs_list = []
        for dockscore, pos in zip(graph.dockscores, graph.docked_pos):
            tmp_graph = graph.clone()
            tmp_graph.pos = pos
            tmp_graph.docked_pos = pos
            tmp_graph.dockscore = dockscore
            tmp_graph.dockscores = None
            graphs_list.append(tmp_graph)
        return graphs_list
