import torch
from torch_geometric.data import InMemoryDataset, Data

import os
import pandas as pd
from tqdm import tqdm

from LambdaZero.chem import DockVina_smi


# TODO: class can be launched with different docker_kwargs, but only one set persists - single raw file - make a warning if newly supplied docker_kwargs mismatch
#  more less the same mechanism as torch_geometric uses for pre_transform and pre_filter
class DockedDataset(InMemoryDataset):
    def __init__(self, root, file_name, alias, transform=None, pre_transform=None, pre_filter=None, **docker_kwargs):
        self.file_name_no_ext = file_name.split('.')[0]
        self.alias = alias
        self.docker_kwargs = docker_kwargs

        if pre_transform is None:
            pre_transform = DockedDataset.get_best_conformation_minimal

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self):
        # extend default behaviour to go extra directory in depth
        # this allows to have multiple processed sets for the same raw file
        return os.path.join(super().processed_dir, self.alias)

    @property
    def raw_file_names(self):
        return f"{self.file_name_no_ext}.pth"

    @property
    def processed_file_names(self):
        return f"{self.file_name_no_ext}.pth"

    def construct_graphs(self):
        """Construct graph (Data object) as a collection of provided attributes and results of docking"""
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
                _, mol_name, dockscores, free_pose, docked_poses = docker.dock(entry['smiles'])
                graph['mol_name'] = mol_name
                graph['dockscores'] = torch.from_numpy(dockscores)
                graph['free_pose'] = torch.from_numpy(free_pose)
                graph['docked_poses'] = torch.from_numpy(docked_poses)
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
    def get_best_score_only_minimal(raw_graph):
        graph = Data()
        graph.idx = raw_graph.idx
        graph.y = raw_graph.dockscores[[0]]  # keep batch dimension, there would be error on collation otherwise
        graph.smiles = raw_graph.smiles
        return graph

    @staticmethod
    def get_best_conformation_minimal(raw_graph):
        graph = Data()
        graph.idx = raw_graph.idx
        graph.y = raw_graph.dockscores[[0]]  # keep batch dimension
        graph.pos = raw_graph.free_pose
        graph.docked_pos = raw_graph.docked_poses[0]
        graph.smiles = raw_graph.smiles
        return graph

    @staticmethod
    def get_all_docked_conformations_minimal(raw_graph):
        graphs_list = []
        for dockscore, pos in zip(raw_graph.dockscores, raw_graph.docked_poses):
            graph = Data()
            graph.idx = raw_graph.idx
            graph.y = [dockscore]  # restore batch dimension
            graph.pos = pos
            graph.smiles = raw_graph.smiles
            graphs_list.append(graph)
        return graphs_list
