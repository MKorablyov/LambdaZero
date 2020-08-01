from LambdaZero.examples.drug_comb.drug_combdb_data import DrugCombDb
from LambdaZero.examples.drug_comb.model.subgraph_embedding_model import SubgraphEmbeddingRegressorModel
from LambdaZero.examples.drug_comb import transforms
from LambdaZero.utils import get_external_dirs, MeanVarianceNormalizer
from torch_geometric.data import Batch
from torchvision.transforms import Compose
from torch.utils.data import TensorDataset, DataLoader
from ray import tune
import torch.nn.functional as F
import ray
import torch
import os
import math
import time

def random_split(num_examples, test_prob, valid_prob):
    nvalid = int(num_examples * valid_prob)
    ntest = int(num_examples * test_prob)
    idx = torch.randperm(num_examples)

    train_idx = idx[ntest + nvalid:]
    val_idx = idx[:nvalid]
    test_idx = idx[:ntest]

    return train_idx, val_idx, test_idx

def train_epoch(ddi_graph, train_idxs, model, optimizer, normalizer, device, config):
    import pdb; pdb.set_trace()
    subgraph_dataset = TensorDataset(ddi_graph.edge_index[:, train_idxs].T,
                                     ddi_graph.edge_attr[train_idxs],
                                     ddi_graph.edge_classes[train_idxs],
                                     ddi_graph.y[train_idxs])

    loader = DataLoader(subgraph_dataset,
                        batch_size=config["batch_size"],
                        pin_memory=device == 'cpu',
                        shuffle=True)
    model.train()

    epoch_targets = []
    epoch_preds = []
    metrics = {"loss": 0, "mse": 0, "mae": 0}
    for i, drug_drug_batch in enumerate(loader):
        optimizer.zero_grad()

        drug_drug_index, edge_attr, edge_classes, y = drug_drug_batch
        drug_drug_index = drug_drug_index.T

        # Here, sg is shorthand for subgraph.  See documentation of
        # get_batch_subgraph_edges for more information on sg_edge_index,
        # sg_nodes, and sg_avging_idx
        sg_edge_index, sg_nodes, sg_avging_idx = get_batch_subgraph_edges(ddi_graph, drug_drug_index)

        preds = model(ddi_graph.protein_ftrs, drug_drug_index, edge_attr, edge_classes,
                      sg_edge_index, sg_nodes, sg_avging_idx)
        loss = F.mse_loss(normalizer.forward_transform(y), preds)

        loss.backward()
        optimizer.step()

        metrics["loss"] += loss.item() * drug_drug_index.shape[1]
        epoch_targets.append(y.detach().cpu().numpy())
        epoch_preds.append(normalizer.backward_transform(preds).detach().cpu().numpy())

    epoch_targets = np.concatenate(epoch_targets,0)
    epoch_preds = np.concatenate(epoch_preds, 0)
    metrics["loss"] = metrics["loss"] / epoch_targets.shape[0]
    metrics["mae"] = np.abs(epoch_targets - epoch_preds).mean()
    metrics["mse"] = ((epoch_targets - epoch_preds)**2).mean()

    return metrics

def eval_epoch(ddi_graph, eval_idxs, model, normalizer, device, config):
    subgraph_dataset = TensorDataset(ddi_graph.edge_index[:, eval_idxs].T,
                                     ddi_graph.edge_attr[eval_idxs],
                                     ddi_graph.edge_classes[eval_idxs],
                                     ddi_graph.y[eval_idxs])

    loader = DataLoader(subgraph_dataset, batch_size=config["batch_size"], pin_memory=device == 'cpu')
    model.eval()

    epoch_targets = []
    epoch_preds = []
    metrics = {"loss": 0, "mse": 0, "mae": 0}
    for drug_drug_batch in loader:
        drug_drug_index, edge_attr, edge_classes, y = drug_drug_batch
        drug_drug_index = drug_drug_index.T

        # Here, sg is shorthand for subgraph.  See documentation of
        # get_batch_subgraph_edges for more information on sg_edge_index,
        # sg_nodes, and sg_avging_idx
        sg_edge_index, sg_nodes, sg_avging_idx = get_batch_subgraph_edges(ddi_graph, drug_drug_index)

        preds = model(ddi_graph.protein_ftrs, drug_drug_index, edge_attr, edge_classes,
                      sg_edge_index, sg_nodes, sg_avging_idx)
        loss = F.mse_loss(normalizer.forward_transform(y), preds)

        metrics["loss"] += loss.item() * drug_drug_index.shape[1]
        epoch_targets.append(y.detach().cpu().numpy())
        epoch_preds.append(normalizer.backward_transform(preds).detach().cpu().numpy())

    epoch_targets = np.concatenate(epoch_targets,0)
    epoch_preds = np.concatenate(epoch_preds, 0)
    metrics["loss"] = metrics["loss"] / epoch_targets.shape[0]
    metrics["mae"] = np.abs(epoch_targets - epoch_preds).mean()
    metrics["mse"] = ((epoch_targets - epoch_preds)**2).mean()

    return metrics

def get_batch_subgraph_edges(ddi_graph, batch_drug_drug_index):
    """Gets values to enable averaging over node embeddings.

    These docs follow the notation conventions of docs in drug_combdb_data.py.

    Arguments
    ---------
    ddi_graph : Data
        The drug-drug graph.  Note that it is formed according to the pre_transform
        to_drug_induced_subgraphs such that each drug in the ddi_graph has its own
        protein subgraph.  For more details, refer to the documentation of
        to_drug_induced_subgraphs.
    batch_drug_drug_index : torch.LongTensor
        A tensor representing the drug-drug edges for this batch.  The tensor
        is of shape (2, num_drug_drug_edges_in_batch).

    Returns
    -------
    edge_index : torch.LongTensor
        The edge index for all the edges in the subgraphs of the batch's drugs.
        Write the set of edges in drug_drug_batch as :math:`B`.  Then write
        the set of drugs in :math:`B` as :math:`D_B = {d | (d, d') or (d', d) \in B}`.
        Following the notation of the docs for the method _build_subgraph in
        transforms.py, edge_index is the union of the edges of the graphs in
        :math:`M_B`, that is, :math:`edge_index = \bigcup_{d \in D_B} E_d`.
        Thus, edge_index is a tensor of shape (2, :math:`|\bigcup_{d \in D_B} E_d|`).
    nodes : torch.LongTensor
        The indices of the protein nodes in each of the batch's drugs subgraphs.
        Note that these indices are specified relative to the global ordering
        of protein nodes as returned by the construction of the DrugCombDb object.
        We follow the notation adopted in the explanation of edge_index here.
        nodes can be represented by the following set allowing duplicates as
        :math:`nodes = \bigcup_{d \in D_B} V_d`.  Thus, nodes is of shape
        (:math:`|\bigcup_{d \in D_B} V_d|`,).  Note that the ordering of the protein
        indices in this tensor are in order and relative to the drug's index. E.g.,
        if :math:`D_B = {1, 4, 7}` and :math:`V_1 = {10, 11, 17}, V_4 = {8, 11},
        V_7 = {9, 10, 14}` (where :math:`V_i` is the node set for drug i's subgraph),
        then nodes will be the tensor:
            torch.tensor([10, 11, 17, 8, 11, 9, 10, 14])

        That is, nodes will be the concatenation of :math:`V_1, V_2, V_3`.
    averaging_idx : torch.LongTensor
        The indices in :math:`D_B` that the items of nodes belong to.
        The tensor is then the same shape as nodes,
        (:math:`|\bigcup_{d \in D_B} V_d|`,).  In the nodes example, drug 1
        is at index 0 of :math:`D_B`, drug 4 is at index 1, and drug 7 is at
        index 2.  Note that the first three nodes in the nodes example
        above are for drug 1, the next two are for drug 4, and the final
        three are for drug 7.  In this case, averaging_idx would be the tensor:
            torch.tensor([0, 0, 0, 1, 1, 2, 2, 2])

        averaging_idx is used by the method scatter_mean to do efficient
        protein node embedding averaging when obtaining graph embeddings.
    """
    batch_drugs = torch.unique(batch_drug_drug_index.flatten())
    subgraph_edge_indices = tuple([
        ddi_graph.drug_idx_to_graph[drug.item()].edge_index for drug in batch_drugs
    ])

    edge_index = torch.cat(subgraph_edge_indices, dim=1)
    edge_index = torch.unique(edge_index, dim=1, sorted=False)

    subgraph_nodes = tuple([ddi_graph.drug_idx_to_graph[drug.item()].nodes for drug in batch_drugs])
    nodes = torch.cat(subgraph_nodes)

    averaging_idx_tensors = tuple([
        torch.full((len(nodes),), i, dtype=torch.long) for i, nodes in enumerate(subgraph_nodes)
    ])

    averaging_idx = torch.cat(averaging_idx_tensors).to(batch_drugs.device)

    return edge_index, nodes, averaging_idx

class SubgraphRegressor(tune.Trainable):
    """A regressor class that plays nicely with ray.tune.

    Attributes
    ----------
    config : Dict
        A dictionary holding the various configurations for the model.
    device : {'cpu', 'cuda'}
        The device that the model will operate on.
    ddi_graph : Data
        The drug-drug graph.  Note that it is formed according to the pre_transform
        to_drug_induced_subgraphs such that each drug in the ddi_graph has its own
        protein subgraph.  For more details, refer to the documentation of
        to_drug_induced_subgraphs.
    train_idxs : torch.tensor
        The indices of ddi_graph.edge_index representing training edges.
    val_idxs : torch.tensor
        The indices of ddi_graph.edge_index representing validation edges.
    test_idxs : torch.tensor
        The indices of ddi_graph.edge_index representing test edges.
    model : SubgraphEmbeddingRegressorModel
        The model to be used for prediction.  Refer to documentation of the
        SubgraphEmbeddingRegressorModel class for more information.
    optim : torch.optim.Adam
        The optimizer to be user in the train step.
    normalizer : MeanVarianceNormalizer
        A normalizer to be applied during loss computation to properly scale
        outputs.
    train_epoch : Function
        The method containing logic for the training step.
    eval_epoch : Function
        The method containing logic for the validation step.
    """
    def _setup(self, config):
        """Sets up the regressor.

        Note that config contains all the arguments, so we detail the
        keys of config in the Arguments section.

        Arguments
        ---------
        score_type : {'hsa', 'zip', 'bliss', 'loewe'}
            The score type to treat as the target for the model.
        use_one_hot : bool
            If True, use one-hot encodings for the protein features.  If False, use
            learnable embeddings for the protein features.
        protein_embedding_size : int
            The size of the learnable protein embedding.
        use_single_cell_line : bool
            If True then only include drug-drug edges for the most populous in the
            dataset.  If False, include all edges.
        num_edges_to_take : int
            If -1, keep all edges in the dataset. If positive, then keep
            the first num_edges_to_take edges in the dataset and throw the rest
            away.
        test_set_prop : float (inclusive interval [0.0, 1.0])
            The proportion of edges to be used in the test set.
        val_set_prop : float (inclusive interval [0.0, 1.0])
            The proportion of edges to be used in the validation set.
        lr : float
            The learning rate to use in optimization.
        train_epoch : Function
            The method to use for doing a train step over an epoch.
        eval_epoch : Function
            The method to use for doing a validation step over an epoch.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dataset = DrugCombDb(transform=self._get_transform(config),
                             pre_transform=transforms.to_drug_induced_subgraphs,
                             prot_ftr_type='laplacian',
                             target_type='response')
        self.ddi_graph = self._extract_ddi_graph(dataset)

        self.train_idxs, self.val_idxs, self.test_idxs = \
            dataset.random_split(config["test_set_prop"], config["val_set_prop"])

        config["in_channels"] = \
            self.ddi_graph.x.shape[1] if config['use_one_hot'] else config["protein_embedding_size"]
        config["out_channels"] = self.ddi_graph.y.shape[1] if len(self.ddi_graph.y.shape) > 1 else 1

        self.model = SubgraphEmbeddingRegressorModel(
            config, torch.unique(self.ddi_graph.edge_classes)).to(self.device)

        to_optimize = list(self.model.parameters())
        if not config['use_one_hot']:
            to_optimize.append(self.ddi_graph.protein_ftrs)

        self.optim = torch.optim.Adam(to_optimize, lr=config["lr"])

        float_y = self.ddi_graph.y.to(dtype=torch.float)
        self.normalizer = MeanVarianceNormalizer((float_y.mean(), float_y.var()))

        self.train_epoch = config["train_epoch"]
        self.eval_epoch = config["eval_epoch"]

    def _get_transform(self, config):
        """
        Gets a transform object to be applied to items of the dataset on indexing.
        Note that config holds all the arguments so we describe the keys of the dict
        here.

        Arguments
        ---------
        score_type : {'hsa', 'zip', 'bliss', 'loewe'}
            The score type to treat as the target for the model.
        use_one_hot : bool
            If True, use one-hot encodings for the protein features.  If False, use
            learnable embeddings for the protein features.
        protein_embedding_size : int
            The size of the learnable protein embedding.
        use_single_cell_line : bool
            If True then only include drug-drug edges for the most populous in the
            dataset.  If False, include all edges.
        num_edges_to_take : int
            If -1, keep all edges in the dataset. If positive, then keep
            the first num_edges_to_take edges in the dataset and throw the rest
            away.

        Returns
        -------
        torchvision.transforms.Compose
            A composition of transforms according to arguments in config.
        """
        transform_list = [transforms.use_score_type_as_target(config['score_type'])]
        if not config['use_one_hot']:
            transform_list.append(
                transforms.subgraph_protein_features_to_embedding(
                    config['protein_embedding_size'], self.device)
            )

        if config['use_single_cell_line']:
            transform_list.append(transforms.use_single_cell_line)

        if config['num_edges_to_take'] != -1:
            transform_list.append(transforms.train_first_k_edges(config['num_edges_to_take']))

        return Compose(transform_list)

    def _extract_ddi_graph(self, dataset):
        """
        The DrugCombDb dataset has only one item in it (i.e., len(DrugCombDb())
        will always return 1), so this method extracts its only graph and moves
        that graph to the device represented by self.device

        Parameters
        ----------
        dataset : DrugCombDb
            The dataset from which to return the single graph from.

        Returns
        -------
        Data
            The single graph held within the dataset object moved to self.device.
        """
        ddi_graph = dataset[0].to(self.device)
        ddi_graph.protein_ftrs = ddi_graph.protein_ftrs.to(self.device)
        for drug, subgraph in ddi_graph.drug_idx_to_graph.items():
            new_subgraph = subgraph.to(self.device)
            if hasattr(new_subgraph, 'nodes'):
                new_subgraph.nodes = subgraph.nodes.to(self.device)

            ddi_graph.drug_idx_to_graph[drug] = new_subgraph

        return ddi_graph

    def _train(self):
        train_scores = self.train_epoch(self.ddi_graph, self.train_idxs, self.model,
                                        self.optim, self.normalizer, self.device, self.config)
        eval_scores = self.eval_epoch(self.ddi_graph, self.val_idxs, self.model,
                                      self.normalizer, self.device, self.config)

        train_scores = [("train_" + k, v) for k, v in train_scores.items()]
        eval_scores = [("eval_" + k, v) for k, v in eval_scores.items()]
        scores = dict(train_scores + eval_scores)
        return scores

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))

# For documentation of the config refer to the docstrings of
# SubgraphRegressor._setup and SubgraphRegressor.
_, _, summaries_dir =  get_external_dirs()
config = {
    "trainer": SubgraphRegressor,
    "trainer_config": {
        "val_set_prop": 0.2,
        "test_set_prop": 0.0,
        "prediction_type": "mlp",#tune.grid_search(["dot_product", "mlp"]),
        "lr": 1e-4,#tune.grid_search([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 5e-1, 5e-2, 5e-3, 5e-4, 5e-5]),
        "use_one_hot": False,#tune.grid_search([True, False]),
        "score_type": 'hsa',
        "weight_initialization_type": "torch_base",
        "protein_embedding_size": 256,
        "train_epoch": train_epoch,
        "eval_epoch": eval_epoch,
        "embed_channels": 256,
        "regressor_hidden_channels": 64,
        "num_epochs": 256,
        "batch_size": 64,
        "conv_dropout_rate": 0.1,
        "linear_dropout_rate": 0.2,
        "skip_gcn": False,#tune.grid_search([False]),
        "use_single_cell_line": True,
        "use_gat": False,
        "num_heads": 1,
        "num_edges_to_take": 1000,
    },
    "summaries_dir": summaries_dir,
    "memory": 20 * 10 ** 9,
    "checkpoint_freq": 200,
    "stop": {"training_iteration": 100},
    "checkpoint_at_end": True,
    "resources_per_trial": {"gpu": 1},
    "name": None
}

if __name__ == "__main__":
    ray.init()

    # Ray has a bug where a race condition can occur and ray may not pick up
    # the available resources on a node and subsequently throw errors saying
    # that insufficient resources are available.  Sleeping for a couple seconds
    # gets around this error, so sleep for a bit here.
    time_to_sleep = 5
    print("Sleeping for %d seconds to appease ray" % time_to_sleep)
    time.sleep(time_to_sleep)
    print("Woke up.. Scheduling")

    config["name"] = "TestFullSingleCellCorr"
    analysis = tune.run(
        config["trainer"],
        name=config["name"],
        config=config["trainer_config"],
        stop=config["stop"],
        resources_per_trial=config["resources_per_trial"],
        num_samples=1,
        checkpoint_at_end=config["checkpoint_at_end"],
        local_dir=config["summaries_dir"],
        checkpoint_freq=config["checkpoint_freq"]
    )

