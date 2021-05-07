"""
Training a Classifier
"""
import torch
import numpy as np
import torch.optim as optim
from torch_geometric.data import Batch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import wandb
from argparse import Namespace
import os

from LambdaZero.environments import BlockMolEnvGraph_v1
from LambdaZero.models.model_atom import MPNNet_v2
from LambdaZero.models.model_egnn import EGNNetRLBO
import LambdaZero

datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()


class GraphNet(nn.Module):
    def __init__(self, model_class, env, device, dim=256, levels=6):
        super().__init__()
        env_space = env.observation_space["mol_graph"]

        model = model_class(num_feat=env_space.num_node_feat,
                            dim=dim, num_out_per_stem=env.num_blocks, levels=levels)
        model = model.to(device)
        self.space = env.observation_space["mol_graph"]
        self.model = model
        self.device = device

    def forward(self, data):
        data = data.to(self.device)
        scalar_outs, data = self.model(data)
        stop_logit = scalar_outs[:, 1:2]
        break_logits = data.jbond_preds.reshape((data.num_graphs, -1))
        add_logits = data.stem_preds.reshape((data.num_graphs, -1))

        actor_logits = torch.cat([stop_logit, break_logits, add_logits], 1)

        return actor_logits


class CustomDataset(Dataset):
    def __init__(self, graphs, targets, classification=True):
        self._graphs = graphs

        if classification:
            targets = torch.argmax(targets, dim=1)
        else:
            targets = targets * 3

        self._targets = targets

    def __getitem__(self, idx):
        sample = {'graph': self._graphs[idx], 'target': self._targets[idx]}
        return sample

    def __len__(self):
        return len(self._graphs)


def collate_fn(data):
    """ Used in dataloader to merge batch """
    _graphs = Batch.from_data_list([x["graph"] for x in data])
    _tgt = torch.stack([x["target"] for x in data])
    return _graphs, _tgt


def train_epoch(ep, loader, model, optimizer, criterion, device, classification=True):
    model.train()

    correct = 0
    total = 0

    running_loss = 0.0

    do_res = (not classification) and (ep + 1) % 50 == 0
    res = dict() if not do_res else dict({k: [] for k in [3, 5, 10, 15]})

    for bidx, (data_graph, data_tgt) in enumerate(loader):
        data_graph, data_tgt = data_graph.to(device), data_tgt.to(device)
        optimizer.zero_grad()
        y_hat = model(data_graph)
        loss = criterion(y_hat, data_tgt)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if not classification:
            if do_res:
                # rank
                tgt_sort = torch.argsort(data_tgt, axis=1)
                pred_sort = torch.argsort(y_hat.data, axis=1)
                for topk in res.keys():
                    for irow in range(len(tgt_sort)):
                        tgt_set = set(tgt_sort[irow][:topk].data.cpu().numpy())
                        pred_set = set(pred_sort[irow][:topk].data.cpu().numpy())
                        match = len(set.intersection(tgt_set, pred_set)) /float(topk)
                        res[topk].append(match)

            data_tgt = torch.argmax(data_tgt, dim=1)

        _, predicted = torch.max(y_hat.data, 1)
        correct += (predicted == data_tgt).sum().item()

        total += data_tgt.size(0)

    if do_res:
        for topk in list(res.keys()):
            res[str(topk)] = np.mean(res.pop(topk))

    return correct / total, running_loss / total, res


def eval_epoch(loader, model, criterion, device, classification=True):
    model.eval()

    correct = 0
    total = 0

    res = dict() if classification else dict({k: [] for k in [3, 5, 10, 15]})

    running_loss = 0.0
    with torch.no_grad():
        for bidx, (data_graph, data_tgt) in enumerate(loader):
            data_graph, data_tgt = data_graph.to(device), data_tgt.to(device)
            y_hat = model(data_graph)
            loss = criterion(y_hat, data_tgt)
            running_loss += loss.item()

            if not classification:
                tgt_sort = torch.argsort(data_tgt, axis=1)
                pred_sort = torch.argsort(y_hat.data, axis=1)
                for topk in res.keys():
                    for irow in range(len(tgt_sort)):
                        tgt_set = set(tgt_sort[irow][:topk].data.cpu().numpy())
                        pred_set = set(pred_sort[irow][:topk].data.cpu().numpy())
                        match = len(set.intersection(tgt_set, pred_set)) /float(topk)
                        res[topk].append(match)

                data_tgt = torch.argmax(data_tgt, dim=1)

            _, predicted = torch.max(y_hat.data, 1)
            total += data_tgt.size(0)
            correct += (predicted == data_tgt).sum().item()

    if not classification:
        for topk in list(res.keys()):
            res[f"e_{topk}"] = np.mean(res.pop(topk))

    return correct / total, running_loss / total, res


def get_data(device):
    # Load env_data
    env_data = np.load(f"{datasets_dir}/eval_env_data.npy", allow_pickle=True).item()

    # Get unpacking method
    env = BlockMolEnvGraph_v1({"random_steps": 3, "allow_removal": True, "max_steps": 1})
    env_space = env.observation_space["mol_graph"]
    _unpack = env_space.unpack

    # Construct data
    _inputs, _targets = [], []
    for smi, smi_data in env_data.items():
        x = _unpack(smi_data["obs"]["mol_graph"].astype(np.uint8))
        act_mask = smi_data["obs"]["action_mask"].astype(np.bool)
        act_r_df = smi_data["next"]["true_r"]
        tgt = torch.zeros(act_mask.shape)
        # tgt[~act_mask] = -20  # as per logits set
        tgt[act_r_df.index.values] = torch.from_numpy(act_r_df.values).float()

        _inputs.append(x)
        _targets.append(tgt)

    # Process Data
    _inputs = [x.to(device) for x in _inputs]

    # Calculate Targets (Argmax target)
    _targets = torch.stack(_targets)
    _targets = _targets.to(device)
    return _inputs, _targets, env


if __name__ == "__main__":
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    args = Namespace(
        batch_size=256,
        train_split=0.8,
        num_epochs=100000,
        device=_device,
        hidden_size=128,
        levels=12,
        eval_freq=50,
        net="EGNNetRLBO",  #"MPNNet_v2", # "EGNNetRLBO"
        classification=False,
        seed=13
    )
    device = torch.device(_device)
    classification = args.classification
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ==============================================================================================
    # Load Data
    _inputs, _targets, env = get_data(device)
    args.dataset_size = len(_inputs)

    # Create DataLoaders
    rnd_state = np.random.RandomState(args.seed)
    rand_idxs = rnd_state.permutation(np.arange(len(_inputs)))
    train_idx = rand_idxs[:round(len(_inputs) * args.train_split)]
    test_idx = rand_idxs[len(train_idx):]
    args.train_size, args.test_size = len(train_idx), len(test_idx)

    def get_idxs(idxs): return [_inputs[i] for i in idxs], _targets[idxs]

    train_set = CustomDataset(*get_idxs(train_idx), classification=classification)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    val_set = CustomDataset(*get_idxs(test_idx), classification=classification)
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

    print(f"Running on | Train set size: {len(train_idx)} | Eval set size: {len(test_idx)} ")
    # ==============================================================================================
    # Load model

    model = GraphNet(globals()[args.net], env, device, dim=args.hidden_size, levels=args.levels)
    model = model.to(device)
    print(model)

    # ==============================================================================================
    # Logger
    with open(f"{summaries_dir}/wandb_key") as f:
        api_key = f.readlines()[0].strip()
    os.environ["WANDB_API_KEY"] = api_key
    wandb.init(project='lztest')
    wandb.config.update(args.__dict__)

    # ==============================================================================================
    # Train
    criterion = nn.CrossEntropyLoss() if classification else nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    eval_freq = args.eval_freq
    training_seen = 0
    best_train_acc = 0
    best_test_acc = 0
    for epoch in range(args.num_epochs):  # loop over the dataset multiple times
        acc, loss, other = train_epoch(
            epoch, train_loader, model, optimizer, criterion, device, classification=classification
        )
        print(f"[T] E: {epoch} | Acc: {acc*100:.3f}% | loss: {loss:.5f}")
        training_seen += len(train_set)

        # Log train
        wandb.log({"loss": loss, "acc": acc*100, "epoch": epoch, "training_seen": training_seen,
                   **other})
        best_train_acc = max(acc, best_train_acc)
        wandb.run.summary["train_accuracy"] = best_train_acc

        if (epoch + 1) % eval_freq == 0 and len(val_set):
            acc, loss, other = eval_epoch(
                val_loader, model, criterion, device, classification=classification
            )
            print(f"\n[E] E: {epoch} | Acc: {acc * 100:.3f}% | loss: {loss:.5f}\n")

            # Log train
            wandb.log({
                "e_loss": loss, "e_acc": acc * 100,
                "epoch": epoch, "training_seen": training_seen,
                **other
            })
            best_test_acc = max(acc, best_test_acc)
            wandb.run.summary["eval_accuracy"] = best_test_acc


